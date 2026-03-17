"""Custom optimizer helpers for experiment training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(
    grad: torch.Tensor, *, steps: int
) -> torch.Tensor:
    """Approximate the nearest orthogonal update with Newton-Schulz iterations."""

    if grad.ndim < 2:
        raise ValueError("Muon requires at least 2D parameters")

    a, b, c = (3.4445, -4.7750, 2.0315)
    update = grad.to(dtype=torch.bfloat16)
    transpose = update.size(-2) > update.size(-1)
    if transpose:
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(int(steps)):
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update
    if transpose:
        update = update.mT
    return update


def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    momentum: float,
    ns_steps: int,
    nesterov: bool,
) -> torch.Tensor:
    """Build one Muon update from a gradient and momentum buffer."""

    momentum_buffer.lerp_(grad, 1.0 - float(momentum))
    update = grad.lerp_(momentum_buffer, float(momentum)) if nesterov else momentum_buffer
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Hybrid optimizer: Muon on hidden matrices, AdamW-style updates elsewhere."""

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        normalized_groups: list[dict[str, Any]] = []
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("Each Muon param group must define use_muon")
            params = list(group["params"])
            if not params:
                continue
            normalized = dict(group)
            normalized["params"] = params
            normalized_groups.append(normalized)
        super().__init__(normalized_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        """Run one optimization step across both Muon and Adam groups."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                if not params:
                    continue
                pad_count = (-len(params)) % world_size
                params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad_count)]
                for base_index in range(0, len(params_pad), world_size):
                    param_index = base_index + rank
                    if param_index < len(params):
                        param = params[param_index]
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        state = self.state[param]
                        if not state:
                            state["momentum_buffer"] = torch.zeros_like(param)
                        update = muon_update(
                            param.grad,
                            state["momentum_buffer"],
                            momentum=group["momentum"],
                            ns_steps=group["ns_steps"],
                            nesterov=group["nesterov"],
                        )
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                        param.add_(update.reshape(param.shape), alpha=-group["lr"])
                    if distributed:
                        dist.all_gather(
                            params_pad[base_index : base_index + world_size],
                            params_pad[base_index + rank],
                        )
                continue

            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["step"] = 0
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = param.grad
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1.0 - beta2)
                bias_correction1 = 1.0 - beta1**state["step"]
                bias_correction2 = 1.0 - beta2**state["step"]
                update = (exp_avg / bias_correction1) / (
                    (exp_avg_sq / bias_correction2).sqrt() + group["eps"]
                )
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"])

        return loss


@dataclass(frozen=True)
class MuonOptimizerConfig:
    """Resolved Muon hyperparameters used to build optimizer groups."""

    adam_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    muon_learning_rate: float
    muon_momentum: float
    muon_ns_steps: int
    muon_nesterov: bool
    weight_decay: float


@dataclass(frozen=True)
class MuonParamPartition:
    """Partition of model parameters for Muon vs auxiliary AdamW updates."""

    muon_params: list[torch.nn.Parameter]
    decay_params: list[torch.nn.Parameter]
    no_decay_params: list[torch.nn.Parameter]


def partition_muon_params(
    model: torch.nn.Module,
    *,
    decay_parameter_names: set[str],
) -> MuonParamPartition:
    """Split model parameters into Muon and AdamW groups."""

    embedding_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_param_ids.update(id(param) for param in module.parameters(recurse=False))

    output_embedding = getattr(model, "get_output_embeddings", lambda: None)()
    if output_embedding is not None and hasattr(output_embedding, "weight"):
        embedding_param_ids.add(id(output_embedding.weight))

    muon_params: list[torch.nn.Parameter] = []
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if param.ndim >= 2 and id(param) not in embedding_param_ids:
            muon_params.append(param)
            continue
        if name in decay_parameter_names:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return MuonParamPartition(
        muon_params=muon_params,
        decay_params=decay_params,
        no_decay_params=no_decay_params,
    )


def build_muon_param_groups(
    model: torch.nn.Module,
    *,
    decay_parameter_names: set[str],
    config: MuonOptimizerConfig,
) -> list[dict[str, Any]]:
    """Create Muon + auxiliary AdamW parameter groups."""

    partition = partition_muon_params(
        model,
        decay_parameter_names=decay_parameter_names,
    )
    groups: list[dict[str, Any]] = []
    if partition.muon_params:
        groups.append(
            {
                "params": partition.muon_params,
                "lr": float(config.muon_learning_rate),
                "momentum": float(config.muon_momentum),
                "ns_steps": int(config.muon_ns_steps),
                "nesterov": bool(config.muon_nesterov),
                "weight_decay": float(config.weight_decay),
                "use_muon": True,
            }
        )
    if partition.decay_params:
        groups.append(
            {
                "params": partition.decay_params,
                "lr": float(config.adam_learning_rate),
                "betas": (float(config.adam_beta1), float(config.adam_beta2)),
                "eps": float(config.adam_epsilon),
                "weight_decay": float(config.weight_decay),
                "use_muon": False,
            }
        )
    if partition.no_decay_params:
        groups.append(
            {
                "params": partition.no_decay_params,
                "lr": float(config.adam_learning_rate),
                "betas": (float(config.adam_beta1), float(config.adam_beta2)),
                "eps": float(config.adam_epsilon),
                "weight_decay": 0.0,
                "use_muon": False,
            }
        )
    return groups
