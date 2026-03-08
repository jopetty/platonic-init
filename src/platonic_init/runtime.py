from __future__ import annotations

import importlib.util
import os
import platform
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def resolve_attn_implementation(prefer_flash_attention_2: bool) -> str | None:
    if not prefer_flash_attention_2:
        return None
    if platform.system() == "Darwin":
        return None
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("flash_attn") is None:
        return None
    return "flash_attention_2"


def model_kwargs(*, bf16: bool, prefer_flash_attention_2: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    attn_impl = resolve_attn_implementation(prefer_flash_attention_2)
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def build_model_from_config(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
    vocab_size: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
):
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    if vocab_size is not None:
        cfg.vocab_size = int(vocab_size)
    if bos_token_id is not None:
        cfg.bos_token_id = int(bos_token_id)
    if eos_token_id is not None:
        cfg.eos_token_id = int(eos_token_id)
    if pad_token_id is not None:
        cfg.pad_token_id = int(pad_token_id)
    return AutoModelForCausalLM.from_config(
        cfg,
        **model_kwargs(bf16=bf16, prefer_flash_attention_2=prefer_flash_attention_2),
    )


def load_pretrained_model(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
):
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs(bf16=bf16, prefer_flash_attention_2=prefer_flash_attention_2),
    )


def resolve_max_length(model: torch.nn.Module, block_size: int) -> int:
    max_length = int(block_size)
    model_ctx = getattr(model.config, "max_position_embeddings", None)
    if model_ctx is not None:
        max_length = min(max_length, int(model_ctx))
    return max_length


def scheduler_kwargs(
    *,
    warmup_steps: int | None,
    warmup_ratio: float,
    min_lr_rate: float,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": float(min_lr_rate)},
    }
    if warmup_steps is not None:
        kwargs["warmup_steps"] = int(warmup_steps)
    else:
        kwargs["warmup_ratio"] = float(warmup_ratio)
    return kwargs


def configure_wandb_env(
    *,
    report_to: list[str] | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    run_name: str | None = None,
    run_group: str | None = None,
) -> None:
    if not report_to or "wandb" not in report_to:
        return
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    if run_name:
        os.environ["WANDB_NAME"] = run_name
    if run_group:
        os.environ["WANDB_RUN_GROUP"] = run_group


def finish_wandb_run(report_to: list[str] | None) -> None:
    if not report_to or "wandb" not in report_to:
        return
    import wandb

    wandb.finish()
