from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from tqdm import tqdm
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from .data import load_saved_tokenizer
from .init_fn import apply_platonic_init, sample_latent
from .runtime import (
    build_model_from_config,
    configure_wandb_env,
    finish_wandb_run,
    load_pretrained_model,
    resolve_max_length,
    scheduler_kwargs,
)


@dataclass(frozen=True)
class TransferProjectionAssets:
    source_vocab: dict[str, int]
    input_embeddings: torch.Tensor
    output_embeddings: torch.Tensor | None


def load_transfer_projection_assets(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> TransferProjectionAssets:
    source_model = load_pretrained_model(
        model_name_or_path,
        bf16=bf16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    source_tokenizer = load_saved_tokenizer(model_name_or_path)
    output_embeddings = source_model.get_output_embeddings()
    return TransferProjectionAssets(
        source_vocab=source_tokenizer.get_vocab(),
        input_embeddings=source_model.get_input_embeddings().weight.detach().to(device="cpu").clone(),
        output_embeddings=(
            output_embeddings.weight.detach().to(device="cpu").clone() if output_embeddings is not None else None
        ),
    )


def _copy_matching_weights(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in source_state.items():
        if k in {"transformer.wte.weight", "lm_head.weight"}:
            continue
        if k in target_state and target_state[k].shape == v.shape:
            filtered[k] = v
    target.load_state_dict(filtered, strict=False)


def _copy_matching_weights_from_state(source_state: dict[str, torch.Tensor], target: torch.nn.Module) -> None:
    target_state = target.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in source_state.items():
        if k in {"transformer.wte.weight", "lm_head.weight"}:
            continue
        if k in target_state and target_state[k].shape == v.shape and torch.is_floating_point(v):
            filtered[k] = v.detach().to(dtype=target_state[k].dtype, device=target_state[k].device)
    target.load_state_dict(filtered, strict=False)


def _project_shared_token_embeddings(
    source_vocab: dict[str, int],
    source_embed: torch.Tensor,
    source_output_embed: torch.Tensor | None,
    target_model,
    target_tokenizer,
) -> int:
    target_vocab = target_tokenizer.get_vocab()
    target_embed = target_model.get_input_embeddings().weight.data
    if source_embed.shape[1] != target_embed.shape[1]:
        return 0

    copied = 0
    for token, sid in source_vocab.items():
        tid = target_vocab.get(token)
        if tid is None:
            continue
        if sid >= source_embed.shape[0] or tid >= target_embed.shape[0]:
            continue
        target_embed[tid].copy_(source_embed[sid])
        copied += 1

    out_embed = target_model.get_output_embeddings()
    if out_embed is not None and source_output_embed is not None and source_output_embed.shape == target_embed.shape:
        out_embed.weight.data.copy_(target_embed)
    return copied


def _apply_prepretrain_projection(
    target_model,
    target_tokenizer,
    embedding_transfer_model_path: str | None,
    *,
    projection_assets: TransferProjectionAssets | None = None,
    copy_non_embedding_weights: bool = False,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> int:
    if projection_assets is None and embedding_transfer_model_path is None:
        return 0
    source_model = None
    if projection_assets is None:
        if embedding_transfer_model_path is None:
            return 0
        if copy_non_embedding_weights:
            source_model = load_pretrained_model(
                embedding_transfer_model_path,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
            source_tokenizer = load_saved_tokenizer(embedding_transfer_model_path)
            projection_assets = TransferProjectionAssets(
                source_vocab=source_tokenizer.get_vocab(),
                input_embeddings=source_model.get_input_embeddings().weight.detach().to(device="cpu").clone(),
                output_embeddings=(
                    source_model.get_output_embeddings().weight.detach().to(device="cpu").clone()
                    if source_model.get_output_embeddings() is not None
                    else None
                ),
            )
        else:
            projection_assets = load_transfer_projection_assets(
                embedding_transfer_model_path,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
    if copy_non_embedding_weights:
        if source_model is None:
            if embedding_transfer_model_path is None:
                raise ValueError("copy_non_embedding_weights requires a loaded source model")
            source_model = load_pretrained_model(
                embedding_transfer_model_path,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
        _copy_matching_weights(source_model, target_model)
    return _project_shared_token_embeddings(
        projection_assets.source_vocab,
        projection_assets.input_embeddings,
        projection_assets.output_embeddings,
        target_model,
        target_tokenizer,
    )


class _TrainStepTqdmCallback(TrainerCallback):
    def __init__(self, total_steps: int, desc: str, position: int = 1) -> None:
        self._total_steps = int(total_steps)
        self._desc = desc
        self._position = int(position)
        self._bar: tqdm | None = None
        self._last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_step = int(state.global_step or 0)
        self._bar = tqdm(
            total=self._total_steps,
            desc=self._desc,
            position=self._position,
            leave=False,
            dynamic_ncols=True,
        )
        if self._last_step > 0:
            self._bar.update(self._last_step)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._bar is None:
            return control
        current = int(state.global_step or 0)
        delta = current - self._last_step
        if delta > 0:
            self._bar.update(delta)
            self._last_step = current
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        return control


def _extract_eval_curve(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    curve: list[dict[str, float]] = []
    for entry in log_history:
        if "eval_loss" not in entry or entry["eval_loss"] is None:
            continue
        step = entry.get("step")
        if step is None:
            continue
        curve.append({"step": float(step), "eval_loss": float(entry["eval_loss"])})
    return curve


def run_variant(
    variant: str,
    model_name_or_path: str,
    tokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    out_dir: Path,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    block_size: int,
    seed: int,
    analytic_subspace: dict[str, Any] | None = None,
    latent_seed: int = 0,
    latent_scale: float = 1.0,
    report_to: list[str] | None = None,
    run_name: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    eval_every: int | None = None,
    transfer_model_path: str | None = None,
    transfer_state_dict: dict[str, torch.Tensor] | None = None,
    embedding_transfer_model_path: str | None = None,
    embedding_transfer_assets: TransferProjectionAssets | None = None,
    step_progress_desc: str | None = None,
    step_progress_position: int = 1,
    warmup_steps: int | None = 500,
    warmup_ratio: float = 0.03,
    min_lr_rate: float = 0.1,
    bf16: bool = False,
    fp16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    configure_wandb_env(
        report_to=report_to,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        run_name=run_name,
        run_group=f"{run_name.rsplit('-', 1)[0]}-inits" if run_name else None,
    )
    model = build_model_from_config(
        model_name_or_path,
        bf16=bf16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    model.resize_token_embeddings(len(tokenizer))
    max_length = resolve_max_length(model, block_size)

    copied_embedding_rows = 0
    if variant == "weight_transfer":
        if transfer_state_dict is not None:
            _copy_matching_weights_from_state(transfer_state_dict, model)
            copied_embedding_rows = _apply_prepretrain_projection(
                target_model=model,
                target_tokenizer=tokenizer,
                embedding_transfer_model_path=embedding_transfer_model_path or transfer_model_path,
                projection_assets=embedding_transfer_assets,
                copy_non_embedding_weights=False,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
        else:
            if transfer_model_path is None:
                raise ValueError("transfer_state_dict or transfer_model_path is required for weight_transfer variant")
            copied_embedding_rows = _apply_prepretrain_projection(
                target_model=model,
                target_tokenizer=tokenizer,
                embedding_transfer_model_path=transfer_model_path,
                projection_assets=embedding_transfer_assets,
                copy_non_embedding_weights=True,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )

    if variant.startswith("platonic"):
        if analytic_subspace is None:
            raise ValueError("analytic_subspace is required for platonic variants")
        latent = None
        if variant == "platonic_sampled":
            latent = sample_latent(analytic_subspace, seed=latent_seed)
        apply_platonic_init(model, analytic_subspace, latent=latent, latent_scale=latent_scale)
    if variant != "weight_transfer":
        copied_embedding_rows = _apply_prepretrain_projection(
            target_model=model,
            target_tokenizer=tokenizer,
            embedding_transfer_model_path=embedding_transfer_model_path,
            projection_assets=embedding_transfer_assets,
            copy_non_embedding_weights=False,
            bf16=bf16,
            prefer_flash_attention_2=prefer_flash_attention_2,
        )

    args = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
        max_length=max_length,
        num_train_epochs=1,
        max_steps=train_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        report_to=report_to or [],
        run_name=run_name,
        save_strategy="no",
        logging_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_every or max(10, train_steps // 5),
        seed=seed,
        bf16=bf16,
        fp16=fp16,
        disable_tqdm=True,
        log_level="error",
        log_level_replica="error",
        **scheduler_kwargs(
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            min_lr_rate=min_lr_rate,
        ),
    )
    callbacks = []
    if step_progress_desc is not None:
        callbacks.append(
            _TrainStepTqdmCallback(
                total_steps=train_steps,
                desc=step_progress_desc,
                position=step_progress_position,
            )
        )
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks or None,
    )
    initial_metrics = trainer.evaluate()
    train_result = trainer.train()
    final_metrics = trainer.evaluate()
    eval_curve = _extract_eval_curve(trainer.state.log_history)
    eval_curve.insert(0, {"step": 0.0, "eval_loss": float(initial_metrics.get("eval_loss", float("nan")))})
    eval_curve.append(
        {
            "step": float(trainer.state.global_step),
            "eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        }
    )
    # Keep the first observation per step to avoid duplicates from repeated evaluate() calls.
    deduped_curve: list[dict[str, float]] = []
    seen_steps: set[float] = set()
    for point in eval_curve:
        step = point["step"]
        if step in seen_steps:
            continue
        seen_steps.add(step)
        deduped_curve.append(point)
    deduped_curve.sort(key=lambda p: p["step"])

    eval_losses = [point["eval_loss"] for point in deduped_curve if point["eval_loss"] == point["eval_loss"]]
    best_eval_loss = min(eval_losses) if eval_losses else float("nan")

    out = {
        "variant": variant,
        "train_loss": float(train_result.training_loss),
        "initial_eval_loss": float(initial_metrics.get("eval_loss", float("nan"))),
        "best_eval_loss": float(best_eval_loss),
        "final_eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        # Backward-compatibility alias used by older analysis notebooks.
        "eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        "eval_curve": deduped_curve,
        "copied_embedding_rows": int(copied_embedding_rows),
    }
    finish_wandb_run(report_to)
    return out
