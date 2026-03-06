from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .data import build_tokenizer, load_init_eval_datasets, load_saved_tokenizer
from .env import load_project_env
from .init_fn import apply_platonic_init, sample_latent
from .paths import pretraining_init_eval_root, prepretraining_seed_dir


def _resolve_attn_implementation(prefer_flash_attention_2: bool) -> str | None:
    if not prefer_flash_attention_2:
        return None
    if platform.system() == "Darwin":
        return None
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("flash_attn") is None:
        return None
    return "flash_attention_2"


def _build_model(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
):
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    model_kwargs = {}
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    attn_impl = _resolve_attn_implementation(prefer_flash_attention_2)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    return AutoModelForCausalLM.from_config(cfg, **model_kwargs)


def _load_pretrained_model(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
):
    model_kwargs = {}
    if bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    attn_impl = _resolve_attn_implementation(prefer_flash_attention_2)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


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


def _project_shared_token_embeddings(source_model, source_tokenizer, target_model, target_tokenizer) -> int:
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    source_embed = source_model.get_input_embeddings().weight.data
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
    if out_embed is not None and out_embed.weight.shape == target_embed.shape:
        out_embed.weight.data.copy_(target_embed)
    return copied


def _apply_prepretrain_projection(
    target_model,
    target_tokenizer,
    embedding_transfer_model_path: str | None,
    *,
    copy_non_embedding_weights: bool = False,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> int:
    if embedding_transfer_model_path is None:
        return 0
    source_model = _load_pretrained_model(
        embedding_transfer_model_path,
        bf16=bf16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    source_tokenizer = load_saved_tokenizer(embedding_transfer_model_path)
    if copy_non_embedding_weights:
        _copy_matching_weights(source_model, target_model)
    copied = _project_shared_token_embeddings(source_model, source_tokenizer, target_model, target_tokenizer)
    return copied


def _configure_wandb_env(wandb_project: str | None, wandb_entity: str | None) -> None:
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity


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
    if report_to and "wandb" in report_to:
        _configure_wandb_env(wandb_project=wandb_project, wandb_entity=wandb_entity)
        if run_name:
            os.environ["WANDB_NAME"] = run_name
            os.environ["WANDB_RUN_GROUP"] = f"{run_name.rsplit('-', 1)[0]}-inits"
    model = _build_model(
        model_name_or_path,
        bf16=bf16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    model.resize_token_embeddings(len(tokenizer))
    max_length = int(block_size)
    model_ctx = getattr(model.config, "max_position_embeddings", None)
    if model_ctx is not None:
        max_length = min(max_length, int(model_ctx))

    copied_embedding_rows = 0
    if variant == "weight_transfer":
        if transfer_state_dict is not None:
            _copy_matching_weights_from_state(transfer_state_dict, model)
            copied_embedding_rows = _apply_prepretrain_projection(
                target_model=model,
                target_tokenizer=tokenizer,
                embedding_transfer_model_path=embedding_transfer_model_path or transfer_model_path,
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
            copy_non_embedding_weights=False,
            bf16=bf16,
            prefer_flash_attention_2=prefer_flash_attention_2,
        )

    scheduler_kwargs: dict[str, Any] = {
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": float(min_lr_rate)},
    }
    if warmup_steps is not None:
        scheduler_kwargs["warmup_steps"] = int(warmup_steps)
    else:
        scheduler_kwargs["warmup_ratio"] = float(warmup_ratio)

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
        **scheduler_kwargs,
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
    if report_to and "wandb" in report_to:
        import wandb

        wandb.finish()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare initialization strategies by downstream fine-tuning validation loss"
    )
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--analytic-subspace", type=str, default="artifacts/analytic_subspace.pt")
    p.add_argument("--out", type=str, default="artifacts/init_eval.json")
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-seed", type=int, default=123)
    p.add_argument("--latent-scale", type=float, default=1.0)
    p.add_argument("--transfer-model-path", type=str, default=None)
    p.add_argument("--include-transfer", action="store_true")
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate every N training steps during downstream fine-tuning",
    )
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)

    train_ds, eval_ds = load_init_eval_datasets(
        cfg=cfg.init_eval_data,
        default_local_path=cfg.data_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)

    analytic_subspace = None
    if Path(args.analytic_subspace).exists():
        import torch

        analytic_subspace = torch.load(args.analytic_subspace, map_location="cpu")

    out_dir = pretraining_init_eval_root(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["random"]
    if analytic_subspace is not None:
        variants += ["platonic_mean", "platonic_sampled"]
    if args.include_transfer:
        variants.append("weight_transfer")

    results = []
    default_transfer_ckpt = prepretraining_seed_dir(cfg, seed=cfg.sweep.seeds[0] if cfg.sweep.seeds else 0)
    embedding_transfer_model_path = str(default_transfer_ckpt) if default_transfer_ckpt.exists() else None
    for variant in variants:
        result = run_variant(
            variant=variant,
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=out_dir / variant,
            train_steps=args.train_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=analytic_subspace,
            latent_seed=args.latent_seed,
            latent_scale=args.latent_scale,
            report_to=cfg.training.report_to,
            run_name=f"{cfg.sweep.experiment_name}-init-eval-{variant}",
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
            transfer_model_path=args.transfer_model_path,
            embedding_transfer_model_path=embedding_transfer_model_path,
        )
        results.append(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
