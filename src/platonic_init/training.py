"""Training workflows for pre-pretraining and initialization evaluation.

This module groups the code that actually builds models, tokenizers, trainers,
and evaluation runs so the experiment lifecycle can be read in one place.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from .config import ExperimentConfig
from .data import (
    build_char_tokenizer_from_text,
    build_tokenizer,
    dataset_cache_key,
    load_or_create_tokenized_dataset,
    load_saved_tokenizer,
    load_text_dataset,
    tokenizer_cache_key,
)
from .initialization import apply_analytic_delta_init
from .support import dataset_cache_root, prepretraining_root, prepretraining_seed_dir


def resolve_attn_implementation(prefer_flash_attention_2: bool) -> str | None:
    """Return the best supported attention implementation for this runtime."""

    if not prefer_flash_attention_2:
        return None
    if platform.system() == "Darwin":
        return None
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("flash_attn") is None:
        return None
    try:
        importlib.import_module("flash_attn_2_cuda")
    except Exception as exc:
        warnings.warn(
            "Flash Attention 2 requested but unavailable in this runtime; "
            "falling back to default attention. "
            f"python={sys.version.split()[0]!s}, "
            f"torch={torch.__version__!s}, "
            f"torch_cuda={torch.version.cuda!s}, "
            f"import failed with: {exc!r}",
            stacklevel=2,
        )
        return None
    return "flash_attention_2"


def resolve_model_dtype(*, bf16: bool, fp16: bool) -> torch.dtype | None:
    """Return the configured runtime dtype for model construction/loading."""

    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def model_kwargs(
    *, bf16: bool, fp16: bool = False, prefer_flash_attention_2: bool
) -> dict[str, Any]:
    """Build common Hugging Face model-loading kwargs."""

    kwargs: dict[str, Any] = {}
    dtype = resolve_model_dtype(bf16=bf16, fp16=fp16)
    if dtype is not None:
        kwargs["dtype"] = dtype
    attn_impl = resolve_attn_implementation(prefer_flash_attention_2)
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def build_model_from_config(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    fp16: bool = False,
    prefer_flash_attention_2: bool = True,
    vocab_size: int | None = None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
):
    """Instantiate a causal LM from config, optionally overriding tokenizer shape."""

    cfg = AutoConfig.from_pretrained(model_name_or_path)
    dtype = resolve_model_dtype(bf16=bf16, fp16=fp16)
    if dtype is not None:
        cfg.torch_dtype = dtype
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
        **model_kwargs(
            bf16=bf16,
            fp16=fp16,
            prefer_flash_attention_2=prefer_flash_attention_2,
        ),
    )


def load_pretrained_model(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    fp16: bool = False,
    prefer_flash_attention_2: bool = True,
):
    """Load a pretrained causal LM with the repo's runtime defaults."""

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs(
            bf16=bf16,
            fp16=fp16,
            prefer_flash_attention_2=prefer_flash_attention_2,
        ),
    )


def build_initialized_state_dict(
    model_name_or_path: str,
    tokenizer,
    *,
    seed: int,
    bf16: bool = False,
    fp16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> dict[str, torch.Tensor]:
    """Instantiate a deterministic fresh model and return its state dict on CPU."""

    set_seed(seed)
    model = build_model_from_config(
        model_name_or_path,
        bf16=bf16,
        fp16=fp16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    model.resize_token_embeddings(len(tokenizer))
    return {
        key: value.detach().to(device="cpu").clone()
        for key, value in model.state_dict().items()
    }


def resolve_max_length(model: torch.nn.Module, block_size: int) -> int:
    """Clamp training sequence length to the model context window."""

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
    """Build scheduler arguments for TRL's `SFTConfig`."""

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
    """Populate W&B environment variables when W&B logging is enabled."""

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
    """Close the active W&B run when the current job uses W&B."""

    if not report_to or "wandb" not in report_to:
        return
    import wandb

    wandb.finish()


def summarize_model(model: torch.nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts for a model."""

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def log_model_summary(
    *,
    model: torch.nn.Module,
    model_name_or_path: str,
    report_to: list[str] | None,
    run_name: str | None = None,
) -> None:
    """Print and log a concise model summary at the start of training."""

    summary = summarize_model(model)
    total_params_m = summary["total_params"] / 1_000_000
    trainable_params_m = summary["trainable_params"] / 1_000_000
    prefix = f"[{run_name}] " if run_name else ""
    print(
        f"{prefix}Model: {model_name_or_path} | "
        f"params={summary['total_params']:,} ({total_params_m:.2f}M) | "
        f"trainable={summary['trainable_params']:,} ({trainable_params_m:.2f}M)",
        flush=True,
    )

    if not report_to or "wandb" not in report_to:
        return
    try:
        import wandb

        if wandb.run is None:
            return
        wandb.config.update(
            {
                "model_name_or_path": model_name_or_path,
                "model_total_params": summary["total_params"],
                "model_trainable_params": summary["trainable_params"],
            },
            allow_val_change=True,
        )
        wandb.log(
            {
                "model/total_params": summary["total_params"],
                "model/trainable_params": summary["trainable_params"],
            },
            step=0,
        )
    except Exception:
        pass


@dataclass(frozen=True)
class TransferProjectionAssets:
    """Embedding assets used to project pretrained tokens into a new tokenizer."""

    source_vocab: dict[str, int]
    input_embeddings: torch.Tensor
    output_embeddings: torch.Tensor | None


@dataclass(frozen=True)
class PretrainJob:
    """One downstream initialization-evaluation variant to execute."""

    label: str
    variant: str
    init_mode: str
    out_name: str
    run_name: str
    analytic_subspace: dict[str, object] | None = None
    transfer_state_dict: dict[str, torch.Tensor] | None = None
    transfer_model_path: str | None = None


def load_transfer_projection_assets(
    model_name_or_path: str,
    *,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> TransferProjectionAssets:
    """Load source embeddings and vocab for token-by-token projection."""

    source_model = load_pretrained_model(
        model_name_or_path,
        bf16=bf16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    source_tokenizer = load_saved_tokenizer(model_name_or_path)
    output_embeddings = source_model.get_output_embeddings()
    return TransferProjectionAssets(
        source_vocab=source_tokenizer.get_vocab(),
        input_embeddings=source_model.get_input_embeddings()
        .weight.detach()
        .to(device="cpu")
        .clone(),
        output_embeddings=(
            output_embeddings.weight.detach().to(device="cpu").clone()
            if output_embeddings is not None
            else None
        ),
    )


def copy_matching_weights(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copy non-embedding weights that have matching names and shapes."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in {"transformer.wte.weight", "lm_head.weight"}:
            continue
        if key in target_state and target_state[key].shape == value.shape:
            filtered[key] = value
    target.load_state_dict(filtered, strict=False)


def copy_matching_weights_from_state(
    source_state: dict[str, torch.Tensor], target: torch.nn.Module
) -> None:
    """Copy matching floating-point tensors from a plain state dict."""

    target_state = target.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key in {"transformer.wte.weight", "lm_head.weight"}:
            continue
        if (
            key in target_state
            and target_state[key].shape == value.shape
            and torch.is_floating_point(value)
        ):
            filtered[key] = value.detach().to(
                dtype=target_state[key].dtype, device=target_state[key].device
            )
    target.load_state_dict(filtered, strict=False)


def project_shared_token_embeddings(
    source_vocab: dict[str, int],
    source_embed: torch.Tensor,
    source_output_embed: torch.Tensor | None,
    target_model,
    target_tokenizer,
) -> int:
    """Copy embeddings for tokens shared between source and target vocabularies."""

    target_vocab = target_tokenizer.get_vocab()
    target_embed = target_model.get_input_embeddings().weight.data
    if source_embed.shape[1] != target_embed.shape[1]:
        return 0

    copied = 0
    for token, source_id in source_vocab.items():
        target_id = target_vocab.get(token)
        if target_id is None:
            continue
        if source_id >= source_embed.shape[0] or target_id >= target_embed.shape[0]:
            continue
        target_embed[target_id].copy_(source_embed[source_id])
        copied += 1

    out_embed = target_model.get_output_embeddings()
    if (
        out_embed is not None
        and source_output_embed is not None
        and source_output_embed.shape == target_embed.shape
    ):
        out_embed.weight.data.copy_(target_embed)
    return copied


def apply_prepretrain_projection(
    target_model,
    target_tokenizer,
    embedding_transfer_model_path: str | None,
    *,
    projection_assets: TransferProjectionAssets | None = None,
    copy_non_embedding_weights: bool = False,
    bf16: bool = False,
    prefer_flash_attention_2: bool = True,
) -> int:
    """Project available pretrained embeddings, optionally copying other weights too."""

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
                input_embeddings=source_model.get_input_embeddings()
                .weight.detach()
                .to(device="cpu")
                .clone(),
                output_embeddings=(
                    source_model.get_output_embeddings()
                    .weight.detach()
                    .to(device="cpu")
                    .clone()
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
                raise ValueError(
                    "copy_non_embedding_weights requires a loaded source model"
                )
            source_model = load_pretrained_model(
                embedding_transfer_model_path,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
        copy_matching_weights(source_model, target_model)

    return project_shared_token_embeddings(
        projection_assets.source_vocab,
        projection_assets.input_embeddings,
        projection_assets.output_embeddings,
        target_model,
        target_tokenizer,
    )


class TrainStepTqdmCallback(TrainerCallback):
    """Show a second progress bar for per-step downstream evaluation runs."""

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


def extract_eval_curve(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Extract just the evaluation-loss curve from TRL trainer logs."""

    curve: list[dict[str, float]] = []
    for entry in log_history:
        if "eval_loss" not in entry or entry["eval_loss"] is None:
            continue
        step = entry.get("step")
        if step is None:
            continue
        curve.append({"step": float(step), "eval_loss": float(entry["eval_loss"])})
    return curve


def extract_train_curve(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Extract train-side metrics such as loss and learning rate from trainer logs."""

    curve: list[dict[str, float]] = []
    for entry in log_history:
        if "loss" not in entry or entry["loss"] is None:
            continue
        step = entry.get("step")
        if step is None:
            continue
        point: dict[str, float] = {"step": float(step), "loss": float(entry["loss"])}
        if entry.get("learning_rate") is not None:
            point["learning_rate"] = float(entry["learning_rate"])
        if entry.get("grad_norm") is not None:
            point["grad_norm"] = float(entry["grad_norm"])
        curve.append(point)
    return curve


def run_variant(
    variant: str,
    model_name_or_path: str,
    tokenizer,
    train_ds: Dataset | IterableDataset,
    eval_ds: Dataset,
    out_dir: Path,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    block_size: int,
    seed: int,
    model_init_seed: int | None = None,
    analytic_subspace: dict[str, Any] | None = None,
    report_to: list[str] | None = None,
    run_name: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    eval_every: int | None = None,
    logging_steps: int | None = None,
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
    """Run one downstream initialization-evaluation job end-to-end."""

    init_seed = seed if model_init_seed is None else int(model_init_seed)
    set_seed(init_seed)
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
        fp16=fp16,
        prefer_flash_attention_2=prefer_flash_attention_2,
    )
    model.resize_token_embeddings(len(tokenizer))
    max_length = resolve_max_length(model, block_size)
    set_seed(seed)

    copied_embedding_rows = 0
    if variant == "weight_transfer":
        if transfer_state_dict is not None:
            copy_matching_weights_from_state(transfer_state_dict, model)
            copied_embedding_rows = apply_prepretrain_projection(
                target_model=model,
                target_tokenizer=tokenizer,
                embedding_transfer_model_path=embedding_transfer_model_path
                or transfer_model_path,
                projection_assets=embedding_transfer_assets,
                copy_non_embedding_weights=False,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )
        else:
            if transfer_model_path is None:
                raise ValueError(
                    "transfer_state_dict or transfer_model_path is required "
                    "for weight_transfer variant"
                )
            copied_embedding_rows = apply_prepretrain_projection(
                target_model=model,
                target_tokenizer=tokenizer,
                embedding_transfer_model_path=transfer_model_path,
                projection_assets=embedding_transfer_assets,
                copy_non_embedding_weights=True,
                bf16=bf16,
                prefer_flash_attention_2=prefer_flash_attention_2,
            )

    if variant == "platonic_delta":
        if analytic_subspace is None:
            raise ValueError("analytic_subspace is required for platonic variants")
        apply_analytic_delta_init(model, analytic_subspace)

    if variant != "weight_transfer":
        copied_embedding_rows = apply_prepretrain_projection(
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
        logging_strategy="steps",
        logging_steps=logging_steps or max(1, min(10, train_steps)),
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
            TrainStepTqdmCallback(
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
    log_model_summary(
        model=model,
        model_name_or_path=model_name_or_path,
        report_to=report_to,
        run_name=run_name,
    )
    initial_metrics = trainer.evaluate()
    train_result = trainer.train()
    final_metrics = trainer.evaluate()
    eval_curve = extract_eval_curve(trainer.state.log_history)
    train_curve = extract_train_curve(trainer.state.log_history)
    eval_curve.insert(
        0,
        {
            "step": 0.0,
            "eval_loss": float(initial_metrics.get("eval_loss", float("nan"))),
        },
    )
    eval_curve.append(
        {
            "step": float(trainer.state.global_step),
            "eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        }
    )

    deduped_curve: list[dict[str, float]] = []
    seen_steps: set[float] = set()
    for point in eval_curve:
        step = point["step"]
        if step in seen_steps:
            continue
        seen_steps.add(step)
        deduped_curve.append(point)
    deduped_curve.sort(key=lambda point: point["step"])

    eval_losses = [
        point["eval_loss"]
        for point in deduped_curve
        if point["eval_loss"] == point["eval_loss"]
    ]
    best_eval_loss = min(eval_losses) if eval_losses else float("nan")

    out = {
        "variant": variant,
        "train_loss": float(train_result.training_loss),
        "initial_eval_loss": float(initial_metrics.get("eval_loss", float("nan"))),
        "best_eval_loss": float(best_eval_loss),
        "final_eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        "eval_loss": float(final_metrics.get("eval_loss", float("nan"))),
        "eval_curve": deduped_curve,
        "train_curve": train_curve,
        "copied_embedding_rows": int(copied_embedding_rows),
    }
    finish_wandb_run(report_to)
    return out


def run_single_seed(config: ExperimentConfig, seed: int, output_dir: str) -> Path:
    """Run one synthetic pre-pretraining seed and save the resulting checkpoint."""

    set_seed(seed)
    run_name = (
        f"{config.training.run_name}-prepretraining-seed{seed}"
        if config.training.run_name
        else f"{config.sweep.experiment_name}-prepretraining-seed{seed}"
    )
    configure_wandb_env(
        report_to=config.training.report_to,
        wandb_project=config.training.wandb_project,
        wandb_entity=config.training.wandb_entity,
        run_name=run_name,
        run_group=f"{config.sweep.experiment_name}-prepretraining",
    )

    if config.training.prepretrain_char_tokenizer:
        tokenizer = build_char_tokenizer_from_text(config.data_path)
    else:
        tokenizer = build_tokenizer(config.training.model_name_or_path)

    dataset = load_or_create_tokenized_dataset(
        load_text_dataset(config.data_path),
        tokenizer,
        block_size=config.training.block_size,
        cache_dir=dataset_cache_root(config),
        cache_key=dataset_cache_key(
            "prepretrain",
            config.data_path,
            config.training.block_size,
            config.training.prepretrain_char_tokenizer,
            tokenizer_cache_key(tokenizer),
        ),
    )
    model = build_model_from_config(
        config.training.model_name_or_path,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        prefer_flash_attention_2=config.training.prefer_flash_attention_2,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.resize_token_embeddings(len(tokenizer))
    max_length = resolve_max_length(model, config.training.block_size)

    warmup_steps = config.training.warmup_steps
    if warmup_steps is None and config.training.max_steps is not None:
        warmup_steps = int(config.training.max_steps * config.training.warmup_ratio)

    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=max_length,
        num_train_epochs=1,
        max_steps=config.training.max_steps
        if config.training.max_steps is not None
        else -1,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        packing=config.training.pretrain_packing,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        report_to=config.training.report_to,
        run_name=run_name,
        seed=seed,
        **scheduler_kwargs(
            warmup_steps=warmup_steps,
            warmup_ratio=config.training.warmup_ratio,
            min_lr_rate=config.training.min_lr_rate,
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    log_model_summary(
        model=model,
        model_name_or_path=config.training.model_name_or_path,
        report_to=config.training.report_to,
        run_name=run_name,
    )
    trainer.train()
    finish_wandb_run(config.training.report_to)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return Path(output_dir)


def sweep(config: ExperimentConfig) -> list[Path]:
    """Run the configured pre-pretraining sweep across all seeds."""

    root = prepretraining_root(config)
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed in config.sweep.seeds:
        out = prepretraining_seed_dir(config, seed)
        out.mkdir(parents=True, exist_ok=True)
        run_single_seed(config, seed=seed, output_dir=str(out))
        outputs.append(out)
    return outputs
