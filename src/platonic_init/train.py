from __future__ import annotations

import argparse
import importlib.util
import os
import platform
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from .config import ExperimentConfig, load_config
from .data import build_char_tokenizer_from_text, build_tokenizer, load_text_dataset
from .env import load_project_env
from .paths import prepretraining_root, prepretraining_seed_dir


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


def _build_model(config: ExperimentConfig, vocab_size: int, bos_token_id: int, eos_token_id: int, pad_token_id: int):
    base_cfg = AutoConfig.from_pretrained(config.training.model_name_or_path)
    base_cfg.vocab_size = int(vocab_size)
    base_cfg.bos_token_id = int(bos_token_id)
    base_cfg.eos_token_id = int(eos_token_id)
    base_cfg.pad_token_id = int(pad_token_id)
    model_kwargs = {}
    if config.training.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    attn_impl = _resolve_attn_implementation(config.training.prefer_flash_attention_2)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_config(base_cfg, **model_kwargs)
    return model


def _configure_wandb_env(config: ExperimentConfig) -> None:
    if "wandb" not in config.training.report_to:
        return
    if config.training.wandb_project:
        os.environ["WANDB_PROJECT"] = config.training.wandb_project
    if config.training.wandb_entity:
        os.environ["WANDB_ENTITY"] = config.training.wandb_entity


def run_single_seed(config: ExperimentConfig, seed: int, output_dir: str) -> Path:
    set_seed(seed)
    _configure_wandb_env(config)
    run_name = (
        f"{config.training.run_name}-prepretraining-seed{seed}"
        if config.training.run_name
        else f"{config.sweep.experiment_name}-prepretraining-seed{seed}"
    )
    if "wandb" in config.training.report_to:
        os.environ["WANDB_NAME"] = run_name
        os.environ["WANDB_RUN_GROUP"] = f"{config.sweep.experiment_name}-prepretraining"
    if config.training.prepretrain_char_tokenizer:
        tokenizer = build_char_tokenizer_from_text(config.data_path)
    else:
        tokenizer = build_tokenizer(config.training.model_name_or_path)
    dataset = load_text_dataset(config.data_path)
    model = _build_model(
        config,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.resize_token_embeddings(len(tokenizer))
    max_length = int(config.training.block_size)
    model_ctx = getattr(model.config, "max_position_embeddings", None)
    if model_ctx is not None:
        max_length = min(max_length, int(model_ctx))

    warmup_steps = config.training.warmup_steps
    if warmup_steps is None and config.training.max_steps is not None:
        warmup_steps = int(config.training.max_steps * config.training.warmup_ratio)
    scheduler_kwargs: dict[str, object] = {
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": float(config.training.min_lr_rate)},
    }
    if warmup_steps is not None:
        scheduler_kwargs["warmup_steps"] = int(warmup_steps)
    else:
        scheduler_kwargs["warmup_ratio"] = float(config.training.warmup_ratio)

    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=max_length,
        num_train_epochs=1,
        max_steps=config.training.max_steps if config.training.max_steps is not None else -1,
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
        **scheduler_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    if "wandb" in config.training.report_to:
        import wandb

        wandb.finish()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return Path(output_dir)


def sweep(config: ExperimentConfig) -> list[Path]:
    root = prepretraining_root(config)
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed in config.sweep.seeds:
        out = prepretraining_seed_dir(config, seed)
        out.mkdir(parents=True, exist_ok=True)
        run_single_seed(config, seed=seed, output_dir=str(out))
        outputs.append(out)
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pre-pretraining sweep over random seeds")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--seed", type=int, default=None, help="Run only one seed")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        out = args.output_dir
        if out is None:
            out = prepretraining_seed_dir(config, args.seed)
        run_single_seed(config, seed=args.seed, output_dir=str(out))
    else:
        sweep(config)


if __name__ == "__main__":
    main()
