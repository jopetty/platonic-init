from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from .config import ExperimentConfig, load_config
from .data import build_tokenizer, load_text_dataset


def _build_model(config: ExperimentConfig):
    base_cfg = AutoConfig.from_pretrained(config.training.model_name_or_path)
    model = AutoModelForCausalLM.from_config(base_cfg)
    return model


def run_single_seed(config: ExperimentConfig, seed: int, output_dir: str) -> Path:
    set_seed(seed)
    tokenizer = build_tokenizer(config.training.model_name_or_path)
    dataset = load_text_dataset(config.data_path)
    model = _build_model(config)
    model.resize_token_embeddings(len(tokenizer))

    args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=config.training.block_size,
        num_train_epochs=1,
        max_steps=config.training.max_steps,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        report_to=[],
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return Path(output_dir)


def sweep(config: ExperimentConfig) -> list[Path]:
    root = (
        Path(config.sweep.output_root)
        / config.sweep.experiment_name
        / config.training.model_name_or_path.replace("/", "_")
    )
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for seed in config.sweep.seeds:
        out = root / f"seed_{seed}"
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
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        out = args.output_dir
        if out is None:
            out = (
                Path(config.sweep.output_root)
                / config.sweep.experiment_name
                / config.training.model_name_or_path.replace("/", "_")
                / f"seed_{args.seed}"
            )
        run_single_seed(config, seed=args.seed, output_dir=str(out))
    else:
        sweep(config)


if __name__ == "__main__":
    main()
