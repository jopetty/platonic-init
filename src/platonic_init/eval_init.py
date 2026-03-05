from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .data import build_tokenizer, load_text_dataset
from .init_fn import apply_platonic_init, sample_latent


def _make_splits(ds: Dataset, eval_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    split = ds.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def _build_model(model_name_or_path: str):
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    return AutoModelForCausalLM.from_config(cfg)


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
) -> dict[str, Any]:
    set_seed(seed)
    model = _build_model(model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    if variant.startswith("platonic"):
        if analytic_subspace is None:
            raise ValueError("analytic_subspace is required for platonic variants")
        latent = None
        if variant == "platonic_sampled":
            latent = sample_latent(analytic_subspace, seed=latent_seed)
        apply_platonic_init(model, analytic_subspace, latent=latent, latent_scale=latent_scale)

    args = SFTConfig(
        output_dir=str(out_dir),
        dataset_text_field="text",
        max_length=block_size,
        num_train_epochs=1,
        max_steps=train_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        report_to=[],
        save_strategy="no",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(10, train_steps // 5),
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()

    out = {
        "variant": variant,
        "train_loss": float(train_result.training_loss),
        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare random and platonic initialization")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--analytic-subspace", type=str, default="artifacts/analytic_subspace.pt")
    p.add_argument("--out", type=str, default="artifacts/init_eval.json")
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-seed", type=int, default=123)
    p.add_argument("--latent-scale", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset = load_text_dataset(cfg.data_path)
    train_ds, eval_ds = _make_splits(dataset, eval_ratio=args.eval_ratio, seed=args.seed)
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)

    analytic_subspace = None
    if Path(args.analytic_subspace).exists():
        import torch

        analytic_subspace = torch.load(args.analytic_subspace, map_location="cpu")

    out_dir = Path("runs") / "init_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["random"]
    if analytic_subspace is not None:
        variants += ["platonic_mean", "platonic_sampled"]

    results = []
    for i, variant in enumerate(variants):
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
            seed=args.seed + i,
            analytic_subspace=analytic_subspace,
            latent_seed=args.latent_seed,
            latent_scale=args.latent_scale,
        )
        results.append(result)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
