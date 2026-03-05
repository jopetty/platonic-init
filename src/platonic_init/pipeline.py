from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analytic import fit_analytic_subspace
from .analyze import _load_state_dict, build_summary, tensorwise_pca
from .config import load_config
from .eval_init import run_variant
from .data import build_tokenizer, load_init_eval_datasets
from .env import load_project_env
from .train import sweep


def _default_checkpoint_dirs(cfg) -> list[Path]:
    root = (
        Path(cfg.sweep.output_root)
        / "prepretraining"
        / cfg.sweep.experiment_name
        / cfg.training.model_name_or_path.replace("/", "_")
    )
    return [root / f"seed_{seed}" for seed in cfg.sweep.seeds]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end platonic initialization experiment pipeline")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Downstream fine-tuning steps used to evaluate initialization quality",
    )
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)

    if not args.skip_train:
        sweep(cfg)

    ckpts = _default_checkpoint_dirs(cfg)
    missing = [p for p in ckpts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    states = [_load_state_dict(p) for p in ckpts]
    subspace = tensorwise_pca(states, cfg.analysis)

    artifacts = Path("artifacts") / "experiments" / cfg.sweep.experiment_name
    analysis_artifacts = artifacts / "analysis"
    pretraining_artifacts = artifacts / "pretraining"
    artifacts.mkdir(parents=True, exist_ok=True)
    analysis_artifacts.mkdir(parents=True, exist_ok=True)
    pretraining_artifacts.mkdir(parents=True, exist_ok=True)

    subspace_path = analysis_artifacts / "weight_subspace.pt"
    torch.save(subspace, subspace_path)
    with (analysis_artifacts / "weight_subspace_summary.json").open("w", encoding="utf-8") as f:
        json.dump(build_summary(subspace), f, indent=2)

    analytic_subspace, fit_report = fit_analytic_subspace(subspace, cfg.analytic_fit)
    analytic_path = analysis_artifacts / "analytic_subspace.pt"
    torch.save(analytic_subspace, analytic_path)
    with (analysis_artifacts / "analytic_fit_report.json").open("w", encoding="utf-8") as f:
        json.dump(fit_report, f, indent=2)

    if args.skip_eval:
        return

    train_ds, eval_ds = load_init_eval_datasets(
        cfg=cfg.init_eval_data,
        default_local_path=cfg.data_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)

    eval_root = Path("runs") / "pretraining" / cfg.sweep.experiment_name / "init_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    results = []
    variants = ["random", "platonic_mean", "platonic_sampled"]
    for variant in variants:
        result = run_variant(
            variant=variant,
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=eval_root / variant,
            train_steps=args.eval_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=analytic_subspace,
            latent_seed=args.seed + 100,
            latent_scale=1.0,
            report_to=cfg.training.report_to,
            run_name=f"{cfg.sweep.experiment_name}-init-eval-{variant}",
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
        )
        results.append(result)

    with (pretraining_artifacts / "init_eval.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)


if __name__ == "__main__":
    main()
