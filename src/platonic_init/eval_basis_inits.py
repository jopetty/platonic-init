from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import build_tokenizer, load_init_eval_datasets
from .env import load_project_env
from .eval_init import run_variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare random vs basis-derived initializations on downstream validation loss curves"
    )
    p.add_argument("--config", type=str, default="configs/experiment_dyck_d10_5k_demo.yaml")
    p.add_argument("--basis-dir", type=str, default=None)
    p.add_argument("--basis", nargs="+", default=["chebyshev", "fourier", "rbf", "poly_exp"])
    p.add_argument(
        "--init-mode",
        type=str,
        default="sampled",
        choices=["mean", "sampled"],
        help="Which platonic variant to use for basis comparisons",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latent-seed", type=int, default=123)
    p.add_argument("--latent-scale", type=float, default=1.0)
    p.add_argument("--transfer-seed", type=int, default=0)
    p.add_argument("--skip-transfer", action="store_true")
    return p.parse_args()


def _default_transfer_checkpoint(cfg, seed: int) -> Path:
    preferred = (
        Path(cfg.sweep.output_root)
        / "prepretraining"
        / cfg.sweep.experiment_name
        / cfg.training.model_name_or_path.replace("/", "_")
        / f"seed_{seed}"
    )
    if preferred.exists():
        return preferred
    # Backward-compatible fallback for runs produced before layout migration.
    legacy = (
        Path(cfg.sweep.output_root)
        / cfg.sweep.experiment_name
        / cfg.training.model_name_or_path.replace("/", "_")
        / f"seed_{seed}"
    )
    return legacy


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

    out_dir = Path("runs") / "pretraining" / cfg.sweep.experiment_name / "init_eval_basis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    random_result = run_variant(
        variant="random",
        model_name_or_path=cfg.training.model_name_or_path,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        out_dir=out_dir / "random",
        train_steps=args.train_steps,
        batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        block_size=cfg.training.block_size,
        seed=args.seed,
        analytic_subspace=None,
        latent_seed=args.latent_seed,
        latent_scale=args.latent_scale,
        report_to=cfg.training.report_to,
        run_name=f"{cfg.sweep.experiment_name}-init-eval-random",
        wandb_project=cfg.training.wandb_project,
        wandb_entity=cfg.training.wandb_entity,
        eval_every=args.eval_every,
    )
    random_result["label"] = "random"
    random_result["basis"] = None
    results.append(random_result)

    basis_dir = (
        Path(args.basis_dir)
        if args.basis_dir is not None
        else Path("artifacts") / "experiments" / cfg.sweep.experiment_name / "analysis" / "basis_sweep"
    )
    platonic_variant = "platonic_mean" if args.init_mode == "mean" else "platonic_sampled"
    for basis in args.basis:
        analytic_path = basis_dir / f"analytic_subspace_{basis}.pt"
        if not analytic_path.exists():
            raise FileNotFoundError(f"Missing analytic subspace for basis '{basis}': {analytic_path}")
        analytic_subspace = torch.load(analytic_path, map_location="cpu")

        basis_result = run_variant(
            variant=platonic_variant,
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=out_dir / basis,
            train_steps=args.train_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=analytic_subspace,
            latent_seed=args.latent_seed,
            latent_scale=args.latent_scale,
            report_to=cfg.training.report_to,
            run_name=f"{cfg.sweep.experiment_name}-init-eval-{basis}-{args.init_mode}",
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
        )
        basis_result["label"] = basis
        basis_result["basis"] = basis
        basis_result["init_mode"] = args.init_mode
        results.append(basis_result)

    if not args.skip_transfer:
        transfer_model_path = _default_transfer_checkpoint(cfg, args.transfer_seed)
        if not transfer_model_path.exists():
            raise FileNotFoundError(
                f"Missing transfer checkpoint for seed {args.transfer_seed}: {transfer_model_path}"
            )
        transfer_result = run_variant(
            variant="weight_transfer",
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=out_dir / "weight_transfer",
            train_steps=args.train_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=None,
            latent_seed=args.latent_seed,
            latent_scale=args.latent_scale,
            report_to=cfg.training.report_to,
            run_name=f"{cfg.sweep.experiment_name}-init-eval-weight-transfer-seed{args.transfer_seed}",
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
            transfer_model_path=str(transfer_model_path),
        )
        transfer_result["label"] = f"weight_transfer_seed_{args.transfer_seed}"
        transfer_result["basis"] = None
        transfer_result["init_mode"] = "transfer"
        results.append(transfer_result)

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path("artifacts")
        / "experiments"
        / cfg.sweep.experiment_name
        / "pretraining"
        / "init_eval_basis_curves.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "basis_dir": str(basis_dir),
        "init_mode": args.init_mode,
        "train_steps": args.train_steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
