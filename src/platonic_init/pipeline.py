from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .analytic import fit_analytic_subspace
from .analyze import _load_state_dict, build_summary, tensorwise_pca
from .config import AnalyticFitBlockConfig, load_config, resolve_analytic_fit_blocks
from .eval_init import run_variant
from .data import build_tokenizer, load_init_eval_datasets
from .env import load_project_env
from .rebasin import align_states_for_pca
from .paths import (
    analysis_artifacts_dir,
    basis_sweep_dir,
    experiment_artifacts_dir,
    pretraining_artifacts_dir,
    pretraining_init_eval_basis_root,
    prepretraining_seed_dir,
)
from .train import sweep

STAGE_PREPRETRAIN = "prepretrain"
STAGE_FIT_INITIALIZATIONS = "fit_initializations"
STAGE_PRETRAIN = "pretrain"
ALL_STAGES = [STAGE_PREPRETRAIN, STAGE_FIT_INITIALIZATIONS, STAGE_PRETRAIN]


def _default_checkpoint_dirs(cfg) -> list[Path]:
    return [prepretraining_seed_dir(cfg, seed) for seed in cfg.sweep.seeds]


def _infer_num_attention_heads(model_dir: Path) -> int | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    for key in ("n_head", "num_attention_heads"):
        if key in raw:
            try:
                value = int(raw[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def _stage_plan(stages: list[str]) -> tuple[bool, bool, bool]:
    run_prepretrain = STAGE_PREPRETRAIN in stages
    run_fit_initializations = STAGE_FIT_INITIALIZATIONS in stages
    run_pretrain = STAGE_PRETRAIN in stages
    return run_prepretrain, run_fit_initializations, run_pretrain


def _fit_block_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Invalid analytic fit block name: {name!r}")
    return slug


def _selected_fit_blocks(cfg: Any, args: argparse.Namespace) -> list[AnalyticFitBlockConfig]:
    all_blocks = list(resolve_analytic_fit_blocks(cfg))
    if not all_blocks:
        raise ValueError("No analytic fit blocks configured")

    names = [b.name for b in all_blocks]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate analytic fit block names in config: {names}")

    slugs = [_fit_block_slug(b.name) for b in all_blocks]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"Analytic fit block names collide after slugify: {names}")

    selected = args.fit_names if args.fit_names is not None else args.basis
    if not selected:
        return all_blocks

    selected_set = set(selected)
    unknown = [name for name in selected if name not in set(names)]
    if unknown:
        raise ValueError(f"Unknown fit names requested: {unknown}. Available: {names}")
    return [block for block in all_blocks if block.name in selected_set]


def _doctor_checks(cfg, args, run_fit_initializations: bool, run_pretrain: bool) -> list[str]:
    issues: list[str] = []
    if run_fit_initializations:
        missing_ckpts = [p for p in _default_checkpoint_dirs(cfg) if not p.exists()]
        if missing_ckpts:
            issues.append(f"Missing pre-pretraining checkpoints: {missing_ckpts}")
    if run_pretrain:
        if not args.skip_transfer:
            transfer_seed_path = prepretraining_seed_dir(cfg, args.transfer_seed)
            if not transfer_seed_path.exists():
                issues.append(f"Missing transfer checkpoint seed_{args.transfer_seed}: {transfer_seed_path}")
        if not run_fit_initializations:
            bs_dir = basis_sweep_dir(cfg, args.basis_dir)
            for block in _selected_fit_blocks(cfg, args):
                slug = _fit_block_slug(block.name)
                p = bs_dir / f"analytic_subspace_{slug}.pt"
                if not p.exists():
                    issues.append(
                        f"Missing analytic subspace for fit '{block.name}': {p} "
                        "(run fit_initializations stage first)"
                    )
    return issues


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end platonic initialization experiment pipeline")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="Pipeline stages to run (default: all stages)",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Downstream fine-tuning steps used to evaluate initialization quality",
    )
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument(
        "--init-mode",
        type=str,
        default="sampled",
        choices=["mean", "sampled"],
        help="Platonic init mode to use for analytic basis comparisons",
    )
    p.add_argument("--transfer-seed", type=int, default=0)
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument(
        "--fit-names",
        nargs="+",
        default=None,
        help="Optional subset of analytic fit block names from config to evaluate",
    )
    p.add_argument(
        "--basis",
        nargs="+",
        default=None,
        help="Legacy alias for --fit-names",
    )
    p.add_argument(
        "--basis-dir",
        type=str,
        default=None,
        help="Directory containing analytic_subspace_<basis>.pt files for pretrain stage reuse",
    )
    p.add_argument(
        "--curves-out",
        type=str,
        default=None,
        help="Optional override for init_eval_basis_curves.json output path",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Validate required inputs for selected stages and exit without running",
    )
    return p.parse_args()


def _fit_initializations_stage(
    cfg: Any,
    args: argparse.Namespace,
    analysis_artifacts: Path,
    basis_sweep_artifacts: Path,
) -> dict[str, dict]:
    ckpts = _default_checkpoint_dirs(cfg)
    missing = [p for p in ckpts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    states = [_load_state_dict(p) for p in ckpts]
    if cfg.rebasin.enabled:
        num_attention_heads = _infer_num_attention_heads(ckpts[0])
        states, rebasin_report = align_states_for_pca(
            states,
            max_iter=int(cfg.rebasin.max_iter),
            seed=int(cfg.rebasin.seed),
            num_attention_heads=num_attention_heads,
        )
        with (analysis_artifacts / "rebasin_report.json").open("w", encoding="utf-8") as f:
            json.dump(rebasin_report, f, indent=2)
    subspace = tensorwise_pca(states, cfg.analysis)

    subspace_path = analysis_artifacts / "weight_subspace.pt"
    torch.save(subspace, subspace_path)
    with (analysis_artifacts / "weight_subspace_summary.json").open("w", encoding="utf-8") as f:
        json.dump(build_summary(subspace), f, indent=2)

    fit_blocks = _selected_fit_blocks(cfg, args)
    basis_subspaces: dict[str, dict] = {}
    fit_manifest: dict[str, dict[str, str]] = {}
    for i, block in enumerate(fit_blocks):
        fit_name = block.name
        slug = _fit_block_slug(fit_name)
        fit_cfg = block.to_fit_config()
        basis_subspace, basis_report = fit_analytic_subspace(subspace, fit_cfg)
        torch.save(basis_subspace, basis_sweep_artifacts / f"analytic_subspace_{slug}.pt")
        with (basis_sweep_artifacts / f"analytic_fit_report_{slug}.json").open("w", encoding="utf-8") as f:
            json.dump(basis_report, f, indent=2)
        basis_subspaces[fit_name] = basis_subspace
        fit_manifest[fit_name] = {"slug": slug, "basis_type": fit_cfg.basis_type}

        if i == 0:
            # Backward-compatible single-file outputs; point to first configured fit block.
            analytic_path = analysis_artifacts / "analytic_subspace.pt"
            torch.save(basis_subspace, analytic_path)
            with (analysis_artifacts / "analytic_fit_report.json").open("w", encoding="utf-8") as f:
                json.dump(basis_report, f, indent=2)

    with (basis_sweep_artifacts / "fit_blocks.json").open("w", encoding="utf-8") as f:
        json.dump(fit_manifest, f, indent=2)
    return basis_subspaces


def _load_basis_subspaces_stage(cfg: Any, args: argparse.Namespace, basis_sweep_artifacts: Path) -> dict[str, dict]:
    basis_subspaces: dict[str, dict] = {}
    for block in _selected_fit_blocks(cfg, args):
        fit_name = block.name
        slug = _fit_block_slug(fit_name)
        analytic_path = basis_sweep_artifacts / f"analytic_subspace_{slug}.pt"
        if not analytic_path.exists():
            raise FileNotFoundError(
                f"Missing analytic subspace for fit '{fit_name}' at {analytic_path}. "
                "Run fit_initializations stage first or include 'fit_initializations' in --stages."
            )
        basis_subspaces[fit_name] = torch.load(analytic_path, map_location="cpu")
    return basis_subspaces


def _pretrain_stage(
    cfg: Any,
    args: argparse.Namespace,
    basis_subspaces: dict[str, dict],
    basis_sweep_artifacts: Path,
    pretraining_artifacts: Path,
) -> None:
    train_ds, eval_ds = load_init_eval_datasets(
        cfg=cfg.init_eval_data,
        default_local_path=cfg.data_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)

    eval_basis_root = pretraining_init_eval_basis_root(cfg)
    eval_basis_root.mkdir(parents=True, exist_ok=True)

    transfer_seed_path = prepretraining_seed_dir(cfg, args.transfer_seed)
    if not transfer_seed_path.exists():
        raise FileNotFoundError(f"Missing transfer checkpoint for seed {args.transfer_seed}: {transfer_seed_path}")
    transfer_model_path = str(transfer_seed_path)

    fit_names = [b.name for b in _selected_fit_blocks(cfg, args)]
    jobs: list[dict[str, Any]] = [
        {
            "label": "random",
            "variant": "random",
            "basis": None,
            "init_mode": "random",
            "out_name": "random",
            "run_name": f"{cfg.sweep.experiment_name}-init-eval-random",
            "analytic_subspace": None,
            "transfer_model_path": None,
        }
    ]
    platonic_variant = "platonic_mean" if args.init_mode == "mean" else "platonic_sampled"
    for fit_name in fit_names:
        jobs.append(
            {
                "label": fit_name,
                "variant": platonic_variant,
                "basis": fit_name,
                "init_mode": args.init_mode,
                "out_name": fit_name,
                "run_name": f"{cfg.sweep.experiment_name}-init-eval-{fit_name}-{args.init_mode}",
                "analytic_subspace": basis_subspaces[fit_name],
                "transfer_model_path": None,
            }
        )
    if not args.skip_transfer:
        jobs.append(
            {
                "label": f"weight_transfer_seed_{args.transfer_seed}",
                "variant": "weight_transfer",
                "basis": None,
                "init_mode": "transfer",
                "out_name": "weight_transfer",
                "run_name": f"{cfg.sweep.experiment_name}-init-eval-weight-transfer-seed{args.transfer_seed}",
                "analytic_subspace": None,
                "transfer_model_path": transfer_model_path,
            }
        )

    results = []
    top_bar = tqdm(total=len(jobs), desc="pretraining 0/0", position=0, leave=True, dynamic_ncols=True)
    for i, job in enumerate(jobs, start=1):
        label = job["label"]
        top_bar.set_description(f"pretraining {i - 1}/{len(jobs)} | init={label}")
        result = run_variant(
            variant=job["variant"],
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=eval_basis_root / job["out_name"],
            train_steps=args.eval_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=job["analytic_subspace"],
            latent_seed=args.seed + 100,
            latent_scale=1.0,
            report_to=cfg.training.report_to,
            run_name=job["run_name"],
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
            transfer_model_path=job["transfer_model_path"],
            embedding_transfer_model_path=transfer_model_path,
            step_progress_desc=f"steps | init={label}",
            step_progress_position=1,
        )
        result["label"] = label
        result["basis"] = job["basis"]
        result["init_mode"] = job["init_mode"]
        results.append(result)
        top_bar.update(1)
    top_bar.set_description(f"pretraining {len(jobs)}/{len(jobs)} | done")
    top_bar.close()

    with (pretraining_artifacts / "init_eval.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)
    curves_out = (
        Path(args.curves_out)
        if args.curves_out is not None
        else pretraining_artifacts / "init_eval_basis_curves.json"
    )
    curves_out.parent.mkdir(parents=True, exist_ok=True)
    with curves_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "basis_dir": str(basis_sweep_artifacts),
                "fit_names": fit_names,
                "init_mode": args.init_mode,
                "train_steps": args.eval_steps,
                "eval_every": args.eval_every,
                "seed": args.seed,
                "results": results,
            },
            f,
            indent=2,
        )


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)

    run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(args.stages)

    if args.doctor:
        issues = _doctor_checks(cfg, args, run_fit_initializations=run_fit_initializations, run_pretrain=run_pretrain)
        if issues:
            for issue in issues:
                print(f"[doctor] ERROR: {issue}")
            raise SystemExit(1)
        print("[doctor] OK")
        return

    if run_prepretrain:
        sweep(cfg)

    artifacts = experiment_artifacts_dir(cfg)
    analysis_artifacts = analysis_artifacts_dir(cfg)
    pretraining_artifacts = pretraining_artifacts_dir(cfg)
    artifacts.mkdir(parents=True, exist_ok=True)
    analysis_artifacts.mkdir(parents=True, exist_ok=True)
    pretraining_artifacts.mkdir(parents=True, exist_ok=True)

    basis_sweep_artifacts = basis_sweep_dir(cfg, args.basis_dir)
    basis_sweep_artifacts.mkdir(parents=True, exist_ok=True)
    basis_subspaces: dict[str, dict] = {}

    if run_fit_initializations:
        basis_subspaces = _fit_initializations_stage(
            cfg=cfg,
            args=args,
            analysis_artifacts=analysis_artifacts,
            basis_sweep_artifacts=basis_sweep_artifacts,
        )
    elif run_pretrain:
        basis_subspaces = _load_basis_subspaces_stage(cfg=cfg, args=args, basis_sweep_artifacts=basis_sweep_artifacts)

    if not run_pretrain:
        return

    _pretrain_stage(
        cfg=cfg,
        args=args,
        basis_subspaces=basis_subspaces,
        basis_sweep_artifacts=basis_sweep_artifacts,
        pretraining_artifacts=pretraining_artifacts,
    )


if __name__ == "__main__":
    main()
