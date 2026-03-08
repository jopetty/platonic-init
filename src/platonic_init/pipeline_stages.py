from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .analytic import fit_analytic_subspace
from .analyze import _load_state_dict, build_summary, tensorwise_pca
from .config import AnalyticFitBlockConfig, ExperimentConfig
from .data import build_tokenizer, load_init_eval_datasets
from .eval_init import run_variant
from .paths import basis_sweep_dir, pretraining_init_eval_basis_root, prepretraining_seed_dir
from .rebasin import align_states_for_pca

STAGE_PREPRETRAIN = "prepretrain"
STAGE_FIT_INITIALIZATIONS = "fit_initializations"
STAGE_PRETRAIN = "pretrain"
ALL_STAGES = [STAGE_PREPRETRAIN, STAGE_FIT_INITIALIZATIONS, STAGE_PRETRAIN]
MERGED_TRANSFER_STATE_NAME = "merged_rebasin_state.pt"


@dataclass(frozen=True)
class PretrainJob:
    label: str
    variant: str
    init_mode: str
    out_name: str
    run_name: str
    analytic_subspace: dict[str, object] | None = None
    transfer_state_dict: dict[str, torch.Tensor] | None = None
    transfer_model_path: str | None = None


def stage_plan(stages: list[str]) -> tuple[bool, bool, bool]:
    run_prepretrain = STAGE_PREPRETRAIN in stages
    run_fit_initializations = STAGE_FIT_INITIALIZATIONS in stages
    run_pretrain = STAGE_PRETRAIN in stages
    return run_prepretrain, run_fit_initializations, run_pretrain


def fit_block_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Invalid analytic fit block name: {name!r}")
    return slug


def selected_fit_blocks(cfg: ExperimentConfig, args: argparse.Namespace) -> list[AnalyticFitBlockConfig]:
    all_blocks = list(cfg.fit_blocks)
    if not all_blocks:
        raise ValueError("No analytic fit blocks configured")

    names = [block.name for block in all_blocks]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate analytic fit block names in config: {names}")

    slugs = [fit_block_slug(block.name) for block in all_blocks]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"Analytic fit block names collide after slugify: {names}")

    if not args.fit_names:
        return all_blocks

    selected = set(args.fit_names)
    unknown = [name for name in args.fit_names if name not in set(names)]
    if unknown:
        raise ValueError(f"Unknown fit names requested: {unknown}. Available: {names}")
    return [block for block in all_blocks if block.name in selected]


def run_fit_jobs(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "skip_fits", False))


def merge_results_by_label(existing: list[dict[str, object]], updated: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label: dict[str, dict[str, object]] = {}
    ordered_labels: list[str] = []
    for row in existing + updated:
        label = str(row.get("label", ""))
        if not label:
            continue
        if label not in by_label:
            ordered_labels.append(label)
        by_label[label] = row
    return [by_label[label] for label in ordered_labels]


def default_checkpoint_dirs(cfg: ExperimentConfig) -> list[Path]:
    return [prepretraining_seed_dir(cfg, seed) for seed in cfg.sweep.seeds]


def infer_num_attention_heads(model_dir: Path) -> int | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("n_head", "num_attention_heads"):
        value = raw.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def build_merged_state(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("Cannot merge empty state list")
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(states[0].keys()):
        ref = states[0][key]
        if not torch.is_tensor(ref):
            continue
        if not all(key in state and tuple(state[key].shape) == tuple(ref.shape) for state in states):
            continue
        if torch.is_floating_point(ref):
            stacked = torch.stack([state[key].detach().to(dtype=torch.float32, device="cpu") for state in states], dim=0)
            merged[key] = stacked.mean(dim=0).to(dtype=ref.dtype)
        else:
            merged[key] = ref.detach().clone()
    return merged


def doctor_checks(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    run_fit_initializations: bool,
    run_pretrain: bool,
) -> list[str]:
    issues: list[str] = []
    if run_fit_initializations:
        missing_ckpts = [path for path in default_checkpoint_dirs(cfg) if not path.exists()]
        if missing_ckpts:
            issues.append(f"Missing pre-pretraining checkpoints: {missing_ckpts}")
    if not run_pretrain:
        return issues
    if not args.skip_transfer:
        transfer_seed_path = prepretraining_seed_dir(cfg, args.transfer_seed)
        if not transfer_seed_path.exists():
            issues.append(f"Missing transfer checkpoint seed_{args.transfer_seed}: {transfer_seed_path}")
    if run_fit_initializations or not run_fit_jobs(args):
        return issues

    sweep_dir = basis_sweep_dir(cfg)
    for block in selected_fit_blocks(cfg, args):
        analytic_path = sweep_dir / f"analytic_subspace_{fit_block_slug(block.name)}.pt"
        if not analytic_path.exists():
            issues.append(f"Missing analytic subspace for fit '{block.name}': {analytic_path} (run fit_initializations stage first)")
    if not args.skip_transfer:
        merged_path = sweep_dir / MERGED_TRANSFER_STATE_NAME
        if not merged_path.exists():
            issues.append(f"Missing merged rebasin transfer state: {merged_path} (run fit_initializations stage first)")
    return issues


def fit_initializations_stage(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    analysis_artifacts: Path,
    basis_sweep_artifacts: Path,
) -> dict[str, dict[str, object]]:
    checkpoints = default_checkpoint_dirs(cfg)
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    states = [_load_state_dict(path) for path in checkpoints]
    if cfg.rebasin.enabled:
        states, rebasin_report = align_states_for_pca(
            states,
            max_iter=int(cfg.rebasin.max_iter),
            seed=int(cfg.rebasin.seed),
            num_attention_heads=infer_num_attention_heads(checkpoints[0]),
        )
        (analysis_artifacts / "rebasin_report.json").write_text(json.dumps(rebasin_report, indent=2), encoding="utf-8")

    merged_state = build_merged_state(states)
    torch.save(merged_state, basis_sweep_artifacts / MERGED_TRANSFER_STATE_NAME)

    subspace = tensorwise_pca(states, cfg.analysis)
    torch.save(subspace, analysis_artifacts / "weight_subspace.pt")
    (analysis_artifacts / "weight_subspace_summary.json").write_text(
        json.dumps(build_summary(subspace), indent=2),
        encoding="utf-8",
    )

    basis_subspaces: dict[str, dict[str, object]] = {}
    fit_manifest: dict[str, dict[str, str]] = {}
    for block in selected_fit_blocks(cfg, args):
        fit_cfg = block.to_fit_config()
        basis_subspace, basis_report = fit_analytic_subspace(subspace, fit_cfg)
        slug = fit_block_slug(block.name)
        torch.save(basis_subspace, basis_sweep_artifacts / f"analytic_subspace_{slug}.pt")
        (basis_sweep_artifacts / f"analytic_fit_report_{slug}.json").write_text(
            json.dumps(basis_report, indent=2),
            encoding="utf-8",
        )
        basis_subspaces[block.name] = basis_subspace
        fit_manifest[block.name] = {"slug": slug, "basis_type": fit_cfg.basis_type}

    (basis_sweep_artifacts / "fit_blocks.json").write_text(json.dumps(fit_manifest, indent=2), encoding="utf-8")
    return basis_subspaces


def load_basis_subspaces_stage(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    basis_sweep_artifacts: Path,
) -> dict[str, dict[str, object]]:
    basis_subspaces: dict[str, dict[str, object]] = {}
    for block in selected_fit_blocks(cfg, args):
        analytic_path = basis_sweep_artifacts / f"analytic_subspace_{fit_block_slug(block.name)}.pt"
        if not analytic_path.exists():
            raise FileNotFoundError(
                f"Missing analytic subspace for fit '{block.name}' at {analytic_path}. "
                "Run fit_initializations stage first or include 'fit_initializations' in --stages."
            )
        basis_subspaces[block.name] = torch.load(analytic_path, map_location="cpu")
    return basis_subspaces


def build_pretrain_jobs(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    basis_subspaces: dict[str, dict[str, object]],
    transfer_model_path: str | None,
    transfer_state_dict: dict[str, torch.Tensor] | None,
) -> list[PretrainJob]:
    jobs: list[PretrainJob] = []
    if not args.skip_random:
        jobs.append(
            PretrainJob(
                label="random",
                variant="random",
                init_mode="random",
                out_name="random",
                run_name=f"{cfg.sweep.experiment_name}-init-eval-random",
            )
        )

    if run_fit_jobs(args):
        platonic_variant = "platonic_mean" if args.init_mode == "mean" else "platonic_sampled"
        for block in selected_fit_blocks(cfg, args):
            jobs.append(
                PretrainJob(
                    label=block.name,
                    variant=platonic_variant,
                    init_mode=args.init_mode,
                    out_name=block.name,
                    run_name=f"{cfg.sweep.experiment_name}-init-eval-{block.name}-{args.init_mode}",
                    analytic_subspace=basis_subspaces[block.name],
                )
            )

    if not args.skip_transfer:
        if transfer_model_path is None or transfer_state_dict is None:
            raise ValueError("Transfer artifacts are required when transfer evaluation is enabled")
        jobs.append(
            PretrainJob(
                label="weight_transfer",
                variant="weight_transfer",
                init_mode="transfer",
                out_name="weight_transfer",
                run_name=f"{cfg.sweep.experiment_name}-init-eval-weight-transfer-seed{args.transfer_seed}",
                transfer_model_path=transfer_model_path,
                transfer_state_dict=transfer_state_dict,
            )
        )
    return jobs


def pretrain_stage(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    basis_subspaces: dict[str, dict[str, object]],
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

    transfer_model_path = None
    transfer_state_dict = None
    if not args.skip_transfer:
        transfer_seed_path = prepretraining_seed_dir(cfg, args.transfer_seed)
        if not transfer_seed_path.exists():
            raise FileNotFoundError(f"Missing transfer checkpoint for seed {args.transfer_seed}: {transfer_seed_path}")
        transfer_model_path = str(transfer_seed_path)
        merged_state_path = basis_sweep_artifacts / MERGED_TRANSFER_STATE_NAME
        if not merged_state_path.exists():
            raise FileNotFoundError(
                f"Missing merged rebasin transfer state: {merged_state_path}. "
                "Run fit_initializations stage first or include 'fit_initializations' in --stages."
            )
        transfer_state_dict = torch.load(merged_state_path, map_location="cpu")

    jobs = build_pretrain_jobs(
        cfg,
        args,
        basis_subspaces=basis_subspaces,
        transfer_model_path=transfer_model_path,
        transfer_state_dict=transfer_state_dict,
    )
    if not jobs:
        raise ValueError(
            "No pretrain initialization jobs selected. "
            "Unset --skip-random/--skip-fits/--skip-transfer to select at least one initialization."
        )

    eval_root = pretraining_init_eval_basis_root(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    top_bar = tqdm(total=len(jobs), desc="pretraining 0/0", position=0, leave=True, dynamic_ncols=True)
    for index, job in enumerate(jobs, start=1):
        top_bar.set_description(f"pretraining {index - 1}/{len(jobs)} | init={job.label}")
        result = run_variant(
            variant=job.variant,
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            out_dir=eval_root / job.out_name,
            train_steps=args.eval_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            analytic_subspace=job.analytic_subspace,
            latent_seed=args.seed + 100,
            latent_scale=1.0,
            report_to=cfg.training.report_to,
            run_name=job.run_name,
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
            transfer_model_path=job.transfer_model_path,
            transfer_state_dict=job.transfer_state_dict,
            embedding_transfer_model_path=transfer_model_path,
            step_progress_desc=f"steps | init={job.label}",
            step_progress_position=1,
            warmup_steps=cfg.training.warmup_steps,
            warmup_ratio=cfg.training.warmup_ratio,
            min_lr_rate=cfg.training.min_lr_rate,
            bf16=cfg.training.bf16,
            fp16=cfg.training.fp16,
            prefer_flash_attention_2=cfg.training.prefer_flash_attention_2,
        )
        result["label"] = job.label
        result["basis"] = job.label if job.analytic_subspace is not None else None
        result["init_mode"] = job.init_mode
        results.append(result)
        top_bar.update(1)
    top_bar.set_description(f"pretraining {len(jobs)}/{len(jobs)} | done")
    top_bar.close()

    init_eval_path = pretraining_artifacts / "init_eval.json"
    merged_results = results
    if init_eval_path.exists():
        try:
            payload = json.loads(init_eval_path.read_text(encoding="utf-8"))
            existing_results = payload.get("results", [])
            if isinstance(existing_results, list):
                merged_results = merge_results_by_label(existing_results, results)
        except Exception:
            pass
    init_eval_path.write_text(json.dumps({"results": merged_results}, indent=2), encoding="utf-8")

    curves_payload = {
        "config": args.config,
        "fit_names": [job.label for job in jobs if job.analytic_subspace is not None],
        "init_mode": args.init_mode,
        "train_steps": args.eval_steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "results": merged_results,
    }
    curves_out = pretraining_artifacts / "init_eval_basis_curves.json"
    if curves_out.exists():
        try:
            payload = json.loads(curves_out.read_text(encoding="utf-8"))
            existing_results = payload.get("results", [])
            if isinstance(existing_results, list):
                curves_payload["results"] = merge_results_by_label(existing_results, results)
        except Exception:
            pass
    curves_out.write_text(json.dumps(curves_payload, indent=2), encoding="utf-8")
