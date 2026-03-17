"""Top-level experiment orchestration for platonic initialization runs.

`main()` is the single CLI entrypoint. The helper functions below keep each
stage readable: first resolve which stages to run, then validate inputs, then
execute synthetic pre-pretraining, analytic fitting, and downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from .config import AnalyticFitBlockConfig, ExperimentConfig, load_config
from .data import (
    build_tokenizer,
    dataset_cache_key,
    load_init_eval_datasets,
    load_or_create_tokenized_dataset,
    load_saved_tokenizer,
    tokenizer_cache_key,
)
from .initialization import (
    fit_analytic_delta,
    load_state_dict,
)
from .rebasin import align_states_for_pca
from .support import (
    analysis_artifacts_dir,
    basis_sweep_dir,
    dataset_cache_root,
    experiment_artifacts_dir,
    load_project_env,
    prepretraining_seed_dir,
    pretraining_artifacts_dir,
    pretraining_init_eval_basis_root,
)
from .training import (
    PretrainJob,
    build_initialized_state_dict,
    load_transfer_projection_assets,
    run_variant,
    sweep,
)

STAGE_PREPRETRAIN = "prepretrain"
STAGE_FIT_INITIALIZATIONS = "fit_initializations"
STAGE_PRETRAIN = "pretrain"
ALL_STAGES = [STAGE_PREPRETRAIN, STAGE_FIT_INITIALIZATIONS, STAGE_PRETRAIN]
MERGED_TRANSFER_STATE_NAME = "merged_rebasin_state.pt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the end-to-end pipeline."""

    parser = argparse.ArgumentParser(
        description="End-to-end platonic initialization experiment pipeline"
    )
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="Pipeline stages to run (default: all stages)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Downstream fine-tuning steps used to evaluate initialization quality",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--transfer-seed", type=int, default=None)
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-fits", action="store_true")
    parser.add_argument("--fit-names", nargs="+", default=None)
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Validate required inputs for selected stages and exit",
    )
    return parser.parse_args()


def stage_plan(stages: list[str]) -> tuple[bool, bool, bool]:
    """Return booleans for the three pipeline stages in execution order."""

    return (
        STAGE_PREPRETRAIN in stages,
        STAGE_FIT_INITIALIZATIONS in stages,
        STAGE_PRETRAIN in stages,
    )


def fit_block_slug(name: str) -> str:
    """Turn a fit-block label into a stable filename slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Invalid analytic fit block name: {name!r}")
    return slug


def selected_fit_blocks(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> list[AnalyticFitBlockConfig]:
    """Return the configured analytic fit blocks selected by CLI filters."""

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

    requested = set(args.fit_names)
    unknown = [name for name in args.fit_names if name not in set(names)]
    if unknown:
        raise ValueError(f"Unknown fit names requested: {unknown}. Available: {names}")
    return [block for block in all_blocks if block.name in requested]


def run_fit_jobs(args: argparse.Namespace) -> bool:
    """Return whether analytic fit jobs should be executed for this run."""

    return not bool(getattr(args, "skip_fits", False))


def merge_results_by_label(
    existing: list[dict[str, object]], updated: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Merge result rows by label while preserving their original order."""

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


def write_pretraining_summaries(
    *,
    pretraining_artifacts: Path,
    args: argparse.Namespace,
    jobs: list[PretrainJob],
    results: list[dict[str, object]],
) -> None:
    """Persist downstream pretraining summaries so partial progress is syncable."""

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
    init_eval_path.write_text(
        json.dumps({"results": merged_results}, indent=2), encoding="utf-8"
    )

    curves_payload = {
        "config": args.config,
        "fit_names": [job.label for job in jobs if job.analytic_subspace is not None],
        "init_mode": "delta",
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
                curves_payload["results"] = merge_results_by_label(
                    existing_results, results
                )
        except Exception:
            pass
    curves_out.write_text(json.dumps(curves_payload, indent=2), encoding="utf-8")


def default_checkpoint_dirs(cfg: ExperimentConfig) -> list[Path]:
    """Return expected seed checkpoint directories for the configured sweep."""

    return [prepretraining_seed_dir(cfg, seed) for seed in cfg.sweep.seeds]


def _seed_from_checkpoint_dir(path: Path) -> int | None:
    """Extract the numeric seed from a checkpoint directory name."""

    match = re.fullmatch(r"seed_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _candidate_prepretrain_loss(seed_dir: Path) -> float | None:
    """Return the best recorded pre-pretraining loss for one seed."""

    summary_path = seed_dir / "prepretrain_metrics.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("best_logged_loss", "train_loss", "training_loss"):
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

    trainer_state_paths = [
        seed_dir / "trainer_state.json",
        *seed_dir.glob("checkpoint-*/trainer_state.json"),
    ]
    best_loss: float | None = None
    for trainer_state_path in trainer_state_paths:
        if not trainer_state_path.exists():
            continue
        try:
            payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in payload.get("log_history", []):
            value = entry.get("loss")
            if value is None:
                continue
            try:
                loss = float(value)
            except (TypeError, ValueError):
                continue
            if best_loss is None or loss < best_loss:
                best_loss = loss
    return best_loss


def select_best_prepretraining_seed(cfg: ExperimentConfig) -> int:
    """Pick the pre-pretraining seed with the lowest recorded loss."""

    candidates: list[tuple[float, int]] = []
    for seed_dir in default_checkpoint_dirs(cfg):
        seed = _seed_from_checkpoint_dir(seed_dir)
        if seed is None or not seed_dir.exists():
            continue
        loss = _candidate_prepretrain_loss(seed_dir)
        if loss is None:
            continue
        candidates.append((loss, seed))
    if not candidates:
        raise FileNotFoundError(
            "Could not determine the best pre-pretraining seed from existing "
            "checkpoints. Expected trainer_state.json or prepretrain_metrics.json."
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def resolve_reference_init_seed(cfg: ExperimentConfig) -> int:
    """Return the configured reference seed or auto-select the best PPT seed."""

    configured = cfg.stages.fit_initializations.reference_init_seed
    if configured is not None:
        return int(configured)
    return select_best_prepretraining_seed(cfg)


def effective_batch_size(cfg: ExperimentConfig) -> int:
    """Return the effective batch size for one optimizer step."""

    return int(cfg.training.per_device_train_batch_size) * int(
        cfg.training.gradient_accumulation_steps
    )


def resolve_pretrain_eval_steps(
    cfg: ExperimentConfig, requested_steps: int | None
) -> int:
    """Scale downstream train steps to preserve token budget across batch sizes."""

    if requested_steps is not None:
        return int(requested_steps)

    base_steps = int(cfg.stages.pretrain_eval.train_steps)
    reference_batch = cfg.stages.pretrain_eval.reference_effective_batch_size
    if reference_batch is None:
        return base_steps

    current_batch = effective_batch_size(cfg)
    if current_batch <= 0:
        raise ValueError(f"Effective batch size must be positive, got {current_batch}")
    scaled_steps = round(base_steps * float(reference_batch) / float(current_batch))
    return max(1, int(scaled_steps))


def infer_num_attention_heads(model_dir: Path) -> int | None:
    """Infer GPT-style attention head count from a saved config file."""

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


def build_merged_state(
    states: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average matching floating-point tensors across aligned seed checkpoints."""

    if not states:
        raise ValueError("Cannot merge empty state list")

    merged: dict[str, torch.Tensor] = {}
    for key in sorted(states[0].keys()):
        ref = states[0][key]
        if not torch.is_tensor(ref):
            continue
        if not all(
            key in state and tuple(state[key].shape) == tuple(ref.shape)
            for state in states
        ):
            continue
        if torch.is_floating_point(ref):
            stacked = torch.stack(
                [
                    state[key].detach().to(dtype=torch.float32, device="cpu")
                    for state in states
                ],
                dim=0,
            )
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
    """Validate that all required inputs exist for the requested stages."""

    issues: list[str] = []
    selected_transfer_seed: int | None = None
    if not args.skip_transfer:
        try:
            selected_transfer_seed = (
                int(args.transfer_seed)
                if args.transfer_seed is not None
                else select_best_prepretraining_seed(cfg)
            )
        except FileNotFoundError as exc:
            issues.append(str(exc))
            selected_transfer_seed = None
    if run_fit_initializations:
        missing_ckpts = [
            path for path in default_checkpoint_dirs(cfg) if not path.exists()
        ]
        if missing_ckpts:
            issues.append(f"Missing pre-pretraining checkpoints: {missing_ckpts}")
    if not run_pretrain:
        return issues

    if not args.skip_transfer:
        if selected_transfer_seed is None:
            return issues
        transfer_seed_path = prepretraining_seed_dir(cfg, selected_transfer_seed)
        if not transfer_seed_path.exists():
            issues.append(
                "Missing transfer checkpoint "
                f"seed_{selected_transfer_seed}: {transfer_seed_path}"
            )
    if run_fit_initializations or not run_fit_jobs(args):
        return issues

    sweep_dir = basis_sweep_dir(cfg)
    for block in selected_fit_blocks(cfg, args):
        analytic_path = sweep_dir / f"analytic_subspace_{fit_block_slug(block.name)}.pt"
        if not analytic_path.exists():
            issues.append(
                f"Missing analytic subspace for fit '{block.name}': "
                f"{analytic_path} (run fit_initializations stage first)"
            )
    if not args.skip_transfer:
        merged_path = sweep_dir / MERGED_TRANSFER_STATE_NAME
        if not merged_path.exists():
            issues.append(
                "Missing merged rebasin transfer state: "
                f"{merged_path} (run fit_initializations stage first)"
            )
    return issues


def fit_initializations_stage(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    analysis_artifacts: Path,
    basis_sweep_artifacts: Path,
) -> dict[str, dict[str, object]]:
    """Align seed checkpoints and fit compact analytic deltas from a base init."""

    checkpoints = default_checkpoint_dirs(cfg)
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    states = [load_state_dict(path) for path in checkpoints]
    if cfg.rebasin.enabled:
        states, rebasin_report = align_states_for_pca(
            states,
            max_iter=int(cfg.rebasin.max_iter),
            seed=int(cfg.rebasin.seed),
            num_attention_heads=infer_num_attention_heads(checkpoints[0]),
        )
        (analysis_artifacts / "rebasin_report.json").write_text(
            json.dumps(rebasin_report, indent=2), encoding="utf-8"
        )

    merged_state = build_merged_state(states)
    torch.save(merged_state, basis_sweep_artifacts / MERGED_TRANSFER_STATE_NAME)
    reference_init_seed = resolve_reference_init_seed(cfg)
    reference_checkpoint = prepretraining_seed_dir(cfg, reference_init_seed)
    if not reference_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing reference checkpoint seed_{reference_init_seed}: "
            f"{reference_checkpoint}"
        )

    reference_tokenizer = load_saved_tokenizer(reference_checkpoint)
    reference_state = build_initialized_state_dict(
        str(reference_checkpoint),
        reference_tokenizer,
        seed=reference_init_seed,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        prefer_flash_attention_2=cfg.training.prefer_flash_attention_2,
    )

    basis_subspaces: dict[str, dict[str, object]] = {}
    fit_manifest: dict[str, dict[str, object]] = {}
    for block in selected_fit_blocks(cfg, args):
        fit_cfg = block.to_fit_config()
        basis_subspace, basis_report = fit_analytic_delta(
            reference_state,
            merged_state,
            fit_cfg,
            reference_init_seed=reference_init_seed,
        )
        slug = fit_block_slug(block.name)
        torch.save(
            basis_subspace, basis_sweep_artifacts / f"analytic_subspace_{slug}.pt"
        )
        (basis_sweep_artifacts / f"analytic_fit_report_{slug}.json").write_text(
            json.dumps(basis_report, indent=2),
            encoding="utf-8",
        )
        basis_subspaces[block.name] = basis_subspace
        fit_manifest[block.name] = {
            "slug": slug,
            "basis_type": fit_cfg.basis_type,
            "reference_init_seed": reference_init_seed,
            "artifact_type": "analytic_delta",
        }

    (basis_sweep_artifacts / "fit_blocks.json").write_text(
        json.dumps(fit_manifest, indent=2), encoding="utf-8"
    )
    return basis_subspaces


def load_basis_subspaces_stage(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    basis_sweep_artifacts: Path,
) -> dict[str, dict[str, object]]:
    """Load previously fit analytic initialization artifacts from disk."""

    basis_subspaces: dict[str, dict[str, object]] = {}
    for block in selected_fit_blocks(cfg, args):
        analytic_path = (
            basis_sweep_artifacts / f"analytic_subspace_{fit_block_slug(block.name)}.pt"
        )
        if not analytic_path.exists():
            raise FileNotFoundError(
                f"Missing analytic subspace for fit '{block.name}' at "
                f"{analytic_path}. Run fit_initializations stage first or "
                "include 'fit_initializations' in --stages."
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
    transfer_seed: int | None = None,
) -> list[PretrainJob]:
    """Build the list of downstream initialization variants to evaluate."""

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
        for block in selected_fit_blocks(cfg, args):
            jobs.append(
                PretrainJob(
                    label=block.name,
                    variant="platonic_delta",
                    init_mode="delta",
                    out_name=block.name,
                    run_name=(
                        f"{cfg.sweep.experiment_name}-init-eval-" f"{block.name}-delta"
                    ),
                    analytic_subspace=basis_subspaces[block.name],
                )
            )

    if not args.skip_transfer:
        if transfer_model_path is None:
            raise ValueError(
                "A transfer checkpoint path is required when transfer "
                "evaluation is enabled"
            )
        jobs.append(
            PretrainJob(
                label="weight_transfer",
                variant="weight_transfer",
                init_mode="transfer",
                out_name="weight_transfer",
                run_name=(
                    f"{cfg.sweep.experiment_name}-init-eval-"
                    f"weight-transfer-seed{transfer_seed}"
                ),
                transfer_model_path=transfer_model_path,
            )
        )
        if transfer_state_dict is not None:
            jobs.append(
                PretrainJob(
                    label="rebasin_weight_transfer",
                    variant="weight_transfer",
                    init_mode="rebasin_transfer",
                    out_name="rebasin_weight_transfer",
                    run_name=(
                        f"{cfg.sweep.experiment_name}-init-eval-"
                        f"rebasin-weight-transfer-seed{transfer_seed}"
                    ),
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
    """Run downstream initialization evaluation across the selected variants."""

    train_ds, eval_ds = load_init_eval_datasets(
        cfg=cfg.init_eval_data,
        default_local_path=cfg.data_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    tokenizer = build_tokenizer(cfg.training.model_name_or_path)
    cache_root = dataset_cache_root(cfg)
    tokenized_train_ds = load_or_create_tokenized_dataset(
        train_ds,
        tokenizer,
        block_size=cfg.training.block_size,
        cache_dir=cache_root,
        cache_key=dataset_cache_key(
            "init-eval-train",
            cfg.init_eval_data.source,
            cfg.init_eval_data.dataset_name,
            cfg.init_eval_data.dataset_config_name,
            cfg.init_eval_data.train_split,
            cfg.init_eval_data.text_field,
            cfg.init_eval_data.local_data_path or cfg.data_path,
            cfg.init_eval_data.max_train_samples,
            args.eval_ratio,
            args.seed,
            cfg.training.block_size,
            tokenizer_cache_key(tokenizer),
        ),
    )
    tokenized_eval_ds = load_or_create_tokenized_dataset(
        eval_ds,
        tokenizer,
        block_size=cfg.training.block_size,
        cache_dir=cache_root,
        cache_key=dataset_cache_key(
            "init-eval-eval",
            cfg.init_eval_data.source,
            cfg.init_eval_data.dataset_name,
            cfg.init_eval_data.dataset_config_name,
            cfg.init_eval_data.eval_split,
            cfg.init_eval_data.text_field,
            cfg.init_eval_data.local_data_path or cfg.data_path,
            cfg.init_eval_data.max_eval_samples,
            args.eval_ratio,
            args.seed,
            cfg.training.block_size,
            tokenizer_cache_key(tokenizer),
        ),
    )

    transfer_model_path = None
    transfer_projection_assets = None
    transfer_state_dict = None
    reference_init_seed = resolve_reference_init_seed(cfg)
    selected_transfer_seed: int | None = None
    if not args.skip_transfer:
        selected_transfer_seed = (
            int(args.transfer_seed)
            if args.transfer_seed is not None
            else select_best_prepretraining_seed(cfg)
        )
        transfer_seed_path = prepretraining_seed_dir(cfg, selected_transfer_seed)
        if not transfer_seed_path.exists():
            raise FileNotFoundError(
                f"Missing transfer checkpoint for seed "
                f"{selected_transfer_seed}: {transfer_seed_path}"
            )
        transfer_model_path = str(transfer_seed_path)
        transfer_projection_assets = load_transfer_projection_assets(
            transfer_model_path,
            bf16=cfg.training.bf16,
            fp16=cfg.training.fp16,
            prefer_flash_attention_2=cfg.training.prefer_flash_attention_2,
        )
        merged_transfer_state_path = basis_sweep_artifacts / MERGED_TRANSFER_STATE_NAME
        if merged_transfer_state_path.exists():
            transfer_state_dict = torch.load(
                merged_transfer_state_path, map_location="cpu"
            )

    jobs = build_pretrain_jobs(
        cfg,
        args,
        basis_subspaces=basis_subspaces,
        transfer_model_path=transfer_model_path,
        transfer_state_dict=transfer_state_dict,
        transfer_seed=selected_transfer_seed,
    )
    if not jobs:
        raise ValueError(
            "No pretrain initialization jobs selected. "
            "Unset --skip-random/--skip-fits/--skip-transfer to select at "
            "least one initialization."
        )

    eval_root = pretraining_init_eval_basis_root(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(
        "Downstream pretraining budget: "
        f"train_steps={args.eval_steps}, "
        f"effective_batch_size={effective_batch_size(cfg)}, "
        "reference_effective_batch_size="
        f"{cfg.stages.pretrain_eval.reference_effective_batch_size}",
        flush=True,
    )

    results: list[dict[str, object]] = []
    top_bar = tqdm(
        total=len(jobs),
        desc="pretraining 0/0",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for index, job in enumerate(jobs, start=1):
        top_bar.set_description(
            f"pretraining {index - 1}/{len(jobs)} | init={job.label}"
        )
        result = run_variant(
            variant=job.variant,
            model_name_or_path=cfg.training.model_name_or_path,
            tokenizer=tokenizer,
            train_ds=tokenized_train_ds,
            eval_ds=tokenized_eval_ds,
            out_dir=eval_root / job.out_name,
            train_steps=args.eval_steps,
            batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            learning_rate=cfg.training.learning_rate,
            block_size=cfg.training.block_size,
            seed=args.seed,
            model_init_seed=reference_init_seed,
            analytic_subspace=job.analytic_subspace,
            report_to=cfg.training.report_to,
            run_name=job.run_name,
            wandb_project=cfg.training.wandb_project,
            wandb_entity=cfg.training.wandb_entity,
            eval_every=args.eval_every,
            logging_steps=cfg.stages.pretrain_eval.logging_steps,
            transfer_model_path=job.transfer_model_path,
            transfer_state_dict=job.transfer_state_dict,
            embedding_transfer_model_path=transfer_model_path,
            embedding_transfer_assets=transfer_projection_assets,
            step_progress_desc=f"steps | init={job.label}",
            step_progress_position=1,
            warmup_steps=cfg.training.warmup_steps,
            warmup_ratio=cfg.training.warmup_ratio,
            min_lr_rate=cfg.training.min_lr_rate,
            optimizer_type=cfg.training.optimizer_type,
            weight_decay=cfg.training.weight_decay,
            adam_beta1=cfg.training.adam_beta1,
            adam_beta2=cfg.training.adam_beta2,
            adam_epsilon=cfg.training.adam_epsilon,
            max_grad_norm=cfg.training.max_grad_norm,
            muon_learning_rate=cfg.training.muon_learning_rate,
            muon_momentum=cfg.training.muon_momentum,
            muon_ns_steps=cfg.training.muon_ns_steps,
            muon_nesterov=cfg.training.muon_nesterov,
            bf16=cfg.training.bf16,
            fp16=cfg.training.fp16,
            prefer_flash_attention_2=cfg.training.prefer_flash_attention_2,
        )
        result["label"] = job.label
        result["basis"] = job.label if job.analytic_subspace is not None else None
        result["init_mode"] = job.init_mode
        results.append(result)
        write_pretraining_summaries(
            pretraining_artifacts=pretraining_artifacts,
            args=args,
            jobs=jobs,
            results=results,
        )
        top_bar.update(1)
    top_bar.set_description(f"pretraining {len(jobs)}/{len(jobs)} | done")
    top_bar.close()


def main() -> None:
    """Run the requested experiment stages."""

    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)
    args.eval_steps = resolve_pretrain_eval_steps(cfg, args.eval_steps)
    if args.eval_every is None:
        args.eval_every = int(cfg.stages.pretrain_eval.eval_every)

    run_prepretrain, run_fit_initializations, run_pretrain = stage_plan(args.stages)
    if args.doctor:
        issues = doctor_checks(
            cfg,
            args,
            run_fit_initializations=run_fit_initializations,
            run_pretrain=run_pretrain,
        )
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
    basis_sweep_artifacts = basis_sweep_dir(cfg)
    for path in (
        artifacts,
        analysis_artifacts,
        pretraining_artifacts,
        basis_sweep_artifacts,
    ):
        path.mkdir(parents=True, exist_ok=True)

    basis_subspaces: dict[str, dict] = {}
    if run_fit_initializations:
        basis_subspaces = fit_initializations_stage(
            cfg,
            args,
            analysis_artifacts=analysis_artifacts,
            basis_sweep_artifacts=basis_sweep_artifacts,
        )
    elif run_pretrain and run_fit_jobs(args):
        basis_subspaces = load_basis_subspaces_stage(
            cfg,
            args,
            basis_sweep_artifacts=basis_sweep_artifacts,
        )

    if run_pretrain:
        pretrain_stage(
            cfg,
            args,
            basis_subspaces=basis_subspaces,
            basis_sweep_artifacts=basis_sweep_artifacts,
            pretraining_artifacts=pretraining_artifacts,
        )


_doctor_checks = doctor_checks
_merge_results_by_label = merge_results_by_label
_stage_plan = stage_plan


if __name__ == "__main__":
    main()
