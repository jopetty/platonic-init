from __future__ import annotations

import argparse

from .config import load_config
from .env import load_project_env
from .paths import analysis_artifacts_dir, basis_sweep_dir, experiment_artifacts_dir, pretraining_artifacts_dir
from .pipeline_stages import (
    ALL_STAGES,
    doctor_checks,
    fit_initializations_stage,
    load_basis_subspaces_stage,
    merge_results_by_label as _merge_results_by_label,
    pretrain_stage,
    run_fit_jobs,
    stage_plan as _stage_plan,
)
from .train import sweep

_doctor_checks = doctor_checks


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
        default=None,
        help="Downstream fine-tuning steps used to evaluate initialization quality",
    )
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--init-mode", type=str, default="sampled", choices=["mean", "sampled"])
    p.add_argument("--transfer-seed", type=int, default=0)
    p.add_argument("--skip-transfer", action="store_true")
    p.add_argument("--skip-random", action="store_true")
    p.add_argument("--skip-fits", action="store_true")
    p.add_argument("--fit-names", nargs="+", default=None)
    p.add_argument("--doctor", action="store_true", help="Validate required inputs for selected stages and exit")
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    cfg = load_config(args.config)
    if args.eval_steps is None:
        args.eval_steps = int(cfg.stages.pretrain_eval.train_steps)
    if args.eval_every is None:
        args.eval_every = int(cfg.stages.pretrain_eval.eval_every)

    run_prepretrain, run_fit_initializations, run_pretrain = _stage_plan(args.stages)
    if args.doctor:
        issues = doctor_checks(cfg, args, run_fit_initializations=run_fit_initializations, run_pretrain=run_pretrain)
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
    for path in (artifacts, analysis_artifacts, pretraining_artifacts, basis_sweep_artifacts):
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


if __name__ == "__main__":
    main()
