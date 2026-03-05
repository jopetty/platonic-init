from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig


def model_key(model_name_or_path: str) -> str:
    return model_name_or_path.replace("/", "_")


def experiment_artifacts_dir(cfg: ExperimentConfig) -> Path:
    return Path("artifacts") / "experiments" / cfg.sweep.experiment_name


def analysis_artifacts_dir(cfg: ExperimentConfig) -> Path:
    return experiment_artifacts_dir(cfg) / "analysis"


def pretraining_artifacts_dir(cfg: ExperimentConfig) -> Path:
    return experiment_artifacts_dir(cfg) / "pretraining"


def prepretraining_root(cfg: ExperimentConfig) -> Path:
    return (
        Path(cfg.sweep.output_root)
        / "prepretraining"
        / cfg.sweep.experiment_name
        / model_key(cfg.training.model_name_or_path)
    )


def prepretraining_seed_dir(cfg: ExperimentConfig, seed: int) -> Path:
    return prepretraining_root(cfg) / f"seed_{seed}"


def pretraining_init_eval_root(cfg: ExperimentConfig) -> Path:
    return Path("runs") / "pretraining" / cfg.sweep.experiment_name / "init_eval"


def pretraining_init_eval_basis_root(cfg: ExperimentConfig) -> Path:
    return Path("runs") / "pretraining" / cfg.sweep.experiment_name / "init_eval_basis"


def basis_sweep_dir(cfg: ExperimentConfig, override: str | None = None) -> Path:
    if override is not None:
        return Path(override)
    return analysis_artifacts_dir(cfg) / "basis_sweep"
