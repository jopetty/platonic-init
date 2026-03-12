"""Shared environment and filesystem helpers used across the package.

This module centralizes repo-root discovery and experiment artifact path
construction so the training and pipeline workflows can read top-down without
switching between tiny helper files.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from .config import ExperimentConfig


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor that looks like the repository root."""

    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


def load_project_env(start: Path | None = None) -> Path | None:
    """Load `.env` from the repository root when it exists."""

    root = find_repo_root(start)
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        return env_path
    return None


def model_key(model_name_or_path: str) -> str:
    """Convert a model identifier into a filesystem-safe directory key."""

    return model_name_or_path.replace("/", "_")


def experiment_artifacts_dir(cfg: ExperimentConfig) -> Path:
    """Return the root artifact directory for one experiment."""

    return Path("artifacts") / "experiments" / cfg.sweep.experiment_name


def analysis_artifacts_dir(cfg: ExperimentConfig) -> Path:
    """Return the directory for analysis-stage outputs."""

    return experiment_artifacts_dir(cfg) / "analysis"


def pretraining_artifacts_dir(cfg: ExperimentConfig) -> Path:
    """Return the directory for downstream pretraining evaluation outputs."""

    return experiment_artifacts_dir(cfg) / "pretraining"


def dataset_cache_root(cfg: ExperimentConfig) -> Path:
    """Return the cache location for tokenized datasets."""

    return experiment_artifacts_dir(cfg) / "cache" / "datasets"


def prepretraining_root(cfg: ExperimentConfig) -> Path:
    """Return the root directory that stores seed checkpoints."""

    return (
        Path(cfg.sweep.output_root)
        / "prepretraining"
        / cfg.sweep.experiment_name
        / model_key(cfg.training.model_name_or_path)
    )


def prepretraining_seed_dir(cfg: ExperimentConfig, seed: int) -> Path:
    """Return the checkpoint directory for one pre-pretraining seed."""

    return prepretraining_root(cfg) / f"seed_{seed}"


def pretraining_init_eval_basis_root(cfg: ExperimentConfig) -> Path:
    """Return the output directory for initialization evaluation runs."""

    return Path("runs") / "pretraining" / cfg.sweep.experiment_name / "init_eval_basis"


def basis_sweep_dir(cfg: ExperimentConfig) -> Path:
    """Return the directory that stores analytic fit sweep artifacts."""

    return analysis_artifacts_dir(cfg) / "basis_sweep"
