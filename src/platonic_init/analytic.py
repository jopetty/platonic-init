from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .basis import build_basis_numpy
from .config import AnalyticFitBlockConfig, AnalyticFitConfig, ExperimentConfig, load_config
from .env import load_project_env
from .paths import basis_sweep_dir


def _basis_params(cfg: AnalyticFitConfig) -> dict[str, Any]:
    return {
        "poly_degree": int(cfg.poly_degree),
        "exp_scales": [float(x) for x in cfg.exp_scales],
        "chebyshev_degree": int(cfg.chebyshev_degree),
        "fourier_degree": int(cfg.fourier_degree),
        "rbf_num_centers": int(cfg.rbf_num_centers),
        "rbf_sigma": float(cfg.rbf_sigma),
    }


def _fit_vector(vec: torch.Tensor, cfg: AnalyticFitConfig) -> tuple[np.ndarray, float]:
    y = vec.detach().cpu().numpy().astype(np.float64)
    basis = build_basis_numpy(len(y), cfg.basis_type, **_basis_params(cfg))
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    recon = basis @ coeffs
    rel_error = float(np.linalg.norm(recon - y) / (np.linalg.norm(y) + 1e-12))
    return coeffs, rel_error


def fit_analytic_subspace(
    subspace: dict[str, Any],
    cfg: AnalyticFitConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out: dict[str, Any] = {}
    report: dict[str, Any] = {"tensors": {}, "mean_relative_error": 0.0}
    all_errors = []

    for key, entry in subspace.items():
        mean = entry["mean"]
        components = entry["components"]
        n = int(mean.numel())

        comp_coeffs = []
        comp_errors = []
        for i in range(components.shape[0]):
            coeffs, err = _fit_vector(components[i], cfg)
            comp_coeffs.append(torch.from_numpy(coeffs).to(torch.float32))
            comp_errors.append(err)
            all_errors.append(err)

        out[key] = {
            "shape": entry["shape"],
            "numel": entry["numel"],
            "mean": mean,
            "basis_type": cfg.basis_type,
            "basis_params": _basis_params(cfg),
            "basis_dim": int(build_basis_numpy(n, cfg.basis_type, **_basis_params(cfg)).shape[1]),
            "component_coeffs": comp_coeffs,
            "explained_variance": entry["explained_variance"],
            "explained_variance_ratio": entry["explained_variance_ratio"],
        }
        report["tensors"][key] = {
            "component_relative_errors": comp_errors,
            "mean_component_relative_error": float(np.mean(comp_errors)) if comp_errors else 0.0,
        }

    report["mean_relative_error"] = float(np.mean(all_errors)) if all_errors else 0.0
    return out, report


def _fit_block_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    if not slug:
        raise ValueError(f"Invalid analytic fit block name: {name!r}")
    return slug


def _resolve_fit_block(cfg: ExperimentConfig, fit_name: str | None) -> AnalyticFitBlockConfig:
    blocks = list(cfg.fit_blocks)
    if not blocks:
        raise ValueError("Config must define at least one fit block under stages.fit_initializations.fit_blocks")
    if fit_name is None:
        if len(blocks) != 1:
            names = [block.name for block in blocks]
            raise ValueError(f"Multiple fit blocks configured; pass --fit-name explicitly. Available: {names}")
        return blocks[0]
    for block in blocks:
        if block.name == fit_name:
            return block
    names = [block.name for block in blocks]
    raise ValueError(f"Unknown fit block {fit_name!r}. Available: {names}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit analytic basis to principal components")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--subspace", type=str, default=None)
    p.add_argument("--fit-name", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--report-out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    exp_cfg = load_config(args.config)
    fit_block = _resolve_fit_block(exp_cfg, args.fit_name)
    fit_slug = _fit_block_slug(fit_block.name)
    sweep_dir = basis_sweep_dir(exp_cfg)
    subspace_path = Path(args.subspace) if args.subspace is not None else sweep_dir.parent / "weight_subspace.pt"
    subspace = torch.load(subspace_path, map_location="cpu")
    analytic, report = fit_analytic_subspace(subspace, fit_block.to_fit_config())

    out_path = Path(args.out) if args.out is not None else sweep_dir / f"analytic_subspace_{fit_slug}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(analytic, out_path)

    report_path = (
        Path(args.report_out) if args.report_out is not None else sweep_dir / f"analytic_fit_report_{fit_slug}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
