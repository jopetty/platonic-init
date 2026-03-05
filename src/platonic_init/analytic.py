from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AnalyticFitConfig, load_config
from .env import load_project_env


def _build_basis(n: int, cfg: AnalyticFitConfig) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    cols = [np.ones_like(x)]

    if cfg.basis_type in {"poly", "poly_exp"}:
        for p in range(1, cfg.poly_degree + 1):
            cols.append(x**p)

    if cfg.basis_type in {"exp", "poly_exp"}:
        for scale in cfg.exp_scales:
            cols.append(np.exp(-scale * x))
            cols.append(np.exp(scale * (x - 1.0)))

    basis = np.stack(cols, axis=1)
    return basis


def _fit_vector(vec: torch.Tensor, cfg: AnalyticFitConfig) -> tuple[np.ndarray, float]:
    y = vec.detach().cpu().numpy().astype(np.float64)
    basis = _build_basis(len(y), cfg)
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
            "poly_degree": cfg.poly_degree,
            "exp_scales": cfg.exp_scales,
            "basis_dim": int(_build_basis(n, cfg).shape[1]),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit analytic basis to principal components")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--subspace", type=str, default="artifacts/weight_subspace.pt")
    p.add_argument("--out", type=str, default="artifacts/analytic_subspace.pt")
    p.add_argument("--report-out", type=str, default="artifacts/analytic_fit_report.json")
    return p.parse_args()


def main() -> None:
    load_project_env()
    args = parse_args()
    exp_cfg = load_config(args.config)
    subspace = torch.load(args.subspace, map_location="cpu")
    analytic, report = fit_analytic_subspace(subspace, exp_cfg.analytic_fit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(analytic, out_path)

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
