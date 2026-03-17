"""Weight-space mechanics for compact delta-based initialization artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm

from .config import AnalyticFitConfig


def load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint state dict from a Hugging Face model directory."""

    safe_path = model_dir / "model.safetensors"
    if safe_path.exists():
        return safe_load_file(str(safe_path))

    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            return state["state_dict"]
        return state

    raise FileNotFoundError(f"No checkpoint file found in {model_dir}")


def filter_float_tensor_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Return floating-point tensor keys that participate in delta fitting."""

    keys = []
    for key, value in state.items():
        if not torch.is_floating_point(value):
            continue
        if "lm_head" in key:
            continue
        keys.append(key)
    return sorted(keys)


def coerce_exp_scales(exp_scales: Iterable[float] | None) -> list[float]:
    """Normalize optional exponential scale parameters."""

    return [float(scale) for scale in (exp_scales or [0.5, 1.0, 2.0, 4.0])]


def build_basis_numpy(
    n: int,
    basis_type: str,
    *,
    poly_degree: int = 5,
    exp_scales: list[float] | None = None,
    chebyshev_degree: int = 12,
    fourier_degree: int = 8,
    rbf_num_centers: int = 8,
    rbf_sigma: float = 0.08,
) -> np.ndarray:
    """Construct a dense basis matrix for analytic fitting."""

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    cols: list[np.ndarray] = [np.ones_like(x)]
    exp_scales = coerce_exp_scales(exp_scales)

    if basis_type in {"poly", "poly_exp"}:
        for degree in range(1, poly_degree + 1):
            cols.append(x**degree)

    if basis_type in {"exp", "poly_exp"}:
        for scale in exp_scales:
            cols.append(np.exp(-scale * x))
            cols.append(np.exp(scale * (x - 1.0)))

    if basis_type == "chebyshev":
        x_cheb = 2.0 * x - 1.0
        t0 = np.ones_like(x_cheb)
        cols = [t0]
        if chebyshev_degree >= 1:
            t1 = x_cheb.copy()
            cols.append(t1)
            prev2, prev1 = t0, t1
            for _ in range(2, chebyshev_degree + 1):
                t_next = 2.0 * x_cheb * prev1 - prev2
                cols.append(t_next)
                prev2, prev1 = prev1, t_next

    if basis_type == "fourier":
        cols = [np.ones_like(x)]
        for degree in range(1, fourier_degree + 1):
            cols.append(np.cos(2.0 * np.pi * degree * x))
            cols.append(np.sin(2.0 * np.pi * degree * x))

    if basis_type == "rbf":
        cols = [np.ones_like(x)]
        centers = np.linspace(0.0, 1.0, max(1, rbf_num_centers), dtype=np.float64)
        sigma = max(float(rbf_sigma), 1e-6)
        for center in centers:
            cols.append(np.exp(-0.5 * ((x - center) / sigma) ** 2))

    if basis_type not in {"poly", "exp", "poly_exp", "chebyshev", "fourier", "rbf"}:
        raise ValueError(f"Unsupported basis_type: {basis_type}")

    return np.stack(cols, axis=1)


def build_basis_torch(
    n: int,
    basis_type: str,
    *,
    poly_degree: int = 5,
    exp_scales: list[float] | None = None,
    chebyshev_degree: int = 12,
    fourier_degree: int = 8,
    rbf_num_centers: int = 8,
    rbf_sigma: float = 0.08,
) -> torch.Tensor:
    """Torch variant of the analytic basis builder used at reconstruction time."""

    x = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    cols: list[torch.Tensor] = [torch.ones_like(x)]
    exp_scales = coerce_exp_scales(exp_scales)

    if basis_type in {"poly", "poly_exp"}:
        for degree in range(1, poly_degree + 1):
            cols.append(x**degree)

    if basis_type in {"exp", "poly_exp"}:
        for scale in exp_scales:
            cols.append(torch.exp(-scale * x))
            cols.append(torch.exp(scale * (x - 1.0)))

    if basis_type == "chebyshev":
        x_cheb = 2.0 * x - 1.0
        t0 = torch.ones_like(x_cheb)
        cols = [t0]
        if chebyshev_degree >= 1:
            t1 = x_cheb.clone()
            cols.append(t1)
            prev2, prev1 = t0, t1
            for _ in range(2, chebyshev_degree + 1):
                t_next = 2.0 * x_cheb * prev1 - prev2
                cols.append(t_next)
                prev2, prev1 = prev1, t_next

    if basis_type == "fourier":
        cols = [torch.ones_like(x)]
        for degree in range(1, fourier_degree + 1):
            cols.append(torch.cos(2.0 * torch.pi * degree * x))
            cols.append(torch.sin(2.0 * torch.pi * degree * x))

    if basis_type == "rbf":
        cols = [torch.ones_like(x)]
        centers = torch.linspace(0.0, 1.0, max(1, rbf_num_centers), dtype=torch.float32)
        sigma = max(float(rbf_sigma), 1e-6)
        for center in centers:
            cols.append(torch.exp(-0.5 * ((x - center) / sigma) ** 2))

    if basis_type not in {"poly", "exp", "poly_exp", "chebyshev", "fourier", "rbf"}:
        raise ValueError(f"Unsupported basis_type: {basis_type}")

    return torch.stack(cols, dim=1)


def basis_params(cfg: AnalyticFitConfig) -> dict[str, Any]:
    """Extract basis hyperparameters from a fit config."""

    return {
        "poly_degree": int(cfg.poly_degree),
        "exp_scales": [float(x) for x in cfg.exp_scales],
        "chebyshev_degree": int(cfg.chebyshev_degree),
        "fourier_degree": int(cfg.fourier_degree),
        "rbf_num_centers": int(cfg.rbf_num_centers),
        "rbf_sigma": float(cfg.rbf_sigma),
    }


def fit_vector(vec: torch.Tensor, cfg: AnalyticFitConfig) -> tuple[np.ndarray, float]:
    """Fit one principal component against the requested basis family."""

    y = vec.detach().cpu().numpy().astype(np.float64)
    basis = build_basis_numpy(len(y), cfg.basis_type, **basis_params(cfg))
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    recon = basis @ coeffs
    rel_error = float(np.linalg.norm(recon - y) / (np.linalg.norm(y) + 1e-12))
    return coeffs, rel_error


def fit_analytic_delta(
    reference_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    cfg: AnalyticFitConfig,
    *,
    reference_init_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a compact analytic approximation to the weight delta from a base init."""

    out: dict[str, Any] = {
        "__meta__": {
            "artifact_type": "analytic_delta",
            "reference_init_seed": int(reference_init_seed),
        }
    }
    report: dict[str, Any] = {"tensors": {}, "mean_relative_error": 0.0}
    all_errors: list[float] = []

    keys = filter_float_tensor_keys(target_state)
    for key in tqdm(keys, desc="Fitting tensor deltas"):
        if key not in reference_state:
            raise ValueError(f"Missing tensor {key} in reference state")
        ref_tensor = reference_state[key]
        target_tensor = target_state[key]
        if tuple(ref_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"Shape mismatch for tensor {key}: "
                "reference="
                f"{tuple(ref_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )

        delta = (
            target_tensor.detach().to(dtype=torch.float32, device="cpu")
            - ref_tensor.detach().to(dtype=torch.float32, device="cpu")
        ).reshape(-1)
        coeffs, err = fit_vector(delta, cfg)
        coeff_tensor = torch.from_numpy(coeffs).to(torch.float32)
        n = int(delta.numel())
        basis_params_dict = basis_params(cfg)
        basis_dim = int(
            build_basis_numpy(n, cfg.basis_type, **basis_params_dict).shape[1]
        )

        out[key] = {
            "shape": tuple(target_tensor.shape),
            "numel": n,
            "basis_type": cfg.basis_type,
            "basis_params": basis_params_dict,
            "basis_dim": basis_dim,
            "delta_coeffs": coeff_tensor,
        }
        report["tensors"][key] = {
            "component_relative_errors": [err],
            "mean_component_relative_error": err,
        }
        all_errors.append(err)

    report["mean_relative_error"] = float(np.mean(all_errors)) if all_errors else 0.0
    return out, report


def reconstruct_component(
    n: int,
    coeffs: torch.Tensor,
    basis_type: str,
    basis_params_dict: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Rebuild one component vector from stored basis coefficients."""

    basis = build_basis_torch(n=n, basis_type=basis_type, **(basis_params_dict or {}))
    return basis @ coeffs.to(torch.float32)


def build_delta_state_dict(
    reference_state: dict[str, torch.Tensor],
    analytic_delta: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Construct a new state dict by adding a fitted analytic delta to a base init."""

    out = {key: value.detach().clone() for key, value in reference_state.items()}

    for key, entry in analytic_delta.items():
        if key.startswith("__"):
            continue
        if key not in out:
            continue

        target = out[key]
        if not torch.is_floating_point(target):
            continue

        if int(np.prod(entry["shape"])) != target.numel():
            continue

        basis_params_dict = entry.get("basis_params", {})
        delta = reconstruct_component(
            target.numel(),
            entry["delta_coeffs"],
            entry["basis_type"],
            basis_params_dict,
        )
        vec = target.detach().to(dtype=torch.float32).reshape(-1) + delta
        out[key] = vec.reshape(target.shape).to(dtype=target.dtype)

    return out


def apply_analytic_delta_init(
    model: torch.nn.Module,
    analytic_delta: dict[str, Any],
) -> torch.nn.Module:
    """Apply a compact fitted delta to a freshly initialized model."""

    new_state = build_delta_state_dict(model.state_dict(), analytic_delta)
    model.load_state_dict(new_state, strict=False)
    return model
