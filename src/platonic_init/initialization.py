"""Weight-space analysis and platonic-initialization mechanics.

The main flow is:
1. Load matching checkpoints into state dicts.
2. Estimate a tensor-wise shared subspace with PCA.
3. Fit each principal direction with a compact analytic basis.
4. Reconstruct a deterministic or sampled initialization from that basis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm

from .config import AnalyticFitConfig, AnalysisConfig


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


def dtype_from_name(name: str) -> torch.dtype:
    """Map config strings to torch dtypes."""

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[name]


def filter_float_tensor_keys(state: dict[str, torch.Tensor]) -> list[str]:
    """Return tensor keys that participate in PCA analysis."""

    keys = []
    for key, value in state.items():
        if not torch.is_floating_point(value):
            continue
        if "lm_head" in key:
            continue
        keys.append(key)
    return sorted(keys)


def stack_tensor_values(
    states: list[dict[str, torch.Tensor]],
    key: str,
    analysis_cfg: AnalysisConfig,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten one tensor across checkpoints into a matrix for PCA."""

    dtype = dtype_from_name(analysis_cfg.dtype)
    flattened = []
    shape = tuple(states[0][key].shape)
    for state in states:
        tensor = state[key].detach().to(dtype=torch.float32, device="cpu")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Shape mismatch for tensor {key}: expected {shape}, got {tuple(tensor.shape)}")
        flattened.append(tensor.reshape(-1))

    matrix = torch.stack(flattened, dim=0)
    if analysis_cfg.max_params_per_tensor is not None and matrix.shape[1] > analysis_cfg.max_params_per_tensor:
        idx = torch.linspace(0, matrix.shape[1] - 1, steps=analysis_cfg.max_params_per_tensor).long()
        matrix = matrix[:, idx]
    return matrix.to(dtype=dtype), shape


def tensorwise_pca(states: list[dict[str, torch.Tensor]], cfg: AnalysisConfig) -> dict[str, Any]:
    """Estimate a shared low-rank subspace for each tensor across checkpoints."""

    if len(states) < 2:
        raise ValueError("Need at least 2 checkpoints to estimate shared subspace")

    keys = filter_float_tensor_keys(states[0])
    for state in states[1:]:
        present = set(filter_float_tensor_keys(state))
        missing = [key for key in keys if key not in present]
        if missing:
            raise ValueError(f"Missing tensor keys in checkpoint: {missing[:5]}")

    stats: dict[str, Any] = {}
    for key in tqdm(keys, desc="Analyzing tensors"):
        matrix, shape = stack_tensor_values(states, key, cfg)
        mean = matrix.mean(dim=0)
        centered = matrix - mean

        max_rank = min(matrix.shape[0] - 1, matrix.shape[1])
        k = min(cfg.top_k_components, max_rank)
        if k <= 0:
            continue

        _, singular_values, v_t = torch.linalg.svd(centered, full_matrices=False)
        components = v_t[:k]
        explained_variance = (singular_values**2) / max(1, matrix.shape[0] - 1)
        explained_ratio = explained_variance / explained_variance.sum().clamp(min=1e-12)

        stats[key] = {
            "shape": shape,
            "numel": int(np.prod(shape)),
            "mean": mean,
            "components": components,
            "explained_variance": explained_variance[:k],
            "explained_variance_ratio": explained_ratio[:k],
        }

    return stats


def build_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable summary of PCA results."""

    summary = {
        "num_tensors": len(stats),
        "total_params_analyzed": int(sum(entry["numel"] for entry in stats.values())),
        "tensors": {},
    }
    for key, entry in stats.items():
        summary["tensors"][key] = {
            "shape": list(entry["shape"]),
            "numel": entry["numel"],
            "explained_variance_ratio": [float(x) for x in entry["explained_variance_ratio"].cpu().tolist()],
        }
    return summary


def save_summary(stats: dict[str, Any], path: str | Path) -> None:
    """Write a subspace summary JSON file."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(build_summary(stats), indent=2), encoding="utf-8")


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


def fit_analytic_subspace(
    subspace: dict[str, Any],
    cfg: AnalyticFitConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compress each PCA component into analytic basis coefficients."""

    out: dict[str, Any] = {}
    report: dict[str, Any] = {"tensors": {}, "mean_relative_error": 0.0}
    all_errors = []

    for key, entry in subspace.items():
        mean = entry["mean"]
        components = entry["components"]
        n = int(mean.numel())

        component_coeffs = []
        component_errors = []
        for index in range(components.shape[0]):
            coeffs, err = fit_vector(components[index], cfg)
            component_coeffs.append(torch.from_numpy(coeffs).to(torch.float32))
            component_errors.append(err)
            all_errors.append(err)

        out[key] = {
            "shape": entry["shape"],
            "numel": entry["numel"],
            "mean": mean,
            "basis_type": cfg.basis_type,
            "basis_params": basis_params(cfg),
            "basis_dim": int(build_basis_numpy(n, cfg.basis_type, **basis_params(cfg)).shape[1]),
            "component_coeffs": component_coeffs,
            "explained_variance": entry["explained_variance"],
            "explained_variance_ratio": entry["explained_variance_ratio"],
        }
        report["tensors"][key] = {
            "component_relative_errors": component_errors,
            "mean_component_relative_error": float(np.mean(component_errors)) if component_errors else 0.0,
        }

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


def build_platonic_state_dict(
    reference_state: dict[str, torch.Tensor],
    analytic_subspace: dict[str, Any],
    latent: dict[str, torch.Tensor] | None = None,
    latent_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Construct a new state dict by applying the analytic subspace to a model."""

    out = {key: value.detach().clone() for key, value in reference_state.items()}

    for key, entry in analytic_subspace.items():
        if key not in out:
            continue
        target = out[key]
        if not torch.is_floating_point(target):
            continue

        mean = entry["mean"].to(torch.float32)
        if mean.numel() != target.numel():
            continue

        entry_basis_params = entry.get("basis_params")
        if entry_basis_params is None:
            entry_basis_params = {
                "poly_degree": int(entry.get("poly_degree", 5)),
                "exp_scales": [float(x) for x in entry.get("exp_scales", [0.5, 1.0, 2.0, 4.0])],
            }
        coeffs = entry["component_coeffs"]

        vec = mean.clone()
        if latent is not None and key in latent and len(coeffs) > 0:
            z = latent[key].to(torch.float32) * float(latent_scale)
            if z.numel() != len(coeffs):
                raise ValueError(f"Latent dim mismatch for {key}: got {z.numel()} expected {len(coeffs)}")
            for index, coeff in enumerate(coeffs):
                component = reconstruct_component(len(vec), coeff, entry["basis_type"], entry_basis_params)
                vec = vec + z[index] * component

        if int(np.prod(entry["shape"])) != target.numel():
            continue
        out[key] = vec.reshape(target.shape).to(dtype=target.dtype)

    return out


def sample_latent(analytic_subspace: dict[str, Any], seed: int = 0) -> dict[str, torch.Tensor]:
    """Sample latent coordinates using the stored explained variances."""

    rng = np.random.default_rng(seed)
    latent: dict[str, torch.Tensor] = {}
    for key, entry in analytic_subspace.items():
        explained_variance = entry.get("explained_variance")
        k = len(entry["component_coeffs"])
        if k == 0:
            continue
        if explained_variance is None:
            std = np.ones(k, dtype=np.float32)
        else:
            std = np.sqrt(torch.as_tensor(explained_variance).cpu().numpy()[:k]).astype(np.float32)
        latent[key] = torch.from_numpy(rng.normal(0.0, std, size=k).astype(np.float32))
    return latent


def apply_platonic_init(
    model: torch.nn.Module,
    analytic_subspace: dict[str, Any],
    latent: dict[str, torch.Tensor] | None = None,
    latent_scale: float = 1.0,
) -> torch.nn.Module:
    """Apply a platonic initialization directly to a live model."""

    new_state = build_platonic_state_dict(model.state_dict(), analytic_subspace, latent=latent, latent_scale=latent_scale)
    model.load_state_dict(new_state, strict=False)
    return model
