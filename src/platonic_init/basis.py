from __future__ import annotations

import numpy as np
import torch


def _coerce_exp_scales(exp_scales: list[float] | None) -> list[float]:
    return [float(s) for s in (exp_scales or [0.5, 1.0, 2.0, 4.0])]


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
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    cols: list[np.ndarray] = [np.ones_like(x)]
    exp_scales = _coerce_exp_scales(exp_scales)

    if basis_type in {"poly", "poly_exp"}:
        for p in range(1, poly_degree + 1):
            cols.append(x**p)

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
        for k in range(1, fourier_degree + 1):
            cols.append(np.cos(2.0 * np.pi * k * x))
            cols.append(np.sin(2.0 * np.pi * k * x))

    if basis_type == "rbf":
        cols = [np.ones_like(x)]
        centers = np.linspace(0.0, 1.0, max(1, rbf_num_centers), dtype=np.float64)
        sigma = max(float(rbf_sigma), 1e-6)
        for c in centers:
            cols.append(np.exp(-0.5 * ((x - c) / sigma) ** 2))

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
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    cols: list[torch.Tensor] = [torch.ones_like(x)]
    exp_scales = _coerce_exp_scales(exp_scales)

    if basis_type in {"poly", "poly_exp"}:
        for p in range(1, poly_degree + 1):
            cols.append(x**p)

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
        for k in range(1, fourier_degree + 1):
            cols.append(torch.cos(2.0 * torch.pi * k * x))
            cols.append(torch.sin(2.0 * torch.pi * k * x))

    if basis_type == "rbf":
        cols = [torch.ones_like(x)]
        centers = torch.linspace(0.0, 1.0, max(1, rbf_num_centers), dtype=torch.float32)
        sigma = max(float(rbf_sigma), 1e-6)
        for c in centers:
            cols.append(torch.exp(-0.5 * ((x - c) / sigma) ** 2))

    if basis_type not in {"poly", "exp", "poly_exp", "chebyshev", "fourier", "rbf"}:
        raise ValueError(f"Unsupported basis_type: {basis_type}")

    return torch.stack(cols, dim=1)
