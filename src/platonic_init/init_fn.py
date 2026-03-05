from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _build_basis(n: int, basis_type: str, poly_degree: int, exp_scales: list[float]) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    cols = [torch.ones_like(x)]

    if basis_type in {"poly", "poly_exp"}:
        for p in range(1, poly_degree + 1):
            cols.append(x**p)

    if basis_type in {"exp", "poly_exp"}:
        for scale in exp_scales:
            cols.append(torch.exp(-scale * x))
            cols.append(torch.exp(scale * (x - 1.0)))

    return torch.stack(cols, dim=1)


def reconstruct_component(
    n: int,
    coeffs: torch.Tensor,
    basis_type: str,
    poly_degree: int,
    exp_scales: list[float],
) -> torch.Tensor:
    basis = _build_basis(n=n, basis_type=basis_type, poly_degree=poly_degree, exp_scales=exp_scales)
    return basis @ coeffs.to(torch.float32)


def build_platonic_state_dict(
    reference_state: dict[str, torch.Tensor],
    analytic_subspace: dict[str, Any],
    latent: dict[str, torch.Tensor] | None = None,
    latent_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    out = {k: v.detach().clone() for k, v in reference_state.items()}

    for key, entry in analytic_subspace.items():
        if key not in out:
            continue
        target = out[key]
        if not torch.is_floating_point(target):
            continue

        mean = entry["mean"].to(torch.float32)
        basis_type = entry["basis_type"]
        poly_degree = int(entry["poly_degree"])
        exp_scales = [float(x) for x in entry["exp_scales"]]
        coeffs = entry["component_coeffs"]

        vec = mean.clone()
        if latent is not None and key in latent and len(coeffs) > 0:
            z = latent[key].to(torch.float32) * float(latent_scale)
            if z.numel() != len(coeffs):
                raise ValueError(f"Latent dim mismatch for {key}: got {z.numel()} expected {len(coeffs)}")
            for i, c in enumerate(coeffs):
                comp = reconstruct_component(len(vec), c, basis_type, poly_degree, exp_scales)
                vec = vec + z[i] * comp

        out[key] = vec.reshape(entry["shape"]).to(dtype=target.dtype)

    return out


def sample_latent(analytic_subspace: dict[str, Any], seed: int = 0) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    latent: dict[str, torch.Tensor] = {}
    for key, entry in analytic_subspace.items():
        ev = entry.get("explained_variance")
        k = len(entry["component_coeffs"])
        if k == 0:
            continue
        if ev is None:
            std = np.ones(k, dtype=np.float32)
        else:
            std = np.sqrt(torch.as_tensor(ev).cpu().numpy()[:k]).astype(np.float32)
        z = rng.normal(0.0, std, size=k).astype(np.float32)
        latent[key] = torch.from_numpy(z)
    return latent


def apply_platonic_init(
    model: torch.nn.Module,
    analytic_subspace: dict[str, Any],
    latent: dict[str, torch.Tensor] | None = None,
    latent_scale: float = 1.0,
) -> torch.nn.Module:
    reference = model.state_dict()
    new_state = build_platonic_state_dict(reference, analytic_subspace, latent=latent, latent_scale=latent_scale)
    model.load_state_dict(new_state, strict=False)
    return model
