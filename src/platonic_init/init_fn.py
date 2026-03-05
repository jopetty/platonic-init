from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .basis import build_basis_torch


def reconstruct_component(
    n: int,
    coeffs: torch.Tensor,
    basis_type: str,
    basis_params: dict[str, Any] | None = None,
) -> torch.Tensor:
    basis = build_basis_torch(n=n, basis_type=basis_type, **(basis_params or {}))
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
        if mean.numel() != target.numel():
            # Skip tensors whose shape changed (e.g., vocab embeddings between PPT and downstream pretraining).
            continue
        basis_type = entry["basis_type"]
        basis_params = entry.get("basis_params")
        if basis_params is None:
            # Backward compatibility with old artifacts.
            basis_params = {
                "poly_degree": int(entry.get("poly_degree", 5)),
                "exp_scales": [float(x) for x in entry.get("exp_scales", [0.5, 1.0, 2.0, 4.0])],
            }
        coeffs = entry["component_coeffs"]

        vec = mean.clone()
        if latent is not None and key in latent and len(coeffs) > 0:
            z = latent[key].to(torch.float32) * float(latent_scale)
            if z.numel() != len(coeffs):
                raise ValueError(f"Latent dim mismatch for {key}: got {z.numel()} expected {len(coeffs)}")
            for i, c in enumerate(coeffs):
                comp = reconstruct_component(len(vec), c, basis_type, basis_params=basis_params)
                vec = vec + z[i] * comp

        if int(np.prod(entry["shape"])) != target.numel():
            continue
        out[key] = vec.reshape(target.shape).to(dtype=target.dtype)

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
