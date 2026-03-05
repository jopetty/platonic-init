from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm

from .config import AnalysisConfig, load_config


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
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


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[name]


def _filter_tensor_keys(state: dict[str, torch.Tensor]) -> list[str]:
    keys = []
    for k, v in state.items():
        if not torch.is_floating_point(v):
            continue
        if "lm_head" in k:
            # lm_head can be tied to embeddings and may be duplicated.
            continue
        keys.append(k)
    return sorted(keys)


def _stack_tensor_values(
    states: list[dict[str, torch.Tensor]],
    key: str,
    analysis_cfg: AnalysisConfig,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    dtype = _dtype_from_str(analysis_cfg.dtype)
    flattened = []
    shape = tuple(states[0][key].shape)
    for s in states:
        t = s[key].detach().to(dtype=torch.float32, device="cpu")
        if tuple(t.shape) != shape:
            raise ValueError(f"Shape mismatch for tensor {key}: expected {shape}, got {tuple(t.shape)}")
        flattened.append(t.reshape(-1))

    x = torch.stack(flattened, dim=0)

    if analysis_cfg.max_params_per_tensor is not None and x.shape[1] > analysis_cfg.max_params_per_tensor:
        idx = torch.linspace(0, x.shape[1] - 1, steps=analysis_cfg.max_params_per_tensor).long()
        x = x[:, idx]

    return x.to(dtype=dtype), shape


def tensorwise_pca(states: list[dict[str, torch.Tensor]], cfg: AnalysisConfig) -> dict[str, Any]:
    if len(states) < 2:
        raise ValueError("Need at least 2 checkpoints to estimate shared subspace")

    keys = _filter_tensor_keys(states[0])
    for s in states[1:]:
        keys2 = set(_filter_tensor_keys(s))
        missing = [k for k in keys if k not in keys2]
        if missing:
            raise ValueError(f"Missing tensor keys in checkpoint: {missing[:5]}")

    stats: dict[str, Any] = {}
    for key in tqdm(keys, desc="Analyzing tensors"):
        x, shape = _stack_tensor_values(states, key, cfg)
        mean = x.mean(dim=0)
        centered = x - mean

        max_rank = min(x.shape[0] - 1, x.shape[1])
        k = min(cfg.top_k_components, max_rank)
        if k <= 0:
            continue

        # SVD on [num_models, num_params]; right singular vectors are principal axes in parameter space.
        _, svals, v_t = torch.linalg.svd(centered, full_matrices=False)
        components = v_t[:k]

        ev = (svals**2) / max(1, x.shape[0] - 1)
        ev_ratio = ev / ev.sum().clamp(min=1e-12)

        stats[key] = {
            "shape": shape,
            "numel": int(np.prod(shape)),
            "mean": mean,
            "components": components,
            "explained_variance": ev[:k],
            "explained_variance_ratio": ev_ratio[:k],
        }

    return stats


def build_summary(stats: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "num_tensors": len(stats),
        "total_params_analyzed": int(sum(v["numel"] for v in stats.values())),
        "tensors": {},
    }
    for k, v in stats.items():
        summary["tensors"][k] = {
            "shape": list(v["shape"]),
            "numel": v["numel"],
            "explained_variance_ratio": [float(x) for x in v["explained_variance_ratio"].cpu().tolist()],
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze multiple checkpoints to find shared core subspace")
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--checkpoints", nargs="+", required=True, help="Model directories from seed runs")
    p.add_argument("--out", type=str, default="artifacts/weight_subspace.pt")
    p.add_argument("--summary-out", type=str, default="artifacts/weight_subspace_summary.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_config(args.config)

    states = [_load_state_dict(Path(path)) for path in args.checkpoints]
    stats = tensorwise_pca(states, exp_cfg.analysis)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out_path)

    summary = build_summary(stats)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
