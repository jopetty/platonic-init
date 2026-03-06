from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AxisPermutation:
    perm_name: str
    group_size: int = 1
    num_blocks: int = 1


@dataclass(frozen=True)
class PermutationSpec:
    perm_to_axes: dict[str, list[tuple[str, int]]]
    axes_to_perm: dict[str, tuple[AxisPermutation | None, ...]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: dict[str, tuple[AxisPermutation | None, ...]],
) -> PermutationSpec:
    perm_to_axes: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, axis_perms in axes_to_perm.items():
        for axis, perm_ref in enumerate(axis_perms):
            if perm_ref is not None:
                perm_to_axes[perm_ref.perm_name].append((key, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


_GPT2_LAYER_PATTERN = re.compile(r"^transformer\.h\.(\d+)\.")


def _gpt2_layer_ids(state: dict[str, torch.Tensor]) -> list[int]:
    layer_ids: set[int] = set()
    for key in state.keys():
        m = _GPT2_LAYER_PATTERN.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    return sorted(layer_ids)


def gpt2_permutation_spec(
    state: dict[str, torch.Tensor],
    *,
    num_attention_heads: int | None = None,
) -> PermutationSpec:
    layer_ids = _gpt2_layer_ids(state)

    axes_to_perm: dict[str, tuple[AxisPermutation | None, ...]] = {}
    for layer_id in layer_ids:
        mlp_perm = f"P_mlp_{layer_id}"
        fc_w = f"transformer.h.{layer_id}.mlp.c_fc.weight"
        fc_b = f"transformer.h.{layer_id}.mlp.c_fc.bias"
        proj_w = f"transformer.h.{layer_id}.mlp.c_proj.weight"
        if fc_w in state and fc_b in state and proj_w in state:
            # GPT-2 Conv1D layout:
            # c_fc.weight: [hidden, mlp]
            # c_proj.weight: [mlp, hidden]
            axes_to_perm[fc_w] = (None, AxisPermutation(mlp_perm))
            axes_to_perm[fc_b] = (AxisPermutation(mlp_perm),)
            axes_to_perm[proj_w] = (AxisPermutation(mlp_perm), None)

        if num_attention_heads is None or num_attention_heads <= 1:
            continue
        attn_perm = f"P_attn_{layer_id}"
        attn_qkv_w = f"transformer.h.{layer_id}.attn.c_attn.weight"
        attn_qkv_b = f"transformer.h.{layer_id}.attn.c_attn.bias"
        attn_proj_w = f"transformer.h.{layer_id}.attn.c_proj.weight"
        if attn_qkv_w not in state or attn_qkv_b not in state or attn_proj_w not in state:
            continue
        hidden = int(state[attn_proj_w].shape[0])
        if hidden <= 0 or hidden % int(num_attention_heads) != 0:
            continue
        head_dim = hidden // int(num_attention_heads)
        axes_to_perm[attn_qkv_w] = (None, AxisPermutation(attn_perm, group_size=head_dim, num_blocks=3))
        axes_to_perm[attn_qkv_b] = (AxisPermutation(attn_perm, group_size=head_dim, num_blocks=3),)
        axes_to_perm[attn_proj_w] = (AxisPermutation(attn_perm, group_size=head_dim, num_blocks=1), None)

    return permutation_spec_from_axes_to_perm(axes_to_perm)


def _expanded_axis_indices(ref: AxisPermutation, perm_idx: torch.Tensor) -> torch.Tensor:
    blocks = []
    n = int(perm_idx.numel())
    g = int(ref.group_size)
    block_size = n * g
    for block in range(int(ref.num_blocks)):
        base = block * block_size
        block_idx = []
        for p in perm_idx.tolist():
            start = base + int(p) * g
            block_idx.extend(range(start, start + g))
        blocks.extend(block_idx)
    return torch.tensor(blocks, dtype=torch.long)


def _flatten_rows_for_matching(w: torch.Tensor, n: int, ref: AxisPermutation) -> torch.Tensor:
    g = int(ref.group_size)
    b = int(ref.num_blocks)
    expected = n * g * b
    if int(w.shape[0]) != expected:
        raise ValueError(f"Unexpected axis size for grouped permutation: got {w.shape[0]}, expected {expected}")
    return w.reshape(b, n, g, -1).permute(1, 0, 2, 3).reshape(n, -1)


def get_permuted_param(
    ps: PermutationSpec,
    perm: dict[str, torch.Tensor],
    key: str,
    params: dict[str, torch.Tensor],
    except_axis: int | None = None,
) -> torch.Tensor:
    w = params[key]
    for axis, perm_ref in enumerate(ps.axes_to_perm.get(key, ())):
        if axis == except_axis or perm_ref is None:
            continue
        idx = _expanded_axis_indices(perm_ref, perm[perm_ref.perm_name]).to(device=w.device)
        w = torch.index_select(w, axis, idx)
    return w


def apply_permutation(
    ps: PermutationSpec,
    perm: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: get_permuted_param(ps, perm, key, params) if key in ps.axes_to_perm else value
        for key, value in params.items()
    }


def weight_matching(
    ps: PermutationSpec,
    params_a: dict[str, torch.Tensor],
    params_b: dict[str, torch.Tensor],
    *,
    max_iter: int = 100,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    perm_sizes: dict[str, int] = {}
    for perm_name, axes in ps.perm_to_axes.items():
        n_candidates = []
        for wk, axis in axes:
            ref = ps.axes_to_perm[wk][axis]
            if ref is None:
                continue
            axis_size = int(params_a[wk].shape[axis])
            denom = int(ref.group_size) * int(ref.num_blocks)
            if denom <= 0 or axis_size % denom != 0:
                raise ValueError(f"Invalid grouped permutation axis for {wk}: size {axis_size}, denom {denom}")
            n_candidates.append(axis_size // denom)
        if not n_candidates:
            continue
        n0 = n_candidates[0]
        if any(n != n0 for n in n_candidates):
            raise ValueError(f"Inconsistent permutation size for {perm_name}: {n_candidates}")
        perm_sizes[perm_name] = n0

    perm = {perm_name: torch.arange(n, dtype=torch.long) for perm_name, n in perm_sizes.items()}
    if not perm:
        return perm

    rng = np.random.default_rng(seed)
    perm_names = list(perm.keys())
    for _ in range(max_iter):
        progress = False
        for perm_index in rng.permutation(len(perm_names)):
            perm_name = perm_names[int(perm_index)]
            n = perm_sizes[perm_name]
            a = torch.zeros((n, n), dtype=torch.float64)
            for wk, axis in ps.perm_to_axes[perm_name]:
                ref = ps.axes_to_perm[wk][axis]
                if ref is None:
                    continue
                w_a = params_a[wk].detach().to(dtype=torch.float64, device="cpu")
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis).detach().to(
                    dtype=torch.float64,
                    device="cpu",
                )
                w_a = torch.movedim(w_a, axis, 0)
                w_b = torch.movedim(w_b, axis, 0)
                w_a = _flatten_rows_for_matching(w_a, n, ref)
                w_b = _flatten_rows_for_matching(w_b, n, ref)
                a = a + (w_a @ w_b.T)

            _, col_ind = linear_sum_assignment(a.numpy(), maximize=True)
            col_t = torch.from_numpy(col_ind).to(dtype=torch.long)

            old_l = float(a[torch.arange(n), perm[perm_name]].sum().item())
            new_l = float(a[torch.arange(n), col_t].sum().item())
            if new_l > old_l + 1e-12:
                progress = True
            perm[perm_name] = col_t

        if not progress:
            break
    return perm


def align_states_for_pca(
    states: list[dict[str, torch.Tensor]],
    *,
    max_iter: int = 50,
    seed: int = 0,
    num_attention_heads: int | None = None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    if len(states) <= 1:
        return states, {"enabled": True, "num_permutations": 0, "per_state": []}

    ps = gpt2_permutation_spec(states[0], num_attention_heads=num_attention_heads)
    if not ps.perm_to_axes:
        return states, {"enabled": True, "num_permutations": 0, "per_state": []}

    aligned = [states[0]]
    report = {
        "enabled": True,
        "num_attention_heads": num_attention_heads,
        "num_permutations": len(ps.perm_to_axes),
        "permutation_names": sorted(ps.perm_to_axes.keys()),
        "per_state": [],
    }
    for idx, state in enumerate(states[1:], start=1):
        perm = weight_matching(ps, states[0], state, max_iter=max_iter, seed=seed + idx)
        aligned_state = apply_permutation(ps, perm, state)
        aligned.append(aligned_state)
        report["per_state"].append(
            {
                "state_index": idx,
                "perm_sizes": {name: int(t.numel()) for name, t in perm.items()},
            }
        )
    return aligned, report
