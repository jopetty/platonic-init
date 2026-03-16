"""Synthetic formal-language generators used for pre-pretraining experiments.

The paper replication path needs more than the current 1-Dyck generator. This
module provides reusable generators for:
- k-Dyck
- k-Shuffle Dyck
- ww copy strings

All exported helpers return whitespace-delimited token sequences so larger
alphabets such as k=64 remain unambiguous and tokenizer-friendly.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def sample_depth(max_depth: int, alpha: float, rng: random.Random) -> int:
    """Sample a nesting depth with a power-law bias toward shallow strings."""

    if max_depth <= 0:
        raise ValueError("max_depth must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    depths = list(range(1, max_depth + 1))
    weights = [depth ** (-alpha) for depth in depths]
    return rng.choices(depths, weights=weights, k=1)[0]


def infer_dataset_stem(
    language: str, n_samples: int, *, k: int | None = None, max_depth: int | None = None
) -> str:
    """Build a predictable dataset filename stem for generated corpora."""

    sample_suffix = (
        f"{int(round(n_samples / 1000))}k"
        if math.isclose(n_samples / 1000, round(n_samples / 1000))
        else str(n_samples)
    )
    parts = [language]
    if k is not None:
        parts.append(f"k{k}")
    if max_depth is not None:
        parts.append(f"d{max_depth}")
    parts.append(sample_suffix)
    return "_".join(parts)


def _pair_tokens(k: int) -> list[tuple[str, str]]:
    """Return `k` distinct opening and closing token pairs."""

    if k <= 0:
        raise ValueError("k must be > 0")
    return [(f"<{i}>", f"</{i}>") for i in range(k)]


def generate_k_dyck_exact_depth(
    depth: int,
    rng: random.Random,
    *,
    k: int = 1,
    max_attempts: int = 2000,
) -> list[str]:
    """Generate one valid k-Dyck string whose maximum depth is exactly `depth`."""

    if depth < 1:
        raise ValueError("depth must be >= 1")
    pairs = _pair_tokens(k)

    for _ in range(max_attempts):
        n_pairs = depth + rng.randint(0, depth)
        total_len = 2 * n_pairs
        opens = 0
        closes = 0
        cur_depth = 0
        max_seen = 0
        stack: list[int] = []
        out: list[str] = []

        for _step in range(total_len):
            can_open = opens < n_pairs and cur_depth < depth
            can_close = closes < opens and stack
            remaining = total_len - len(out)
            must_close = (opens - closes) == remaining

            if must_close:
                action = "close"
            elif can_open and can_close:
                action = (
                    "open"
                    if rng.random() < (0.6 if cur_depth < depth - 1 else 0.3)
                    else "close"
                )
            elif can_open:
                action = "open"
            elif can_close:
                action = "close"
            else:
                break

            if action == "open":
                pair_idx = rng.randrange(k)
                stack.append(pair_idx)
                out.append(pairs[pair_idx][0])
                opens += 1
                cur_depth += 1
                max_seen = max(max_seen, cur_depth)
            else:
                pair_idx = stack.pop()
                out.append(pairs[pair_idx][1])
                closes += 1
                cur_depth -= 1

            if cur_depth < 0:
                break

        if (
            len(out) == total_len
            and opens == closes == n_pairs
            and max_seen == depth
            and not stack
        ):
            return out

    # Deterministic fallback preserves exact depth.
    pair_idx = rng.randrange(k)
    open_tok, close_tok = pairs[pair_idx]
    return [open_tok] * depth + [close_tok] * depth


def generate_shuffle_dyck(
    depth: int,
    rng: random.Random,
    *,
    k: int = 2,
) -> list[str]:
    """Generate a k-Shuffle Dyck sequence from interleaved 1-Dyck strings."""

    if k < 2:
        raise ValueError("shuffle_dyck requires k >= 2")

    components = [generate_k_dyck_exact_depth(depth, rng, k=1) for _ in range(k)]
    renamed_components: list[list[str]] = []
    for index, component in enumerate(components):
        open_tok, close_tok = _pair_tokens(k)[index]
        renamed_components.append(
            [open_tok if token == "<0>" else close_tok for token in component]
        )

    # Interleave while preserving each component's local order.
    cursors = [0] * k
    out: list[str] = []
    while True:
        active = [
            i for i, cursor in enumerate(cursors) if cursor < len(renamed_components[i])
        ]
        if not active:
            break
        chosen = active[rng.randrange(len(active))]
        out.append(renamed_components[chosen][cursors[chosen]])
        cursors[chosen] += 1
    return out


def generate_ww(
    rng: random.Random,
    *,
    alphabet_size: int = 16,
    min_half_length: int = 8,
    max_half_length: int = 64,
) -> list[str]:
    """Generate a `ww` copy-language example over a finite token alphabet."""

    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be > 0")
    if min_half_length <= 0 or max_half_length < min_half_length:
        raise ValueError("Require 0 < min_half_length <= max_half_length")

    alphabet = [f"a{i}" for i in range(alphabet_size)]
    half_length = rng.randint(min_half_length, max_half_length)
    prefix = [alphabet[rng.randrange(alphabet_size)] for _ in range(half_length)]
    return prefix + prefix


def render_tokens(tokens: Sequence[str], *, compact_single_dyck: bool = False) -> str:
    """Render a token sequence as one line of training text."""

    if compact_single_dyck and set(tokens).issubset({"<0>", "</0>"}):
        return "".join("(" if token == "<0>" else ")" for token in tokens)
    return " ".join(tokens)


def generate_formal_language_lines(
    *,
    language: str,
    n_samples: int,
    seed: int,
    max_depth: int = 10,
    alpha: float = 1.5,
    k: int = 1,
    ww_alphabet_size: int = 16,
    ww_min_half_length: int = 8,
    ww_max_half_length: int = 64,
    compact_single_dyck: bool = False,
) -> list[str]:
    """Generate newline-ready examples for one supported formal language."""

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(n_samples):
        if language == "dyck":
            depth = sample_depth(max_depth, alpha, rng)
            tokens = generate_k_dyck_exact_depth(depth, rng, k=k)
        elif language == "shuffle_dyck":
            depth = sample_depth(max_depth, alpha, rng)
            tokens = generate_shuffle_dyck(depth, rng, k=k)
        elif language == "ww":
            tokens = generate_ww(
                rng,
                alphabet_size=ww_alphabet_size,
                min_half_length=ww_min_half_length,
                max_half_length=ww_max_half_length,
            )
        else:
            raise ValueError(f"Unsupported language: {language}")
        lines.append(render_tokens(tokens, compact_single_dyck=compact_single_dyck))
    return lines


def is_valid_k_dyck(tokens: Sequence[str], *, k: int) -> bool:
    """Return whether a token sequence belongs to the k-Dyck language."""

    pair_lookup = {
        open_tok: idx for idx, (open_tok, _close_tok) in enumerate(_pair_tokens(k))
    }
    close_lookup = {
        close_tok: idx for idx, (_open_tok, close_tok) in enumerate(_pair_tokens(k))
    }
    stack: list[int] = []
    for token in tokens:
        if token in pair_lookup:
            stack.append(pair_lookup[token])
        elif token in close_lookup:
            if not stack or stack.pop() != close_lookup[token]:
                return False
        else:
            return False
    return not stack


def is_valid_shuffle_dyck(tokens: Sequence[str], *, k: int) -> bool:
    """Return whether a token sequence belongs to the shuffle-Dyck family."""

    if k < 2:
        return False
    pair_tokens = _pair_tokens(k)
    per_type: list[list[str]] = [[] for _ in range(k)]
    lookup: dict[str, int] = {}
    for idx, (open_tok, close_tok) in enumerate(pair_tokens):
        lookup[open_tok] = idx
        lookup[close_tok] = idx
    for token in tokens:
        idx = lookup.get(token)
        if idx is None:
            return False
        per_type[idx].append("<0>" if token == pair_tokens[idx][0] else "</0>")
    return all(is_valid_k_dyck(component, k=1) for component in per_type)


def is_valid_ww(tokens: Sequence[str]) -> bool:
    """Return whether a token sequence belongs to the copy language `ww`."""

    if len(tokens) % 2 != 0:
        return False
    half = len(tokens) // 2
    return list(tokens[:half]) == list(tokens[half:])
