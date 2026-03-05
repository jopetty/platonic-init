#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def sample_depth(max_depth: int, alpha: float, rng: random.Random) -> int:
    depths = list(range(1, max_depth + 1))
    weights = [d ** (-alpha) for d in depths]
    return rng.choices(depths, weights=weights, k=1)[0]


def generate_dyck_exact_depth(depth: int, rng: random.Random, max_attempts: int = 2000) -> str:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    for _ in range(max_attempts):
        n_pairs = depth + rng.randint(0, depth)
        total_len = 2 * n_pairs
        opens = 0
        closes = 0
        cur_depth = 0
        max_seen = 0
        out: list[str] = []

        for _step in range(total_len):
            can_open = opens < n_pairs and cur_depth < depth
            can_close = closes < opens

            remaining = total_len - len(out)
            must_close = (opens - closes) == remaining

            if must_close:
                token = ")"
            elif can_open and can_close:
                # Encourage exploration while preserving validity.
                p_open = 0.55 if cur_depth < depth - 1 else 0.25
                token = "(" if rng.random() < p_open else ")"
            elif can_open:
                token = "("
            elif can_close:
                token = ")"
            else:
                break

            out.append(token)
            if token == "(":
                opens += 1
                cur_depth += 1
                max_seen = max(max_seen, cur_depth)
            else:
                closes += 1
                cur_depth -= 1

            if cur_depth < 0:
                break

        if len(out) == total_len and opens == closes == n_pairs and max_seen == depth:
            return "".join(out)

    # Guaranteed fallback: exact depth via pure nesting.
    return "(" * depth + ")" * depth


def infer_dataset_stem(n_samples: int, max_depth: int) -> str:
    k = n_samples / 1000
    if math.isclose(k, round(k)):
        return f"dyck_d{max_depth}_{int(round(k))}k"
    return f"dyck_d{max_depth}_{n_samples}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dyck-string synthetic data with power-law depth")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.5, help="Power-law exponent for depth sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise ValueError("--n-samples must be > 0")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be > 0")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0")

    rng = random.Random(args.seed)

    lines = []
    for _ in range(args.n_samples):
        d = sample_depth(args.max_depth, args.alpha, rng)
        lines.append(generate_dyck_exact_depth(d, rng))

    if args.output is None:
        stem = infer_dataset_stem(args.n_samples, args.max_depth)
        out_path = Path("data") / f"{stem}.txt"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(lines)} samples to {out_path}")


if __name__ == "__main__":
    main()
