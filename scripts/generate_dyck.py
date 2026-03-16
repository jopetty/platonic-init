#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from platonic_init.formal_language import (
    generate_formal_language_lines,
    infer_dataset_stem,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dyck-string synthetic data with power-law depth"
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument(
        "--alpha", type=float, default=1.5, help="Power-law exponent for depth sampling"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    lines = generate_formal_language_lines(
        language="dyck",
        n_samples=args.n_samples,
        seed=args.seed,
        max_depth=args.max_depth,
        alpha=args.alpha,
        k=1,
        compact_single_dyck=True,
    )
    if args.output is None:
        stem = infer_dataset_stem("dyck", args.n_samples, max_depth=args.max_depth)
        out_path = Path("data") / f"{stem}.txt"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} samples to {out_path}")


if __name__ == "__main__":
    main()
