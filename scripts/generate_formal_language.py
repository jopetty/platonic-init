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
        description="Generate formal-language synthetic data for pre-pretraining"
    )
    parser.add_argument(
        "--language", type=str, choices=["dyck", "shuffle_dyck", "ww"], required=True
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--k", type=int, default=1, help="Bracket/alphabet family size")
    parser.add_argument("--ww-alphabet-size", type=int, default=16)
    parser.add_argument("--ww-min-half-length", type=int, default=8)
    parser.add_argument("--ww-max-half-length", type=int, default=64)
    parser.add_argument(
        "--compact-single-dyck",
        action="store_true",
        help=(
            "Render 1-Dyck using raw parentheses for compatibility with the old script"
        ),
    )
    args = parser.parse_args()

    lines = generate_formal_language_lines(
        language=args.language,
        n_samples=args.n_samples,
        seed=args.seed,
        max_depth=args.max_depth,
        alpha=args.alpha,
        k=args.k,
        ww_alphabet_size=args.ww_alphabet_size,
        ww_min_half_length=args.ww_min_half_length,
        ww_max_half_length=args.ww_max_half_length,
        compact_single_dyck=args.compact_single_dyck,
    )
    if args.output is None:
        stem = infer_dataset_stem(
            args.language,
            args.n_samples,
            k=args.k if args.language != "ww" else args.ww_alphabet_size,
            max_depth=args.max_depth if args.language != "ww" else None,
        )
        out_path = Path("data") / f"{stem}.txt"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} samples to {out_path}")


if __name__ == "__main__":
    main()
