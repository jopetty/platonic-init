from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compatibility wrapper: run basis init evaluation through platonic_init.pipeline"
    )
    p.add_argument("--config", type=str, default="configs/experiment_dyck_d10_20k_demo.yaml")
    p.add_argument("--basis-dir", type=str, default=None)
    p.add_argument("--fit-names", nargs="+", default=None)
    p.add_argument("--basis", nargs="+", default=None, help="Legacy alias for --fit-names")
    p.add_argument("--init-mode", type=str, default="sampled", choices=["mean", "sampled"])
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--transfer-seed", type=int, default=0)
    p.add_argument("--skip-transfer", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stages = ["pretrain"] if args.basis_dir is not None else ["fit_initializations", "pretrain"]
    cmd = [
        sys.executable,
        "-m",
        "platonic_init.pipeline",
        "--config",
        args.config,
        "--stages",
        *stages,
        "--eval-steps",
        str(args.train_steps),
        "--eval-every",
        str(args.eval_every),
        "--eval-ratio",
        str(args.eval_ratio),
        "--seed",
        str(args.seed),
        "--init-mode",
        args.init_mode,
        "--transfer-seed",
        str(args.transfer_seed),
    ]
    selected = args.fit_names if args.fit_names is not None else args.basis
    if selected:
        cmd.extend(["--fit-names", *selected])
    if args.basis_dir is not None:
        cmd.extend(["--basis-dir", args.basis_dir])
    if args.out is not None:
        cmd.extend(["--curves-out", args.out])
    if args.skip_transfer:
        cmd.append("--skip-transfer")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
