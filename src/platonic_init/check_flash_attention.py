"""Runtime diagnostics for FlashAttention availability inside the training env."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import platform
import sys

import torch
import torch.version

from .training import resolve_attn_implementation


def _device_summary() -> list[str]:
    """Return concise CUDA device diagnostics."""

    lines: list[str] = []
    if not torch.cuda.is_available():
        lines.append("cuda_available: no")
        return lines
    lines.append("cuda_available: yes")
    lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        lines.append(
            "device"
            f"[{idx}]: name={props.name}, capability={props.major}.{props.minor}, "
            f"total_memory_gb={props.total_memory / (1024**3):.1f}"
        )
    return lines


def _import_status(module_name: str) -> str:
    """Return whether a module can be found and imported."""

    if importlib.util.find_spec(module_name) is None:
        return "missing"
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"present_but_import_fails: {exc!r}"
    return "ok"


def main() -> int:
    """Print FlashAttention diagnostics and optionally fail if unavailable."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-fa2",
        action="store_true",
        help="Exit nonzero unless flash_attention_2 is usable in this runtime.",
    )
    args = parser.parse_args()

    lines = [
        f"platform: {platform.platform()}",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"torch_cuda: {torch.version.cuda}",
        f"flash_attn: {_import_status('flash_attn')}",
        f"flash_attn_2_cuda: {_import_status('flash_attn_2_cuda')}",
    ]
    lines.extend(_device_summary())

    attn_impl = resolve_attn_implementation(prefer_flash_attention_2=True)
    lines.append(f"resolved_attn_implementation: {attn_impl}")

    if attn_impl == "flash_attention_2":
        lines.append("result: FlashAttention 2 is available for Transformers.")
        print("\n".join(lines))
        return 0

    lines.append(
        "result: FlashAttention 2 is not available. "
        "Most often this means flash-attn was installed without "
        "a usable CUDA extension."
    )
    print("\n".join(lines))
    return 1 if args.require_fa2 else 0


if __name__ == "__main__":
    raise SystemExit(main())
