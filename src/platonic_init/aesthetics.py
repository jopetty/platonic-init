from __future__ import annotations

import colorsys
import importlib
from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

Color = tuple[float, float, float]
Palette = dict[str, Color]

PAPER_WIDTH_IN: float = 5.5
FIG_HEIGHT_SINGLE_ROW_IN: float = 1.3
FIG_HEIGHT_DOUBLE_ROW_IN: float = 2.8

NICE_FORMATTER = mtick.EngFormatter(places=0, sep="")
PCT_FORMATTER = mtick.PercentFormatter(1.0)

DEFAULT_RCS: dict[str, Any] = {
    "font.size": 10.0,
    "axes.labelsize": "medium",
    "axes.titlesize": "medium",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "figure.dpi": 120,
}
# Alias used by notebooks/scripts that want an explicit rc payload.
rcs = DEFAULT_RCS


def _to_rgb_tuple(color: str | Color) -> Color:
    if isinstance(color, str):
        rgb = sns.color_palette([color])[0]
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    c = tuple(float(x) for x in color)
    if len(c) != 3:
        raise ValueError(f"Expected RGB tuple with 3 values, got {color!r}")
    if any(x > 1.0 for x in c):
        return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
    return (c[0], c[1], c[2])


def _darken_color(color: str | Color, by: float) -> Color:
    by = min(max(float(by), 0.0), 1.0)
    rgb = _to_rgb_tuple(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    darker = colorsys.hls_to_rgb(h, l * (1.0 - by), s)
    return (float(darker[0]), float(darker[1]), float(darker[2]))


def darken(color: str | Color | Mapping[str, str | Color] | Iterable[str | Color], by: float = 0.2):
    if isinstance(color, Mapping):
        return {k: _darken_color(v, by) for k, v in color.items()}
    if isinstance(color, str):
        return _darken_color(color, by)
    if isinstance(color, tuple) and len(color) == 3 and all(isinstance(x, (int, float)) for x in color):
        return _darken_color(color, by)
    return [_darken_color(c, by) for c in color]


NONSEMANTIC_COLOR: Color = darken("#ffcc66", by=0.3)

INIT_PALETTE: Palette = darken(
    {
        "random": sns.color_palette("Greys", n_colors=5)[2],
        "weight_transfer": sns.color_palette("Greens", n_colors=5)[3],
        "chebyshev": sns.color_palette("Blues", n_colors=6)[3],
        "chebyshev_d24": sns.color_palette("Blues", n_colors=6)[5],
        "fourier": sns.color_palette("Purples", n_colors=6)[3],
        "rbf": sns.color_palette("Oranges", n_colors=6)[3],
        "poly_exp": NONSEMANTIC_COLOR,
    },
    by=0.25,
)


def set_theme(style: str = "ticks", context: str = "notebook", rc: dict[str, Any] | None = None) -> None:
    merged = dict(DEFAULT_RCS)
    if rc:
        merged.update(rc)
    sns.set_theme(style=style, context=context, rc=merged)
    plt.rcParams["figure.dpi"] = merged["figure.dpi"]


def with_context(context: str = "notebook", rc: dict[str, Any] | None = None):
    merged = dict(DEFAULT_RCS)
    if rc:
        merged.update(rc)
    return sns.plotting_context(context=context, rc=merged)


def make_init_palette(labels: Iterable[str]) -> dict[str, Color]:
    labels = list(labels)
    missing = [x for x in labels if x not in INIT_PALETTE]
    if not missing:
        return {k: INIT_PALETTE[k] for k in labels}

    fallback = sns.color_palette("tab20", n_colors=max(1, len(missing)))
    out: dict[str, Color] = {k: INIT_PALETTE[k] for k in labels if k in INIT_PALETTE}
    for i, label in enumerate(missing):
        c = fallback[i]
        out[label] = (float(c[0]), float(c[1]), float(c[2]))
    return out


def make_seed_palette(labels: Iterable[str]) -> dict[str, Color]:
    labels = list(labels)
    if not labels:
        return {}
    base = sns.color_palette("Oranges", n_colors=len(labels) + 2)[2:]
    out: dict[str, Color] = {}
    for label, color in zip(labels, base):
        out[label] = (float(color[0]), float(color[1]), float(color[2]))
    return out


def enable_notebook_autoreload(module_name: str = "platonic_init.aesthetics") -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    except Exception:
        # Autoreload still helps even if import/reload fails initially.
        pass
    return True


__all__ = [
    "Color",
    "Palette",
    "PAPER_WIDTH_IN",
    "FIG_HEIGHT_SINGLE_ROW_IN",
    "FIG_HEIGHT_DOUBLE_ROW_IN",
    "NICE_FORMATTER",
    "PCT_FORMATTER",
    "DEFAULT_RCS",
    "rcs",
    "NONSEMANTIC_COLOR",
    "INIT_PALETTE",
    "darken",
    "set_theme",
    "with_context",
    "make_init_palette",
    "make_seed_palette",
    "enable_notebook_autoreload",
]
