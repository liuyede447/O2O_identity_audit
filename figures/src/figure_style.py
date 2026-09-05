"""Shared publication style for every final statistical figure.

The values are defined at final physical size for a 178 mm double-column figure.
All plotting scripts must import this module rather than redefining fonts, colors,
markers, line widths, or export settings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


MM_PER_INCH = 25.4
DOUBLE_COLUMN_MM = 178.0
SINGLE_COLUMN_MM = 85.0
WHITE = "#FFFFFF"
INK = "#222222"
MUTED = "#7F8C8D"
GUIDE = "#E5E5E5"

BRANCH = {
    "O2O": "#1F4E79",
    "O2M": "#E67E22",
    "Paired": "#8E5AA7",
    "Auxiliary": MUTED,
}
BRANCH_MARKER = {"O2O": "o", "O2M": "s", "Paired": "D"}

MECHANISM = {
    "Eligibility": "#009E73",
    "Within-set rank": "#56B4E9",
    "Top-k": "#E69F00",
    "Conflict": "#4D4D4D",
    "Other/reserved": "#B8C0C2",
}

FONT = {
    "panel_label": 8.0,
    "panel_title": 7.8,
    "axis_label": 7.0,
    "legend": 7.0,
    "tick": 7.0,
    "annotation": 7.0,
    "minimum": 7.0,
}

LINE = {
    "main": 1.15,
    "secondary": 0.95,
    "ci": 0.75,
    "axis": 0.70,
    "reference": 0.65,
    "guide": 0.35,
}


def mm(value: float) -> float:
    """Convert millimetres to inches for Matplotlib."""

    return value / MM_PER_INCH


def resolve_font() -> tuple[str, str]:
    """Resolve the single permitted sans-serif family and return family/path."""

    for family in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        if Path(path).is_file():
            return family, path
    raise RuntimeError("Arial, Helvetica, or Liberation Sans is required")


def apply_style() -> dict[str, str]:
    """Apply the frozen figure style and return the resolved font record."""

    family, path = resolve_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family],
            "font.size": FONT["tick"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": LINE["axis"],
            "axes.titlesize": FONT["panel_title"],
            "axes.titleweight": "bold",
            "axes.labelsize": FONT["axis_label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": FONT["legend"],
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return {"family": family, "path": str(Path(path).resolve())}


def panel_label(ax, label: str, *, x: float = -0.13, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=FONT["panel_label"],
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def zero_line(ax, *, axis: str = "x") -> None:
    method = ax.axvline if axis == "x" else ax.axhline
    method(0.0, color=MUTED, linewidth=LINE["reference"], zorder=0)


def finish_axes(ax, *, horizontal_guides: bool = False) -> None:
    ax.tick_params(width=LINE["axis"], length=2.5)
    if horizontal_guides:
        ax.grid(axis="y", color=GUIDE, linewidth=LINE["guide"], zorder=0)
    else:
        ax.grid(False)


def save_figure(fig, output_stem: Path, *, dpi: int = 600) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    outputs = []
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": dpi}),
    ):
        path = output_stem.with_suffix(f".{extension}")
        # Preserve the declared physical canvas exactly. Tight bounding boxes
        # make nominally identical 178-mm figures drift in width and alter the
        # effective raster PPI at manuscript placement.
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs
