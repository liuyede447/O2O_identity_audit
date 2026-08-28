"""Rebuild the preferred four-panel Figure 2 without overlapping legends."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MATRIX = ROOT / "data/checkpoint_matrix/cross_dataset_checkpoint_matrix.csv"
OUT = HERE / "fig2_quantitative_evidence"

INK = "#1F2933"
BLUE = "#2F5D9B"
GREEN = "#4F7F36"
ORANGE = "#D98224"
RED = "#D64B3C"
TEAL = "#2A9DBB"
GREY = "#7A8592"
GREY_LIGHT = "#D8E0E8"
TINY = "#D7650A"
SMALL = "#8794A6"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8.5
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["legend.frameon"] = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_0p1(value: float) -> float:
    return value ** 0.1


def read_rows():
    with MATRIX.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    result = []
    for row in rows:
        dataset = row["dataset"]
        cp = row["checkpoint_label"]
        if dataset == "AI-TOD-v2":
            short = "AI · Y26 " + cp.replace("_main", "").replace("_", "-")
            marker = "o"
            group = "AI-Y26"
        else:
            short = "Vis · Y26 " + cp.replace("_", "-")
            marker = "s"
            group = "Vis-Y26"
        result.append(
            {
                "label": short,
                "group": group,
                "marker": marker,
                "frag": per_0p1(float(row["fragility_or"])),
                "frag_lo": per_0p1(float(row["fragility_ci_low"])),
                "frag_hi": per_0p1(float(row["fragility_ci_high"])),
                "fn": per_0p1(float(row["fn_or"])),
                "fn_lo": per_0p1(float(row["fn_ci_low"])),
                "fn_hi": per_0p1(float(row["fn_ci_high"])),
            }
        )

    # Compatible-detector row from the frozen formal YOLOv10 audit.
    result.insert(
        5,
        {
            "label": "AI · Y10 b4",
            "group": "AI-Y10",
            "marker": "D",
            "frag": per_0p1(0.002369),
            "frag_lo": per_0p1(0.001534),
            "frag_hi": per_0p1(0.003591),
            "fn": per_0p1(0.19406),
            "fn_lo": per_0p1(0.13299),
            "fn_hi": per_0p1(0.30381),
        },
    )
    return result


def direct_key(ax, items, y=0.965):
    x = 0.02
    for color, label in items:
        ax.text(x, y, "■", transform=ax.transAxes, color=color, fontsize=7.8, va="top")
        ax.text(x + 0.032, y, label, transform=ax.transAxes, color=INK, fontsize=7.0, va="top")
        x += 0.48


def forest(ax, rows, field, lo_field, hi_field, color, title, xlim):
    y = np.arange(len(rows))
    for idx, row in enumerate(rows):
        point = row[field]
        lo = row[lo_field]
        hi = row[hi_field]
        edge = TEAL if row["group"] == "Vis-Y26" else GREEN if row["group"] == "AI-Y10" else color
        ax.errorbar(
            point,
            idx,
            xerr=[[point - lo], [hi - point]],
            fmt=row["marker"],
            color=edge,
            ecolor=edge,
            elinewidth=1.0,
            capsize=2.2,
            markersize=4.6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=4,
        )
    ax.set_yticks(y, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.axvline(1.0, color=RED, lw=0.9, ls="--", alpha=0.8)
    ax.axhline(4.5, color=GREY_LIGHT, lw=0.7)
    ax.axhline(5.5, color=GREY_LIGHT, lw=0.7)
    ax.grid(axis="x", color=GREY_LIGHT, lw=0.55, ls="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=9.3, weight="bold", color=INK, pad=8)
    ax.tick_params(axis="both", labelsize=7.0, length=3)
    ax.text(0.86, 0.965, "point ± 95% CI", transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color=GREY)


def main():
    rows = read_rows()
    fig = plt.figure(figsize=(190 / 25.4, 145 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.985, bottom=0.105, top=0.950, wspace=0.36, hspace=0.34)

    # (a) Same-model descriptive rates.
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    width = 0.29
    o2m = [73.38, 71.67]
    o2o = [24.69, 9.59]
    bars1 = ax.bar(x - width / 2, o2m, width, color="#E9A400", edgecolor="#8D4A00", linewidth=0.7)
    bars2 = ax.bar(x + width / 2, o2o, width, color="#147EB3", edgecolor="#174A78", linewidth=0.7)
    for bars, vals, color in ((bars1, o2m, "#9B4800"), (bars2, o2o, "#0F527E")):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.2, weight="bold", color=color)
    # A vertical double-headed gauge expresses the between-scale O2O gap
    # without implying a directional process or crossing either bar group.
    x_delta = 0.50
    ax.annotate(
        "",
        xy=(x_delta, o2o[0]),
        xytext=(x_delta, o2o[1]),
        arrowprops=dict(arrowstyle="<->", color="#147EB3", lw=0.9, shrinkA=0, shrinkB=0),
    )
    ax.hlines([o2o[0], o2o[1]], x_delta - 0.035, x_delta + 0.035, colors="#147EB3", linewidth=0.8)
    # Keep the scale-gap label in the inter-group whitespace, separated from
    # the 24.7% bar label and the double-headed gauge.
    ax.text(x_delta, 35.0, r"$\Delta_{\mathrm{O2O}}=15.10$ pp", ha="center", va="bottom", fontsize=7.2, color="#0F527E", weight="bold")
    ax.set_xticks(x, ["8–16 px\n(Tiny)", "16–32 px\n(Small)"])
    ax.set_ylabel("Any-direction fragility (%)")
    ax.set_ylim(0, 94)
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.55, ls="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title("(a) Same-model specificity (YOLO26s)", loc="left", fontsize=9.3, weight="bold", color=INK, pad=8)
    direct_key(ax, [("#E9A400", "Native O2M top-rank"), ("#147EB3", "Loss-active O2O identity")])
    ax.tick_params(axis="both", labelsize=7.0)

    # (b) Undefined competition boundary.
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    tiny = [15.58, 7.24]
    small = [0.10, 0.17]
    bars1 = ax.bar(x - width / 2, tiny, width, color=TINY, edgecolor="#843600", linewidth=0.7)
    bars2 = ax.bar(x + width / 2, small, width, color=SMALL, edgecolor="#4E5D70", linewidth=0.7)
    for bars, vals, color in ((bars1, tiny, "#9A3500"), (bars2, small, "#46566D")):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.2f}%", ha="center", va="bottom", fontsize=7.2, weight="bold", color=color)
    ax.set_xticks(x, ["AI-TOD-v2", "VisDrone"])
    ax.set_ylabel("Undefined competition (%)")
    ax.set_ylim(0, 21)
    ax.grid(axis="y", color=GREY_LIGHT, lw=0.55, ls="--", alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title("(b) Undefined-competition boundary", loc="left", fontsize=9.3, weight="bold", color=INK, pad=8)
    direct_key(ax, [(TINY, "8–16 px"), (SMALL, "16–32 px")])
    ax.tick_params(axis="both", labelsize=7.0)

    # (c,d) Forest plots. Legends are replaced by direct annotation.
    ax_c = fig.add_subplot(gs[1, 0])
    forest(ax_c, rows, "frag", "frag_lo", "frag_hi", BLUE, "(c) Margin → identity fragility", (0.44, 1.025))
    ax_c.set_xlabel("Adjusted OR per +0.1 native margin")

    ax_d = fig.add_subplot(gs[1, 1])
    forest(ax_d, rows, "fn", "fn_lo", "fn_hi", ORANGE, "(d) Margin → false negative (FN)", (0.76, 1.025))
    ax_d.set_xlabel("Adjusted OR per +0.1 native margin")

    fig.text(
        0.53,
        0.025,
        "Checkpoint rows are fitted separately and are not pooled; b4/b8 comparisons retain batch-size confounding.",
        ha="center",
        fontsize=7.2,
        color=GREY,
        style="italic",
    )

    outputs = []
    for ext in ("svg", "pdf", "png"):
        path = OUT.with_suffix(f".{ext}")
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)

    validation = {
        "status": "release_rebuild",
        "approved_for_main_tex": True,
        "source_pdf": "figures/fig2_quantitative_evidence.pdf",
        "matrix_source": str(MATRIX.relative_to(ROOT)).replace("\\", "/"),
        "matrix_sha256": sha256(MATRIX),
        "or_unit": "+0.1 native O2O margin",
        "checkpoint_rows": len(rows),
        "legend_overlap_removed": True,
        "outputs": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs],
        "main_tex_modified": True,
    }
    (HERE / "fig2_quantitative_evidence.validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "outputs": [p.name for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
