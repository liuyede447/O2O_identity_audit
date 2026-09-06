"""Render Figure 5 from frozen Rank-Set and positive-count artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures_final"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    BRANCH,
    FONT,
    GUIDE,
    INK,
    LINE,
    MUTED,
    apply_style,
    finish_axes,
    mm,
    panel_label,
    save_figure,
    zero_line,
)


SOURCES = {
    "rank_fixed": (
        "runs/20260831T161235_rank_set_fixed_v1/artifact/rank_set_summary.json",
        "649627210a886208014733f881429f6572f414d306bb051fe52c6ef2758ed4b4",
        "EV-RANK-SET-FIXED-V1",
    ),
    "rank_equivalent": (
        "runs/20260831T162551_rank_set_normalized_v1/artifact/rank_set_summary.json",
        "7f1a83d6b538143dbf1ff38b8deecc7627795298d73733a72abfd4b7e102b11d",
        "EV-RANK-SET-NORMALIZED-V1",
    ),
    "count_summary": (
        "runs/20260905_o2m_positive_count_scale_controls_v5/artifact/summary.json",
        "2b232e52716cb5a45ef97f4bf64dd6458308fb639283762a30824e90cc056e9f",
        "SOURCE_ADDENDUM",
    ),
    "count_strata": (
        "runs/20260905_o2m_positive_count_scale_controls_v5/artifact/exact_positive_count_strata.csv",
        "8a68539c42a30b5db38ac3ca948f6b862990a885a84b21be335aa99de3c0f365",
        "SOURCE_ADDENDUM",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict:
    records = {}
    for key, (relative, expected, authority) in SOURCES.items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA mismatch for {relative}: {actual}")
        records[key] = {"path": relative, "sha256": actual, "authority": authority}
    return records


def load_json(key: str) -> dict:
    return json.loads((ROOT / SOURCES[key][0]).read_text(encoding="utf-8"))


def source_rows() -> tuple[list[dict], dict]:
    sources = verify()
    fixed = load_json("rank_fixed")
    equivalent = load_json("rank_equivalent")
    count = load_json("count_summary")
    strata = pd.read_csv(ROOT / SOURCES["count_strata"][0])
    rows: list[dict] = []

    metrics = [
        ("Exact set equality", "exact_set_equal", "set"),
        ("Jaccard", "jaccard", "set"),
        ("Base retention", "base_retention", "set"),
        ("Assigned-set Top-1 flip", "assigned_set_q_top1_flip", "rank"),
        ("Rank-only among flips", "rank_only_fraction_among_assigned_top1_flips", "rank"),
    ]
    for contract, payload, source_key in (
        ("Fixed 1 px", fixed, "rank_fixed"),
        ("Equivalent-area-side", equivalent, "rank_equivalent"),
    ):
        for label, key, group in metrics:
            name = f"t_8_16_minus_s_16_32.{key}"
            low, high = payload["ci_95"][name]
            rows.append(
                {
                    "panel": "b",
                    "contract": contract,
                    "metric": label,
                    "metric_group": group,
                    "estimate": -100 * payload["point"][name],
                    "ci_low": -100 * high,
                    "ci_high": -100 * low,
                    "orientation": "16-32 minus 8-16",
                    "source_orientation": "8-16 minus 16-32; sign and interval endpoints inverted for display",
                    "source_path": sources[source_key]["path"],
                    "source_sha256": sources[source_key]["sha256"],
                    "evidence_id": sources[source_key]["authority"],
                }
            )

    total_tiny = float(strata["object_weight_tiny"].sum())
    total_small = float(strata["object_weight_small"].sum())
    for record in strata.to_dict("records"):
        for size, total in (("8-16 px", total_tiny), ("16-32 px", total_small)):
            weight = float(record["object_weight_tiny" if size == "8-16 px" else "object_weight_small"])
            rows.append(
                {
                    "panel": "c",
                    "size": size,
                    "positive_count": int(record["positive_count"]),
                    "weighted_proportion": weight / total if total else 0.0,
                    "adequate_overlap": bool(record["adequate_overlap_support"]),
                    "source_path": sources["count_strata"]["path"],
                    "source_sha256": sources["count_strata"]["sha256"],
                    "evidence_id": "SOURCE_ADDENDUM",
                }
            )

    standard_metrics = [
        ("Object Top-1 fragility", "object_any_top1_fragility"),
        ("Directional Top-1 flip", "direction_top1_flip"),
        ("Positive-set turnover", "rank_set_turnover"),
    ]
    for label, key in standard_metrics:
        item = count["standardized_analyses"][key]["adequate_overlap_support"]
        bundle = count["standardized_analyses"][key]
        point = item["point"]
        interval = item["ci_95_gap_small_minus_tiny"]
        full_interval = bundle["full_crude_ci_95_gap_small_minus_tiny"]
        overlap_interval = bundle["adequate_overlap_crude_ci_95_gap_small_minus_tiny"]
        rows.append(
            {
                "panel": "d",
                "metric": label,
                "crude": 100 * point["crude_gap_small_minus_tiny"],
                "crude_ci_low": 100 * full_interval["low"],
                "crude_ci_high": 100 * full_interval["high"],
                "overlap_crude": 100 * point["overlap_crude_gap_small_minus_tiny"],
                "overlap_ci_low": 100 * overlap_interval["low"],
                "overlap_ci_high": 100 * overlap_interval["high"],
                "standardized": 100 * point["standardized_gap_small_minus_tiny"],
                "std_ci_low": 100 * interval["low"],
                "std_ci_high": 100 * interval["high"],
                "orientation": "16-32 minus 8-16",
                "coverage_tiny": point["coverage_tiny"],
                "coverage_small": point["coverage_small"],
                "support_note": "full crude, K=3-7 overlap crude, and K=3-7 standardized estimates separate restriction from count-composition changes",
                "source_path": sources["count_summary"]["path"],
                "source_sha256": sources["count_summary"]["sha256"],
                "evidence_id": "SOURCE_ADDENDUM",
            }
        )
    metadata = {
        "status": "SOURCE_BOUND_WITH_ADDENDUM",
        "figure": "Figure 5",
        "sources": sources,
        "rank_set_support": {
            "fixed": {"images": fixed["images"], "focal_gt": fixed["focal_gt"], "directions": fixed["direction_rows"]},
            "equivalent": {"images": equivalent["images"], "focal_gt": equivalent["focal_gt"], "directions": equivalent["direction_rows"]},
        },
        "count_support": {"common_valid_gt": 6377, "contributing_images": 288, "selected_images": 300, "adequate_overlap_k": [3, 4, 5, 6, 7]},
        "bootstrap": {"rank_set": 5000, "count_standardization": count["bootstrap"]},
        "warnings": [
            "K=0 is retained in the distribution although full_common_support begins at K=1.",
            "Full-support crude, overlap-support crude, and overlap-support standardized estimates are shown separately.",
            "Positive-count controls are registered as a figure-only SOURCE_ADDENDUM, not a new final_evidence_v3 entry.",
        ],
    }
    return rows, metadata


def write_sources(rows: list[dict], metadata: dict) -> None:
    directory = FIG / "source_data"
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    csv_path = directory / "fig5_source.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata["source_csv"] = str(csv_path.relative_to(ROOT)).replace("\\", "/")
    metadata["source_csv_sha256"] = sha256(csv_path)
    (directory / "fig5_source.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def draw_set_case(ax, x0: float, title: str, base: list[str], shifted: list[str], base_top: str, shift_top: str) -> None:
    ax.text(x0 + 0.13, 0.92, title, transform=ax.transAxes, fontsize=FONT["axis_label"], fontweight="bold", ha="center", va="top", color=INK)
    ax.text(x0 + 0.025, 0.67, "Base", transform=ax.transAxes, fontsize=FONT["annotation"], color=INK, ha="left")
    ax.text(x0 + 0.025, 0.30, "Shift", transform=ax.transAxes, fontsize=FONT["annotation"], color=INK, ha="left")
    for row_y, members, top in ((0.70, base, base_top), (0.33, shifted, shift_top)):
        for index, member in enumerate(members):
            cx = x0 + 0.145 + 0.072 * index
            face = BRANCH["O2M"] if member == top else "white"
            circle = Circle((cx, row_y), 0.032, transform=ax.transAxes, facecolor=face, edgecolor=BRANCH["O2M"], linewidth=LINE["main"], clip_on=False)
            ax.add_patch(circle)
            ax.text(cx, row_y, member, transform=ax.transAxes, fontsize=8.4, ha="center", va="center", color=INK)


def render(rows: list[dict]) -> None:
    font = apply_style()
    data = pd.DataFrame(rows)
    orientations = set(data.loc[data.panel.isin(["b", "d"]), "orientation"].dropna())
    if orientations != {"16-32 minus 8-16"}:
        raise RuntimeError(f"Figure 5 contrast orientations are inconsistent: {sorted(orientations)}")
    fig, axes = plt.subplots(2, 2, figsize=(mm(178), mm(116)))
    axa, axb, axc, axd = axes.flat

    axa.set_axis_off()
    draw_set_case(axa, 0.00, "Rank-only", ["A", "B", "C"], ["A", "B", "C"], "A", "B")
    draw_set_case(axa, 0.34, "Rank + set", ["A", "B", "C"], ["B", "C", "D"], "A", "B")
    draw_set_case(axa, 0.68, "Set-only; Top-1 stable", ["A", "B", "C"], ["A", "C", "D"], "A", "A")
    axa.legend(handles=[
        Line2D([], [], marker="o", color=BRANCH["O2M"], markerfacecolor=BRANCH["O2M"], linestyle="none", label="Top-1"),
        Line2D([], [], marker="o", color=BRANCH["O2M"], markerfacecolor="white", linestyle="none", label="Other positive"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2,
       handletextpad=0.3, columnspacing=0.7, borderaxespad=0.0)
    panel_label(axa, "a", x=-0.02, y=1.01)

    pb = data[data.panel == "b"].copy()
    metric_order = ["Exact set equality", "Jaccard", "Base retention", "Assigned-set Top-1 flip", "Rank-only among flips"]
    ybase = np.arange(len(metric_order))[::-1]
    for offset, contract, open_marker in ((0.12, "Fixed 1 px", False), (-0.12, "Equivalent-area-side", True)):
        subset = pb[pb.contract == contract].set_index("metric").loc[metric_order]
        for index, record in enumerate(subset.itertuples()):
            axb.errorbar(record.estimate, ybase[index] + offset, xerr=[[record.estimate - record.ci_low], [record.ci_high - record.estimate]], fmt="s" if open_marker else "o", markersize=4.0, markerfacecolor="white" if open_marker else BRANCH["O2M"], markeredgecolor=BRANCH["O2M"], ecolor=BRANCH["O2M"], elinewidth=LINE["ci"], capsize=2, label=("Eq.-area" if open_marker else "Fixed") if index == 0 else None)
    zero_line(axb)
    axb.axhspan(1.5, 4.5, color=GUIDE, alpha=0.16, zorder=0)
    axb.set_yticks(ybase, metric_order)
    axb.set_xlabel("16–32 minus 8–16 px contrast (pp)")
    axb.set_title("Native Rank-Set contrasts", loc="left", pad=4)
    axb.text(0.985, 0.88, "SET PERSISTENCE", transform=axb.transAxes,
             ha="right", va="center", fontsize=FONT["minimum"],
             color=MUTED, fontweight="bold")
    axb.text(0.985, 0.18, "RANK TURNOVER", transform=axb.transAxes,
             ha="right", va="center", fontsize=FONT["minimum"],
             color=MUTED, fontweight="bold")
    axb.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
               borderaxespad=0.0, columnspacing=0.8, handletextpad=0.35)
    finish_axes(axb)
    panel_label(axb, "b")

    pc = data[data.panel == "c"].copy()
    axc.axvspan(2.5, 7.5, color=GUIDE, alpha=0.18, zorder=0)
    for size, marker, color, offset in (("8-16 px", "o", BRANCH["O2O"], -0.08),
                                        ("16-32 px", "^", BRANCH["O2M"], 0.08)):
        subset = pc[pc["size"] == size].sort_values("positive_count")
        x = subset.positive_count.to_numpy(float) + offset
        y = subset.weighted_proportion.to_numpy(float)
        axc.vlines(x, 0, y, color=color, linewidth=LINE["ci"], alpha=0.75)
        axc.plot(x, y, marker=marker, markersize=3.8, linestyle="none", color=color, label=size)
    axc.set_xticks(range(0, 11))
    axc.set_xlabel("Native positive count K")
    axc.set_ylabel("Weighted proportion")
    axc.set_title("Positive-count distribution", loc="left", pad=4)
    axc.legend(loc="upper right")
    axc.text(5, axc.get_ylim()[1] * 0.92, "K=3–7 overlap", ha="center", fontsize=FONT["annotation"], color=MUTED)
    finish_axes(axc)
    panel_label(axc, "c")

    pd_ = data[data.panel == "d"].copy().reset_index(drop=True)
    y = np.arange(len(pd_))[::-1]
    for index, record in pd_.iterrows():
        axd.plot([record.crude, record.overlap_crude, record.standardized], [y[index]] * 3, color=GUIDE, linewidth=LINE["secondary"], zorder=1)
        axd.errorbar(record.crude, y[index], xerr=[[record.crude-record.crude_ci_low], [record.crude_ci_high-record.crude]], fmt="o", color=MUTED, ecolor=MUTED, markersize=4.0, elinewidth=LINE["ci"], capsize=2, label="Full crude" if index == 0 else None)
        axd.errorbar(record.overlap_crude, y[index], xerr=[[record.overlap_crude-record.overlap_ci_low], [record.overlap_ci_high-record.overlap_crude]], fmt="o", markerfacecolor="white", markeredgecolor=MUTED, ecolor=MUTED, markersize=4.0, elinewidth=LINE["ci"], capsize=2, label="Overlap crude" if index == 0 else None)
        axd.errorbar(record.standardized, y[index], xerr=[[record.standardized - record.std_ci_low], [record.std_ci_high - record.standardized]], fmt="D", markersize=4.2, color=BRANCH["Paired"], ecolor=BRANCH["Paired"], elinewidth=LINE["ci"], capsize=2, label="K=3–7 standardized" if index == 0 else None)
    zero_line(axd)
    axd.set_xlim(-8, 52)
    axd.set_yticks(y, pd_["metric"])
    axd.set_xlabel("16–32 minus 8–16 px contrast (pp)")
    axd.set_title("Support restriction precedes K standardisation", loc="left", pad=4)
    axd.legend(loc="upper right", frameon=True, facecolor="white",
               edgecolor="none", framealpha=1.0)
    finish_axes(axd)
    panel_label(axd, "d")

    stem = FIG / "draft" / "Fig5"
    save_figure(fig, stem, dpi=600)
    validation = {
        "status": "PASS",
        "figure": "Fig5",
        "resolved_font": font,
        "source_rows": len(rows),
        "source_addendum_used": True,
        "warnings_preserved": ["K=0 retained", "support restriction and K standardisation are separated"],
        "interpretation_guards": {"o2m_top1_unique_positive": False, "causal_count_explanation": False},
        "contrast_orientation": "16-32 minus 8-16 in panels b and d",
        "outputs": {ext: sha256(stem.with_suffix(f".{ext}")) for ext in ("pdf", "svg", "png")},
    }
    path = FIG / "qa" / "Fig5.validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    rows, metadata = source_rows()
    write_sources(rows, metadata)
    render(rows)


if __name__ == "__main__":
    main()
