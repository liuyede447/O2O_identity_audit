"""Render Figure 6 from dense-grid v3 and held-out diagnostic artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
    "events": (
        "runs/20260902_boundary_full_grid_1024_population_v1/per_direction_boundary_events.csv",
        "ee33d0927ebbb9953a8e0e3f6c02956ef02cb6570beb3660e03ab172527e897d",
        "EV-BOUNDARY-GEOMETRY-V3",
    ),
    "margin": (
        "runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/margin_construct.csv",
        "e818cfd5a12b167b8aea1f174368a5d2f1457fa84a31251a6b4d03c6b565ea22",
        "EV-MARGIN-RHOR-CONSTRUCT-V3",
    ),
    "rho_status": (
        "runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/rho_R_status.csv",
        "852482d938a5c10e61834a3327d342a5b6134b42b895cc291066bc7d45c43bab",
        "EV-MARGIN-RHOR-CONSTRUCT-V3",
    ),
    "increment_ai": (
        "results/reviewer_controls_20260829/formal_s0e300_k00625/incremental_value_summary_foldwise.json",
        "0f555db9c5e1ecf3d738607551d45beb75f6f0a6f3e0820683b27d9aa5b32a88",
        "SOURCE_ADDENDUM",
    ),
    "increment_vis": (
        "results/reviewer_controls_20260829/vis_b4e150_equiv_k00625/incremental_value_summary_foldwise.json",
        "8227b831cfe73cc3533ae1b953e21e62c138e5f9205b35e54eba38c38371ed5c",
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


def source_rows() -> tuple[list[dict], dict, pd.DataFrame]:
    sources = verify()
    usecols = ["image_id", "size_bin", "sampling_weight", "o2o_base_margin", "rho_R_event_observed", "rho_R_radius"]
    events = pd.read_csv(ROOT / SOURCES["events"][0], usecols=usecols)
    observed = events[events["rho_R_event_observed"].eq(1)].dropna(subset=["o2o_base_margin", "rho_R_radius"]).copy()
    if len(observed) != 1593 or observed["image_id"].nunique() != 173:
        raise RuntimeError("observed-crossing support differs from frozen contract")
    margin = pd.read_csv(ROOT / SOURCES["margin"][0])
    status = pd.read_csv(ROOT / SOURCES["rho_status"][0])
    rows: list[dict] = []

    for record in status[(status.size_group == "all") & status.status.isin(["observed", "competing_censored", "right_censored", "undefined"])].to_dict("records"):
        rows.append({"panel": "b", "status": record["status"], "n_rows": int(record["n_rows"]), "estimate": record["estimate"], "ci_low": record["ci_low"], "ci_high": record["ci_high"], "source_path": sources["rho_status"]["path"], "source_sha256": sources["rho_status"]["sha256"], "evidence_id": sources["rho_status"]["authority"]})

    display = {"all": "All", "t_8_16": "8-16 px", "s_16_32": "16-32 px"}
    analysis_labels = {
        "margin_vs_observed_rho_R": "Rank crossing",
        "specificity_control_margin_vs_eligibility_cause_overall_radius": "Eligibility boundary",
    }
    for record in margin[margin.analysis.isin(analysis_labels)].to_dict("records"):
        rows.append({"panel": "c", "size": display[record["size_group"]], "analysis": analysis_labels[record["analysis"]], "estimate": record["estimate"], "ci_low": record["ci_low"], "ci_high": record["ci_high"], "n_rows": int(record["n_rows"]), "n_images": int(record["n_images"]), "source_path": sources["margin"]["path"], "source_sha256": sources["margin"]["sha256"], "evidence_id": sources["margin"]["authority"]})

    metric_map = [
        ("ROC-AUC", "delta_roc_auc", 1.0),
        ("PR-AUC", "delta_pr_auc", 1.0),
        ("Log loss", "delta_log_loss", -1.0),
        ("Brier", "delta_brier", -1.0),
    ]
    for dataset, key, checkpoint, payload in (
        ("AI-TOD-v2", "increment_ai", "4b57787f7351...", load_json("increment_ai")),
        ("VisDrone", "increment_vis", "e7cffd98...", load_json("increment_vis")),
    ):
        for label, metric, sign in metric_map:
            raw = payload["M3_minus_M2_point"][metric]
            low, high = payload["M3_minus_M2_ci_95"][metric]
            oriented_low, oriented_high = ((low, high) if sign > 0 else (-high, -low))
            rows.append({"panel": "d", "dataset": dataset, "metric": label, "estimate": 1000 * sign * raw, "ci_low": 1000 * oriented_low, "ci_high": 1000 * oriented_high, "orientation": "right is held-out improvement; log loss and Brier signs inverted", "n_gt": payload["n_gt"], "n_images": payload["n_images"], "checkpoint": checkpoint, "source_path": sources[key]["path"], "source_sha256": sources[key]["sha256"], "evidence_id": "SOURCE_ADDENDUM"})

    metadata = {
        "status": "SOURCE_BOUND_WITH_ADDENDUM",
        "figure": "Figure 6",
        "sources": sources,
        "observed_crossing_support": {"rows": 1593, "images": 173, "conditioning": "observed fixed-pair crossings only"},
        "trajectory_status_support": {"rows": 29724, "bootstrap": "5000 stratified image-cluster percentile replicates"},
        "incremental_support": {"AI-TOD-v2": {"gt": 6410, "images": 288}, "VisDrone": {"gt": 13946, "images": 300}},
        "warnings": ["correlations use separately observed supports and are not censoring-adjusted", "held-out incremental artifacts are figure-only SOURCE_ADDENDUM records"],
        "scatter_rule": "raw observed trajectories plus ten descriptive margin-decile medians; no regression fit",
    }
    return rows, metadata, observed


def write_sources(rows: list[dict], metadata: dict, observed: pd.DataFrame) -> None:
    directory = FIG / "source_data"
    directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    csv_path = directory / "fig6_source.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    raw_path = directory / "fig6_observed_crossings.csv"
    observed.to_csv(raw_path, index=False, lineterminator="\n")
    metadata.update({"source_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"), "source_csv_sha256": sha256(csv_path), "observed_crossings_csv": str(raw_path.relative_to(ROOT)).replace("\\", "/"), "observed_crossings_csv_sha256": sha256(raw_path)})
    (directory / "fig6_source.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def render(rows: list[dict], observed: pd.DataFrame) -> None:
    font = apply_style()
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(mm(178), mm(116)))
    axa, axb, axc, axd = axes.flat

    axa.scatter(observed["o2o_base_margin"], observed["rho_R_radius"], s=5, alpha=0.18, color=BRANCH["O2O"], edgecolors="none", rasterized=False)
    deciles = observed.assign(
        margin_decile=pd.qcut(observed["o2o_base_margin"], q=10, labels=False, duplicates="drop")
    ).groupby("margin_decile", as_index=False).agg(
        margin_median=("o2o_base_margin", "median"),
        crossing_median=("rho_R_radius", "median"),
    )
    axa.plot(deciles["margin_median"], deciles["crossing_median"],
             color=BRANCH["O2O"], linewidth=LINE["secondary"], marker="o",
             markersize=3.2, markerfacecolor="white", markeredgewidth=0.8,
             label="decile medians (descriptive)")
    axa.set_xlabel("Native active-relative margin")
    axa.set_ylabel("Observed rank-crossing distance")
    axa.set_title("Observed fixed-pair crossings only\nn=1,593 · ρ=0.628 · no regression fit", loc="left", pad=4)
    axa.legend(loc="upper left")
    finish_axes(axa)
    panel_label(axa, "a")

    pb = data[data.panel == "b"].copy()
    order = ["observed", "competing_censored", "right_censored", "undefined"]
    colors = ["#A9D5E8", "#AEB6BF", "#7F8C8D", "#D5D8DC"]
    labels = ["Observed crossing", "Competing censor", "Right censor", "Undefined"]
    left = 0.0
    observed_value = None
    for key, label, color in zip(order, labels, colors):
        value = float(pb.loc[pb.status == key, "estimate"].iloc[0])
        if key == "observed":
            observed_value = value
        axb.barh([0], [value], left=left, color=color, edgecolor="white", linewidth=0.6, height=0.38, label=label)
        if value > 0.08:
            axb.text(left + value / 2, 0, f"{100*value:.1f}%", ha="center", va="center", fontsize=FONT["annotation"], color=INK)
        left += value
    if observed_value is None:
        raise RuntimeError("Observed-crossing proportion is missing")
    axb.annotate(
        f"{100 * observed_value:.1f}% observed",
        xy=(observed_value / 2, 0.19), xytext=(0.16, 0.48),
        ha="center", va="bottom", fontsize=FONT["annotation"], color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": LINE["reference"]},
    )
    axb.set_xlim(0, 1)
    axb.set_ylim(-0.72, 0.72)
    axb.set_yticks([])
    axb.set_xlabel("Weighted trajectory proportion")
    axb.set_title("Most trajectories do not yield an observed crossing", loc="left", pad=4)
    axb.legend(loc="lower center", ncol=2)
    finish_axes(axb)
    panel_label(axb, "b")

    pc = data[data.panel == "c"].copy()
    sizes = ["All", "8-16 px", "16-32 px"]
    y = np.arange(len(sizes))[::-1]
    for offset, analysis, marker, color in ((0.11, "Rank crossing", "o", BRANCH["O2O"]), (-0.11, "Eligibility boundary", "s", MUTED)):
        subset = pc[pc.analysis == analysis].set_index("size").loc[sizes]
        for index, record in enumerate(subset.itertuples()):
            axc.errorbar(record.estimate, y[index] + offset, xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]], fmt=marker, markersize=4.1, color=color, ecolor=color, elinewidth=LINE["ci"], capsize=2, label=analysis if index == 0 else None)
    axc.set_yticks(y, sizes)
    axc.set_xlim(0, 0.76)
    axc.set_xlabel("Conditional Spearman ρ")
    axc.set_title("Rank alignment is stronger than eligibility alignment", loc="left", pad=4)
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    finish_axes(axc)
    panel_label(axc, "c")

    pd_ = data[data.panel == "d"].copy()
    metrics = ["ROC-AUC", "PR-AUC", "Log loss", "Brier"]
    y = np.arange(len(metrics))[::-1]
    for offset, dataset, marker, filled in ((0.11, "AI-TOD-v2", "o", True), (-0.11, "VisDrone", "s", False)):
        subset = pd_[pd_.dataset == dataset].set_index("metric").loc[metrics]
        for index, record in enumerate(subset.itertuples()):
            axd.errorbar(record.estimate, y[index] + offset, xerr=[[record.estimate-record.ci_low], [record.ci_high-record.estimate]], fmt=marker, markersize=4.1, markerfacecolor=BRANCH["Paired"] if filled else "white", markeredgecolor=BRANCH["Paired"], ecolor=BRANCH["Paired"], elinewidth=LINE["ci"], capsize=2, label=dataset if index == 0 else None)
    zero_line(axd)
    axd.set_yticks(y, metrics)
    # Arial lacks U+207B (superscript minus); U+02D7 is the corresponding
    # raised modifier-minus glyph and keeps the complete label in embedded
    # Arial rather than silently introducing a second PDF math font.
    axd.set_xlabel("Held-out improvement after adding margin (10˗³)")
    axd.set_title("No stable incremental downstream diagnostic value", loc="left", pad=4)
    axd.legend(loc="lower right")
    axd.text(0.98, 0.96, "Positive = improvement\nLog loss/Brier sign-reversed",
             transform=axd.transAxes, ha="right", va="top",
             fontsize=FONT["annotation"], color=MUTED)
    finish_axes(axd)
    panel_label(axd, "d")

    stem = FIG / "draft" / "Fig6"
    save_figure(fig, stem, dpi=600)
    validation = {"status": "PASS", "figure": "Fig6", "resolved_font": font, "source_rows": len(rows), "observed_rows": len(observed), "descriptive_decile_bins": len(deciles), "source_addendum_used": True, "interpretation_guards": {"population_predictor": False, "censoring_adjusted_claim": False, "causal_claim": False, "regression_fit": False}, "outputs": {ext: sha256(stem.with_suffix(f".{ext}")) for ext in ("pdf", "svg", "png")}}
    path = FIG / "qa" / "Fig6.validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    rows, metadata, observed = source_rows()
    write_sources(rows, metadata, observed)
    render(rows, observed)


if __name__ == "__main__":
    main()
