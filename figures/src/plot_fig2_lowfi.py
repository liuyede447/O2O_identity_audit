"""Generate the source-bound Figure 2 submission draft.

Every estimate and interval is loaded from a SHA-verified frozen artifact. The
script writes panel source CSV/JSON before rendering SVG, PDF, and 600-dpi PNG.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    BRANCH,
    BRANCH_MARKER,
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
    "fixed": (
        "evidence/artifacts/results/measurement_validation_20260831/fixed_branch_replay_v1/normalized_perturbation_summary.json",
        "aece532fd9ea842247ed22ea03557e13ece6209144d83f3598beadd38ba0bcfe",
    ),
    "equivalent": (
        "evidence/artifacts/results/reviewer_controls_20260829/formal_s0e300_k00625/normalized_perturbation_summary.json",
        "5d09278684cf683a8b6227b7e792f162f01f3e65718327b6cf51e0fdc669b37b",
    ),
    "kappa": (
        "evidence/artifacts/results/stress_contract_sensitivity_20260829/frozen_matrix/kappa_sensitivity_matrix.csv",
        "38ee31b39b02f383cba7b816aa3b34fdccc7440d051cb725524bfb879815f0a8",
    ),
    "dynamics": (
        "evidence/artifacts/results/five_experiment_upgrade_20260830/learning_dynamics/learning_dynamics_trajectory.csv",
        "3cb0c2d4bce8ca85d458a69a063c241457a72244feafa82465bf4ceaf1705aa9",
    ),
    "yolov10": (
        "evidence/artifacts/results/five_experiment_upgrade_20260830/yolov10_normalized_k00625/normalized_perturbation_summary.json",
        "ca344ce4f99d92d4654b65c1c1e97818bf385f9d3a57695bc3e37d6f4af275eb",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> dict[str, dict[str, str]]:
    records = {}
    for key, (relative, expected) in SOURCES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"source SHA mismatch for {key}: {actual} != {expected}")
        records[key] = {"path": relative, "sha256": actual}
    return records


def load_json(key: str) -> dict:
    return json.loads((ROOT / SOURCES[key][0]).read_text(encoding="utf-8"))


def append_row(rows: list[dict], **values) -> None:
    rows.append(values)


def build_source_rows() -> tuple[list[dict], dict]:
    verified = verify_sources()
    fixed = load_json("fixed")
    equivalent = load_json("equivalent")
    y10 = load_json("yolov10")
    rows: list[dict] = []

    metrics = (
        ("O2O", "o2o_scale_contrast", "O2O"),
        ("O2M", "o2m_scale_contrast", "O2M"),
        ("Paired", "paired_difference_in_scale_gradients", "Paired"),
    )
    for contract, payload, source_key in (
        ("Fixed 1 px", fixed, "fixed"),
        ("Equivalent-area-side", equivalent, "equivalent"),
    ):
        for label, metric, branch in metrics:
            low, high = payload["ci_95"][metric]
            append_row(
                rows,
                panel="a",
                row=f"{contract} · {label}",
                series=branch,
                x=contract,
                estimate=100.0 * payload["point"][metric],
                ci_low=100.0 * low,
                ci_high=100.0 * high,
                unit="percentage points",
                evidence_id="EV-STRESS-CONTRACT-PRIMARY-V1",
                source_path=verified[source_key]["path"],
                source_sha256=verified[source_key]["sha256"],
            )

    kappa = pd.read_csv(ROOT / SOURCES["kappa"][0], dtype={"dataset": str})
    for record in kappa.to_dict("records"):
        append_row(
            rows,
            panel="b",
            row=f"{record['dataset']} · κ={record['kappa']}",
            series=record["dataset"],
            x=float(record["kappa"]),
            estimate=100.0 * float(record["paired_difference_in_scale_gradients"]),
            ci_low=100.0 * float(record["paired_difference_in_scale_gradients_ci_low"]),
            ci_high=100.0 * float(record["paired_difference_in_scale_gradients_ci_high"]),
            unit="percentage points",
            evidence_id="EV-STRESS-CONTRACT-KAPPA-DATASET-V1",
            source_path=verified["kappa"]["path"],
            source_sha256=verified["kappa"]["sha256"],
        )

    dynamics = pd.read_csv(ROOT / SOURCES["dynamics"][0])
    for record in dynamics.to_dict("records"):
        label = "Fixed 1 px" if record["stress_mode"] == "fixed_1px" else "Equivalent-area-side"
        append_row(
            rows,
            panel="c",
            row=f"epoch {int(record['trajectory_epoch'])} · {label}",
            series=label,
            x=float(record["trajectory_epoch"]),
            estimate=float(record["paired_contrast_pp"]),
            ci_low=float(record["paired_ci_low_pp"]),
            ci_high=float(record["paired_ci_high_pp"]),
            unit="percentage points",
            evidence_id="EV-LEARNING-DYNAMICS-V1",
            source_path=verified["dynamics"]["path"],
            source_sha256=verified["dynamics"]["sha256"],
            checkpoint_sha256=record["checkpoint_sha256"],
            checkpoint_file=record["checkpoint_file"],
        )

    for detector, payload, source_key in (
        ("YOLO26s", equivalent, "equivalent"),
        ("YOLOv10-S", y10, "yolov10"),
    ):
        evidence_id = "EV-STRESS-CONTRACT-PRIMARY-V1" if detector == "YOLO26s" else "EV-YOLOV10-NORMALIZED-V1"
        for label, metric, branch in metrics:
            low, high = payload["ci_95"][metric]
            append_row(
                rows,
                panel="d",
                row=f"{detector} · {label}",
                series=branch,
                x=detector,
                estimate=100.0 * payload["point"][metric],
                ci_low=100.0 * low,
                ci_high=100.0 * high,
                unit="percentage points",
                evidence_id=evidence_id,
                source_path=verified[source_key]["path"],
                source_sha256=verified[source_key]["sha256"],
            )

    metadata = {
        "status": "SOURCE_BOUND",
        "figure": "Figure 2 low-fidelity",
        "scientific_claim": "Observed scale response depends on the stress contract; the paired sign persists with different branch-wise decomposition.",
        "sources": verified,
        "source_row_count": len(rows),
        "default_statistical_unit": "common-valid focal GT clustered by sampled image",
        "default_weighting": "inverse image-inclusion weights",
        "default_bootstrap": "5000 stratified image-cluster replicates",
        "exceptions": {
            "learning_dynamics": "row-specific archived checkpoint; no interpolation",
            "yolov10": "compatible native detector contract, not architecture generalisation",
        },
    }
    return rows, metadata


def write_sources(rows: list[dict], metadata: dict) -> None:
    source_dir = FIG_ROOT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    csv_path = source_dir / "fig2_source.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        **metadata,
        "fig2_source_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "fig2_source_csv_sha256": sha256(csv_path),
    }
    (source_dir / "fig2_source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def forest(ax, frame: pd.DataFrame, title: str) -> None:
    frame = frame.reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    for index, record in frame.iterrows():
        branch = record["series"]
        marker = BRANCH_MARKER[branch]
        open_marker = "Equivalent-area-side" in record["row"]
        ax.errorbar(
            record["estimate"],
            y[index],
            xerr=[[record["estimate"] - record["ci_low"]], [record["ci_high"] - record["estimate"]]],
            fmt=marker,
            markersize=4.2,
            markerfacecolor="white" if open_marker else BRANCH[branch],
            markeredgecolor=BRANCH[branch],
            markeredgewidth=0.9,
            ecolor=BRANCH[branch],
            elinewidth=LINE["ci"],
            capsize=2.0,
            zorder=3,
        )
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["row"].str.replace("Equivalent-area-side", "Eq.-area", regex=False))
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("8–16 minus 16–32 px contrast (pp)")
    finish_axes(ax)


def render(rows: list[dict]) -> None:
    font = apply_style()
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(mm(178), mm(116)))
    axa, axb, axc, axd = axes.flat

    forest(axa, data[data.panel == "a"], "Primary branch decomposition")
    axa.set_xlim(-28, 42)
    panel_label(axa, "a")

    panel_b = data[data.panel == "b"].copy()
    for dataset, marker, filled in (("AI-TOD-v2", "o", True), ("VisDrone", "s", False)):
        subset = panel_b[panel_b.series == dataset].sort_values("x")
        axb.errorbar(
            subset.x,
            subset.estimate,
            yerr=[subset.estimate - subset.ci_low, subset.ci_high - subset.estimate],
            fmt=marker,
            markersize=4.4,
            markerfacecolor=BRANCH["Paired"] if filled else "white",
            markeredgecolor=BRANCH["Paired"],
            markeredgewidth=0.9,
            ecolor=BRANCH["Paired"],
            elinewidth=LINE["ci"],
            capsize=2,
            label=dataset,
        )
    axb.set_title("Tested stress magnitudes", loc="left", pad=4)
    axb.set_xlabel("Equivalent-area-side κ")
    axb.set_ylabel("Paired O2O−O2M contrast (pp)")
    axb.set_xticks([0.03125, 0.0625, 0.125], ["0.03125", "0.0625", "0.125"])
    axb.legend(loc="center right", bbox_to_anchor=(0.98, 0.64), frameon=True,
               facecolor="white", edgecolor="none", framealpha=1.0)
    finish_axes(axb)
    panel_label(axb, "b")

    panel_c = data[data.panel == "c"].copy()
    stage_position = {0.0: 0.0, 20.0: 1.0, 40.0: 2.0, 60.0: 3.0, 80.0: 4.0, 300.0: 5.0}
    for label, filled in (("Fixed 1 px", True), ("Equivalent-area-side", False)):
        subset = panel_c[panel_c.series == label].sort_values("x")
        plot_x = subset.x.map(stage_position).to_numpy(float)
        axc.errorbar(
            plot_x,
            subset.estimate,
            yerr=[subset.estimate - subset.ci_low, subset.ci_high - subset.estimate],
            fmt="D",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=BRANCH["Paired"] if filled else "white",
            markeredgecolor=BRANCH["Paired"],
            markeredgewidth=0.9,
            ecolor=BRANCH["Paired"],
            elinewidth=LINE["ci"],
            capsize=2,
            label="Fixed" if filled else "Eq.-area",
        )
    # Keep the archive gap annotation in the panel's upper whitespace, away from
    # the 80/300 ticks and the final checkpoint marker.
    y_top = axc.get_ylim()[1]
    axc.text(0.35, y_top - 0.25, "No archive:\nepochs 100–280",
             ha="left", va="top", fontsize=FONT["minimum"], color=MUTED)
    axc.plot([4.05, 4.95], [y_top - 1.95, y_top - 1.95], color=MUTED, lw=0.8)
    axc.set_title("Physically archived checkpoints", loc="left", pad=4)
    axc.set_xlabel("Archived checkpoint")
    axc.set_ylabel("Paired contrast (pp)")
    axc.set_xlim(-0.3, 5.3)
    axc.set_xticks([0, 1, 2, 3, 4, 5], ["0", "20", "40", "60", "80", "300\n(final)"])
    axc.legend(loc="upper right", ncol=2)
    finish_axes(axc)
    panel_label(axc, "c")

    forest(axd, data[data.panel == "d"], "Compatible native-contract replication")
    axd.set_xlim(-32, 42)
    axd.set_ylim(-0.8, 5.5)
    panel_label(axd, "d")

    output_stem = FIG_ROOT / "draft" / "Fig2"
    save_figure(fig, output_stem, dpi=600)
    qa = {
        "status": "PASS",
        "resolved_font": font,
        "source_rows": len(rows),
        "outputs": {suffix: sha256(output_stem.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".png")},
        "interpretation_guards": {
            "same_mechanism_claim": False,
            "continuous_dose_response_claim": False,
            "archived_stage_interpolation": False,
            "architecture_generalisation_claim": False,
        },
    }
    qa_path = FIG_ROOT / "qa" / "Fig2.validation.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    rows, metadata = build_source_rows()
    write_sources(rows, metadata)
    render(rows)


if __name__ == "__main__":
    main()
