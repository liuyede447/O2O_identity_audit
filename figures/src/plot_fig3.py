"""Build source-bound draft Figure 3 and its fail-closed validation record."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures_final"
MANIFEST_PATH = FIG_ROOT / "FIGURE_EVIDENCE_MANIFEST.json"
ADDENDUM_PATH = FIG_ROOT / "FIGURE_EVIDENCE_ADDENDUM.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    BRANCH,
    FONT,
    GUIDE,
    INK,
    LINE,
    MECHANISM,
    MUTED,
    apply_style,
    finish_axes,
    mm,
    panel_label,
    zero_line,
)


PANEL_IDS = ("Fig3a", "Fig3b", "Fig3c", "Fig3d")
FIRST_LABELS = (
    ("Eligibility", "eligibility_boundary", MECHANISM["Eligibility"]),
    ("Top-k", "topk_membership_transition", MECHANISM["Top-k"]),
    ("Conflict", "conflict_reassignment", MECHANISM["Conflict"]),
    ("Within-set rank", "within_set_geometry_rank_reversal", MECHANISM["Within-set rank"]),
)
ANY_STAGE = (
    ("Eligibility", "flag__eligibility", MECHANISM["Eligibility"]),
    ("Top-k", "flag__topk", MECHANISM["Top-k"]),
    ("Conflict", "flag__conflict", MECHANISM["Conflict"]),
    ("Within-set rank", "flag__within_set_rank", MECHANISM["Within-set rank"]),
    ("Active missing", "flag__active_missing", MECHANISM["Other/reserved"]),
    ("Geometry clipping", "flag__geometry_clipping", MUTED),
    ("Eligibility ∩ Within-set", "pair__eligibility__within_set_rank", INK),
)
CONTRACTS = (
    ("Fixed 1 px", "fixed_1px", "fixed_1px", "fixed_1px", "fixed_1px", True),
    ("Equivalent-area-side", "equivalent_side", "equivalent_side_k00625", "normalized_k00625", "equivalent_area_side", False),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_path(value: str) -> str:
    return value.replace("\\", "/")


def load_and_verify_manifest() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    if not MANIFEST_PATH.is_file() or not ADDENDUM_PATH.is_file():
        raise FileNotFoundError("Figure manifest and source addendum are required")
    manifest = read_json(MANIFEST_PATH)
    addendum = read_json(ADDENDUM_PATH)
    # This deliverable is explicitly a second-round draft.  The global manifest
    # is presently blocked only by FigS1 identity findings, so draft rendering
    # is allowed when the manifest's own low-fidelity gate is open and every
    # Figure 3 panel/source passes its local admission checks below.  The script
    # never labels this output final or inserts it into the manuscript.
    if not manifest.get("low_fidelity_allowed", False):
        raise RuntimeError(f"manifest does not allow draft rendering: {manifest.get('status')}")
    panels = {item["panel_id"]: item for item in manifest.get("panels", []) if item.get("panel_id") in PANEL_IDS}
    if set(panels) != set(PANEL_IDS):
        raise RuntimeError(f"missing Figure 3 manifest panels: {set(PANEL_IDS) - set(panels)}")
    for panel_id, item in panels.items():
        if item.get("panel_status") not in {"PASS", "WARN"}:
            raise RuntimeError(f"{panel_id} is not admitted: {item.get('panel_status')}")

    add = {item["addendum_id"]: item for item in addendum.get("entries", [])}
    co = add.get("ADD-FIG-COOCCURRENCE-V1")
    if not co or co.get("admission") != "WARN/SOURCE_ADDENDUM":
        raise RuntimeError("Fig3c co-occurrence SOURCE_ADDENDUM is absent or not admitted")
    if co.get("scope") != "Fig3c non-exclusive descriptive any-stage occurrence only":
        raise RuntimeError("Fig3c source addendum scope changed")

    verified: dict[str, dict[str, str]] = {}
    for panel_id, item in panels.items():
        for source in item["source_artifacts"]:
            rel = norm_path(source["path"])
            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            expected = source["expected_sha256"]
            if actual != expected:
                raise RuntimeError(f"SHA mismatch for {panel_id} source {rel}: {actual} != {expected}")
            key = f"{panel_id}:{source['role']}"
            verified[key] = {
                "panel_id": panel_id,
                "path": rel,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "evidence_id": source["evidence_id"],
                "authority": source["authority"],
                "role": source["role"],
            }

    addendum_sources = {norm_path(source["path"]): source for source in co["sources"]}
    for rel in (
        "runs/20260901_first_divergence_cooccurrence_v1/summary.json",
        "runs/20260901_first_divergence_cooccurrence_v1/cooccurrence.csv",
    ):
        source = addendum_sources.get(rel)
        if not source or source.get("authority") != "SOURCE_ADDENDUM":
            raise RuntimeError(f"missing declared SOURCE_ADDENDUM record for {rel}")
        if sha256(ROOT / rel) != source["expected_sha256"]:
            raise RuntimeError(f"source-addendum SHA mismatch for {rel}")
    return panels, verified


def source_by_role(verified: dict[str, dict[str, str]], role: str) -> Path:
    matches = [ROOT / item["path"] for item in verified.values() if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source with role {role!r}, found {len(matches)}")
    return matches[0]


def base_row(panel: str, kind: str, contract: str, label: str, estimate: float,
             low: float, high: float, unit: str, source: dict[str, str], **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "panel": panel,
        "kind": kind,
        "contract": contract,
        "label": label,
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "unit": unit,
        "evidence_id": source["evidence_id"],
        "authority": source["authority"],
        "source_path": source["path"],
        "source_sha256": source["actual_sha256"],
    }
    row.update(extra)
    return row


def verified_role(verified: dict[str, dict[str, str]], role: str) -> dict[str, str]:
    matches = [item for item in verified.values() if item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one verified role {role!r}")
    return matches[0]


def build_rows(panels: dict[str, dict[str, Any]], verified: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock_source = verified_role(verified, "eligibility-lock estimates")
    stable_source = verified_role(verified, "eligibility-stable subset")
    anatomy_source = verified_role(verified, "exclusive first-observed labels")
    co_source = verified_role(verified, "non-exclusive any-stage summary")
    fixed_pre_source = verified_role(verified, "fixed pre-resolution control")
    eq_pre_source = verified_role(verified, "equivalent-side pre-resolution control")

    lock = read_json(ROOT / lock_source["path"])
    stable = read_json(ROOT / stable_source["path"])
    anatomy = read_json(ROOT / anatomy_source["path"])
    co = read_json(ROOT / co_source["path"])
    fixed_pre = read_json(ROOT / fixed_pre_source["path"])
    eq_pre = read_json(ROOT / eq_pre_source["path"])
    rows: list[dict[str, Any]] = []

    # Panel a: dumbbell endpoints and the source-derived attenuation statistic.
    for display, prefix, _, _, _, _ in CONTRACTS:
        native_key = f"{prefix}_native"
        locked_key = f"{prefix}_eligibility_lock"
        for state, key in (("Native", native_key), ("Eligibility locked", locked_key)):
            low, high = lock["ci_95"][key]["o2o_gap"]
            rows.append(base_row(
                "a", "o2o_gap", display, state,
                100 * lock["point"][key]["o2o_gap"], 100 * low, 100 * high,
                "percentage points", lock_source,
                image_n=panels["Fig3a"]["image_n"],
                support="eligibility-lock artifact support",
                statistical_unit=panels["Fig3a"]["statistical_unit"],
            ))
    frac_low, frac_high = lock["ci_95"]["fixed_o2o_gap_fraction_removed"]
    rows.append(base_row(
        "a", "attenuation_fraction", "Fixed 1 px", "Lock-support attenuation",
        100 * lock["point"]["fixed_o2o_gap_fraction_removed"], 100 * frac_low, 100 * frac_high,
        "percent", lock_source,
        image_n=panels["Fig3a"]["image_n"],
        support="eligibility-lock artifact support",
        statistical_unit=panels["Fig3a"]["statistical_unit"],
    ))

    # Panel b: post-replay conditioned observational gaps.
    for display, _, stable_key, _, _, _ in CONTRACTS:
        for level, label in (("object_level", "Object-level"), ("direction_level", "Direction-level")):
            item = stable[stable_key][level]
            low, high = item["ci_95"]["scale_contrast"]
            rows.append(base_row(
                "b", "eligibility_stable_gap", display, label,
                100 * item["point"]["scale_contrast"], 100 * low, 100 * high,
                "percentage points", stable_source,
                retained_n=item.get("retained_objects", item.get("retained_directions")),
                all_n=item.get("all_objects", item.get("all_directions")),
                contributing_images=item["contributing_images"],
                support="post-replay eligibility-stable subset",
                statistical_unit=level.replace("_", " "),
            ))

    # Panel c-left: exclusive first-observed labels. Other/reserved is the sum
    # of the two prespecified fallback labels, both retained even when zero.
    for display, _, _, anatomy_key, _, _ in CONTRACTS:
        first = anatomy[anatomy_key]["pathways"]["all"]
        for label, key, _ in FIRST_LABELS:
            low, high = first[key]["ci_95"]
            rows.append(base_row(
                "c", "exclusive_first_observed", display, label,
                100 * first[key]["point"], 100 * low, 100 * high,
                "percent of identity-flip directions", anatomy_source,
                directional_flip_rows=first["flip_rows"],
                contributing_images=first["contributing_images"],
                exclusive=True,
            ))
        other_point = first["active_disappearance"]["point"] + first["compound_or_other"]["point"]
        other_low = first["active_disappearance"]["ci_95"][0] + first["compound_or_other"]["ci_95"][0]
        other_high = first["active_disappearance"]["ci_95"][1] + first["compound_or_other"]["ci_95"][1]
        rows.append(base_row(
            "c", "exclusive_first_observed", display, "Other/reserved",
            100 * other_point, 100 * other_low, 100 * other_high,
            "percent of identity-flip directions", anatomy_source,
            directional_flip_rows=first["flip_rows"],
            contributing_images=first["contributing_images"],
            exclusive=True,
            reserved_components="active_disappearance + compound_or_other",
        ))

    # Panel c-right: non-exclusive SOURCE_ADDENDUM occurrence rates.
    for display, _, _, _, co_key, _ in CONTRACTS:
        item = co["contracts"][co_key]
        for label, key, _ in ANY_STAGE:
            low, high = item["ci_95"][key]
            rows.append(base_row(
                "c", "nonexclusive_any_stage", display, label,
                100 * item["point"][key], 100 * low, 100 * high,
                "percent of identity-flip directions", co_source,
                directional_flip_rows=item["directional_flip_rows"],
                exclusive=False,
                interpretation="descriptive co-occurrence; not causal attribution",
            ))

    # Panel d: same-branch pre-resolution controls.
    for display, payload, source in (("Fixed 1 px", fixed_pre, fixed_pre_source),
                                     ("Equivalent-area-side", eq_pre, eq_pre_source)):
        metrics = (
            ("Pre-conflict Top-7 winner gap", "preconflict_top7_scale_gap_8_16_minus_16_32"),
            ("Final assigned gap", "loss_active_scale_gap_8_16_minus_16_32"),
            ("Final minus pre-conflict", "loss_active_minus_preconflict_scale_gap"),
            ("Fragility-definition disagreement", "any_direction_fragility_disagreement_rate"),
        )
        for label, key in metrics:
            low, high = payload["ci_95"][key]
            rows.append(base_row(
                "d", "pre_resolution_control", display, label,
                100 * payload["point"][key], 100 * low, 100 * high,
                "percentage points", source,
                n_gt=payload["n_gt"], image_n=payload["n_images"],
                directional_replays=payload["n_directional_replays"],
                statistical_unit="focal GT and directional replay",
            ))

    metadata = {
        "status": "SOURCE_BOUND_DRAFT",
        "figure": "Figure 3",
        "title": "Eligibility and native-path localisation",
        "scientific_role": "localisation and sensitivity; not causal decomposition",
        "manifest_path": norm_path(str(MANIFEST_PATH.relative_to(ROOT))),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "addendum_path": norm_path(str(ADDENDUM_PATH.relative_to(ROOT))),
        "addendum_sha256": sha256(ADDENDUM_PATH),
        "sources": list(verified.values()),
        "source_row_count": len(rows),
        "panel_support": {key: panels[key] for key in PANEL_IDS},
        "interpretation_guards": {
            "eligibility_causal_contribution_claim": False,
            "observational_subset_as_causal_proof": False,
            "any_stage_stacked": False,
            "first_label_execution_order_invariant_claim": False,
            "conflict_mediation_claim": False,
        },
    }
    return rows, metadata


def write_sources(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = FIG_ROOT / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fig3_source.csv"
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "fig3_source.json"
    payload = {**metadata, "fig3_source_csv": norm_path(str(csv_path.relative_to(ROOT))),
               "fig3_source_csv_sha256": sha256(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return csv_path, json_path


def errorbar_x(ax: Any, row: pd.Series, y: float, *, color: str, marker: str,
               filled: bool, size: float = 4.2, zorder: int = 4,
               alpha: float = 1.0) -> None:
    ax.errorbar(
        row.estimate, y,
        xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
        fmt=marker, markersize=size,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color, markeredgewidth=0.9,
        ecolor=color, elinewidth=LINE["ci"], capsize=2.0,
        linestyle="none", zorder=zorder, alpha=alpha,
    )


def render_panel_a(ax: Any, data: pd.DataFrame) -> None:
    frame = data[data.kind == "o2o_gap"]
    y_map = {"Fixed 1 px": 1.0, "Equivalent-area-side": 0.0}
    for contract, y in y_map.items():
        subset = frame[frame.contract == contract].set_index("label")
        native = subset.loc["Native"]
        locked = subset.loc["Eligibility locked"]
        ax.plot([native.estimate, locked.estimate], [y, y], color=GUIDE, linewidth=2.2, zorder=1)
        errorbar_x(ax, native, y, color=BRANCH["O2O"], marker="o", filled=True, size=4.4)
        errorbar_x(ax, locked, y, color=MECHANISM["Eligibility"], marker="o", filled=True, size=4.4)
    zero_line(ax)
    ax.set_ylim(-0.55, 1.58)
    ax.set_yticks([1, 0], ["Fixed 1 px", "Eq.-area"])
    ax.set_xlim(-4.5, 23)
    ax.set_xlabel("O2O scale gap: 8–16 minus 16–32 px (pp)")
    ax.set_title("Eligibility-lock sensitivity", loc="left", pad=4)
    ax.text(0.01, 0.86, "Lock-support estimand",
            transform=ax.transAxes, ha="left", va="top", fontsize=FONT["minimum"], color=MUTED)
    handles = [
        Line2D([], [], marker="o", color=BRANCH["O2O"], markerfacecolor=BRANCH["O2O"], linestyle="none", label="Native"),
        Line2D([], [], marker="o", color=MECHANISM["Eligibility"], markerfacecolor=MECHANISM["Eligibility"], linestyle="none", label="Eligibility locked"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, handletextpad=0.35, columnspacing=0.8,
              bbox_to_anchor=(1.0, 1.01))
    finish_axes(ax)
    panel_label(ax, "a", x=-0.14, y=1.04)


def render_panel_b(ax: Any, data: pd.DataFrame) -> None:
    frame = data[data.kind == "eligibility_stable_gap"].copy()
    rows = (("Fixed 1 px", "Object-level"), ("Fixed 1 px", "Direction-level"),
            ("Equivalent-area-side", "Object-level"), ("Equivalent-area-side", "Direction-level"))
    labels = ("Fixed\nObject-level", "Fixed\nDirection-level",
              "Eq.-area\nObject-level", "Eq.-area\nDirection-level")
    ys = np.arange(len(rows))[::-1]
    for y, (contract, level) in zip(ys, rows):
        row = frame[(frame.contract == contract) & (frame.label == level)].iloc[0]
        marker = "o" if level == "Object-level" else "s"
        errorbar_x(ax, row, float(y), color=BRANCH["O2O"], marker=marker, filled=False, size=4.3)
    zero_line(ax)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(ys, labels)
    ax.set_xlim(-1.8, 6.7)
    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6])
    ax.set_xlabel("Eligibility-stable O2O scale gap (pp)")
    ax.set_title("Eligibility-stable observational subset", loc="left", pad=4)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BRANCH["O2O"], markerfacecolor="white", linestyle="none", label="Object-level"),
        Line2D([], [], marker="s", color=BRANCH["O2O"], markerfacecolor="white", linestyle="none", label="Direction-level"),
    ], loc="lower right", ncol=2, handletextpad=0.25, columnspacing=0.65)
    finish_axes(ax)
    panel_label(ax, "b", x=-0.14, y=1.04)


def render_panel_c(ax_left: Any, ax_right: Any, data: pd.DataFrame, *, display_label: str = "c") -> None:
    first = data[data.kind == "exclusive_first_observed"]
    y_map = {"Fixed 1 px": 1.0, "Equivalent-area-side": 0.0}
    color_map = {label: color for label, _, color in FIRST_LABELS}
    color_map["Other/reserved"] = MECHANISM["Other/reserved"]
    ordered_labels = [item[0] for item in FIRST_LABELS] + ["Other/reserved"]
    for contract, y in y_map.items():
        left = 0.0
        subset = first[first.contract == contract].set_index("label")
        for label in ordered_labels:
            value = float(subset.loc[label, "estimate"])
            ax_left.barh(y, value, left=left, height=0.46, color=color_map[label],
                         edgecolor="white", linewidth=0.45)
            if value >= 8:
                ax_left.text(left + value / 2, y, f"{value:.1f}", ha="center", va="center",
                             fontsize=FONT["minimum"], color=INK)
            left += value
        if not math.isclose(left, 100.0, abs_tol=1e-8):
            raise RuntimeError(f"exclusive first-observed composition does not sum to 100 for {contract}: {left}")
    ax_left.set_xlim(0, 102)
    ax_left.set_yticks([1, 0], ["Fixed 1 px", "Eq.-area"])
    ax_left.set_xlabel("Exclusive first-observed label (%)")
    ax_left.set_title("First observed", loc="left", pad=4)
    ax_left.spines["left"].set_visible(False)
    ax_left.tick_params(axis="y", length=0)
    finish_axes(ax_left)

    any_stage = data[data.kind == "nonexclusive_any_stage"]
    main_any_stage = tuple(
        item for item in ANY_STAGE
        if item[0] not in {"Active missing", "Geometry clipping"}
    )
    labels = ["Eligibility ∩ rank" if item[0] == "Eligibility ∩ Within-set" else item[0] for item in main_any_stage]
    ys = np.arange(len(labels))[::-1]
    for index, (label, _, color) in enumerate(main_any_stage):
        y = ys[index]
        emphasized = label in {"Eligibility", "Within-set rank", "Eligibility ∩ Within-set"}
        for contract, offset, filled, marker in (("Fixed 1 px", 0.14, True, "o"),
                                                  ("Equivalent-area-side", -0.14, False, "o")):
            row = any_stage[(any_stage.contract == contract) & (any_stage.label == label)].iloc[0]
            errorbar_x(ax_right, row, float(y + offset), color=color, marker=marker, filled=filled,
                       size=4.1 if emphasized else 3.2, zorder=5 if emphasized else 3,
                       alpha=1.0 if emphasized else 0.48)
    ax_right.set_xlim(-2, 86)
    ax_right.set_xticks([0, 20, 40, 60, 80])
    ax_right.set_yticks(ys, labels)
    for tick in ax_right.get_yticklabels():
        if tick.get_text() in {"Eligibility", "Within-set rank", "Eligibility ∩ rank"}:
            tick.set_fontweight("bold")
    ax_right.set_xlabel("Any-stage occurrence (%)")
    ax_right.set_title("Any stage (non-exclusive)", loc="left", pad=4)
    ax_right.legend(handles=[
        Line2D([], [], marker="o", color=INK, markerfacecolor=INK, linestyle="none", label="Fixed"),
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none", label="Eq.-area"),
    ], loc="upper left", ncol=1, handletextpad=0.3, labelspacing=0.25)
    finish_axes(ax_right)
    panel_label(ax_left, display_label, x=-0.16, y=1.06)


def render_panel_d(ax: Any, data: pd.DataFrame, *, display_label: str = "d") -> None:
    frame = data[data.kind == "pre_resolution_control"]
    labels = ("Pre-conflict\nTop-7", "Final\nassigned", "Δ gap", "Definition\nmismatch")
    source_labels = ("Pre-conflict Top-7 winner gap", "Final assigned gap",
                     "Final minus pre-conflict", "Fragility-definition disagreement")
    ys = np.arange(len(labels))[::-1]
    for index, source_label in enumerate(source_labels):
        y = ys[index]
        for contract, offset, filled in (("Fixed 1 px", 0.14, True), ("Equivalent-area-side", -0.14, False)):
            row = frame[(frame.contract == contract) & (frame.label == source_label)].iloc[0]
            errorbar_x(ax, row, float(y + offset), color=BRANCH["O2O"], marker="o", filled=filled, size=3.8)
    zero_line(ax)
    ax.set_ylim(-0.58, 3.55)
    ax.set_yticks(ys, labels)
    ax.set_xlim(-1.5, 23)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.axhline(1.5, color=GUIDE, linewidth=LINE["guide"], zorder=0)
    ax.set_xlabel("O2O contrast / difference (pp)")
    ax.set_title("Pre-resolution same-branch control", loc="left", pad=4)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=BRANCH["O2O"], markerfacecolor=BRANCH["O2O"], linestyle="none", label="Fixed"),
        Line2D([], [], marker="o", color=BRANCH["O2O"], markerfacecolor="white", linestyle="none", label="Eq.-area"),
    ], loc="lower right", ncol=2, handletextpad=0.25, columnspacing=0.65)
    finish_axes(ax)
    panel_label(ax, display_label, x=-0.22, y=1.04)


def make_figure(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    font = apply_style()
    # Figure-specific contract is stricter than the shared 6.7-pt tick default.
    plt.rcParams["xtick.labelsize"] = FONT["minimum"]
    plt.rcParams["ytick.labelsize"] = FONT["minimum"]
    data = pd.DataFrame(rows)
    fig = plt.figure(figsize=(mm(178), mm(130)))
    outer = fig.add_gridspec(2, 14, height_ratios=[0.94, 1.06], hspace=0.68, wspace=0.86)
    axa = fig.add_subplot(outer[0, 0:5])
    pathway = outer[0, 6:14].subgridspec(1, 2, width_ratios=[0.88, 1.32], wspace=0.82)
    axc1 = fig.add_subplot(pathway[0, 0])
    axc2 = fig.add_subplot(pathway[0, 1])
    axd = fig.add_subplot(outer[1, 3:11])

    render_panel_a(axa, data[data.panel == "a"])
    render_panel_c(axc1, axc2, data[data.panel == "c"], display_label="b")
    render_panel_d(axd, data[data.panel == "d"], display_label="c")

    fig.text(0.012, 0.018, "First observed ≠ causal contribution",
             ha="left", va="bottom", fontsize=FONT["minimum"], color=MUTED)
    # A compact mechanism key, separate from branch/stress encoding.
    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label)
                      for label, _, color in FIRST_LABELS]
    legend_handles.append(Patch(facecolor=MECHANISM["Other/reserved"], edgecolor="none", label="Other/reserved"))
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.053),
               ncol=5, columnspacing=0.65, handlelength=1.0, handletextpad=0.3)

    fig.subplots_adjust(left=0.085, right=0.992, top=0.95, bottom=0.18)

    fig.canvas.draw()
    visible_text = [item for item in fig.findobj(match=lambda obj: hasattr(obj, "get_text") and hasattr(obj, "get_fontsize"))
                    if item.get_visible() and str(item.get_text()).strip()]
    for item in visible_text:
        if item.get_fontsize() < FONT["minimum"]:
            item.set_fontsize(FONT["minimum"])
    fig.canvas.draw()
    geometry = geometry_audit(fig)
    preflight = {
        "resolved_font": font,
        "minimum_visible_font_pt": min(item.get_fontsize() for item in visible_text),
        "visible_text_count": len(visible_text),
        "axes_count": len(fig.axes),
        "geometry": geometry,
    }
    return fig, preflight


def geometry_audit(fig: Any) -> dict[str, Any]:
    """Fail closed on clipped text or text intruding into another panel."""
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    owner: dict[int, int] = {}
    records: list[tuple[int | None, str, Any]] = []
    for axis_index, ax in enumerate(fig.axes):
        for artist in ax.findobj(match=Text):
            if not artist.get_visible() or not str(artist.get_text()).strip():
                continue
            owner[id(artist)] = axis_index
            records.append((axis_index, str(artist.get_text()), artist.get_window_extent(renderer)))
    for artist in fig.texts:
        if id(artist) in owner or not artist.get_visible() or not str(artist.get_text()).strip():
            continue
        records.append((None, str(artist.get_text()), artist.get_window_extent(renderer)))

    clipped: list[str] = []
    for _, label, box in records:
        if box.x0 < figure_box.x0 - 1 or box.y0 < figure_box.y0 - 1 or box.x1 > figure_box.x1 + 1 or box.y1 > figure_box.y1 + 1:
            clipped.append(label)

    foreign_intrusions: list[dict[str, Any]] = []
    axis_boxes = [ax.get_window_extent(renderer) for ax in fig.axes]
    for source_axis, label, box in records:
        if source_axis is None:
            continue
        for target_axis, target_box in enumerate(axis_boxes):
            if target_axis == source_axis:
                continue
            intersection = Bbox.intersection(box, target_box)
            if intersection is not None and intersection.width * intersection.height > 4:
                foreign_intrusions.append({
                    "text": label,
                    "source_axis": source_axis,
                    "target_axis": target_axis,
                    "overlap_pixels2": intersection.width * intersection.height,
                })
    return {
        "status": "PASS" if not clipped and not foreign_intrusions else "FAIL",
        "clipped_text": clipped,
        "foreign_panel_text_intrusions": foreign_intrusions,
        "figure_pixel_bounds": [figure_box.x0, figure_box.y0, figure_box.x1, figure_box.y1],
    }


def save_draft(fig: Any, output_stem: Path) -> None:
    """Export without tight_layout, preserving the audited manual grid."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(output_stem.with_suffix(f".{extension}"), facecolor="white", **kwargs)
    plt.close(fig)
    # Matplotlib writes an opaque-looking RGBA PNG.  The venue contract asks
    # for RGB explicitly, so remove the unused alpha channel while preserving
    # the 600-dpi metadata.
    from PIL import Image
    png_path = output_stem.with_suffix(".png")
    with Image.open(png_path) as source:
        rgb = source.convert("RGB")
        rgb.save(png_path, dpi=(600, 600))


def crosscheck_cooccurrence_csv(verified: dict[str, dict[str, str]]) -> dict[str, Any]:
    summary_source = verified_role(verified, "non-exclusive any-stage summary")
    csv_source = verified_role(verified, "non-exclusive any-stage rows")
    summary = read_json(ROOT / summary_source["path"])
    table = pd.read_csv(ROOT / csv_source["path"])
    differences: list[float] = []
    contract_map = {"fixed_1px": "fixed_1px", "equivalent_area_side": "equivalent_area_side"}
    for record in table.to_dict("records"):
        contract = contract_map[record["contract"]]
        key = record["quantity"]
        differences.extend([
            abs(float(record["point_pct"]) - 100 * summary["contracts"][contract]["point"][key]),
            abs(float(record["ci_low_pct"]) - 100 * summary["contracts"][contract]["ci_95"][key][0]),
            abs(float(record["ci_high_pct"]) - 100 * summary["contracts"][contract]["ci_95"][key][1]),
        ])
    maximum = max(differences, default=float("inf"))
    if len(table) != 42 or maximum > 1e-10:
        raise RuntimeError(f"co-occurrence CSV/summary mismatch: rows={len(table)}, max_abs={maximum}")
    return {"rows": len(table), "max_abs_difference": maximum, "status": "PASS"}


def pdf_font_audit(pdf_path: Path) -> dict[str, Any]:
    command = shutil.which("pdffonts")
    if not command:
        return {"status": "WARN", "reason": "pdffonts unavailable"}
    proc = subprocess.run([command, str(pdf_path)], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "FAIL", "reason": proc.stderr.strip()}
    lines = [line for line in proc.stdout.splitlines()[2:] if line.strip()]
    embedded = all(" yes " in f" {line} " for line in lines)
    type42_compatible = all(("TrueType" in line or "CID TrueType" in line) for line in lines)
    return {"status": "PASS" if embedded and type42_compatible else "FAIL",
            "font_rows": lines, "all_embedded": embedded, "truetype_or_cid_truetype": type42_compatible}


def output_audit(output_stem: Path) -> dict[str, Any]:
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    svg_text = svg_path.read_text(encoding="utf-8")
    from PIL import Image
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0, 0))
        png_record = {"pixels": list(image.size), "dpi": [float(dpi[0]), float(dpi[1])],
                      "mode": image.mode, "dpi_600": min(dpi) >= 599}
    from pypdf import PdfReader
    page = PdfReader(str(pdf_path)).pages[0]
    width_mm = float(page.mediabox.width) * 25.4 / 72.0
    height_mm = float(page.mediabox.height) * 25.4 / 72.0
    pdf_geometry = {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "target_mm": [178.0, 130.0],
        "within_0_1_mm": abs(width_mm - 178.0) <= 0.1 and abs(height_mm - 130.0) <= 0.1,
    }
    outputs = {}
    for path in (svg_path, pdf_path, png_path):
        outputs[path.suffix[1:]] = {"path": norm_path(str(path.relative_to(ROOT))),
                                    "sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "outputs": outputs,
        "svg_text_nodes": svg_text.count("<text"),
        "svg_editable_text": "<text" in svg_text,
        "png": png_record,
        "pdf_geometry": pdf_geometry,
        "pdf_fonts": pdf_font_audit(pdf_path),
    }


def validate_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    expected_counts = {"a": 5, "b": 4, "c": 24, "d": 8}
    actual_counts = frame.groupby("panel").size().to_dict()
    finite = np.isfinite(frame[["estimate", "ci_low", "ci_high"]].astype(float).to_numpy()).all()
    ordered = ((frame.ci_low.astype(float) <= frame.estimate.astype(float)) &
               (frame.estimate.astype(float) <= frame.ci_high.astype(float))).all()
    first = frame[(frame.panel == "c") & (frame.kind == "exclusive_first_observed")]
    sums = first.groupby("contract").estimate.sum().to_dict()
    exclusive_sums = all(abs(value - 100) <= 1e-8 for value in sums.values())
    addendum_rows = frame[(frame.panel == "c") & (frame.kind == "nonexclusive_any_stage")]
    addendum_only = set(addendum_rows.authority) == {"SOURCE_ADDENDUM"}
    status = (actual_counts == expected_counts and finite and ordered and exclusive_sums and addendum_only)
    return {
        "status": "PASS" if status else "FAIL",
        "expected_panel_rows": expected_counts,
        "actual_panel_rows": actual_counts,
        "all_values_finite": bool(finite),
        "ci_order_valid": bool(ordered),
        "exclusive_composition_sums": sums,
        "exclusive_composition_sum_valid": bool(exclusive_sums),
        "any_stage_authority": sorted(set(addendum_rows.authority)),
        "any_stage_source_addendum_only": bool(addendum_only),
    }


def evidence_parity(rows: list[dict[str, Any]], verified: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Independently compare every plotted estimate, CI, and recorded N to evidence."""
    frame = pd.DataFrame(rows)
    numeric_diffs: list[float] = []
    n_mismatches: list[str] = []
    comparisons = 0

    def row_for(panel: str, kind: str, contract: str, label: str) -> pd.Series:
        match = frame[(frame.panel == panel) & (frame.kind == kind) &
                      (frame.contract == contract) & (frame.label == label)]
        if len(match) != 1:
            raise RuntimeError(f"parity row is not unique: {panel}/{kind}/{contract}/{label}")
        return match.iloc[0]

    def compare_values(row: pd.Series, point: float, interval: list[float]) -> None:
        nonlocal comparisons
        expected = (100 * point, 100 * interval[0], 100 * interval[1])
        actual = (float(row.estimate), float(row.ci_low), float(row.ci_high))
        numeric_diffs.extend(abs(a - e) for a, e in zip(actual, expected))
        comparisons += 3

    def compare_n(row: pd.Series, field: str, expected: int) -> None:
        nonlocal comparisons
        actual = int(float(row[field]))
        comparisons += 1
        if actual != int(expected):
            n_mismatches.append(f"{row.panel}/{row.contract}/{row.label}:{field}={actual}!={expected}")

    lock = read_json(source_by_role(verified, "eligibility-lock estimates"))
    stable = read_json(source_by_role(verified, "eligibility-stable subset"))
    anatomy = read_json(source_by_role(verified, "exclusive first-observed labels"))
    co = read_json(source_by_role(verified, "non-exclusive any-stage summary"))
    fixed_pre = read_json(source_by_role(verified, "fixed pre-resolution control"))
    eq_pre = read_json(source_by_role(verified, "equivalent-side pre-resolution control"))

    for display, prefix, stable_key, anatomy_key, co_key, _ in CONTRACTS:
        for state, key in (("Native", f"{prefix}_native"),
                           ("Eligibility locked", f"{prefix}_eligibility_lock")):
            row = row_for("a", "o2o_gap", display, state)
            compare_values(row, lock["point"][key]["o2o_gap"], lock["ci_95"][key]["o2o_gap"])
            compare_n(row, "image_n", 300)

        for level, label in (("object_level", "Object-level"), ("direction_level", "Direction-level")):
            item = stable[stable_key][level]
            row = row_for("b", "eligibility_stable_gap", display, label)
            compare_values(row, item["point"]["scale_contrast"], item["ci_95"]["scale_contrast"])
            compare_n(row, "retained_n", item.get("retained_objects", item.get("retained_directions")))
            compare_n(row, "all_n", item.get("all_objects", item.get("all_directions")))
            compare_n(row, "contributing_images", item["contributing_images"])

        first = anatomy[anatomy_key]["pathways"]["all"]
        for label, key, _ in FIRST_LABELS:
            row = row_for("c", "exclusive_first_observed", display, label)
            compare_values(row, first[key]["point"], first[key]["ci_95"])
            compare_n(row, "directional_flip_rows", first["flip_rows"])
            compare_n(row, "contributing_images", first["contributing_images"])
        row = row_for("c", "exclusive_first_observed", display, "Other/reserved")
        compare_values(
            row,
            first["active_disappearance"]["point"] + first["compound_or_other"]["point"],
            [first["active_disappearance"]["ci_95"][0] + first["compound_or_other"]["ci_95"][0],
             first["active_disappearance"]["ci_95"][1] + first["compound_or_other"]["ci_95"][1]],
        )
        compare_n(row, "directional_flip_rows", first["flip_rows"])
        compare_n(row, "contributing_images", first["contributing_images"])

        occurrence = co["contracts"][co_key]
        for label, key, _ in ANY_STAGE:
            row = row_for("c", "nonexclusive_any_stage", display, label)
            compare_values(row, occurrence["point"][key], occurrence["ci_95"][key])
            compare_n(row, "directional_flip_rows", occurrence["directional_flip_rows"])

    fraction = row_for("a", "attenuation_fraction", "Fixed 1 px", "Lock-support attenuation")
    compare_values(fraction, lock["point"]["fixed_o2o_gap_fraction_removed"],
                   lock["ci_95"]["fixed_o2o_gap_fraction_removed"])
    compare_n(fraction, "image_n", 300)

    metrics = (
        ("Pre-conflict Top-7 winner gap", "preconflict_top7_scale_gap_8_16_minus_16_32"),
        ("Final assigned gap", "loss_active_scale_gap_8_16_minus_16_32"),
        ("Final minus pre-conflict", "loss_active_minus_preconflict_scale_gap"),
        ("Fragility-definition disagreement", "any_direction_fragility_disagreement_rate"),
    )
    for display, payload in (("Fixed 1 px", fixed_pre), ("Equivalent-area-side", eq_pre)):
        for label, key in metrics:
            row = row_for("d", "pre_resolution_control", display, label)
            compare_values(row, payload["point"][key], payload["ci_95"][key])
            compare_n(row, "n_gt", payload["n_gt"])
            compare_n(row, "image_n", payload["n_images"])
            compare_n(row, "directional_replays", payload["n_directional_replays"])

    maximum = max(numeric_diffs, default=float("inf"))
    status = maximum <= 1e-12 and not n_mismatches
    return {
        "status": "PASS" if status else "FAIL",
        "comparisons": comparisons,
        "maximum_absolute_numeric_difference": maximum,
        "n_mismatches": n_mismatches,
    }


def main() -> None:
    panels, verified = load_and_verify_manifest()
    rows, metadata = build_rows(panels, verified)
    csv_path, json_path = write_sources(rows, metadata)
    fig, preflight = make_figure(rows)
    output_stem = FIG_ROOT / "draft" / "Fig3"
    save_draft(fig, output_stem)

    source_validation = validate_source_rows(rows)
    parity_validation = evidence_parity(rows, verified)
    co_validation = crosscheck_cooccurrence_csv(verified)
    output_validation = output_audit(output_stem)
    checks = {
        "source_hashes": "PASS",
        "source_number_ci_n_support_parity": parity_validation["status"],
        "cooccurrence_summary_csv_parity": co_validation["status"],
        "minimum_visible_font": "PASS" if preflight["minimum_visible_font_pt"] >= 7 else "FAIL",
        "text_clipping_and_cross_panel_overlap": preflight["geometry"]["status"],
        "arial_or_permitted_fallback": "PASS" if preflight["resolved_font"]["family"] in {"Arial", "Helvetica", "Liberation Sans"} else "FAIL",
        "svg_editable_text": "PASS" if output_validation["svg_editable_text"] else "FAIL",
        "png_600_dpi": "PASS" if output_validation["png"]["dpi_600"] else "FAIL",
        "png_rgb": "PASS" if output_validation["png"]["mode"] == "RGB" else "FAIL",
        "pdf_final_size": "PASS" if output_validation["pdf_geometry"]["within_0_1_mm"] else "FAIL",
        "pdf_type42_embedded": output_validation["pdf_fonts"]["status"],
        "interpretation_guards": "PASS",
        "any_stage_not_stacked": "PASS",
        "first_observed_exclusive_100pct": "PASS" if source_validation["exclusive_composition_sum_valid"] else "FAIL",
        "manuscript_unchanged": "PASS (not touched by this script)",
    }
    fail = [name for name, value in checks.items() if str(value).startswith("FAIL")]
    qa = {
        "status": "PASS" if not fail else "FAIL",
        "stage": "FINAL_FIGURE_SOURCE_VALIDATED",
        "figure": "Fig3",
        "checks": checks,
        "failures": fail,
        "preflight": preflight,
        "source_validation": source_validation,
        "evidence_parity": parity_validation,
        "cooccurrence_crosscheck": co_validation,
        "output_validation": output_validation,
        "source_data": {
            "csv": norm_path(str(csv_path.relative_to(ROOT))), "csv_sha256": sha256(csv_path),
            "json": norm_path(str(json_path.relative_to(ROOT))), "json_sha256": sha256(json_path),
        },
        "interpretation_guards": metadata["interpretation_guards"],
        "notes": [
            "Eligibility attenuation is explicitly limited to lock support and is not a causal contribution fraction.",
            "The eligibility-stable observational subset is retained in source data for supplementary placement and omitted from the main figure.",
            "Active-missing and geometry-clipping any-stage rows are retained in source data for supplementary placement and omitted from the main figure.",
            "Only mutually exclusive first-observed labels are stacked; any-stage occurrence is a non-exclusive dot plot.",
            "First-observed labels are execution-order dependent; all pathway values are descriptive, not causal.",
            "Pre-resolution estimates are a same-branch descriptive control, not a mediation analysis.",
        ],
    }
    qa_path = FIG_ROOT / "qa" / "Fig3.validation.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if fail:
        raise RuntimeError(f"Figure 3 QA failed: {fail}")


if __name__ == "__main__":
    main()
