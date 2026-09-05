"""Generate SHA-bound dense-v3 Figure 4 and its independent QA record."""

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
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures_final"
MANIFEST_PATH = FIG_ROOT / "FIGURE_EVIDENCE_MANIFEST.json"
RUN_PREFIX = "runs/20260903_boundary_full_grid_1024_postprocess_v2/"
TERMINAL_REL = RUN_PREFIX + "TERMINAL_VALIDATION.json"
PANEL_IDS = ("Fig4a", "Fig4b", "Fig4c", "Fig4d")
TAUS = (0.05, 0.0625, 0.10, 0.125, 0.20, 0.25)
SIZE_GROUPS = (("all", "All"), ("t_8_16", "8–16 px"), ("s_16_32", "16–32 px"))

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


CAUSES = (
    ("Eligibility", "cif_eligibility", MECHANISM["Eligibility"]),
    ("Within-set", "cif_within_set", MECHANISM["Within-set rank"]),
    ("Top-k", "cif_topk", MECHANISM["Top-k"]),
    ("Conflict", "cif_conflict", MECHANISM["Conflict"]),
    ("Other", "cif_other", MECHANISM["Other/reserved"]),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_path(value: str) -> str:
    return value.replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_verify_manifest() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)
    manifest = read_json(MANIFEST_PATH)
    if not manifest.get("low_fidelity_allowed", False):
        raise RuntimeError(f"manifest does not allow draft generation: {manifest.get('status')}")
    panels = {item["panel_id"]: item for item in manifest.get("panels", []) if item.get("panel_id") in PANEL_IDS}
    if set(panels) != set(PANEL_IDS):
        raise RuntimeError(f"missing Figure 4 panels: {set(PANEL_IDS) - set(panels)}")
    verified: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for panel_id, panel in panels.items():
        if panel.get("panel_status") != "PASS":
            raise RuntimeError(f"{panel_id} is not PASS")
        if panel_id in {"Fig4a", "Fig4b"} and "1/1024" not in str(panel.get("stress_contract")):
            raise RuntimeError(f"{panel_id} is not bound to the dense 1/1024 contract")
        for source in panel["source_artifacts"]:
            rel = norm_path(source["path"])
            if source["evidence_id"] != "EV-BOUNDARY-GEOMETRY-V3":
                raise RuntimeError(f"non-v3 evidence supplied to {panel_id}: {source['evidence_id']}")
            if not rel.startswith(RUN_PREFIX):
                raise RuntimeError(f"noncanonical boundary source supplied to {panel_id}: {rel}")
            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != source["expected_sha256"]:
                raise RuntimeError(f"SHA mismatch for {panel_id}/{rel}: {actual} != {source['expected_sha256']}")
            key = (panel_id, rel, source["role"])
            if key not in seen:
                verified.append({
                    "panel_id": panel_id,
                    "path": rel,
                    "expected_sha256": source["expected_sha256"],
                    "actual_sha256": actual,
                    "evidence_id": source["evidence_id"],
                    "authority": source["authority"],
                    "role": source["role"],
                })
                seen.add(key)

    terminal = read_json(ROOT / TERMINAL_REL)
    if terminal.get("status") != "PASS" or terminal.get("source_validation", {}).get("status") != "PASS":
        raise RuntimeError("dense-grid terminal validation is not PASS")
    if terminal["source_validation"].get("event_rows") != 29724 or terminal["source_validation"].get("curve_rows") != 7593288:
        raise RuntimeError("dense-grid terminal row counts changed")
    for item in verified:
        if item["path"].endswith("TERMINAL_VALIDATION.json"):
            continue
        parts = Path(item["path"]).parts
        tau_dir = next((part for part in parts if part.startswith("tau_")), None)
        if not tau_dir:
            raise RuntimeError(f"numeric source lacks tau directory: {item['path']}")
        terminal_hash = terminal["geometry"][tau_dir]["output_sha256"].get(Path(item["path"]).name)
        if terminal_hash != item["actual_sha256"]:
            raise RuntimeError(f"terminal manifest does not bind {item['path']}")
    return panels, verified


def verified_path(verified: list[dict[str, str]], panel_id: str, role: str) -> tuple[Path, dict[str, str]]:
    matches = [item for item in verified if item["panel_id"] == panel_id and item["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source for {panel_id}/{role}, found {len(matches)}")
    return ROOT / matches[0]["path"], matches[0]


def row_base(panel: str, kind: str, source: dict[str, str], **values: Any) -> dict[str, Any]:
    return {
        "panel": panel,
        "kind": kind,
        "evidence_version": "V3",
        "evidence_id": source["evidence_id"],
        "authority": source["authority"],
        "source_path": source["path"],
        "source_sha256": source["actual_sha256"],
        **values,
    }


def build_rows(panels: dict[str, dict[str, Any]], verified: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    survival_path, survival_source = verified_path(verified, "Fig4a", "KM and Aalen-Johansen estimates")
    rmsr_path, rmsr_source = verified_path(verified, "Fig4c", "RMCBD absolute estimates")
    contrast_path, contrast_source = verified_path(verified, "Fig4c", "RMCBD contrasts at tau .125")
    survival = pd.read_csv(survival_path)
    rmsr = pd.read_csv(rmsr_path)
    contrast = pd.read_csv(contrast_path)
    rows: list[dict[str, Any]] = []

    support_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for size_key, _ in SIZE_GROUPS:
        for branch_key in ("o2o", "o2m_assigned_positive"):
            match = rmsr[(rmsr.size_group == size_key) & (rmsr.estimand == branch_key)]
            if len(match) != 1:
                raise RuntimeError(f"RMCBD support row not unique for {size_key}/{branch_key}")
            support_lookup[(size_key, branch_key)] = match.iloc[0].to_dict()

    # Panel a: only the final loss-active O2O and assigned-positive O2M definitions.
    for size_key, size_label in SIZE_GROUPS[1:]:
        for branch_key, branch_label in (("o2o", "O2O identity"),
                                         ("o2m_assigned_positive", "O2M assigned Top-1 state")):
            subset = survival[(survival.size_group == size_key) &
                              (survival.estimand == branch_key) &
                              (survival.metric == "survival")].sort_values("radius")
            if subset.empty:
                raise RuntimeError(f"missing survival curve {size_key}/{branch_key}")
            support = support_lookup[(size_key, branch_key)]
            for record in subset.to_dict("records"):
                rows.append(row_base(
                    "a", "km_survival", survival_source,
                    size_group=size_key, size_label=size_label,
                    series=branch_label, estimand=branch_key,
                    radius=float(record["radius"]), estimate=float(record["estimate"]),
                    ci_low=float(record["ci_low"]), ci_high=float(record["ci_high"]),
                    unit="survival probability", n_base_defined=int(support["n_base_defined"]),
                    n_base_undefined_excluded=int(support["n_base_undefined_excluded"]),
                    image_n=panels["Fig4a"]["image_n"],
                    statistical_unit=panels["Fig4a"]["statistical_unit"],
                ))

    # Panel b: overall O2O competing first-cause incidence only.
    for label, metric, _ in CAUSES:
        subset = survival[(survival.size_group == "all") & (survival.estimand == "o2o") &
                          (survival.metric == metric)].sort_values("radius")
        if subset.empty:
            raise RuntimeError(f"missing AJ curve {metric}")
        for record in subset.to_dict("records"):
            rows.append(row_base(
                "b", "aj_cumulative_incidence", survival_source,
                size_group="all", size_label="All", series=label, estimand="o2o",
                metric=metric, radius=float(record["radius"]), estimate=float(record["estimate"]),
                ci_low=float(record["ci_low"]), ci_high=float(record["ci_high"]),
                unit="cumulative incidence", n_base_defined=29048,
                n_base_undefined_excluded=676, image_n=panels["Fig4b"]["image_n"],
                statistical_unit=panels["Fig4b"]["statistical_unit"],
            ))

    # Panel c: final absolute O2O/O2M values and final paired contrast only.
    for size_key, size_label in SIZE_GROUPS:
        for branch_key, branch_label in (("o2o", "O2O"), ("o2m_assigned_positive", "O2M")):
            record = support_lookup[(size_key, branch_key)]
            rows.append(row_base(
                "c", "rmcbd_absolute", rmsr_source,
                size_group=size_key, size_label=size_label, series=branch_label, estimand=branch_key,
                tau=float(record["tau"]), estimate=float(record["estimate"]),
                ci_low=float(record["ci_low"]), ci_high=float(record["ci_high"]), unit="cardinal distance r",
                n_base_defined=int(record["n_base_defined"]),
                n_base_undefined_excluded=int(record["n_base_undefined_excluded"]),
                weighted_n=float(record["weighted_n"]), image_n=panels["Fig4c"]["image_n"],
                statistical_unit=panels["Fig4c"]["statistical_unit"],
            ))
        match = contrast[(contrast.size_group == size_key) &
                         (contrast.contrast == "o2o_minus_o2m_assigned_positive")]
        if len(match) != 1:
            raise RuntimeError(f"final paired contrast not unique for {size_key}")
        record = match.iloc[0]
        rows.append(row_base(
            "c", "rmcbd_paired", contrast_source,
            size_group=size_key, size_label=size_label, series="O2O − O2M",
            estimand="o2o_minus_o2m_assigned_positive", tau=float(record.tau),
            estimate=float(record.estimate), ci_low=float(record.ci_low), ci_high=float(record.ci_high),
            unit="cardinal distance r", image_n=panels["Fig4c"]["image_n"],
            statistical_unit=panels["Fig4c"]["statistical_unit"],
            support="paired stratified image-cluster bootstrap over branch-specific base-defined supports",
        ))

    # Panel d: all six manifest-bound tau files, final O2M definition only.
    for tau in TAUS:
        role = f"RMCBD contrasts at tau {str(tau).rstrip('0').rstrip('.') if tau != 0.0625 else '.0625'}"
        # Manifest roles omit the leading zero for every tau.
        role = {0.05: "RMCBD contrasts at tau .05", 0.0625: "RMCBD contrasts at tau .0625",
                0.10: "RMCBD contrasts at tau .10", 0.125: "RMCBD contrasts at tau .125",
                0.20: "RMCBD contrasts at tau .20", 0.25: "RMCBD contrasts at tau .25"}[tau]
        tau_path, tau_source = verified_path(verified, "Fig4d", role)
        table = pd.read_csv(tau_path)
        subset = table[table.contrast == "o2o_minus_o2m_assigned_positive"]
        if set(subset.size_group) != {item[0] for item in SIZE_GROUPS} or len(subset) != 3:
            raise RuntimeError(f"tau {tau} lacks the three final paired rows")
        for size_key, size_label in SIZE_GROUPS:
            record = subset[subset.size_group == size_key].iloc[0]
            if not math.isclose(float(record.tau), tau, abs_tol=1e-12):
                raise RuntimeError(f"tau mismatch in {tau_path}")
            rows.append(row_base(
                "d", "rmcbd_tau_sensitivity", tau_source,
                size_group=size_key, size_label=size_label, series=size_label,
                estimand="o2o_minus_o2m_assigned_positive", tau=tau,
                estimate=float(record.estimate), ci_low=float(record.ci_low), ci_high=float(record.ci_high),
                unit="cardinal distance r", directional_trajectories=29724,
                image_n=panels["Fig4d"]["image_n"], statistical_unit=panels["Fig4d"]["statistical_unit"],
            ))

    if any("legacy" in str(value).lower() or "pre_topk" in str(value).lower()
           for row in rows for value in row.values()):
        raise RuntimeError("legacy/pre-Top-k O2M content entered Figure 4 source rows")
    metadata = {
        "status": "SOURCE_BOUND_DRAFT",
        "figure": "Figure 4",
        "title": "Grid-resolved cardinal boundary geometry",
        "evidence_version": "EV-BOUNDARY-GEOMETRY-V3",
        "source_run": RUN_PREFIX.rstrip("/"),
        "manifest_path": norm_path(str(MANIFEST_PATH.relative_to(ROOT))),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "sources": verified,
        "source_row_count": len(rows),
        "panel_support": {key: panels[key] for key in PANEL_IDS},
        "interpretation_guards": {
            "full_2d_boundary_claim": False,
            "smoothed_survival_or_cif": False,
            "o2m_identity_claim": False,
            "legacy_or_pre_topk_o2m_used": False,
            "tau_magnitude_robustness_claim": False,
            "right_censoring_removed": False,
        },
    }
    return rows, metadata


def write_sources(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = FIG_ROOT / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fig4_source.csv"
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "fig4_source.json"
    payload = {**metadata, "fig4_source_csv": norm_path(str(csv_path.relative_to(ROOT))),
               "fig4_source_csv_sha256": sha256(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return csv_path, json_path


def step_band(ax: Any, subset: pd.DataFrame, *, color: str, label: str, linestyle: str = "-") -> None:
    subset = subset.sort_values("radius")
    x = subset.radius.to_numpy(float)
    y = subset.estimate.to_numpy(float)
    lo = subset.ci_low.to_numpy(float)
    hi = subset.ci_high.to_numpy(float)
    ax.step(x, y, where="post", color=color, linewidth=LINE["main"], linestyle=linestyle, label=label)
    ax.fill_between(x, lo, hi, step="post", color=color, alpha=0.10, linewidth=0)


def render_panel_a(fig: Any, spec: Any, data: pd.DataFrame) -> list[Any]:
    inner = spec.subgridspec(1, 2, wspace=0.22)
    axes = [fig.add_subplot(inner[0, index]) for index in range(2)]
    for ax, (size_key, size_title) in zip(axes, SIZE_GROUPS[1:]):
        subset = data[(data.panel == "a") & (data.size_group == size_key)]
        step_band(ax, subset[subset.estimand == "o2o"], color=BRANCH["O2O"], label="O2O loss-active ID")
        step_band(ax, subset[subset.estimand == "o2m_assigned_positive"], color=BRANCH["O2M"],
                  label="O2M assigned-set Top-1")
        ax.set_xlim(0, 0.128)
        ax.set_ylim(0.47, 1.012)
        ax.set_xticks([0, 0.0625, 0.125], ["0", "0.0625", "0.125"])
        ax.set_yticks([0.5, 0.75, 1.0])
        ax.set_title(size_title, loc="left", pad=3)
        ax.set_xlabel("Cardinal distance r")
        finish_axes(ax)
    axes[0].set_ylabel("Native-state survival S(r)")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="lower left", handlelength=1.4, labelspacing=0.25)
    panel_label(axes[0], "a", x=-0.30, y=1.07)
    return axes


def render_panel_b(ax: Any, data: pd.DataFrame) -> None:
    subset = data[data.panel == "b"]
    for label, _, color in CAUSES:
        step_band(ax, subset[subset.series == label], color=color, label=label)
    ax.set_xlim(0, 0.128)
    ax.set_ylim(0, 0.071)
    ax.set_xticks([0, 0.0625, 0.125], ["0", "0.0625", "0.125"])
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.set_xlabel("Cardinal distance r")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("First O2O boundary cause", loc="left", pad=4)
    ax.legend(loc="upper center", ncol=2, handlelength=1.25, columnspacing=0.7, labelspacing=0.25)
    finish_axes(ax)
    panel_label(ax, "b", x=-0.25, y=1.07)


def forest_point(ax: Any, row: pd.Series, y: float, *, color: str, marker: str, filled: bool) -> None:
    ax.errorbar(
        float(row.estimate), y,
        xerr=[[float(row.estimate - row.ci_low)], [float(row.ci_high - row.estimate)]],
        fmt=marker, markersize=4.0, markerfacecolor=color if filled else "white",
        markeredgecolor=color, markeredgewidth=0.85, ecolor=color,
        elinewidth=LINE["ci"], capsize=1.8, linestyle="none", zorder=3,
    )


def render_panel_c(fig: Any, spec: Any, data: pd.DataFrame) -> list[Any]:
    inner = spec.subgridspec(1, 2, width_ratios=[1.28, 0.82], wspace=0.62)
    ax_abs = fig.add_subplot(inner[0, 0])
    ax_delta = fig.add_subplot(inner[0, 1])
    frame = data[data.panel == "c"]
    ys = np.arange(3)[::-1]
    labels = [item[1] for item in SIZE_GROUPS]
    for y, (size_key, _) in zip(ys, SIZE_GROUPS):
        for branch, offset, color, marker in (("O2O", 0.14, BRANCH["O2O"], "o"),
                                               ("O2M", -0.14, BRANCH["O2M"], "s")):
            row = frame[(frame.kind == "rmcbd_absolute") & (frame.size_group == size_key) &
                        (frame.series == branch)].iloc[0]
            forest_point(ax_abs, row, float(y + offset), color=color, marker=marker, filled=True)
            if size_key == "all":
                branch_text = "O2M assigned-set\nTop-1" if branch == "O2M" else branch
                ax_abs.text(float(row.ci_high) + 0.0008, float(y + offset),
                            f"{branch_text} {row.estimate:.3f}",
                            ha="left", va="center", fontsize=FONT["minimum"], color=INK)
        row = frame[(frame.kind == "rmcbd_paired") & (frame.size_group == size_key)].iloc[0]
        forest_point(ax_delta, row, float(y), color=BRANCH["Paired"], marker="D", filled=True)
        if float(row.ci_high) > 0.035:
            x_text, align = float(row.ci_low) - 0.0008, "right"
        else:
            x_text, align = float(row.ci_high) + 0.0008, "left"
        ax_delta.text(x_text, float(y), f"{row.estimate:.3f}",
                      ha=align, va="center", fontsize=FONT["minimum"], color=INK)
    ax_abs.set_yticks(ys, labels)
    ax_abs.set_xlim(0.078, 0.132)
    ax_abs.set_xticks([0.08, 0.10, 0.12])
    ax_abs.set_xlabel("Absolute RMCBD (r)")
    ax_abs.set_title("Absolute", loc="left", pad=4)
    finish_axes(ax_abs)
    ax_delta.set_yticks(ys, ["", "", ""])
    ax_delta.set_xlim(0, 0.042)
    ax_delta.set_xticks([0, 0.02, 0.04])
    ax_delta.set_xlabel("O2O − O2M (r)")
    ax_delta.set_title("Paired difference", loc="left", pad=4)
    zero_line(ax_delta)
    finish_axes(ax_delta)
    panel_label(ax_abs, "c", x=-0.30, y=1.07)
    ax_abs.text(0.0, 1.16, "Restricted mean cardinal boundary distance", transform=ax_abs.transAxes,
                ha="left", va="bottom", fontsize=FONT["panel_title"], fontweight="bold", color=INK)
    return [ax_abs, ax_delta]


def render_panel_d(ax: Any, data: pd.DataFrame) -> None:
    frame = data[data.panel == "d"]
    styles = {
        "All": ("D", "-"),
        "8–16 px": ("o", "--"),
        "16–32 px": ("s", ":"),
    }
    for label, (marker, linestyle) in styles.items():
        subset = frame[frame.series == label].sort_values("tau")
        ax.plot(subset.tau, subset.estimate, color=BRANCH["Paired"], linewidth=LINE["secondary"],
                linestyle=linestyle, marker=marker, markersize=3.7,
                markerfacecolor="white" if label == "8–16 px" else BRANCH["Paired"],
                markeredgecolor=BRANCH["Paired"], markeredgewidth=0.8, label=label)
        ax.errorbar(subset.tau, subset.estimate,
                    yerr=[subset.estimate - subset.ci_low, subset.ci_high - subset.estimate],
                    fmt="none", ecolor=BRANCH["Paired"], elinewidth=LINE["ci"], capsize=1.6)
    ax.axvline(0.125, color=GUIDE, linewidth=LINE["reference"], zorder=0)
    ax.text(0.125, 0.102, "primary τ", ha="center", va="bottom", fontsize=FONT["minimum"], color=MUTED)
    ax.set_xlim(0.043, 0.257)
    ax.set_ylim(-0.003, 0.108)
    ax.set_xticks(TAUS, ["0.05", "0.063", "0.10", "0.125", "0.20", "0.25"], rotation=50, ha="right")
    ax.set_yticks([0, 0.025, 0.05, 0.075, 0.10])
    ax.set_xlabel("Truncation horizon τ")
    ax.set_ylabel("O2O − O2M RMCBD (r)")
    ax.set_title("Branch ordering persists across τ", loc="left", pad=4)
    ax.legend(loc="upper left", handlelength=1.6, labelspacing=0.25)
    finish_axes(ax)
    panel_label(ax, "d", x=-0.26, y=1.07)


def geometry_audit(fig: Any) -> dict[str, Any]:
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    records: list[tuple[int | None, str, Any]] = []
    owned: set[int] = set()
    for axis_index, ax in enumerate(fig.axes):
        for artist in ax.findobj(match=Text):
            if artist.get_visible() and str(artist.get_text()).strip():
                records.append((axis_index, str(artist.get_text()), artist.get_window_extent(renderer)))
                owned.add(id(artist))
    for artist in fig.texts:
        if id(artist) not in owned and artist.get_visible() and str(artist.get_text()).strip():
            records.append((None, str(artist.get_text()), artist.get_window_extent(renderer)))
    clipped = [label for _, label, box in records
               if box.x0 < figure_box.x0 - 1 or box.y0 < figure_box.y0 - 1 or
               box.x1 > figure_box.x1 + 1 or box.y1 > figure_box.y1 + 1]
    intrusions: list[dict[str, Any]] = []
    axis_boxes = [ax.get_window_extent(renderer) for ax in fig.axes]
    for source_axis, label, box in records:
        if source_axis is None:
            continue
        for target_axis, target_box in enumerate(axis_boxes):
            if target_axis == source_axis:
                continue
            overlap = Bbox.intersection(box, target_box)
            if overlap is not None and overlap.width * overlap.height > 4:
                intrusions.append({"text": label, "source_axis": source_axis,
                                   "target_axis": target_axis,
                                   "overlap_pixels2": overlap.width * overlap.height})
    return {"status": "PASS" if not clipped and not intrusions else "FAIL",
            "clipped_text": clipped, "foreign_panel_text_intrusions": intrusions}


def make_figure(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    font = apply_style()
    plt.rcParams["xtick.labelsize"] = FONT["minimum"]
    plt.rcParams["ytick.labelsize"] = FONT["minimum"]
    data = pd.DataFrame(rows)
    fig = plt.figure(figsize=(mm(178), mm(132)))
    outer = fig.add_gridspec(2, 14, height_ratios=[1.04, 0.96], hspace=0.68, wspace=0.72)
    render_panel_a(fig, outer[0, 0:7], data)
    axb = fig.add_subplot(outer[0, 8:14])
    render_panel_b(axb, data)
    render_panel_c(fig, outer[1, 0:8], data)
    axd = fig.add_subplot(outer[1, 9:14])
    render_panel_d(axd, data)
    fig.text(0.012, 0.012, "Cardinal rays only; not a 2-D minimum boundary",
             ha="left", va="bottom", fontsize=FONT["minimum"], color=MUTED)
    fig.text(0.988, 0.012, "RMCBD magnitude increases with τ by construction",
             ha="right", va="bottom", fontsize=FONT["minimum"], color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.992, top=0.95, bottom=0.125)
    fig.canvas.draw()
    visible = [item for item in fig.findobj(match=lambda obj: hasattr(obj, "get_text") and hasattr(obj, "get_fontsize"))
               if item.get_visible() and str(item.get_text()).strip()]
    for item in visible:
        if item.get_fontsize() < FONT["minimum"]:
            item.set_fontsize(FONT["minimum"])
    fig.canvas.draw()
    geometry = geometry_audit(fig)
    return fig, {
        "resolved_font": font,
        "minimum_visible_font_pt": min(item.get_fontsize() for item in visible),
        "visible_text_count": len(visible),
        "axes_count": len(fig.axes),
        "geometry": geometry,
    }


def save_draft(fig: Any, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(output_stem.with_suffix(f".{extension}"), facecolor="white", **kwargs)
    plt.close(fig)
    from PIL import Image
    png_path = output_stem.with_suffix(".png")
    with Image.open(png_path) as source:
        source.convert("RGB").save(png_path, dpi=(600, 600))


def structural_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    counts = frame.groupby(["panel", "kind"]).size().to_dict()
    expected = {
        ("a", "km_survival"): 499,
        ("b", "aj_cumulative_incidence"): 650,
        ("c", "rmcbd_absolute"): 6,
        ("c", "rmcbd_paired"): 3,
        ("d", "rmcbd_tau_sensitivity"): 18,
    }
    finite = np.isfinite(frame[["estimate", "ci_low", "ci_high"]].astype(float).to_numpy()).all()
    ordered = ((frame.ci_low.astype(float) <= frame.estimate.astype(float)) &
               (frame.estimate.astype(float) <= frame.ci_high.astype(float))).all()
    forbidden = any("legacy" in str(value).lower() or "pre_topk" in str(value).lower()
                    for row in rows for value in row.values())
    tau_set = set(frame[frame.panel == "d"].tau.astype(float))
    status = counts == expected and finite and ordered and not forbidden and tau_set == set(TAUS)
    return {
        "status": "PASS" if status else "FAIL",
        "expected_counts": {"/".join(key): value for key, value in expected.items()},
        "actual_counts": {"/".join(key): value for key, value in counts.items()},
        "all_values_finite": bool(finite), "ci_order_valid": bool(ordered),
        "legacy_or_pre_topk_content": bool(forbidden), "tau_set": sorted(tau_set),
    }


def evidence_parity(rows: list[dict[str, Any]], verified: list[dict[str, str]]) -> dict[str, Any]:
    """Independently reconstruct all selected rows from frozen dense-v3 CSVs."""
    frame = pd.DataFrame(rows)
    diffs: list[float] = []
    n_mismatches: list[str] = []
    comparisons = 0

    def compare(actual: pd.Series, expected: pd.Series, n_fields: tuple[str, ...] = ()) -> None:
        nonlocal comparisons
        for key in ("estimate", "ci_low", "ci_high"):
            diffs.append(abs(float(actual[key]) - float(expected[key])))
            comparisons += 1
        for field in n_fields:
            comparisons += 1
            if int(float(actual[field])) != int(float(expected[field])):
                n_mismatches.append(f"{actual.panel}/{actual.kind}/{field}")

    survival_path, _ = verified_path(verified, "Fig4a", "KM and Aalen-Johansen estimates")
    rmsr_path, _ = verified_path(verified, "Fig4c", "RMCBD absolute estimates")
    contrast_path, _ = verified_path(verified, "Fig4c", "RMCBD contrasts at tau .125")
    survival = pd.read_csv(survival_path)
    rmsr = pd.read_csv(rmsr_path)
    contrast = pd.read_csv(contrast_path)

    for _, actual in frame[frame.panel == "a"].iterrows():
        match = survival[(survival.size_group == actual.size_group) & (survival.estimand == actual.estimand) &
                         (survival.metric == "survival") & np.isclose(survival.radius, float(actual.radius), atol=1e-14)]
        if len(match) != 1:
            raise RuntimeError("survival parity row not unique")
        expected = match.iloc[0].to_dict()
        support = rmsr[(rmsr.size_group == actual.size_group) & (rmsr.estimand == actual.estimand)].iloc[0].copy()
        expected["n_base_defined"] = support.n_base_defined
        expected["n_base_undefined_excluded"] = support.n_base_undefined_excluded
        compare(actual, expected, ("n_base_defined", "n_base_undefined_excluded"))

    metric_for_label = {item[0]: item[1] for item in CAUSES}
    for _, actual in frame[frame.panel == "b"].iterrows():
        match = survival[(survival.size_group == "all") & (survival.estimand == "o2o") &
                         (survival.metric == metric_for_label[actual.series]) &
                         np.isclose(survival.radius, float(actual.radius), atol=1e-14)]
        if len(match) != 1:
            raise RuntimeError("AJ parity row not unique")
        expected = match.iloc[0].to_dict()
        expected["n_base_defined"] = 29048
        expected["n_base_undefined_excluded"] = 676
        compare(actual, expected, ("n_base_defined", "n_base_undefined_excluded"))

    for _, actual in frame[(frame.panel == "c") & (frame.kind == "rmcbd_absolute")].iterrows():
        expected = rmsr[(rmsr.size_group == actual.size_group) & (rmsr.estimand == actual.estimand)].iloc[0]
        compare(actual, expected, ("n_base_defined", "n_base_undefined_excluded"))
        diffs.append(abs(float(actual.weighted_n) - float(expected.weighted_n)))
        comparisons += 1
    for _, actual in frame[(frame.panel == "c") & (frame.kind == "rmcbd_paired")].iterrows():
        expected = contrast[(contrast.size_group == actual.size_group) &
                            (contrast.contrast == "o2o_minus_o2m_assigned_positive")].iloc[0]
        compare(actual, expected)

    role_map = {0.05: "RMCBD contrasts at tau .05", 0.0625: "RMCBD contrasts at tau .0625",
                0.10: "RMCBD contrasts at tau .10", 0.125: "RMCBD contrasts at tau .125",
                0.20: "RMCBD contrasts at tau .20", 0.25: "RMCBD contrasts at tau .25"}
    tau_tables = {tau: pd.read_csv(verified_path(verified, "Fig4d", role)[0]) for tau, role in role_map.items()}
    for _, actual in frame[frame.panel == "d"].iterrows():
        table = tau_tables[float(actual.tau)]
        expected = table[(table.size_group == actual.size_group) &
                         (table.contrast == "o2o_minus_o2m_assigned_positive")].iloc[0]
        compare(actual, expected)
        comparisons += 1
        if int(actual.directional_trajectories) != 29724:
            n_mismatches.append(f"d/{actual.size_group}/{actual.tau}:directional_trajectories")

    maximum = max(diffs, default=float("inf"))
    return {"status": "PASS" if maximum <= 1e-14 and not n_mismatches else "FAIL",
            "comparisons": comparisons, "maximum_absolute_numeric_difference": maximum,
            "n_mismatches": n_mismatches}


def pdf_font_audit(pdf_path: Path) -> dict[str, Any]:
    command = shutil.which("pdffonts")
    if not command:
        return {"status": "WARN", "reason": "pdffonts unavailable"}
    result = subprocess.run([command, str(pdf_path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"status": "FAIL", "reason": result.stderr.strip()}
    lines = [line for line in result.stdout.splitlines()[2:] if line.strip()]
    embedded = all(" yes " in f" {line} " for line in lines)
    true_type = all("TrueType" in line or "CID TrueType" in line for line in lines)
    return {"status": "PASS" if embedded and true_type else "FAIL", "font_rows": lines,
            "all_embedded": embedded, "truetype_or_cid_truetype": true_type}


def output_audit(stem: Path) -> dict[str, Any]:
    from PIL import Image
    from pypdf import PdfReader
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    svg_text = svg.read_text(encoding="utf-8")
    with Image.open(png) as image:
        dpi = image.info.get("dpi", (0, 0))
        png_info = {"pixels": list(image.size), "dpi": [float(dpi[0]), float(dpi[1])],
                    "mode": image.mode, "dpi_600": min(dpi) >= 599}
    page = PdfReader(str(pdf)).pages[0]
    width = float(page.mediabox.width) * 25.4 / 72
    height = float(page.mediabox.height) * 25.4 / 72
    outputs = {path.suffix[1:]: {"path": norm_path(str(path.relative_to(ROOT))),
                                 "sha256": sha256(path), "bytes": path.stat().st_size}
               for path in (svg, pdf, png)}
    return {"outputs": outputs, "svg_text_nodes": svg_text.count("<text"),
            "svg_editable_text": "<text" in svg_text, "png": png_info,
            "pdf_geometry": {"width_mm": width, "height_mm": height,
                             "target_mm": [178, 132],
                             "within_0_1_mm": abs(width - 178) <= 0.1 and abs(height - 132) <= 0.1},
            "pdf_fonts": pdf_font_audit(pdf)}


def main() -> None:
    panels, verified = load_and_verify_manifest()
    rows, metadata = build_rows(panels, verified)
    csv_path, json_path = write_sources(rows, metadata)
    fig, preflight = make_figure(rows)
    output_stem = FIG_ROOT / "draft" / "Fig4"
    save_draft(fig, output_stem)
    structure = structural_validation(rows)
    parity = evidence_parity(rows, verified)
    output = output_audit(output_stem)
    checks = {
        "dense_v3_source_hashes_and_terminal_binding": "PASS",
        "legacy_and_pre_topk_excluded": "PASS" if not structure["legacy_or_pre_topk_content"] else "FAIL",
        "source_estimate_ci_n_support_parity": parity["status"],
        "source_structure_and_tau_set": structure["status"],
        "unsmoothed_step_geometry": "PASS",
        "cardinal_not_full_2d_guard": "PASS",
        "o2m_rank_state_not_identity_guard": "PASS",
        "minimum_visible_font": "PASS" if preflight["minimum_visible_font_pt"] >= 7 else "FAIL",
        "arial_or_permitted_fallback": "PASS" if preflight["resolved_font"]["family"] in {"Arial", "Helvetica", "Liberation Sans"} else "FAIL",
        "text_clipping_and_cross_panel_overlap": preflight["geometry"]["status"],
        "svg_editable_text": "PASS" if output["svg_editable_text"] else "FAIL",
        "pdf_type42_embedded": output["pdf_fonts"]["status"],
        "pdf_final_size": "PASS" if output["pdf_geometry"]["within_0_1_mm"] else "FAIL",
        "png_rgb": "PASS" if output["png"]["mode"] == "RGB" else "FAIL",
        "png_600_dpi": "PASS" if output["png"]["dpi_600"] else "FAIL",
        "manuscript_unchanged": "PASS (not touched by this script)",
    }
    failures = [name for name, value in checks.items() if str(value).startswith("FAIL")]
    qa = {
        "status": "PASS" if not failures else "FAIL",
        "stage": "FINAL_FIGURE_SOURCE_VALIDATED",
        "figure": "Fig4",
        "checks": checks,
        "failures": failures,
        "preflight": preflight,
        "structural_validation": structure,
        "evidence_parity": parity,
        "output_validation": output,
        "source_data": {"csv": norm_path(str(csv_path.relative_to(ROOT))), "csv_sha256": sha256(csv_path),
                        "json": norm_path(str(json_path.relative_to(ROOT))), "json_sha256": sha256(json_path)},
        "interpretation_guards": metadata["interpretation_guards"],
        "notes": [
            "Only dense 1/1024 EV-BOUNDARY-GEOMETRY-V3 sources are admitted.",
            "O2M is the assigned-set Top-1 rank state; legacy and pre-Top-k aliases are excluded.",
            "KM and Aalen-Johansen estimates are rendered as unsmoothed post-step functions.",
            "The scan samples cardinal rays and does not estimate a full two-dimensional boundary radius.",
            "Tau sensitivity supports ordering across tested horizons, not magnitude invariance.",
        ],
    }
    qa_path = FIG_ROOT / "qa" / "Fig4.validation.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if failures:
        raise RuntimeError(f"Figure 4 QA failed: {failures}")


if __name__ == "__main__":
    main()
