"""Render primary-checkpoint qualitative assignment traces for Figure S1."""

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
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "figures_final"
MANIFEST_PATH = FIG_ROOT / "FIGURE_EVIDENCE_MANIFEST.json"
TRACE_REL = "figures_final/source_data/figs1_primary_trace/primary_qualitative_trace.json"
VALIDATION_REL = "figures_final/source_data/figs1_primary_trace/validation.json"
IMAGE_REL = "figures_final/source_data/figs1_primary_trace/0000170_00401_d_0000001__160_0.png"
REPLAY_REL = "results/measurement_validation_20260831/fixed_branch_replay_v1/per_direction.csv"
PRIMARY_CHECKPOINT_SHA = "4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa"
EXPECTED_CASES = (
    ("Stable", 40, "down", 5552, 5452, 5552, 0),
    ("Fragile", 28, "up", 11582, 6264, 6264, 1),
    ("Undefined", 17, "right", 3930, None, None, 1),
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    BRANCH,
    FONT,
    INK,
    LINE,
    MECHANISM,
    MUTED,
    apply_style,
    mm,
)


ACTIVE_COLOR = BRANCH["O2O"]
RUNNER_COLOR = BRANCH["Paired"]
FOCAL_GT_COLOR = MECHANISM["Eligibility"]
SHIFT_GT_COLOR = "#C23B33"
OVERLAY_STYLE = {
    "focal_gt": (FOCAL_GT_COLOR, "-", 0.78),
    "shifted_gt": (SHIFT_GT_COLOR, "--", 0.72),
    "active_candidate": (ACTIVE_COLOR, "-", 0.56),
    "runner_candidate": (RUNNER_COLOR, "--", 0.50),
}


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


def normalise_id(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return None
    return int(float(value))


def load_and_verify_sources() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, dict[str, str]]]:
    manifest = read_json(MANIFEST_PATH)
    panels = [panel for panel in manifest.get("panels", []) if panel.get("panel_id") == "FigS1"]
    if len(panels) != 1:
        raise RuntimeError("current manifest must contain exactly one FigS1 panel")
    panel = panels[0]
    if panel.get("panel_status") != "WARN" or panel.get("evidence_class") != "figure-only qualitative SOURCE_ADDENDUM":
        raise RuntimeError("FigS1 is not admitted as the expected figure-only SOURCE_ADDENDUM")
    expected_roles = {
        "primary-checkpoint qualitative candidate trace": TRACE_REL,
        "primary-checkpoint trace validation": VALIDATION_REL,
        "real AI-TOD-v2 source image": IMAGE_REL,
        "frozen fixed-pixel replay identities": REPLAY_REL,
    }
    verified: dict[str, dict[str, str]] = {}
    for source in panel["source_artifacts"]:
        role = source["role"]
        rel = norm_path(source["path"])
        if role not in expected_roles or rel != expected_roles[role]:
            raise RuntimeError(f"unexpected or old FigS1 source: {role}/{rel}")
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != source["expected_sha256"]:
            raise RuntimeError(f"FigS1 SHA mismatch for {rel}")
        verified[role] = {"path": rel, "sha256": actual, "evidence_id": source["evidence_id"],
                          "authority": source["authority"], "role": role}
    if set(verified) != set(expected_roles):
        raise RuntimeError("FigS1 manifest source set is incomplete")

    trace = read_json(ROOT / TRACE_REL)
    validation = read_json(ROOT / VALIDATION_REL)
    replay = pd.read_csv(ROOT / REPLAY_REL)
    if trace.get("status") != "PASS" or validation.get("status") != "PASS":
        raise RuntimeError("primary trace validation is not PASS")
    if trace.get("checkpoint_sha256") != PRIMARY_CHECKPOINT_SHA or validation.get("checkpoint_sha256") != PRIMARY_CHECKPOINT_SHA:
        raise RuntimeError("trace does not use the primary checkpoint")
    if trace.get("source_image") != IMAGE_REL or trace.get("fixed_replay_source") != REPLAY_REL:
        raise RuntimeError("trace source identity changed")
    if trace.get("source_image_sha256") != verified["real AI-TOD-v2 source image"]["sha256"]:
        raise RuntimeError("trace/source-image hash conflict")
    if trace.get("fixed_replay_source_sha256") != verified["frozen fixed-pixel replay identities"]["sha256"]:
        raise RuntimeError("trace/frozen-replay hash conflict")
    if validation.get("trace_sha256") != verified["primary-checkpoint qualitative candidate trace"]["sha256"]:
        raise RuntimeError("validation sidecar does not bind trace")
    if validation.get("image_sha256") != verified["real AI-TOD-v2 source image"]["sha256"]:
        raise RuntimeError("validation sidecar does not bind image")
    if validation.get("records") != 3 or validation.get("identity_checks") != 9:
        raise RuntimeError("validation-sidecar record/identity counts changed")
    return trace, validation, replay, verified


def validate_identities(trace: dict[str, Any], replay: pd.DataFrame) -> dict[str, Any]:
    records = trace.get("records", [])
    if len(records) != 3:
        raise RuntimeError("trace must contain exactly three cases")
    checks = 0
    for expected, record in zip(EXPECTED_CASES, records):
        case, gt_id, direction, active, runner, shift_active, flip = expected
        if (record["case"], int(record["gt_id"]), record["direction"]) != (case, gt_id, direction):
            raise RuntimeError(f"unexpected case identity: {record['case']}/{record['gt_id']}/{record['direction']}")
        frozen = record["frozen_row"]
        expected_values = (active, runner, shift_active, flip)
        actual_values = (normalise_id(frozen["active"]), normalise_id(frozen["runner"]),
                         normalise_id(frozen["shift_active"]), int(frozen["o2o_flip"]))
        if actual_values != expected_values:
            raise RuntimeError(f"trace frozen-row identity mismatch for GT {gt_id}")
        row = replay[(replay.image_id == trace["image_id"]) & (replay.gt_id == gt_id) &
                     (replay.direction == direction)]
        if len(row) != 1:
            raise RuntimeError(f"frozen replay row is not unique for GT {gt_id}/{direction}")
        row = row.iloc[0]
        replay_values = (normalise_id(row.o2o_active), normalise_id(row.o2o_runner),
                         normalise_id(row.o2o_active_shift), int(row.o2o_flip))
        if replay_values != expected_values:
            raise RuntimeError(f"frozen per_direction identity mismatch for GT {gt_id}")
        if normalise_id(record["base"]["active"]["candidate"]) != active:
            raise RuntimeError(f"base active trace mismatch for GT {gt_id}")
        if normalise_id(record["base"].get("runner", {}).get("candidate") if record["base"].get("runner") else None) != runner:
            raise RuntimeError(f"base runner trace mismatch for GT {gt_id}")
        if normalise_id(record["shift"].get("active", {}).get("candidate") if record["shift"].get("active") else None) != shift_active:
            raise RuntimeError(f"shift active trace mismatch for GT {gt_id}")
        checks += 7
    return {"status": "PASS", "case_records": 3, "identity_checks": checks}


def validate_shift_coordinates(trace: dict[str, Any]) -> dict[str, Any]:
    expected_delta = {"down": (0.0, 1.0, 0.0, 1.0),
                      "up": (0.0, -1.0, 0.0, -1.0),
                      "right": (1.0, 0.0, 1.0, 0.0)}
    maximum = 0.0
    records = []
    for record in trace["records"]:
        base = np.asarray(record["gt_xyxy"], dtype=float)
        shifted = np.asarray(record["shift_gt_xyxy"], dtype=float)
        observed = shifted - base
        expected = np.asarray(expected_delta[record["direction"]], dtype=float)
        difference = float(np.max(np.abs(observed - expected)))
        maximum = max(maximum, difference)
        records.append({"gt_id": int(record["gt_id"]), "direction": record["direction"],
                        "observed_delta_xyxy": observed.tolist(), "expected_delta_xyxy": expected.tolist(),
                        "max_abs_difference": difference})
    if maximum > 1e-9:
        raise RuntimeError(f"shift coordinate mismatch: {maximum}")
    return {"status": "PASS", "maximum_absolute_difference": maximum, "records": records}


def fixed_crop(center_x: float, center_y: float, size: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    half = size / 2
    x1, x2 = center_x - half, center_x + half
    y1, y2 = center_y - half, center_y + half
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_width:
        x1 -= x2 - image_width
        x2 = image_width
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = image_height
    return (x1, y1, x2, y2)


def overlay_row(record: dict[str, Any], column: str, role: str, xyxy: list[float],
                source_field: str, candidate_id: int | None, crop: tuple[float, float, float, float],
                source: dict[str, str]) -> dict[str, Any]:
    color, linestyle, linewidth = OVERLAY_STYLE[role]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return {
        "figure_row": record["case"], "gt_id": int(record["gt_id"]),
        "direction": record["direction"], "column": column, "role": role,
        "source_field": source_field, "candidate_id": candidate_id,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "crop_x1": crop[0], "crop_y1": crop[1], "crop_x2": crop[2], "crop_y2": crop[3],
        "mapped_x1": x1 - crop[0], "mapped_y1": y1 - crop[1],
        "mapped_x2": x2 - crop[0], "mapped_y2": y2 - crop[1],
        "color": color, "linestyle": linestyle, "linewidth_pt": linewidth,
        "source_path": source["path"], "source_sha256": source["sha256"],
        "evidence_id": source["evidence_id"], "authority": source["authority"],
    }


def build_rows(trace: dict[str, Any], verified: dict[str, dict[str, str]], image_size: tuple[int, int],
               content_bbox: tuple[int, int, int, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_width, image_height = image_size
    content_x1, content_y1, content_x2, content_y2 = content_bbox
    if (content_x1, content_y1) != (0, 0):
        raise RuntimeError(f"unexpected nonzero valid-content origin: {content_bbox}")
    trace_source = verified["primary-checkpoint qualitative candidate trace"]
    rows: list[dict[str, Any]] = []
    crop_records = []
    for record in trace["records"]:
        base_gt = np.asarray(record["gt_xyxy"], dtype=float)
        shifted_gt = np.asarray(record["shift_gt_xyxy"], dtype=float)
        center_x = float((min(base_gt[0], shifted_gt[0]) + max(base_gt[2], shifted_gt[2])) / 2)
        center_y = float((min(base_gt[1], shifted_gt[1]) + max(base_gt[3], shifted_gt[3])) / 2)
        # Crop against the detected non-padding content extent rather than the
        # 800x800 model canvas.  This shifts the GT28 context window upward but
        # leaves every trace box and its coordinate mapping unchanged.
        context_crop = fixed_crop(center_x, center_y, 180.0, content_x2, content_y2)
        # Tight, case-specific pixel windows enlarge the tiny target without
        # inventing detail. Every overlay must still fit inside the crop and is
        # checked below against the frozen trace coordinates.
        detail_size = {"Stable": 60.0, "Fragile": 48.0, "Undefined": 32.0}[record["case"]]
        detail_crop = fixed_crop(center_x, center_y, detail_size, content_x2, content_y2)
        crop_records.append({"case": record["case"], "gt_id": int(record["gt_id"]),
                             "context_crop_xyxy": list(context_crop), "detail_crop_xyxy": list(detail_crop),
                             "detail_crop_size_px": detail_size})

        rows.append(overlay_row(record, "context", "focal_gt", record["gt_xyxy"], "gt_xyxy", None,
                                context_crop, trace_source))
        rows.append(overlay_row(record, "base", "focal_gt", record["gt_xyxy"], "gt_xyxy", None,
                                detail_crop, trace_source))
        rows.append(overlay_row(record, "base", "active_candidate", record["base"]["active"]["decoded_xyxy"],
                                "base.active.decoded_xyxy", int(record["base"]["active"]["candidate"]), detail_crop, trace_source))
        if record["base"].get("runner") is not None:
            rows.append(overlay_row(record, "base", "runner_candidate", record["base"]["runner"]["decoded_xyxy"],
                                    "base.runner.decoded_xyxy", int(record["base"]["runner"]["candidate"]), detail_crop, trace_source))

        rows.append(overlay_row(record, "shift", "focal_gt", record["gt_xyxy"], "gt_xyxy", None,
                                detail_crop, trace_source))
        rows.append(overlay_row(record, "shift", "shifted_gt", record["shift_gt_xyxy"], "shift_gt_xyxy", None,
                                detail_crop, trace_source))
        if record["shift"].get("active") is not None:
            rows.append(overlay_row(record, "shift", "active_candidate", record["shift"]["active"]["decoded_xyxy"],
                                    "shift.active.decoded_xyxy", int(record["shift"]["active"]["candidate"]), detail_crop, trace_source))
        if record["shift"].get("runner") is not None:
            rows.append(overlay_row(record, "shift", "runner_candidate", record["shift"]["runner"]["decoded_xyxy"],
                                    "shift.runner.decoded_xyxy", int(record["shift"]["runner"]["candidate"]), detail_crop, trace_source))

    metadata = {
        "status": "SOURCE_BOUND_DRAFT",
        "figure": "Supplementary Figure S1",
        "artifact_role": trace["artifact_role"],
        "checkpoint_sha256": trace["checkpoint_sha256"],
        "image_id": trace["image_id"], "image_size_wh": [image_width, image_height],
        "source_image": trace["source_image"], "source_image_sha256": trace["source_image_sha256"],
        "valid_nonpadding_content_bbox_xyxy": list(content_bbox),
        "source_row_count": len(rows), "crops": crop_records,
        "sources": list(verified.values()),
        "manifest_path": norm_path(str(MANIFEST_PATH.relative_to(ROOT))),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "interpretation_guards": {
            "final_detection_claim": False,
            "aggregate_estimate_claim": False,
            "old_qualitative_pack_used": False,
            "image_or_feature_shifted": False,
            "missing_candidate_imputed": False,
        },
    }
    return rows, metadata


def write_sources(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = FIG_ROOT / "source_data"
    csv_path = output_dir / "figs1_source.csv"
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "figs1_source.json"
    payload = {**metadata, "figs1_source_csv": norm_path(str(csv_path.relative_to(ROOT))),
               "figs1_source_csv_sha256": sha256(csv_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return csv_path, json_path


def add_rectangle(ax: Any, row: pd.Series) -> None:
    if row.role in {"active_candidate", "runner_candidate"}:
        # Preserve the exact audited xyxy extent while exposing the source
        # pixels: candidate overlays are rendered as corner brackets rather
        # than opaque-looking full rectangles around a tiny target.
        x1, y1, x2, y2 = map(float, (row.x1, row.y1, row.x2, row.y2))
        length = 0.22 * min(x2 - x1, y2 - y1)
        segments = (
            ((x1, y1 + length), (x1, y1), (x1 + length, y1)),
            ((x2 - length, y1), (x2, y1), (x2, y1 + length)),
            ((x2, y2 - length), (x2, y2), (x2 - length, y2)),
            ((x1 + length, y2), (x1, y2), (x1, y2 - length)),
        )
        for xs in segments:
            ax.plot([point[0] for point in xs], [point[1] for point in xs],
                    color=row.color, linewidth=float(row.linewidth_pt),
                    linestyle=row.linestyle, solid_capstyle="butt",
                    dash_capstyle="butt", zorder=4)
        return
    ax.add_patch(Rectangle(
        (float(row.x1), float(row.y1)), float(row.x2 - row.x1), float(row.y2 - row.y1),
        fill=False, edgecolor=row.color, linewidth=float(row.linewidth_pt),
        linestyle=row.linestyle, joinstyle="miter", zorder=4,
    ))


def draw_pixel_mosaic(ax: Any, image: np.ndarray, crop: np.ndarray) -> None:
    """Render source pixels as vector cells without interpolation or invented detail."""

    x1 = max(0, int(np.floor(crop[0])))
    y1 = max(0, int(np.floor(crop[1])))
    x2 = min(image.shape[1], int(np.ceil(crop[2])))
    y2 = min(image.shape[0], int(np.ceil(crop[3])))
    tile = image[y1:y2, x1:x2]
    if tile.dtype.kind in {"u", "i"}:
        colors = tile.astype(float) / np.iinfo(tile.dtype).max
    else:
        colors = np.clip(tile.astype(float), 0.0, 1.0)
    values = np.zeros((y2 - y1, x2 - x1), dtype=float)
    mesh = ax.pcolormesh(
        np.arange(x1, x2 + 1), np.arange(y1, y2 + 1), values,
        shading="flat", edgecolors="none", linewidth=0.0, antialiased=False,
    )
    mesh.set_array(None)
    mesh.set_facecolor(colors.reshape(-1, colors.shape[-1]))


def state_text(record: dict[str, Any]) -> str:
    active = normalise_id(record["frozen_row"]["active"])
    shifted = normalise_id(record["frozen_row"]["shift_active"])
    shifted_text = "undefined" if shifted is None else str(shifted)
    case = "Identity-disappearance departure" if record["case"] == "Undefined" else record["case"]
    return f"{case} · GT {record['gt_id']} · {record['direction']} · assigned O2O {active} → {shifted_text}"


def geometry_audit(fig: Any, image_axes: list[Any]) -> dict[str, Any]:
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
    image_boxes = [ax.get_window_extent(renderer) for ax in image_axes]
    for source_axis, label, box in records:
        for target_axis, image_box in enumerate(image_boxes):
            # Titles are outside their own image axis and are permitted; any
            # actual intersection with an image rectangle is not.
            overlap = Bbox.intersection(box, image_box)
            if overlap is not None and overlap.width * overlap.height > 4:
                intrusions.append({"text": label, "text_owner_axis": source_axis,
                                   "image_axis": target_axis,
                                   "overlap_pixels2": overlap.width * overlap.height})
    return {"status": "PASS" if not clipped and not intrusions else "FAIL",
            "clipped_text": clipped, "text_over_image_intrusions": intrusions,
            "in_image_annotation_count": sum(len(ax.texts) for ax in image_axes)}


def make_figure(trace: dict[str, Any], rows: list[dict[str, Any]], image: np.ndarray) -> tuple[Any, dict[str, Any]]:
    font = apply_style()
    plt.rcParams["xtick.labelsize"] = FONT["minimum"]
    plt.rcParams["ytick.labelsize"] = FONT["minimum"]
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["image.resample"] = False
    fig, axes = plt.subplots(3, 3, figsize=(mm(178), mm(170)))
    frame = pd.DataFrame(rows)
    column_names = ("Context", "Base state", "Shift state")
    column_keys = ("context", "base", "shift")
    for row_index, record in enumerate(trace["records"]):
        for column_index, column in enumerate(column_keys):
            ax = axes[row_index, column_index]
            subset = frame[(frame.figure_row == record["case"]) & (frame.column == column)]
            crop = subset.iloc[0][["crop_x1", "crop_y1", "crop_x2", "crop_y2"]].astype(float).to_numpy()
            if column == "context":
                ax.imshow(image, interpolation="none", resample=False)
            else:
                draw_pixel_mosaic(ax, image, crop)
            ax.set_xlim(crop[0], crop[2])
            ax.set_ylim(crop[3], crop[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for _, overlay in subset.iterrows():
                add_rectangle(ax, overlay)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row_index, 0].set_title(state_text(record), loc="left", pad=3,
                                     fontsize=FONT["panel_title"], fontweight="bold", color=INK)

    fig.subplots_adjust(left=0.035, right=0.992, top=0.875, bottom=0.13, hspace=0.30, wspace=0.08)
    fig.canvas.draw()
    for column_index, name in enumerate(column_names):
        position = axes[0, column_index].get_position()
        fig.text((position.x0 + position.x1) / 2, 0.905, name, ha="center", va="bottom",
                 fontsize=FONT["panel_title"], fontweight="bold", color=INK)
    fig.suptitle("Representative audited assignment-state cases", x=0.012, y=0.982,
                 ha="left", va="top", fontsize=FONT["panel_label"], fontweight="bold", color=INK)
    handles = [
        Line2D([], [], color=FOCAL_GT_COLOR, linewidth=0.78, linestyle="-", label="Focal GT"),
        Line2D([], [], color=SHIFT_GT_COLOR, linewidth=0.72, linestyle="--", label="Shifted GT"),
        Line2D([], [], color=ACTIVE_COLOR, linewidth=0.56, linestyle="-", label="Assigned O2O candidate (corner marks)"),
        Line2D([], [], color=RUNNER_COLOR, linewidth=0.50, linestyle="--", label="Runner candidate (corner marks)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=4,
               handlelength=2.0, columnspacing=1.2, handletextpad=0.45)
    fig.text(0.012, 0.018, "Overlays are audited assignment states, not final detections.",
             ha="left", va="bottom", fontsize=FONT["minimum"], color=INK)
    fig.text(0.012, 0.041, "Detail insets render source pixels as vector cells without interpolation or enhancement; only the focal GT centre moves by one input pixel.",
             ha="left", va="bottom", fontsize=FONT["minimum"], color=MUTED)
    fig.canvas.draw()
    visible = [item for item in fig.findobj(match=lambda obj: hasattr(obj, "get_text") and hasattr(obj, "get_fontsize"))
               if item.get_visible() and str(item.get_text()).strip()]
    for item in visible:
        if item.get_fontsize() < FONT["minimum"]:
            item.set_fontsize(FONT["minimum"])
    fig.canvas.draw()
    geometry = geometry_audit(fig, list(axes.flat))
    return fig, {"resolved_font": font, "minimum_visible_font_pt": min(item.get_fontsize() for item in visible),
                 "visible_text_count": len(visible), "axes_count": len(fig.axes), "geometry": geometry}


def save_draft(fig: Any, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("svg", {"dpi": 600}), ("pdf", {"dpi": 600}), ("png", {"dpi": 600})):
        fig.savefig(output_stem.with_suffix(f".{extension}"), facecolor="white", **kwargs)
    plt.close(fig)
    png = output_stem.with_suffix(".png")
    with Image.open(png) as source:
        source.convert("RGB").save(png, dpi=(600, 600))


def coordinate_parity(rows: list[dict[str, Any]], trace: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    maximum = 0.0
    comparisons = 0
    candidate_mismatches: list[str] = []
    for record in trace["records"]:
        expected: list[tuple[str, str, list[float], int | None]] = [
            ("context", "focal_gt", record["gt_xyxy"], None),
            ("base", "focal_gt", record["gt_xyxy"], None),
            ("base", "active_candidate", record["base"]["active"]["decoded_xyxy"], int(record["base"]["active"]["candidate"])),
            ("shift", "focal_gt", record["gt_xyxy"], None),
            ("shift", "shifted_gt", record["shift_gt_xyxy"], None),
        ]
        if record["base"].get("runner") is not None:
            expected.append(("base", "runner_candidate", record["base"]["runner"]["decoded_xyxy"], int(record["base"]["runner"]["candidate"])))
        if record["shift"].get("active") is not None:
            expected.append(("shift", "active_candidate", record["shift"]["active"]["decoded_xyxy"], int(record["shift"]["active"]["candidate"])))
        if record["shift"].get("runner") is not None:
            expected.append(("shift", "runner_candidate", record["shift"]["runner"]["decoded_xyxy"], int(record["shift"]["runner"]["candidate"])))
        subset = frame[frame.figure_row == record["case"]]
        if len(subset) != len(expected):
            raise RuntimeError(f"overlay row count mismatch for {record['case']}")
        for column, role, xyxy, candidate in expected:
            match = subset[(subset.column == column) & (subset.role == role)]
            if len(match) != 1:
                raise RuntimeError(f"overlay row not unique: {record['case']}/{column}/{role}")
            actual = match.iloc[0]
            actual_xyxy = np.asarray([actual.x1, actual.y1, actual.x2, actual.y2], dtype=float)
            maximum = max(maximum, float(np.max(np.abs(actual_xyxy - np.asarray(xyxy, dtype=float)))))
            mapped = actual_xyxy - np.asarray([actual.crop_x1, actual.crop_y1, actual.crop_x1, actual.crop_y1], dtype=float)
            stored_mapped = np.asarray([actual.mapped_x1, actual.mapped_y1, actual.mapped_x2, actual.mapped_y2], dtype=float)
            maximum = max(maximum, float(np.max(np.abs(mapped - stored_mapped))))
            comparisons += 8
            if normalise_id(actual.candidate_id) != candidate:
                candidate_mismatches.append(f"{record['case']}/{column}/{role}")
    return {"status": "PASS" if maximum <= 1e-12 and not candidate_mismatches else "FAIL",
            "coordinate_comparisons": comparisons, "maximum_absolute_difference": maximum,
            "candidate_mismatches": candidate_mismatches}


def box_and_crop_validation(rows: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    frame = pd.DataFrame(rows)
    boxes_valid = ((frame.x1 >= 0) & (frame.y1 >= 0) & (frame.x2 <= width) & (frame.y2 <= height) &
                   (frame.x2 > frame.x1) & (frame.y2 > frame.y1)).all()
    contained = ((frame.x1 >= frame.crop_x1) & (frame.y1 >= frame.crop_y1) &
                 (frame.x2 <= frame.crop_x2) & (frame.y2 <= frame.crop_y2)).all()
    expected_counts = {"Stable": 8, "Fragile": 8, "Undefined": 5}
    counts = frame.groupby("figure_row").size().to_dict()
    return {"status": "PASS" if boxes_valid and contained and counts == expected_counts else "FAIL",
            "boxes_within_image": bool(boxes_valid), "overlays_within_crops": bool(contained),
            "expected_case_rows": expected_counts, "actual_case_rows": counts,
            "total_overlay_rows": len(frame)}


def detect_nonpadding_content_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bbox of rows/columns containing any non-black pixel."""
    nonblack = np.any(image != 0, axis=2)
    ys, xs = np.where(nonblack)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("source image contains no non-black content")
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def padding_crop_validation(rows: list[dict[str, Any]], image: np.ndarray,
                            content_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    """Prove that every displayed crop stays inside non-padding image content."""
    frame = pd.DataFrame(rows)
    unique = frame[["figure_row", "column", "crop_x1", "crop_y1", "crop_x2", "crop_y2"]].drop_duplicates()
    x1_valid, y1_valid, x2_valid, y2_valid = content_bbox
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for record in unique.to_dict("records"):
        x1, y1 = int(round(float(record["crop_x1"]))), int(round(float(record["crop_y1"])))
        x2, y2 = int(round(float(record["crop_x2"]))), int(round(float(record["crop_y2"])))
        within = x1 >= x1_valid and y1 >= y1_valid and x2 <= x2_valid and y2 <= y2_valid
        crop = image[y1:y2, x1:x2]
        all_black_rows = int(np.sum(np.all(crop == 0, axis=(1, 2)))) if crop.size else -1
        item = {"case": record["figure_row"], "column": record["column"],
                "crop_xyxy": [float(record["crop_x1"]), float(record["crop_y1"]),
                               float(record["crop_x2"]), float(record["crop_y2"])],
                "within_nonpadding_bbox": bool(within), "all_black_rows": all_black_rows}
        records.append(item)
        if not within or all_black_rows != 0:
            failures.append(item)
    return {"status": "PASS" if not failures else "FAIL",
            "detected_nonpadding_bbox_xyxy": list(content_bbox),
            "displayed_crops": len(records), "records": records, "failures": failures}


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
    from pypdf import PdfReader
    svg, pdf, png = stem.with_suffix(".svg"), stem.with_suffix(".pdf"), stem.with_suffix(".png")
    svg_text = svg.read_text(encoding="utf-8")
    with Image.open(png) as image:
        dpi = image.info.get("dpi", (0, 0))
        png_info = {"pixels": list(image.size), "dpi": [float(dpi[0]), float(dpi[1])],
                    "mode": image.mode, "dpi_600": min(dpi) >= 599}
    page = PdfReader(str(pdf)).pages[0]
    width = float(page.mediabox.width) * 25.4 / 72
    height = float(page.mediabox.height) * 25.4 / 72
    return {
        "outputs": {path.suffix[1:]: {"path": norm_path(str(path.relative_to(ROOT))),
                                       "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in (svg, pdf, png)},
        "svg_text_nodes": svg_text.count("<text"), "svg_image_nodes": svg_text.count("<image"),
        "svg_editable_text": "<text" in svg_text, "svg_contains_raster_source": "<image" in svg_text,
        "png": png_info,
        "pdf_geometry": {"width_mm": width, "height_mm": height, "target_mm": [178, 170],
                         "within_0_1_mm": abs(width - 178) <= 0.1 and abs(height - 170) <= 0.1},
        "pdf_fonts": pdf_font_audit(pdf),
    }


def main() -> None:
    trace, _, replay, verified = load_and_verify_sources()
    identity = validate_identities(trace, replay)
    shift_coordinates = validate_shift_coordinates(trace)
    with Image.open(ROOT / IMAGE_REL) as source_image:
        if source_image.size != tuple(reversed(trace["input_hw"])):
            raise RuntimeError(f"source image dimensions {source_image.size} != trace input {trace['input_hw']}")
        image = np.asarray(source_image.convert("RGB"))
        image_size = source_image.size
    content_bbox = detect_nonpadding_content_bbox(image)
    rows, metadata = build_rows(trace, verified, image_size, content_bbox)
    csv_path, json_path = write_sources(rows, metadata)
    coordinate = coordinate_parity(rows, trace)
    boxes = box_and_crop_validation(rows, image_size)
    padding = padding_crop_validation(rows, image, content_bbox)
    fig, preflight = make_figure(trace, rows, image)
    stem = FIG_ROOT / "draft" / "FigS1"
    save_draft(fig, stem)
    output = output_audit(stem)
    checks = {
        "source_sha_and_primary_checkpoint_identity": "PASS",
        "old_qualitative_pack_excluded": "PASS",
        "frozen_per_direction_identity": identity["status"],
        "one_pixel_shift_coordinates": shift_coordinates["status"],
        "trace_coordinate_and_candidate_parity": coordinate["status"],
        "boxes_within_source_and_crops": boxes["status"],
        "all_displayed_crops_exclude_black_padding": padding["status"],
        "labels_outside_images": "PASS" if preflight["geometry"]["in_image_annotation_count"] == 0 else "FAIL",
        "text_clipping_and_image_overlap": preflight["geometry"]["status"],
        "minimum_visible_font": "PASS" if preflight["minimum_visible_font_pt"] >= 7 else "FAIL",
        "arial_or_permitted_fallback": "PASS" if preflight["resolved_font"]["family"] in {"Arial", "Helvetica", "Liberation Sans"} else "FAIL",
        "svg_editable_text_and_raster_image": "PASS" if output["svg_editable_text"] and output["svg_contains_raster_source"] else "FAIL",
        "pdf_type42_embedded": output["pdf_fonts"]["status"],
        "pdf_final_size": "PASS" if output["pdf_geometry"]["within_0_1_mm"] else "FAIL",
        "png_rgb": "PASS" if output["png"]["mode"] == "RGB" else "FAIL",
        "png_600_dpi": "PASS" if output["png"]["dpi_600"] else "FAIL",
        "interpretation_guard_present": "PASS",
        "manuscript_unchanged": "PASS (not touched by this script)",
    }
    failures = [name for name, value in checks.items() if str(value).startswith("FAIL")]
    qa = {
        "status": "PASS" if not failures else "FAIL", "stage": "FINAL_FIGURE_SOURCE_VALIDATED",
        "figure": "FigS1", "checks": checks, "failures": failures,
        "identity_validation": identity, "shift_coordinate_validation": shift_coordinates,
        "coordinate_parity": coordinate, "box_and_crop_validation": boxes,
        "padding_crop_validation": padding,
        "preflight": preflight, "output_validation": output,
        "source_data": {"csv": norm_path(str(csv_path.relative_to(ROOT))), "csv_sha256": sha256(csv_path),
                        "json": norm_path(str(json_path.relative_to(ROOT))), "json_sha256": sha256(json_path)},
        "interpretation_guards": metadata["interpretation_guards"],
        "notes": [
            "Only the primary-checkpoint qualitative trace and its SHA-bound source image are used.",
            "GT and candidate rectangles are copied exactly from trace xyxy fields; no box is hand positioned.",
            "All semantic labels and the legend are outside image axes.",
            "Base/shift detail insets render each source pixel as a same-colour vector cell; no interpolation, enhancement, or detail synthesis is applied.",
            "Active and runner candidates use exact-coordinate corner marks to reduce target occlusion; GT extents remain full rectangles.",
            "Overlays are audited assignment states, not final detections.",
        ],
    }
    qa_path = FIG_ROOT / "qa" / "FigS1.validation.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if failures:
        raise RuntimeError(f"Figure S1 QA failed: {failures}")


if __name__ == "__main__":
    main()
