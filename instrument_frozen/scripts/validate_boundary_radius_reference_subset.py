"""Validate continuous O2O boundary radii with an independent NumPy assigner.

The validator chooses an outcome-blind SHA-256 subset of a frozen
``selected-images`` manifest.  Every chosen image is forwarded exactly once;
the probabilities, decoded boxes, anchors, and GT state required by the
reference contract are serialized to NPZ.  All radius scans are then executed
from those serialized arrays through ``reference_o2o_assigner_numpy`` only.

Reference trajectories are compared with an existing production run from
``analyze_assignment_boundary_radius.py``.  Event/censor status and first
divergence must agree exactly.  Event radii and brackets may differ by at most
one requested fine-grid step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

# Deliberately do not import margin_o2o_replay or any production taxonomy
# helper.  Assignment and stage localisation below use only the independent
# NumPy reference implementation.
from reference_o2o_assigner_numpy import (
    active_by_gt,
    assign_reference,
    first_divergence_reference,
)
from ultralytics.utils.tal import make_anchors


DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))


class SelectedDataset:
    """Minimal index view used only by the Ultralytics data loader."""

    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.collate_fn = dataset.collate_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--production-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmax", type=float, required=True)
    parser.add_argument("--coarse-step", type=float, required=True)
    parser.add_argument("--fine-step", type=float, required=True)
    parser.add_argument("--hash-threshold", type=int)
    parser.add_argument("--hash-modulus", type=int, default=10000)
    parser.add_argument("--selection-seed", type=int, default=20260831)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def image_hash(seed: int, image_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def choose_subset(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.hash_modulus <= 0:
        raise ValueError("hash-modulus must be positive")
    if args.hash_threshold is not None and not 0 <= args.hash_threshold <= args.hash_modulus:
        raise ValueError("hash-threshold must be in [0, hash-modulus]")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    if args.hash_threshold is None and args.max_images is None and not args.smoke:
        raise ValueError("specify hash-threshold/modulus or max-images")

    ranked: list[dict] = []
    for row in rows:
        copied = dict(row)
        value = image_hash(args.selection_seed, str(copied["image_id"]))
        copied["reference_subset_hash_u64"] = value
        copied["reference_subset_hash_remainder"] = value % args.hash_modulus
        if args.hash_threshold is None or value % args.hash_modulus < args.hash_threshold:
            ranked.append(copied)
    ranked.sort(key=lambda row: (int(row["reference_subset_hash_u64"]), str(row["image_id"])))
    if args.max_images is not None:
        ranked = ranked[: args.max_images]
    if args.smoke:
        ranked = ranked[:1]
    if not ranked:
        raise RuntimeError("SHA-256 subset selection produced zero images")
    return ranked


def size_bin(area: float) -> str:
    side = math.sqrt(max(float(area), 0.0))
    if side < 8.0:
        return "vt_lt8"
    if side < 16.0:
        return "t_8_16"
    if side < 32.0:
        return "s_16_32"
    return "m_ge32"


def forward_grid(start: float, stop: float, step: float) -> list[float]:
    """Increasing points in ``(start, stop]``, including the exact stop."""
    if stop <= start:
        return []
    points: list[float] = []
    index = 1
    while start + index * step < stop - 1e-12:
        points.append(start + index * step)
        index += 1
    points.append(stop)
    return points


def legal_radius_limit(
    box: np.ndarray, direction: str, image_w: float, image_h: float, scale: float
) -> float:
    if direction == "left":
        pixels = float(box[0])
    elif direction == "right":
        pixels = image_w - float(box[2])
    elif direction == "up":
        pixels = float(box[1])
    elif direction == "down":
        pixels = image_h - float(box[3])
    else:
        raise ValueError(direction)
    return max(0.0, pixels / scale)


def shifted_box(
    box: np.ndarray, dx: float, dy: float, image_w: float, image_h: float
) -> np.ndarray | None:
    result = np.asarray(box, dtype=np.float32).copy()
    result[[0, 2]] += np.float32(dx)
    result[[1, 3]] += np.float32(dy)
    if (
        result[0] < 0.0
        or result[1] < 0.0
        or result[2] > image_w
        or result[3] > image_h
        or result[2] <= result[0]
        or result[3] <= result[1]
    ):
        return None
    return result


def guarded_scan_limit(
    box: np.ndarray,
    direction: str,
    unit_dx: int,
    unit_dy: int,
    image_w: float,
    image_h: float,
    scale: float,
    rmax: float,
) -> tuple[float, str, float]:
    """Independent NumPy implementation of the frozen float32 endpoint guard."""
    image_limit = legal_radius_limit(box, direction, image_w, image_h, scale)
    scan_limit = min(rmax, image_limit)
    reason = "rmax" if rmax <= image_limit + 1e-12 else "image_boundary"
    if shifted_box(box, unit_dx * scan_limit * scale, unit_dy * scan_limit * scale, image_w, image_h) is not None or scan_limit <= 0.0:
        return scan_limit, reason, 0.0
    epsilon = float(np.finfo(np.float32).eps)
    for multiplier in (8.0, 16.0, 32.0, 64.0, 128.0):
        guard_pixels = multiplier * epsilon * max(image_w, image_h, 1.0)
        guarded = max(0.0, scan_limit - guard_pixels / scale)
        if shifted_box(box, unit_dx * guarded * scale, unit_dy * guarded * scale, image_w, image_h) is not None:
            return guarded, reason, scan_limit * scale - guarded * scale
    raise RuntimeError("unable to construct a float32-safe legal image-boundary scan limit")


def active_identity(state, gt: int) -> int | None:
    return active_by_gt(state)[gt]


def scan_reference_trajectory(
    *,
    base_state,
    class_scores: np.ndarray,
    pred_boxes: np.ndarray,
    anchors: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    valid_gt: np.ndarray,
    gt: int,
    direction: str,
    unit_dx: int,
    unit_dy: int,
    image_w: float,
    image_h: float,
    rmax: float,
    coarse_step: float,
    fine_step: float,
) -> dict:
    box = gt_boxes[gt]
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    scale = math.sqrt(width * height)
    scan_limit, censor_reason, boundary_guard_pixels = guarded_scan_limit(
        box, direction, unit_dx, unit_dy, image_w, image_h, scale, rmax
    )
    base_active = active_identity(base_state, gt)
    cache: dict[float, tuple[int | None, object]] = {0.0: (base_active, base_state)}

    def evaluate(radius: float):
        key = round(float(radius), 12)
        if key not in cache:
            moved = shifted_box(
                box,
                unit_dx * radius * scale,
                unit_dy * radius * scale,
                image_w,
                image_h,
            )
            if moved is None:
                raise RuntimeError("legal-radius calculation admitted an illegal shift")
            altered = gt_boxes.copy()
            altered[gt] = moved
            state = assign_reference(
                class_scores,
                pred_boxes,
                anchors,
                gt_labels,
                altered,
                valid_gt,
                topk=7,
                topk2=1,
            )
            cache[key] = (active_identity(state, gt), state)
        return cache[key]

    bracket: tuple[float, float] | None = None
    previous = 0.0
    for radius in forward_grid(0.0, scan_limit, coarse_step):
        active, _ = evaluate(radius)
        if active != base_active:
            bracket = (previous, radius)
            break
        previous = radius

    if bracket is None:
        return {
            "o2o_base_active": base_active,
            "o2o_event_observed": 0,
            "o2o_radius": None,
            "o2o_right_censored": 1,
            "o2o_censor_radius": scan_limit,
            "o2o_censor_reason": censor_reason,
            "o2o_bracket_low": None,
            "o2o_bracket_high": None,
            "o2o_event_active": None,
            "o2o_first_divergence": "right_censored",
            "scan_limit": scan_limit,
            "scan_limit_reason": censor_reason,
            "scan_limit_float32_guard_pixels": boundary_guard_pixels,
        }

    low, high = bracket
    refined_low = low
    event_radius = high
    event_active, event_state = evaluate(high)
    for radius in forward_grid(low, high, fine_step):
        active, state = evaluate(radius)
        if active != base_active:
            event_radius, event_active, event_state = radius, active, state
            break
        refined_low = radius
    return {
        "o2o_base_active": base_active,
        "o2o_event_observed": 1,
        "o2o_radius": event_radius,
        "o2o_right_censored": 0,
        "o2o_censor_radius": None,
        "o2o_censor_reason": None,
        "o2o_bracket_low": refined_low,
        "o2o_bracket_high": event_radius,
        "o2o_event_active": event_active,
        "o2o_first_divergence": first_divergence_reference(base_state, event_state, gt),
        "scan_limit": scan_limit,
        "scan_limit_reason": censor_reason,
        "scan_limit_float32_guard_pixels": boundary_guard_pixels,
    }


def optional_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def same_optional_int(left, right) -> bool:
    left, right = optional_value(left), optional_value(right)
    if left is None or right is None:
        return left is None and right is None
    return int(left) == int(right)


def close_optional(left, right, tolerance: float) -> bool:
    left, right = optional_value(left), optional_value(right)
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def validate_production_contract(args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    manifest_path = args.production_output_dir / "manifest.json"
    events_path = args.production_output_dir / "per_direction_boundary_events.csv"
    if not manifest_path.is_file() or not events_path.is_file():
        raise FileNotFoundError("production output lacks manifest or per-direction events")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("rmax", args.rmax),
        ("coarse_step", args.coarse_step),
        ("fine_step", args.fine_step),
    ):
        if not math.isclose(float(manifest[field]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"production {field} differs from requested reference value")
    events = pd.read_csv(events_path)
    required = {
        "image_id",
        "gt_id",
        "direction",
        "o2o_base_active",
        "o2o_event_observed",
        "o2o_radius",
        "o2o_right_censored",
        "o2o_censor_radius",
        "o2o_censor_reason",
        "o2o_bracket_low",
        "o2o_bracket_high",
        "o2o_event_active",
        "o2o_first_divergence",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"production event table missing fields: {sorted(missing)}")
    return manifest, events


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data or selected-images manifest missing")
    if not (args.rmax > 0.0 and args.coarse_step > 0.0 and args.fine_step > 0.0):
        raise ValueError("rmax and grid steps must be positive")
    if args.fine_step > args.coarse_step:
        raise ValueError("fine-step must not exceed coarse-step")

    production_manifest, production_events = validate_production_contract(args)
    production_checkpoint_hash = str(production_manifest.get("checkpoint_sha256", "")).lower()
    if production_checkpoint_hash != args.expected_sha256.lower():
        raise RuntimeError("production and reference checkpoint hashes differ")
    source_rows = pd.read_csv(args.selected_images).to_dict("records")
    required_selection = {"dataset_index", "image_id"}
    if not source_rows or required_selection.difference(source_rows[0]):
        raise ValueError("selected-images manifest is empty or incomplete")
    # The independent reference may only select trajectories that the supplied
    # production run actually evaluated.  This matters for smoke runs, whose
    # production manifest intentionally contains only a prefix of the frozen
    # 300-image roster.  Hashing the full source manifest first can otherwise
    # select a different image and turn every comparison into a missing-row
    # failure rather than a semantic parity test.
    production_image_ids = set(production_events["image_id"].astype(str))
    comparable_source_rows = [row for row in source_rows if str(row["image_id"]) in production_image_ids]
    if not comparable_source_rows:
        raise RuntimeError("production run has no images in the frozen selected-images manifest")
    selected_rows = choose_subset(comparable_source_rows, args)

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if (
        head.__class__.__name__ != "Detect"
        or not model.end2end
        or not head.end2end
        or int(head.reg_max) != 1
    ):
        raise RuntimeError("protocol is frozen for the native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    criterion = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if int(criterion.assigner.topk) != 7 or int(criterion.assigner.topk2) != 1:
        raise RuntimeError("unexpected native O2O Top-7/Top-1 contract")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": args.imgsz,
            "batch": 1,
            "workers": args.workers,
            "rect": False,
            "cache": False,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        data["val"],
        batch=1,
        data=data,
        mode="val",
        stride=max(int(model.stride.max()), 32),
    )
    for row in selected_rows:
        index = int(row["dataset_index"])
        observed = Path(dataset.labels[index]["im_file"]).stem
        if observed != str(row["image_id"]):
            raise RuntimeError("selected-images manifest does not match frozen dataset index")

    indices = [int(row["dataset_index"]) for row in selected_rows]
    loader = build_dataloader(
        SelectedDataset(dataset, indices),
        batch=1,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
    )

    args.output_dir.mkdir(parents=True)
    serialized_dir = args.output_dir / "serialized_inputs"
    serialized_dir.mkdir()
    write_rows(args.output_dir / "selected_images.csv", selected_rows)
    started = time.monotonic()
    serialized_rows: list[dict] = []
    forward_calls = 0

    # Phase 1: exactly one network forward per selected image, followed by a
    # compact serialization of only the classes present among valid GTs.
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            forward_calls += 1
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw dual-branch detector output")
            raw = output[1]["one2one"]
            targets = torch.cat(
                (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                1,
            )
            image_h, image_w = batch["img"].shape[-2:]
            targets = criterion.preprocess(
                targets,
                1,
                torch.tensor(
                    [image_w, image_h, image_w, image_h],
                    device=device,
                    dtype=batch["img"].dtype,
                ),
            )
            labels, gt_boxes = targets.split((1, 4), 2)
            valid = gt_boxes.sum(2, keepdim=True).gt_(0)
            pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
            logits = raw["scores"].permute(0, 2, 1).contiguous()
            anchors, stride = make_anchors(raw["feats"], criterion.stride, 0.5)
            decoded = criterion.bbox_decode(anchors, pred_distri)

            valid_np = valid[0, :, 0].detach().cpu().numpy().astype(bool)
            original_labels = labels[0, :, 0].long().detach().cpu().numpy()
            class_ids = np.unique(original_labels[valid_np]).astype(np.int64)
            if len(class_ids) == 0:
                continue
            remap = {int(value): index for index, value in enumerate(class_ids.tolist())}
            remapped_labels = np.array(
                [remap.get(int(value), 0) for value in original_labels], dtype=np.int64
            )
            scores = logits.sigmoid()[0, :, class_ids.tolist()].detach().cpu().numpy().astype(np.float32)
            pred_boxes = (decoded * stride)[0].detach().cpu().numpy().astype(np.float32)
            anchor_pixels = (anchors * stride).detach().cpu().numpy().astype(np.float32)
            gt_boxes_np = gt_boxes[0].detach().cpu().numpy().astype(np.float32)
            image_id = Path(batch["im_file"][0]).stem
            target = serialized_dir / f"{image_id}.npz"
            np.savez_compressed(
                target,
                class_scores=scores,
                pred_boxes=pred_boxes,
                anchors=anchor_pixels,
                gt_labels=remapped_labels,
                original_gt_labels=original_labels.astype(np.int64),
                class_ids=class_ids,
                gt_boxes=gt_boxes_np,
                valid_gt=valid_np,
                image_size=np.array([image_h, image_w], dtype=np.int64),
            )
            serialized_rows.append(
                {
                    "image_id": image_id,
                    "path": str(target.relative_to(args.output_dir)),
                    "sha256": sha256(target),
                    "candidates": len(pred_boxes),
                    "valid_gt": int(valid_np.sum()),
                    "stored_classes": len(class_ids),
                }
            )

    if forward_calls != len(selected_rows) or len(serialized_rows) != len(selected_rows):
        raise RuntimeError("one-forward-per-selected-image serialization contract failed")
    write_rows(args.output_dir / "serialized_inputs_manifest.csv", serialized_rows)

    # Phase 2: pure NumPy reference scans from serialized inputs.
    reference_rows: list[dict] = []
    for serialized in serialized_rows:
        image_id = str(serialized["image_id"])
        with np.load(args.output_dir / str(serialized["path"]), allow_pickle=False) as data_npz:
            class_scores = data_npz["class_scores"]
            pred_boxes = data_npz["pred_boxes"]
            anchors = data_npz["anchors"]
            gt_labels = data_npz["gt_labels"]
            gt_boxes = data_npz["gt_boxes"]
            valid_gt = data_npz["valid_gt"]
            image_h, image_w = map(float, data_npz["image_size"])
        base_state = assign_reference(
            class_scores,
            pred_boxes,
            anchors,
            gt_labels,
            gt_boxes,
            valid_gt,
            topk=7,
            topk2=1,
        )
        for gt in np.where(valid_gt)[0].tolist():
            box = gt_boxes[gt]
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            area = width * height
            bucket = size_bin(area)
            if bucket not in {"t_8_16", "s_16_32"}:
                continue
            if shifted_box(box, 0.0, 0.0, image_w, image_h) is None:
                continue
            for direction, unit_dx, unit_dy in DIRECTIONS:
                result = scan_reference_trajectory(
                    base_state=base_state,
                    class_scores=class_scores,
                    pred_boxes=pred_boxes,
                    anchors=anchors,
                    gt_labels=gt_labels,
                    gt_boxes=gt_boxes,
                    valid_gt=valid_gt,
                    gt=gt,
                    direction=direction,
                    unit_dx=unit_dx,
                    unit_dy=unit_dy,
                    image_w=image_w,
                    image_h=image_h,
                    rmax=args.rmax,
                    coarse_step=args.coarse_step,
                    fine_step=args.fine_step,
                )
                reference_rows.append(
                    {
                        "image_id": image_id,
                        "gt_id": gt,
                        "direction": direction,
                        "size_bin": bucket,
                        **result,
                    }
                )
    write_rows(args.output_dir / "reference_trajectories.csv", reference_rows)

    selected_ids = {str(row["image_id"]) for row in selected_rows}
    production_events = production_events[
        production_events["image_id"].astype(str).isin(selected_ids)
    ].copy()
    production_index: dict[tuple[str, int, str], dict] = {}
    for row in production_events.to_dict("records"):
        key = (str(row["image_id"]), int(row["gt_id"]), str(row["direction"]))
        if key in production_index:
            raise RuntimeError(f"duplicate production trajectory: {key}")
        production_index[key] = row

    comparisons: list[dict] = []
    mismatches: list[dict] = []
    bracket_tolerance = args.fine_step + 1e-9
    radius_differences: list[float] = []
    for reference in reference_rows:
        key = (str(reference["image_id"]), int(reference["gt_id"]), str(reference["direction"]))
        production = production_index.get(key)
        if production is None:
            checks = {"production_row_present": False}
            comparison = {
                "image_id": key[0],
                "gt_id": key[1],
                "direction": key[2],
                "pass": 0,
                "mismatch": "production_row_present",
            }
        else:
            checks = {
                "base_active": same_optional_int(reference["o2o_base_active"], production["o2o_base_active"]),
                "event_observed": int(reference["o2o_event_observed"]) == int(production["o2o_event_observed"]),
                "right_censored": int(reference["o2o_right_censored"]) == int(production["o2o_right_censored"]),
                "event_active": same_optional_int(reference["o2o_event_active"], production["o2o_event_active"]),
                "radius": close_optional(reference["o2o_radius"], production["o2o_radius"], bracket_tolerance),
                "bracket_low": close_optional(reference["o2o_bracket_low"], production["o2o_bracket_low"], bracket_tolerance),
                "bracket_high": close_optional(reference["o2o_bracket_high"], production["o2o_bracket_high"], bracket_tolerance),
                "censor_radius": close_optional(reference["o2o_censor_radius"], production["o2o_censor_radius"], bracket_tolerance),
                "censor_reason": str(optional_value(reference["o2o_censor_reason"])) == str(optional_value(production["o2o_censor_reason"])),
                "first_divergence": str(reference["o2o_first_divergence"]) == str(production["o2o_first_divergence"]),
            }
            ref_radius, prod_radius = optional_value(reference["o2o_radius"]), optional_value(production["o2o_radius"])
            radius_difference = None
            if ref_radius is not None and prod_radius is not None:
                radius_difference = abs(float(ref_radius) - float(prod_radius))
                radius_differences.append(radius_difference)
            failed = [name for name, passed in checks.items() if not passed]
            comparison = {
                "image_id": key[0],
                "gt_id": key[1],
                "direction": key[2],
                "reference_event_observed": reference["o2o_event_observed"],
                "production_event_observed": production["o2o_event_observed"],
                "reference_radius": reference["o2o_radius"],
                "production_radius": production["o2o_radius"],
                "radius_abs_difference": radius_difference,
                "reference_bracket_low": reference["o2o_bracket_low"],
                "production_bracket_low": production["o2o_bracket_low"],
                "reference_bracket_high": reference["o2o_bracket_high"],
                "production_bracket_high": production["o2o_bracket_high"],
                "reference_first_divergence": reference["o2o_first_divergence"],
                "production_first_divergence": production["o2o_first_divergence"],
                "pass": int(not failed),
                "mismatch": ";".join(failed),
            }
        comparisons.append(comparison)
        if not int(comparison["pass"]):
            mismatches.append(comparison)

    reference_keys = {
        (str(row["image_id"]), int(row["gt_id"]), str(row["direction"]))
        for row in reference_rows
    }
    unexpected_production = sorted(set(production_index).difference(reference_keys))
    for image_id, gt_id, direction in unexpected_production:
        row = {
            "image_id": image_id,
            "gt_id": gt_id,
            "direction": direction,
            "pass": 0,
            "mismatch": "unexpected_production_row",
        }
        comparisons.append(row)
        mismatches.append(row)

    write_rows(args.output_dir / "trajectory_comparison.csv", comparisons)
    write_rows(args.output_dir / "mismatches.csv", mismatches)
    summary = {
        "status": "PASS" if not mismatches else "FAIL",
        "protocol": "continuous_boundary_numpy_reference_subset_v1",
        "selected_images": len(selected_rows),
        "forward_calls": forward_calls,
        "one_forward_per_image": forward_calls == len(selected_rows),
        "reference_trajectories": len(reference_rows),
        "production_trajectories_in_subset": len(production_index),
        "mismatches": len(mismatches),
        "allowed_radius_and_bracket_tolerance": bracket_tolerance,
        "maximum_event_radius_absolute_difference": max(radius_differences, default=0.0),
        "elapsed_sec": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **summary,
        "read_only": True,
        "reference_assignment": "reference_o2o_assigner_numpy.assign_reference",
        "production_results_used_only_for_comparison": True,
        "subset": {
            "selection_seed": args.selection_seed,
            "hash_expression": "SHA256(f'{seed}|{image_id}') first 64 bits, big endian",
            "hash_threshold": args.hash_threshold,
            "hash_modulus": args.hash_modulus,
            "max_images_after_hash_ranking": args.max_images,
            "smoke": bool(args.smoke),
        },
        "scan": {
            "units": "equivalent-side",
            "rmax": args.rmax,
            "coarse_step": args.coarse_step,
            "fine_step": args.fine_step,
            "algorithm": "forward coarse scan to first active-identity change, then forward fine scan within the first bracket",
            "monotonicity_assumed": False,
            "image_boundary_numeric_guard": "independent float32 implementation of the prespecified 8,16,32,64,128 epsilon-times-max-image-dimension inward endpoint guard",
            "base_box_domain": "exclude focal GTs whose unshifted box is not fully inside the model input; never clip",
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256.lower(),
        "data": str(args.data),
        "selected_images_source": str(args.selected_images),
        "production_output_dir": str(args.production_output_dir),
        "production_protocol": production_manifest.get("protocol"),
        "source_sha256": {
            "validator": sha256(Path(__file__)),
            "reference_assigner": sha256(ROOT / "scripts" / "reference_o2o_assigner_numpy.py"),
            "selected_images": sha256(args.selected_images),
            "production_manifest": sha256(args.production_output_dir / "manifest.json"),
            "production_events": sha256(args.production_output_dir / "per_direction_boundary_events.csv"),
        },
        "outputs": [
            "selected_images.csv",
            "serialized_inputs/",
            "serialized_inputs_manifest.csv",
            "reference_trajectories.csv",
            "trajectory_comparison.csv",
            "mismatches.csv",
            "summary.json",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
