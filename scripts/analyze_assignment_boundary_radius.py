"""Read-only continuous assignment-boundary audit on frozen detector outputs.

For each focal 8--16 px or 16--32 px GT and each cardinal direction, this
script scans an equivalent-side displacement radius on a forward coarse grid.
The first observed base-identity ``same -> changed`` bracket is then refined by
another *forward* grid.  No monotonicity assumption or binary search is used.

The audit records the O2O loss-active boundary and its native first-divergence
type.  In parallel it records three deliberately distinct O2M rank estimands
(legacy unrestricted alignment rank, pre-Top-k rank, and assigned-positive
rank) plus the native positive-set Jaccard/retention trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o
from run_reviewer_killer_controls import SelectedDataset, image_stratum, size_bin

DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))
IDENTITY_KEYS = ("o2o", "o2m_legacy", "o2m_pre_topk", "o2m_assigned_positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmax", type=float, required=True, help="Maximum equivalent-side radius")
    parser.add_argument("--coarse-step", type=float, required=True, help="Forward coarse-grid radius step")
    parser.add_argument("--fine-step", type=float, required=True, help="Forward refinement-grid radius step")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use only the first selected image")
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
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def forward_grid(start: float, stop: float, step: float) -> list[float]:
    """Return increasing points in ``(start, stop]``, always including stop."""
    if stop <= start:
        return []
    points: list[float] = []
    index = 1
    while start + index * step < stop - 1e-12:
        points.append(start + index * step)
        index += 1
    points.append(stop)
    return points


def ranked_from_mask(state, gt: int, candidate_mask: torch.Tensor) -> int | None:
    ids = torch.where(candidate_mask)[0]
    if ids.numel() == 0:
        return None
    scores = state.align[0, gt, ids]
    return int(ids[scores.argmax()].item())


def o2m_metrics(state, gt: int) -> dict:
    scores = state.align[0, gt]
    legacy = ranked_from_mask(state, gt, scores > 0)
    pre_topk = ranked_from_mask(state, gt, state.pre[0, gt])
    assigned_mask = state.fg_mask[0] & state.target_gt_idx[0].eq(gt)
    assigned = ranked_from_mask(state, gt, assigned_mask)
    positive_set = frozenset(map(int, torch.where(assigned_mask)[0].tolist()))
    return {
        "o2m_legacy": legacy,
        "o2m_pre_topk": pre_topk,
        "o2m_assigned_positive": assigned,
        "o2m_positive_set": positive_set,
    }


def pair_metrics(base, shifted, gt: int, active: int | None, runner: int | None) -> dict:
    """Summarise fixed base A/B comparability without retaining replay states."""
    if active is None or runner is None or active == runner:
        return {
            "pair_q_active": None,
            "pair_q_runner": None,
            "pair_q_gap": None,
            "pair_eligibility_comparable": 0,
            "pair_pre_topk_comparable": 0,
            "pair_conflict_comparable": 0,
            "pair_comparability_reason": "base_pair_undefined",
        }
    eligibility = not pair_changed(base.eligible[0, gt], shifted.eligible[0, gt], active, runner)
    topk = not pair_changed(base.pre[0, gt], shifted.pre[0, gt], active, runner)
    conflict = not pair_changed(base.conflict[0, gt], shifted.conflict[0, gt], active, runner)
    if not eligibility:
        reason = "eligibility_boundary"
    elif not topk:
        reason = "topk_membership_transition"
    elif not conflict:
        reason = "conflict_reassignment"
    else:
        reason = None
    q_active = float(shifted.align[0, gt, active].item())
    q_runner = float(shifted.align[0, gt, runner].item())
    return {
        "pair_q_active": q_active,
        "pair_q_runner": q_runner,
        "pair_q_gap": q_active - q_runner,
        "pair_eligibility_comparable": int(eligibility),
        "pair_pre_topk_comparable": int(topk),
        "pair_conflict_comparable": int(conflict),
        "pair_comparability_reason": reason,
    }


def pair_changed(mask0: torch.Tensor, mask1: torch.Tensor, a: int | None, b: int | None) -> bool:
    candidates = {candidate for candidate in (a, b) if candidate is not None}
    return any(bool(mask0[candidate]) != bool(mask1[candidate]) for candidate in candidates)


def first_divergence(base, shifted, gt: int, active0: int | None, active1: int | None) -> str:
    eligibility = pair_changed(base.eligible[0, gt], shifted.eligible[0, gt], active0, active1)
    topk = pair_changed(base.pre[0, gt], shifted.pre[0, gt], active0, active1)
    conflict = pair_changed(base.conflict[0, gt], shifted.conflict[0, gt], active0, active1)
    reversal = False
    if active0 is not None and active1 is not None and active0 != active1:
        reversal = bool(
            base.align[0, gt, active0] > base.align[0, gt, active1]
            and shifted.align[0, gt, active0] < shifted.align[0, gt, active1]
        )
    if eligibility:
        return "eligibility_boundary"
    if topk:
        return "topk_membership_transition"
    if conflict:
        return "conflict_reassignment"
    if reversal:
        return "within_set_geometry_rank_reversal"
    if active1 is None:
        return "active_disappearance"
    return "compound_or_other"


def positive_set_overlap(base: frozenset[int], shifted: frozenset[int]) -> tuple[int, int, float | None, float | None, str]:
    intersection = len(base & shifted)
    union = len(base | shifted)
    if not base and not shifted:
        structural_state = "empty_empty"
    elif not base:
        structural_state = "empty_to_nonempty"
    elif not shifted:
        structural_state = "nonempty_to_empty"
    else:
        structural_state = "nonempty_nonempty"
    jaccard = None if union == 0 else intersection / union
    retention = None if len(base) == 0 else intersection / len(base)
    return intersection, union, jaccard, retention, structural_state


def legal_radius_limit(box: torch.Tensor, direction: str, image_w: float, image_h: float, scale: float) -> float:
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


def guarded_scan_limit(
    box: torch.Tensor,
    direction: str,
    unit_dx: int,
    unit_dy: int,
    image_w: float,
    image_h: float,
    scale: float,
    rmax: float,
) -> tuple[float, str, float]:
    """Return a float32-safe legal scan limit and the applied pixel guard.

    A limit obtained by dividing the exact remaining image margin by the
    equivalent-side scale can round outward when it is multiplied and added
    back to a float32 GT box.  Only that terminal censoring point is moved
    inward; all interior grid points and event definitions remain unchanged.
    """
    image_limit = legal_radius_limit(box, direction, image_w, image_h, scale)
    scan_limit = min(rmax, image_limit)
    reason = "rmax" if rmax <= image_limit + 1e-12 else "image_boundary"
    shifted = legal_shift(box, unit_dx * scan_limit * scale, unit_dy * scan_limit * scale, image_w, image_h)
    if shifted is not None or scan_limit <= 0.0:
        return scan_limit, reason, 0.0
    dtype_epsilon = float(torch.finfo(box.dtype).eps)
    for multiplier in (8.0, 16.0, 32.0, 64.0, 128.0):
        guard_pixels = multiplier * dtype_epsilon * max(image_w, image_h, 1.0)
        guarded = max(0.0, scan_limit - guard_pixels / scale)
        shifted = legal_shift(box, unit_dx * guarded * scale, unit_dy * guarded * scale, image_w, image_h)
        if shifted is not None:
            return guarded, reason, scan_limit * scale - guarded * scale
    raise RuntimeError(
        "unable to construct a float32-safe legal image-boundary scan limit: "
        f"direction={direction}, box={box.detach().cpu().tolist()}, image=({image_w},{image_h}), "
        f"scale={scale}, rmax={rmax}, image_limit={image_limit}, initial_scan_limit={scan_limit}"
    )


def event_result(
    key: str,
    base_identity: int | None,
    coarse_states: list[tuple[float, dict]],
    evaluate,
    phases: dict[float, set[str]],
    fine_step: float,
    scan_limit: float,
    censor_reason: str,
) -> dict:
    bracket: tuple[float, float] | None = None
    previous_radius = 0.0
    for radius, metrics in coarse_states:
        if metrics[key] != base_identity:
            bracket = (previous_radius, radius)
            break
        previous_radius = radius
    if bracket is None:
        return {
            f"{key}_event_observed": 0,
            f"{key}_radius": None,
            f"{key}_right_censored": 1,
            f"{key}_censor_radius": scan_limit,
            f"{key}_censor_reason": censor_reason,
            f"{key}_bracket_low": None,
            f"{key}_bracket_high": None,
        }

    low, high = bracket
    event_radius = high
    refined_low = low
    for radius in forward_grid(low, high, fine_step):
        phases[round(radius, 12)].add(f"fine_{key}")
        metrics = evaluate(radius)
        if metrics[key] != base_identity:
            event_radius = radius
            break
        refined_low = radius
    return {
        f"{key}_event_observed": 1,
        f"{key}_radius": event_radius,
        f"{key}_right_censored": 0,
        f"{key}_censor_radius": None,
        f"{key}_censor_reason": None,
        f"{key}_bracket_low": refined_low,
        f"{key}_bracket_high": event_radius,
    }


def rank_boundary_result(
    *,
    base_active: int | None,
    base_runner: int | None,
    base_q_gap: float | None,
    coarse_states: list[tuple[float, dict]],
    evaluate,
    phases: dict[float, set[str]],
    fine_step: float,
    scan_limit: float,
    censor_reason: str,
) -> dict:
    """First fixed-pair q crossing, with discrete loss of comparability competing."""
    empty = {
        "rho_R_event_observed": 0,
        "rho_R_radius": None,
        "rho_R_right_censored": 0,
        "rho_R_competing_censored": 0,
        "rho_R_censor_radius": None,
        "rho_R_censor_reason": None,
        "rho_R_bracket_low": None,
        "rho_R_bracket_high": None,
    }
    if base_active is None:
        return {**empty, "rho_R_censor_reason": "base_active_missing"}
    if base_runner is None:
        return {**empty, "rho_R_censor_reason": "base_runner_missing"}
    if base_q_gap == 0.0:
        return {**empty, "rho_R_censor_reason": "base_pair_tie"}
    if base_q_gap is None or base_q_gap < 0.0:
        return {**empty, "rho_R_censor_reason": "base_order_not_active_over_runner"}

    bracket: tuple[float, float] | None = None
    previous_radius = 0.0
    for radius, metrics in coarse_states:
        if metrics["pair_comparability_reason"] is not None or float(metrics["pair_q_gap"]) <= 0.0:
            bracket = (previous_radius, radius)
            break
        previous_radius = radius
    if bracket is None:
        return {
            **empty,
            "rho_R_right_censored": 1,
            "rho_R_censor_radius": scan_limit,
            "rho_R_censor_reason": censor_reason,
        }

    low, high = bracket
    refined_low = low
    terminal_radius = high
    terminal_metrics = evaluate(high)
    for radius in forward_grid(low, high, fine_step):
        phases[round(radius, 12)].add("fine_rho_R")
        metrics = evaluate(radius)
        if metrics["pair_comparability_reason"] is not None or float(metrics["pair_q_gap"]) <= 0.0:
            terminal_radius = radius
            terminal_metrics = metrics
            break
        refined_low = radius

    if terminal_metrics["pair_comparability_reason"] is not None:
        return {
            **empty,
            "rho_R_competing_censored": 1,
            "rho_R_censor_radius": terminal_radius,
            "rho_R_censor_reason": terminal_metrics["pair_comparability_reason"],
            "rho_R_bracket_low": refined_low,
            "rho_R_bracket_high": terminal_radius,
        }
    return {
        **empty,
        "rho_R_event_observed": 1,
        "rho_R_radius": terminal_radius,
        "rho_R_bracket_low": refined_low,
        "rho_R_bracket_high": terminal_radius,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data or selected-images manifest missing")
    if not (args.rmax > 0 and args.coarse_step > 0 and args.fine_step > 0):
        raise ValueError("rmax and grid steps must be positive")
    if args.fine_step > args.coarse_step:
        raise ValueError("fine-step must not exceed coarse-step")

    import pandas as pd
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("protocol is frozen for the native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if int(o2m.assigner.topk) != 10 or int(o2o.assigner.topk) != 7:
        raise RuntimeError("unexpected native Top-k contract")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)

    selected_rows = pd.read_csv(args.selected_images).to_dict("records")
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    if not selected_rows or required.difference(selected_rows[0]):
        raise ValueError("selected-images manifest is empty or incomplete")
    for row in selected_rows:
        index, stratum = int(row["dataset_index"]), str(row["stratum"])
        observed = Path(dataset.labels[index]["im_file"]).stem
        if observed != str(row["image_id"]) or index not in strata[stratum]:
            raise RuntimeError("selected-images manifest does not match the frozen dataset index")
    selected_rows.sort(key=lambda row: str(row["image_id"]))
    source_selected_count = len(selected_rows)
    if args.smoke:
        selected_rows = selected_rows[:1]
    selected_indices = [int(row["dataset_index"]) for row in selected_rows]
    row_by_image = {str(row["image_id"]): row for row in selected_rows}
    loader = build_dataloader(SelectedDataset(dataset, selected_indices), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "selected_images.csv", selected_rows)
    manifest = {
        "status": "running",
        "protocol": "continuous_assignment_boundary_radius_v1",
        "read_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256.lower(),
        "data": str(args.data),
        "selected_images_source": str(args.selected_images),
        "selected_images_source_sha256": sha256(args.selected_images),
        "source_selected_images": source_selected_count,
        "audited_selected_images": len(selected_rows),
        "radius_units": "equivalent-side; pixel displacement = radius * sqrt(focal GT area)",
        "rmax": args.rmax,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "directions": [item[0] for item in DIRECTIONS],
        "search_algorithm": "forward coarse grid to first observed base-identity same-to-changed bracket, then forward fine grid within that bracket",
        "monotonicity_assumed": False,
        "binary_search_used": False,
        "right_censoring": "at min(rmax, exact image-boundary legal radius) when no identity event is observed",
        "base_box_domain": "a continuous legal trajectory is defined only when the unshifted focal GT box is fully inside the model input; base-outside GTs are retained as explicit exclusion counts and are never clipped",
        "image_boundary_numeric_guard": "if the exact terminal censor point rounds outside under native float32 arithmetic, move only that endpoint inward by the smallest successful value in 8,16,32,64,128 times float32 epsilon times max image dimension; record the pixel guard per trajectory",
        "o2o_estimand": "native post-conflict loss-active identity",
        "rho_R_estimand": "first q_A <= q_B crossing for fixed base active A and base runner B while both candidates remain stage-comparable to base",
        "rho_R_competing_censor_order": "eligibility -> pre-Top-k -> conflict; a discrete comparability loss at an evaluated radius takes precedence over q crossing",
        "rho_R_tie_rule": "exact float32 q_A - q_B == 0 at base is undefined",
        "o2m_estimands": {
            "legacy": "highest q among all q>0 candidates",
            "pre_topk": "highest q among native pre-conflict Top-10 members",
            "assigned_positive": "highest q among native post-conflict positives assigned to focal GT",
            "positive_set": "native post-conflict positives assigned to focal GT",
        },
        "o2m_empty_set_contract": "empty/empty is a separate structural state with undefined Jaccard and retention; transitions to or from empty are retained explicitly",
        "script_sha256": sha256(Path(__file__)),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    event_rows: list[dict] = []
    focal_counts: Counter[str] = Counter()
    base_outside_counts: Counter[str] = Counter()
    audited_focal_counts: Counter[str] = Counter()
    curve_path = args.output_dir / "o2m_rank_set_radius_curve.csv"
    curve_writer = None
    curve_row_count = 0
    started = time.monotonic()
    with torch.no_grad(), curve_path.open("w", newline="", encoding="utf-8") as curve_stream:
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw dual-branch output")
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_o2o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            base_o2m = trace_o2o(o2m, raw["one2many"], labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            selection = row_by_image[image_id]

            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                focal_counts[bucket] += 1
                if legal_shift(box, 0.0, 0.0, float(image_w), float(image_h)) is None:
                    base_outside_counts[bucket] += 1
                    continue
                audited_focal_counts[bucket] += 1
                scale = math.sqrt(area)
                active0, runner0, margin0 = active_candidate_and_margin(base_o2o, 0, gt)
                base_m = o2m_metrics(base_o2m, gt)
                base_pair = pair_metrics(base_o2o, base_o2o, gt, active0, runner0)
                base_metrics = {"o2o": active0, **base_m, **base_pair}

                for direction, unit_dx, unit_dy in DIRECTIONS:
                    scan_limit, censor_reason, boundary_guard_pixels = guarded_scan_limit(
                        box, direction, unit_dx, unit_dy, float(image_w), float(image_h), scale, args.rmax
                    )
                    # Cache only CPU scalars and small Python sets.  Full O2OState/O2MState
                    # tensors are deliberately transient to keep the formal scan bounded.
                    cache: dict[float, dict] = {0.0: base_metrics}
                    phases: dict[float, set[str]] = defaultdict(set)
                    phases[0.0].add("base")

                    def trace_at(radius: float):
                        shifted_box = legal_shift(box, unit_dx * radius * scale, unit_dy * radius * scale, float(image_w), float(image_h))
                        if shifted_box is None:
                            raise RuntimeError("internal legal-radius calculation admitted an illegal shift")
                        altered = boxes.clone()
                        altered[0, gt] = shifted_box
                        shifted_o2o = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
                        shifted_o2m = trace_o2o(o2m, raw["one2many"], labels, altered, mask)
                        return shifted_o2o, shifted_o2m

                    def evaluate(radius: float) -> dict:
                        cache_key = round(float(radius), 12)
                        if cache_key in cache:
                            return cache[cache_key]
                        shifted_o2o, shifted_o2m = trace_at(radius)
                        active, _, _ = active_candidate_and_margin(shifted_o2o, 0, gt)
                        metrics = {
                            "o2o": active,
                            **o2m_metrics(shifted_o2m, gt),
                            **pair_metrics(base_o2o, shifted_o2o, gt, active0, runner0),
                        }
                        cache[cache_key] = metrics
                        del shifted_o2o, shifted_o2m
                        return metrics

                    coarse_states: list[tuple[float, dict]] = []
                    for radius in forward_grid(0.0, scan_limit, args.coarse_step):
                        phases[round(radius, 12)].add("coarse")
                        coarse_states.append((radius, evaluate(radius)))

                    row = {
                        "image_id": image_id,
                        "gt_id": gt,
                        "class_id": int(labels[0, gt, 0]),
                        "direction": direction,
                        "size_bin": bucket,
                        "width": width,
                        "height": height,
                        "area": area,
                        "stratum": str(selection["stratum"]),
                        "sampling_weight": float(selection["sampling_weight"]),
                        "scan_limit": scan_limit,
                        "scan_limit_reason": censor_reason,
                        "scan_limit_float32_guard_pixels": boundary_guard_pixels,
                        "o2o_base_active": active0,
                        "o2o_base_runner": runner0,
                        "o2o_base_margin": margin0,
                        "rho_R_base_q_active": base_pair["pair_q_active"],
                        "rho_R_base_q_runner": base_pair["pair_q_runner"],
                        "rho_R_base_q_gap": base_pair["pair_q_gap"],
                        "o2m_legacy_base": base_m["o2m_legacy"],
                        "o2m_pre_topk_base": base_m["o2m_pre_topk"],
                        "o2m_assigned_positive_base": base_m["o2m_assigned_positive"],
                        "o2m_positive_count_base": len(base_m["o2m_positive_set"]),
                    }
                    for identity_key in IDENTITY_KEYS:
                        row.update(event_result(identity_key, base_metrics[identity_key], coarse_states, evaluate, phases, args.fine_step, scan_limit, censor_reason))
                    row.update(rank_boundary_result(
                        base_active=active0,
                        base_runner=runner0,
                        base_q_gap=base_pair["pair_q_gap"],
                        coarse_states=coarse_states,
                        evaluate=evaluate,
                        phases=phases,
                        fine_step=args.fine_step,
                        scan_limit=scan_limit,
                        censor_reason=censor_reason,
                    ))

                    if int(row["o2o_event_observed"]):
                        event_state = evaluate(float(row["o2o_radius"]))
                        row["o2o_event_active"] = event_state["o2o"]
                        event_radius = float(row["o2o_radius"])
                        event_box = legal_shift(box, unit_dx * event_radius * scale, unit_dy * event_radius * scale, float(image_w), float(image_h))
                        if event_box is None:
                            raise RuntimeError("observed O2O event radius became illegal on exact replay")
                        event_boxes = boxes.clone()
                        event_boxes[0, gt] = event_box
                        event_o2o = trace_o2o(o2o, raw["one2one"], labels, event_boxes, mask)
                        row["o2o_first_divergence"] = first_divergence(base_o2o, event_o2o, gt, active0, event_state["o2o"])
                        del event_o2o, event_boxes
                    else:
                        row["o2o_event_active"] = None
                        row["o2o_first_divergence"] = "right_censored"
                    event_rows.append(row)

                    for radius_key in sorted(cache):
                        metrics = cache[radius_key]
                        intersection, union, jaccard, retention, structural_state = positive_set_overlap(base_m["o2m_positive_set"], metrics["o2m_positive_set"])
                        curve_record = {
                            "image_id": image_id,
                            "gt_id": gt,
                            "direction": direction,
                            "size_bin": bucket,
                            "stratum": str(selection["stratum"]),
                            "sampling_weight": float(selection["sampling_weight"]),
                            "radius": radius_key,
                            "shift_pixels": radius_key * scale,
                            "evaluation_phase": "|".join(sorted(phases[radius_key])),
                            "o2o_active": metrics["o2o"],
                            "o2m_legacy_rank": metrics["o2m_legacy"],
                            "o2m_pre_topk_rank": metrics["o2m_pre_topk"],
                            "o2m_assigned_positive_rank": metrics["o2m_assigned_positive"],
                            "o2m_positive_count_base": len(base_m["o2m_positive_set"]),
                            "o2m_positive_count": len(metrics["o2m_positive_set"]),
                            "o2m_positive_intersection_count": intersection,
                            "o2m_positive_union_count": union,
                            "o2m_positive_set_jaccard": jaccard,
                            "o2m_positive_set_retention": retention,
                            "o2m_positive_set_structural_state": structural_state,
                            "o2m_positive_set_exact_change": int(metrics["o2m_positive_set"] != base_m["o2m_positive_set"]),
                            "rho_R_pair_q_gap": metrics["pair_q_gap"],
                            "rho_R_pair_eligibility_comparable": metrics["pair_eligibility_comparable"],
                            "rho_R_pair_pre_topk_comparable": metrics["pair_pre_topk_comparable"],
                            "rho_R_pair_conflict_comparable": metrics["pair_conflict_comparable"],
                            "rho_R_pair_comparability_reason": metrics["pair_comparability_reason"],
                        }
                        if curve_writer is None:
                            curve_writer = csv.DictWriter(curve_stream, fieldnames=list(curve_record))
                            curve_writer.writeheader()
                        curve_writer.writerow(curve_record)
                        curve_row_count += 1

    write_rows(args.output_dir / "per_direction_boundary_events.csv", event_rows)
    event_counts = {key: sum(int(row[f"{key}_event_observed"]) for row in event_rows) for key in IDENTITY_KEYS}
    event_counts["rho_R"] = sum(int(row["rho_R_event_observed"]) for row in event_rows)
    first_divergence_counts = Counter(row["o2o_first_divergence"] for row in event_rows)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "selected_images": len(selected_rows),
        "focal_direction_rows": len(event_rows),
        "focal_gt_by_size_bin": dict(focal_counts),
        "excluded_base_box_outside_image_by_size_bin": dict(base_outside_counts),
        "audited_focal_gt_by_size_bin": dict(audited_focal_counts),
        "radius_curve_rows": curve_row_count,
        "event_counts": event_counts,
        "o2o_first_divergence_counts": dict(sorted(first_divergence_counts.items())),
        "elapsed_sec": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({"status": summary["status"], "elapsed_sec": summary["elapsed_sec"], "outputs": ["selected_images.csv", "per_direction_boundary_events.csv", "o2m_rank_set_radius_curve.csv", "summary.json"]})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
