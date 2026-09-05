"""Read-only continuous assignment-boundary audit on frozen detector outputs.

For each focal 8--16 px or 16--32 px GT and each cardinal direction, this
script scans an equivalent-side displacement radius.  The legacy mode uses a
forward coarse grid and refines the first observed base-identity
``same -> changed`` bracket.  The full-grid mode instead evaluates every point
on the fine grid over the complete legal radius domain, so no fine-grid state
is skipped.  Neither mode assumes monotonicity or uses binary search.

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
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o
from run_reviewer_killer_controls import SelectedDataset, image_stratum, size_bin

DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))
IDENTITY_KEYS = ("o2o", "o2m_legacy", "o2m_pre_topk", "o2m_assigned_positive")


@dataclass
class FocalRowContext:
    criterion: object
    base_state: object
    pred_scores: torch.Tensor
    pred_boxes: torch.Tensor
    anchor_points: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor


@dataclass
class FocalRowState:
    eligible: torch.Tensor
    pre: torch.Tensor
    conflict: torch.Tensor
    post: torch.Tensor
    align: torch.Tensor
    overlaps: torch.Tensor


@dataclass
class FocalReplayPlan:
    context: FocalRowContext
    gt: int
    nonfocal_pre_count: torch.Tensor
    nonfocal_overlap_max: torch.Tensor
    nonfocal_overlap_index: torch.Tensor


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
    parser.add_argument(
        "--search-mode",
        choices=("legacy", "full-grid"),
        default="legacy",
        help="legacy coarse-triggered refinement (default) or exhaustive full-domain fine-grid evaluation",
    )
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--replay-batch-size",
        type=int,
        default=16,
        help="Number of radii replayed together on the device (default: 16); use 1 for the original serial execution path",
    )
    parser.add_argument(
        "--replay-engine",
        choices=("native", "focal-row"),
        default="focal-row",
        help="full native replay or exact focal-row incremental replay (default)",
    )
    parser.add_argument(
        "--incremental-assert-native",
        action="store_true",
        help="also run full native replay and assert focal eligible/pre/conflict/post and derived metrics exactly",
    )
    parser.add_argument(
        "--max-focal-gt-per-image",
        type=int,
        default=0,
        help="Validation-only cap on audited focal GTs per image; 0 audits all focal GTs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from completed per-image shards in output-dir")
    parser.add_argument(
        "--max-images-this-run",
        type=int,
        default=0,
        help="Stop cleanly after this many new images; 0 processes every pending image",
    )
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


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def concatenate_csv(paths: list[Path], destination: Path) -> None:
    """Concatenate same-schema CSV shards without reserialising any value."""
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as output:
        wrote_header = False
        for path in paths:
            with path.open("rb") as source:
                header = source.readline()
                if not header:
                    continue
                if not wrote_header:
                    output.write(header)
                    wrote_header = True
                for line in source:
                    output.write(line)
    temporary.replace(destination)


def density_label(total_gt: int) -> str:
    if total_gt < 32:
        return "low_lt32"
    if total_gt < 128:
        return "medium_32_127"
    return "high_ge128"


def completed_shards(output_dir: Path, selected_rows: list[dict]) -> dict[str, dict]:
    shards_root = output_dir / "image_shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    for row in selected_rows:
        image_id = str(row["image_id"])
        final_dir = shards_root / image_id
        partial_dir = shards_root / f"{image_id}.partial"
        if not final_dir.exists() and (partial_dir / "complete.json").is_file():
            partial_dir.replace(final_dir)
        marker_path = final_dir / "complete.json"
        if not marker_path.is_file():
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("image_id") != image_id or int(marker.get("dataset_index", -1)) != int(row["dataset_index"]):
            raise RuntimeError(f"completed shard identity mismatch: {final_dir}")
        for filename, expected_hash in marker.get("sha256", {}).items():
            shard_file = final_dir / filename
            if not shard_file.is_file() or sha256(shard_file) != expected_hash:
                raise RuntimeError(f"completed shard hash mismatch: {shard_file}")
        completed[image_id] = marker
    return completed


def progress_payload(selected_rows: list[dict], completed: dict[str, dict], status: str) -> dict:
    markers = [completed[str(row["image_id"])] for row in selected_rows if str(row["image_id"]) in completed]
    processed = len(markers)
    elapsed = sum(float(marker["elapsed_sec"]) for marker in markers)
    density = Counter(str(marker["density"]) for marker in markers)
    remaining = len(selected_rows) - processed
    eta = None if processed == 0 else elapsed / processed * remaining
    return {
        "status": status,
        "processed_images": processed,
        "total_images": len(selected_rows),
        "completed_image_ids": [str(marker["image_id"]) for marker in markers],
        "focal_gt": sum(int(marker["focal_gt"]) for marker in markers),
        "trajectories": sum(int(marker["trajectories"]) for marker in markers),
        "curve_rows": sum(int(marker["curve_rows"]) for marker in markers),
        "elapsed_sec": elapsed,
        "density": dict(sorted(density.items())),
        "eta_sec": eta,
        "eta_method": "completed-image mean elapsed time times remaining images",
    }


def write_progress(output_dir: Path, payload: dict, append_history: bool = False) -> None:
    write_json_atomic(output_dir / "progress.json", payload)
    if append_history:
        with (output_dir / "progress_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


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


def expand_raw_batch(raw: dict, batch_size: int) -> dict:
    """Expand a one-image raw detector branch for batched assignment replay."""
    if batch_size == 1:
        return raw

    def expand(value):
        if isinstance(value, torch.Tensor):
            if value.shape[0] != 1:
                raise ValueError("radius replay expects raw predictions for exactly one source image")
            return value.expand(batch_size, *value.shape[1:])
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, tuple):
            return tuple(expand(item) for item in value)
        raise TypeError(f"unsupported raw prediction value: {type(value)!r}")

    return {key: expand(value) for key, value in raw.items()}


def build_focal_row_context(criterion, raw: dict, base_state, labels: torch.Tensor, mask: torch.Tensor) -> FocalRowContext:
    """Freeze all prediction-side tensors and non-focal GT rows for incremental replay."""
    if criterion.assigner.use_swd_assigner:
        raise RuntimeError("focal-row replay is not valid for the globally normalized SWD assigner")
    pred_scores = raw["scores"].permute(0, 2, 1).contiguous().detach().sigmoid()
    pred_boxes = (base_state.decoded.detach() * base_state.stride).type(base_state.overlaps.dtype)
    return FocalRowContext(
        criterion=criterion,
        base_state=base_state,
        pred_scores=pred_scores,
        pred_boxes=pred_boxes,
        anchor_points=base_state.anchors * base_state.stride,
        labels=labels,
        mask=mask,
    )


def max_except_row(values: torch.Tensor, excluded_row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Columnwise maximum excluding one row, preserving native first-index tie-breaking."""
    row_count, column_count = values.shape
    if row_count == 1:
        return (
            torch.full((column_count,), -torch.inf, dtype=values.dtype, device=values.device),
            torch.full((column_count,), -1, dtype=torch.long, device=values.device),
        )
    left_values = values[:excluded_row]
    right_values = values[excluded_row + 1 :]
    if left_values.shape[0] == 0:
        maximum, index = right_values.max(0)
        return maximum, index + excluded_row + 1
    if right_values.shape[0] == 0:
        return left_values.max(0)
    left_max, left_index = left_values.max(0)
    right_max, right_index = right_values.max(0)
    take_left = left_max >= right_max
    maximum = torch.where(take_left, left_max, right_max)
    index = torch.where(take_left, left_index, right_index + excluded_row + 1)
    return maximum, index


def prepare_focal_replay(context: FocalRowContext, gt: int) -> FocalReplayPlan:
    """Precompute all non-focal quantities reused by every shift of one GT."""
    base_pre = context.base_state.pre[0]
    total_pre_count = base_pre.sum(0)
    nonfocal_max, nonfocal_index = max_except_row(context.base_state.overlaps[0], gt)
    return FocalReplayPlan(
        context=context,
        gt=gt,
        nonfocal_pre_count=total_pre_count - base_pre[gt].to(total_pre_count.dtype),
        nonfocal_overlap_max=nonfocal_max,
        nonfocal_overlap_index=nonfocal_index,
    )


def replay_focal_rows(plan: FocalReplayPlan, shifted_boxes: torch.Tensor) -> FocalRowState:
    """Recompute one GT row and resolve its native cross-GT assignment exactly."""
    context, gt = plan.context, plan.gt
    batch_size = shifted_boxes.shape[0]
    assigner = context.criterion.assigner
    assigner.bs, assigner.n_max_boxes = batch_size, 1
    focal_boxes = shifted_boxes.unsqueeze(1)
    focal_labels = context.labels[:, gt : gt + 1].expand(batch_size, -1, -1)
    focal_mask = context.mask[:, gt : gt + 1].expand(batch_size, -1, -1)
    pred_scores = context.pred_scores.expand(batch_size, -1, -1)
    pred_boxes = context.pred_boxes.expand(batch_size, -1, -1)

    eligible = assigner.select_candidates_in_gts(context.anchor_points, focal_boxes, focal_mask).bool()
    align, overlaps = assigner.get_box_metrics(
        pred_scores,
        pred_boxes,
        focal_labels,
        focal_boxes,
        eligible * focal_mask,
    )
    topk_mask = focal_mask.expand(-1, -1, assigner.topk).bool()
    topk = assigner.select_topk_candidates(align, topk_mask=topk_mask)
    pre = (topk * eligible * focal_mask).bool()

    multi = plan.nonfocal_pre_count.unsqueeze(0) + pre[:, 0].to(plan.nonfocal_pre_count.dtype) > 1
    focal_overlap = overlaps[:, 0]
    focal_wins = (focal_overlap > plan.nonfocal_overlap_max) | (
        focal_overlap.eq(plan.nonfocal_overlap_max) & (gt < plan.nonfocal_overlap_index)
    )
    conflict = torch.where(multi, focal_wins, pre[:, 0]).unsqueeze(1).bool()

    if assigner.topk2 != assigner.topk:
        selected = torch.topk(align * conflict, assigner.topk2, dim=-1, largest=True).indices
        topk2 = torch.zeros_like(conflict)
        topk2.scatter_(-1, selected, True)
        post = conflict & topk2
    else:
        post = conflict
    return FocalRowState(eligible, pre, conflict, post, align, overlaps)


def focal_active(row: FocalRowState, batch_index: int) -> int | None:
    ids = torch.where(row.post[batch_index, 0])[0]
    return int(ids.item()) if ids.numel() == 1 else None


def o2m_metrics_focal(row: FocalRowState, batch_index: int) -> dict:
    scores = row.align[batch_index, 0]
    legacy_ids = torch.where(scores > 0)[0]
    legacy = int(legacy_ids[scores[legacy_ids].argmax()].item()) if legacy_ids.numel() else None
    pre_ids = torch.where(row.pre[batch_index, 0])[0]
    pre_topk = int(pre_ids[scores[pre_ids].argmax()].item()) if pre_ids.numel() else None
    assigned_ids = torch.where(row.post[batch_index, 0])[0]
    assigned = int(assigned_ids[scores[assigned_ids].argmax()].item()) if assigned_ids.numel() else None
    positive_set = frozenset(map(int, assigned_ids.tolist()))
    return {
        "o2m_legacy": legacy,
        "o2m_pre_topk": pre_topk,
        "o2m_assigned_positive": assigned,
        "o2m_positive_set": positive_set,
    }


def pair_metrics_focal(base, shifted: FocalRowState, gt: int, active: int | None, runner: int | None, batch_index: int) -> dict:
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
    eligibility = not pair_changed(base.eligible[0, gt], shifted.eligible[batch_index, 0], active, runner)
    topk = not pair_changed(base.pre[0, gt], shifted.pre[batch_index, 0], active, runner)
    conflict = not pair_changed(base.conflict[0, gt], shifted.conflict[batch_index, 0], active, runner)
    if not eligibility:
        reason = "eligibility_boundary"
    elif not topk:
        reason = "topk_membership_transition"
    elif not conflict:
        reason = "conflict_reassignment"
    else:
        reason = None
    q_active = float(shifted.align[batch_index, 0, active].item())
    q_runner = float(shifted.align[batch_index, 0, runner].item())
    return {
        "pair_q_active": q_active,
        "pair_q_runner": q_runner,
        "pair_q_gap": q_active - q_runner,
        "pair_eligibility_comparable": int(eligibility),
        "pair_pre_topk_comparable": int(topk),
        "pair_conflict_comparable": int(conflict),
        "pair_comparability_reason": reason,
    }


def first_divergence_focal(base, shifted: FocalRowState, gt: int, active0: int | None, active1: int | None) -> str:
    eligibility = pair_changed(base.eligible[0, gt], shifted.eligible[0, 0], active0, active1)
    topk = pair_changed(base.pre[0, gt], shifted.pre[0, 0], active0, active1)
    conflict = pair_changed(base.conflict[0, gt], shifted.conflict[0, 0], active0, active1)
    reversal = False
    if active0 is not None and active1 is not None and active0 != active1:
        reversal = bool(
            base.align[0, gt, active0] > base.align[0, gt, active1]
            and shifted.align[0, 0, active0] < shifted.align[0, 0, active1]
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


def assert_focal_state_exact(name: str, incremental: FocalRowState, native, gt: int) -> None:
    """Fail on the first focal-row state difference against full native replay."""
    for field in ("eligible", "pre", "conflict", "post", "align", "overlaps"):
        observed = getattr(incremental, field)[:, 0]
        expected = getattr(native, field)[:, gt]
        if not torch.equal(observed, expected):
            unequal = int(observed.ne(expected).sum().item())
            max_abs = None
            if observed.dtype.is_floating_point:
                max_abs = float((observed - expected).abs().max().item())
            raise AssertionError(
                f"incremental {name} {field} mismatch for gt={gt}: unequal={unequal}, max_abs={max_abs}"
            )
    for batch_index in range(incremental.post.shape[0]):
        observed_active = focal_active(incremental, batch_index)
        expected_active, _, _ = active_candidate_and_margin(native, batch_index, gt)
        if observed_active != expected_active:
            raise AssertionError(
                f"incremental {name} active mismatch for gt={gt}, batch={batch_index}: "
                f"observed={observed_active}, expected={expected_active}"
            )


def ranked_from_mask(state, batch_index: int, gt: int, candidate_mask: torch.Tensor) -> int | None:
    ids = torch.where(candidate_mask)[0]
    if ids.numel() == 0:
        return None
    scores = state.align[batch_index, gt, ids]
    return int(ids[scores.argmax()].item())


def o2m_metrics(state, gt: int, batch_index: int = 0) -> dict:
    scores = state.align[batch_index, gt]
    legacy = ranked_from_mask(state, batch_index, gt, scores > 0)
    pre_topk = ranked_from_mask(state, batch_index, gt, state.pre[batch_index, gt])
    assigned_mask = state.fg_mask[batch_index] & state.target_gt_idx[batch_index].eq(gt)
    assigned = ranked_from_mask(state, batch_index, gt, assigned_mask)
    positive_set = frozenset(map(int, torch.where(assigned_mask)[0].tolist()))
    return {
        "o2m_legacy": legacy,
        "o2m_pre_topk": pre_topk,
        "o2m_assigned_positive": assigned,
        "o2m_positive_set": positive_set,
    }


def pair_metrics(base, shifted, gt: int, active: int | None, runner: int | None, shifted_batch: int = 0) -> dict:
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
    eligibility = not pair_changed(base.eligible[0, gt], shifted.eligible[shifted_batch, gt], active, runner)
    topk = not pair_changed(base.pre[0, gt], shifted.pre[shifted_batch, gt], active, runner)
    conflict = not pair_changed(base.conflict[0, gt], shifted.conflict[shifted_batch, gt], active, runner)
    if not eligibility:
        reason = "eligibility_boundary"
    elif not topk:
        reason = "topk_membership_transition"
    elif not conflict:
        reason = "conflict_reassignment"
    else:
        reason = None
    q_active = float(shifted.align[shifted_batch, gt, active].item())
    q_runner = float(shifted.align[shifted_batch, gt, runner].item())
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


def first_divergence(base, shifted, gt: int, active0: int | None, active1: int | None, shifted_batch: int = 0) -> str:
    eligibility = pair_changed(base.eligible[0, gt], shifted.eligible[shifted_batch, gt], active0, active1)
    topk = pair_changed(base.pre[0, gt], shifted.pre[shifted_batch, gt], active0, active1)
    conflict = pair_changed(base.conflict[0, gt], shifted.conflict[shifted_batch, gt], active0, active1)
    reversal = False
    if active0 is not None and active1 is not None and active0 != active1:
        reversal = bool(
            base.align[0, gt, active0] > base.align[0, gt, active1]
            and shifted.align[shifted_batch, gt, active0] < shifted.align[shifted_batch, gt, active1]
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
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    if args.resume and not args.output_dir.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {args.output_dir}")
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data or selected-images manifest missing")
    if not (args.rmax > 0 and args.coarse_step > 0 and args.fine_step > 0):
        raise ValueError("rmax and grid steps must be positive")
    if args.fine_step > args.coarse_step:
        raise ValueError("fine-step must not exceed coarse-step")
    if args.replay_batch_size < 1:
        raise ValueError("replay-batch-size must be at least 1")
    if args.max_focal_gt_per_image < 0:
        raise ValueError("max-focal-gt-per-image must be nonnegative")
    if args.max_images_this_run < 0:
        raise ValueError("max-images-this-run must be nonnegative")
    if args.incremental_assert_native and args.replay_engine != "focal-row":
        raise ValueError("incremental-assert-native requires replay-engine=focal-row")

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
    row_by_image = {str(row["image_id"]): row for row in selected_rows}

    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    full_grid = args.search_mode == "full-grid"
    requested_manifest = {
        "status": "running",
        "protocol": "continuous_assignment_boundary_radius_full_grid_v2" if full_grid else "continuous_assignment_boundary_radius_v1",
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
        "search_mode": args.search_mode,
        "replay_batch_size": args.replay_batch_size,
        "replay_engine": args.replay_engine,
        "incremental_assert_native": bool(args.incremental_assert_native),
        "max_focal_gt_per_image": args.max_focal_gt_per_image,
        "directions": [item[0] for item in DIRECTIONS],
        "search_algorithm": (
            "evaluate every fine-grid point from zero through the complete legal scan domain; report the first sampled base-identity change"
            if full_grid
            else "forward coarse grid to first observed base-identity same-to-changed bracket, then forward fine grid within that bracket"
        ),
        "grid_completeness": (
            "all legal fine-grid points are evaluated; no event visible at the declared fine-step resolution can be skipped, but no claim is made about intervals narrower than the fine step"
            if full_grid
            else "coarse-triggered refinement can skip transient intervals that contain no coarse-grid point"
        ),
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
        "sharding": "one atomic directory per image; complete.json written last; final CSVs concatenated in selected-image order",
    }
    manifest_path = args.output_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file() or not (args.output_dir / "selected_images.csv").is_file():
            raise RuntimeError("resume requires manifest.json and selected_images.csv")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "protocol", "checkpoint_sha256", "data", "selected_images_source_sha256",
            "audited_selected_images", "rmax", "coarse_step", "fine_step", "search_mode",
            "replay_batch_size", "replay_engine", "incremental_assert_native",
            "max_focal_gt_per_image", "script_sha256", "smoke",
        )
        mismatches = [key for key in immutable_keys if manifest.get(key) != requested_manifest.get(key)]
        if mismatches:
            raise RuntimeError(f"resume contract mismatch: {mismatches}")
        manifest["status"] = "running"
        manifest["resume_invocations"] = int(manifest.get("resume_invocations", 0)) + 1
    else:
        manifest = requested_manifest
        manifest["resume_invocations"] = 0
        write_rows(args.output_dir / "selected_images.csv", selected_rows)
    write_json_atomic(manifest_path, manifest)

    completed = completed_shards(args.output_dir, selected_rows)
    write_progress(args.output_dir, progress_payload(selected_rows, completed, "running"))
    pending_rows = [row for row in selected_rows if str(row["image_id"]) not in completed]
    if args.max_images_this_run:
        pending_rows = pending_rows[: args.max_images_this_run]
    selected_indices = [int(row["dataset_index"]) for row in pending_rows]
    loader = (
        build_dataloader(
            SelectedDataset(dataset, selected_indices), batch=1, workers=args.workers,
            shuffle=False, rank=-1, drop_last=False, pin_memory=False,
        )
        if selected_indices
        else []
    )

    focal_counts: Counter[str] = Counter()
    base_outside_counts: Counter[str] = Counter()
    audited_focal_counts: Counter[str] = Counter()
    curve_row_count = 0
    incremental_validation_scenarios = 0
    newly_completed_images = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            image_started = time.monotonic()
            image_id = Path(batch["im_file"][0]).stem
            selection = row_by_image[image_id]
            partial_dir = args.output_dir / "image_shards" / f"{image_id}.partial"
            partial_dir.mkdir(parents=True, exist_ok=True)
            image_event_rows: list[dict] = []
            curve_writer = None
            image_curve_path = partial_dir / "o2m_rank_set_radius_curve.csv"
            curve_stream = image_curve_path.open("w", newline="", encoding="utf-8")
            image_curve_start = curve_row_count
            image_validation_start = incremental_validation_scenarios
            focal_counts_start = focal_counts.copy()
            base_outside_start = base_outside_counts.copy()
            audited_focal_start = audited_focal_counts.copy()
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
            if args.replay_engine == "focal-row":
                o2o_focal_context = build_focal_row_context(o2o, raw["one2one"], base_o2o, labels, mask)
                o2m_focal_context = build_focal_row_context(o2m, raw["one2many"], base_o2m, labels, mask)
            else:
                o2o_focal_context = None
                o2m_focal_context = None
            processed_focal_for_image = 0
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                if args.max_focal_gt_per_image and processed_focal_for_image >= args.max_focal_gt_per_image:
                    break
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
                processed_focal_for_image += 1
                scale = math.sqrt(area)
                active0, runner0, margin0 = active_candidate_and_margin(base_o2o, 0, gt)
                base_m = o2m_metrics(base_o2m, gt)
                base_pair = pair_metrics(base_o2o, base_o2o, gt, active0, runner0)
                base_metrics = {"o2o": active0, **base_m, **base_pair}
                if o2o_focal_context is not None and o2m_focal_context is not None:
                    o2o_focal_plan = prepare_focal_replay(o2o_focal_context, gt)
                    o2m_focal_plan = prepare_focal_replay(o2m_focal_context, gt)
                else:
                    o2o_focal_plan = None
                    o2m_focal_plan = None

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

                    def evaluate_radius_batch(radii: list[float]) -> list[dict]:
                        nonlocal incremental_validation_scenarios
                        if args.replay_engine == "native" and len(radii) == 1:
                            shifted_o2o, shifted_o2m = trace_at(radii[0])
                            active, _, _ = active_candidate_and_margin(shifted_o2o, 0, gt)
                            metrics = [{
                                "o2o": active,
                                **o2m_metrics(shifted_o2m, gt),
                                **pair_metrics(base_o2o, shifted_o2o, gt, active0, runner0),
                            }]
                            del shifted_o2o, shifted_o2m
                            return metrics

                        altered = boxes.expand(len(radii), *boxes.shape[1:]).clone()
                        shifted_focal_boxes: list[torch.Tensor] = []
                        for batch_index, radius in enumerate(radii):
                            shifted_box = legal_shift(
                                box,
                                unit_dx * radius * scale,
                                unit_dy * radius * scale,
                                float(image_w),
                                float(image_h),
                            )
                            if shifted_box is None:
                                raise RuntimeError("internal legal-radius calculation admitted an illegal batched shift")
                            altered[batch_index, gt] = shifted_box
                            shifted_focal_boxes.append(shifted_box)
                        focal_box_batch = torch.stack(shifted_focal_boxes)
                        batch_labels = labels.expand(len(radii), *labels.shape[1:])
                        batch_mask = mask.expand(len(radii), *mask.shape[1:])

                        if args.replay_engine == "focal-row":
                            if o2o_focal_plan is None or o2m_focal_plan is None:
                                raise RuntimeError("missing focal-row replay context")
                            focal_o2o = replay_focal_rows(o2o_focal_plan, focal_box_batch)
                            focal_o2m = replay_focal_rows(o2m_focal_plan, focal_box_batch)
                            metrics = []
                            for batch_index in range(len(radii)):
                                active = focal_active(focal_o2o, batch_index)
                                metrics.append({
                                    "o2o": active,
                                    **o2m_metrics_focal(focal_o2m, batch_index),
                                    **pair_metrics_focal(base_o2o, focal_o2o, gt, active0, runner0, batch_index),
                                })

                            if args.incremental_assert_native:
                                native_o2o = trace_o2o(
                                    o2o,
                                    expand_raw_batch(raw["one2one"], len(radii)),
                                    batch_labels,
                                    altered,
                                    batch_mask,
                                )
                                assert_focal_state_exact("o2o", focal_o2o, native_o2o, gt)
                                native_pair_metrics: list[dict] = []
                                native_actives: list[int | None] = []
                                for batch_index in range(len(radii)):
                                    native_active, _, _ = active_candidate_and_margin(native_o2o, batch_index, gt)
                                    native_actives.append(native_active)
                                    native_pair_metrics.append(
                                        pair_metrics(base_o2o, native_o2o, gt, active0, runner0, batch_index)
                                    )
                                del native_o2o

                                native_o2m = trace_o2o(
                                    o2m,
                                    expand_raw_batch(raw["one2many"], len(radii)),
                                    batch_labels,
                                    altered,
                                    batch_mask,
                                )
                                assert_focal_state_exact("o2m", focal_o2m, native_o2m, gt)
                                for batch_index in range(len(radii)):
                                    expected = {
                                        "o2o": native_actives[batch_index],
                                        **o2m_metrics(native_o2m, gt, batch_index),
                                        **native_pair_metrics[batch_index],
                                    }
                                    if metrics[batch_index] != expected:
                                        raise AssertionError(
                                            f"incremental derived-metric mismatch for gt={gt}, batch={batch_index}"
                                        )
                                del native_o2m
                                incremental_validation_scenarios += len(radii)
                            del focal_o2o, focal_o2m, altered, focal_box_batch
                            return metrics

                        shifted_o2o = trace_o2o(
                            o2o,
                            expand_raw_batch(raw["one2one"], len(radii)),
                            batch_labels,
                            altered,
                            batch_mask,
                        )
                        metrics: list[dict] = []
                        for batch_index in range(len(radii)):
                            active, _, _ = active_candidate_and_margin(shifted_o2o, batch_index, gt)
                            metrics.append({
                                "o2o": active,
                                **pair_metrics(base_o2o, shifted_o2o, gt, active0, runner0, batch_index),
                            })
                        del shifted_o2o

                        shifted_o2m = trace_o2o(
                            o2m,
                            expand_raw_batch(raw["one2many"], len(radii)),
                            batch_labels,
                            altered,
                            batch_mask,
                        )
                        for batch_index in range(len(radii)):
                            metrics[batch_index].update(o2m_metrics(shifted_o2m, gt, batch_index))
                        del shifted_o2m, altered
                        return metrics

                    def evaluate_many(radii: list[float]) -> None:
                        pending: list[tuple[float, float]] = []
                        pending_keys: set[float] = set()
                        for radius in radii:
                            cache_key = round(float(radius), 12)
                            if cache_key not in cache and cache_key not in pending_keys:
                                pending.append((cache_key, radius))
                                pending_keys.add(cache_key)
                        for offset in range(0, len(pending), args.replay_batch_size):
                            chunk = pending[offset : offset + args.replay_batch_size]
                            chunk_metrics = evaluate_radius_batch([radius for _, radius in chunk])
                            for (cache_key, _), metrics in zip(chunk, chunk_metrics, strict=True):
                                cache[cache_key] = metrics

                    def evaluate(radius: float) -> dict:
                        cache_key = round(float(radius), 12)
                        evaluate_many([radius])
                        return cache[cache_key]

                    scan_states: list[tuple[float, dict]] = []
                    scan_step = args.fine_step if full_grid else args.coarse_step
                    scan_phase = "full_grid" if full_grid else "coarse"
                    scan_radii = forward_grid(0.0, scan_limit, scan_step)
                    evaluate_many(scan_radii)
                    for radius in scan_radii:
                        phases[round(radius, 12)].add(scan_phase)
                        scan_states.append((radius, evaluate(radius)))

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
                        row.update(event_result(identity_key, base_metrics[identity_key], scan_states, evaluate, phases, args.fine_step, scan_limit, censor_reason))
                    row.update(rank_boundary_result(
                        base_active=active0,
                        base_runner=runner0,
                        base_q_gap=base_pair["pair_q_gap"],
                        coarse_states=scan_states,
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
                        if args.replay_engine == "focal-row":
                            if o2o_focal_plan is None:
                                raise RuntimeError("missing focal-row replay context")
                            event_focal = replay_focal_rows(o2o_focal_plan, event_box.unsqueeze(0))
                            row["o2o_first_divergence"] = first_divergence_focal(
                                base_o2o, event_focal, gt, active0, event_state["o2o"]
                            )
                            del event_focal
                        else:
                            event_boxes = boxes.clone()
                            event_boxes[0, gt] = event_box
                            event_o2o = trace_o2o(o2o, raw["one2one"], labels, event_boxes, mask)
                            row["o2o_first_divergence"] = first_divergence(
                                base_o2o, event_o2o, gt, active0, event_state["o2o"]
                            )
                            del event_o2o, event_boxes
                    else:
                        row["o2o_event_active"] = None
                        row["o2o_first_divergence"] = "right_censored"
                    image_event_rows.append(row)

                    curve_records: list[dict] = []
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
                        curve_records.append(curve_record)
                    curve_writer.writerows(curve_records)
                    curve_row_count += len(curve_records)

            curve_stream.flush()
            os.fsync(curve_stream.fileno())
            curve_stream.close()
            image_events_path = partial_dir / "per_direction_boundary_events.csv"
            write_rows(image_events_path, image_event_rows)
            focal_delta = focal_counts - focal_counts_start
            outside_delta = base_outside_counts - base_outside_start
            audited_delta = audited_focal_counts - audited_focal_start
            image_summary = {
                "status": "complete",
                "image_id": image_id,
                "dataset_index": int(selection["dataset_index"]),
                "total_gt": int(mask[0, :, 0].sum().item()),
                "density": density_label(int(mask[0, :, 0].sum().item())),
                "focal_gt": sum(audited_delta.values()),
                "focal_gt_by_size_bin": dict(focal_delta),
                "excluded_base_box_outside_image_by_size_bin": dict(outside_delta),
                "audited_focal_gt_by_size_bin": dict(audited_delta),
                "trajectories": len(image_event_rows),
                "curve_rows": curve_row_count - image_curve_start,
                "incremental_native_validation_scenarios": (
                    incremental_validation_scenarios - image_validation_start
                ),
                "elapsed_sec": time.monotonic() - image_started,
            }
            image_summary_path = partial_dir / "summary.json"
            write_json_atomic(image_summary_path, image_summary)
            marker = {
                **image_summary,
                "sha256": {
                    "per_direction_boundary_events.csv": sha256(image_events_path),
                    "o2m_rank_set_radius_curve.csv": sha256(image_curve_path),
                    "summary.json": sha256(image_summary_path),
                },
            }
            write_json_atomic(partial_dir / "complete.json", marker)
            final_dir = args.output_dir / "image_shards" / image_id
            partial_dir.replace(final_dir)
            completed[image_id] = marker
            newly_completed_images += 1
            progress_status = "complete" if len(completed) == len(selected_rows) else "running"
            write_progress(
                args.output_dir,
                progress_payload(selected_rows, completed, progress_status),
                append_history=True,
            )

    completed = completed_shards(args.output_dir, selected_rows)
    ordered_shard_dirs = [
        args.output_dir / "image_shards" / str(row["image_id"])
        for row in selected_rows
        if str(row["image_id"]) in completed
    ]
    concatenate_csv(
        [path / "per_direction_boundary_events.csv" for path in ordered_shard_dirs],
        args.output_dir / "per_direction_boundary_events.csv",
    )
    concatenate_csv(
        [path / "o2m_rank_set_radius_curve.csv" for path in ordered_shard_dirs],
        args.output_dir / "o2m_rank_set_radius_curve.csv",
    )
    all_event_rows: list[dict] = []
    for shard_dir in ordered_shard_dirs:
        with (shard_dir / "per_direction_boundary_events.csv").open(newline="", encoding="utf-8") as stream:
            all_event_rows.extend(csv.DictReader(stream))
    event_counts = {key: sum(int(row[f"{key}_event_observed"]) for row in all_event_rows) for key in IDENTITY_KEYS}
    event_counts["rho_R"] = sum(int(row["rho_R_event_observed"]) for row in all_event_rows)
    first_divergence_counts = Counter(row["o2o_first_divergence"] for row in all_event_rows)
    aggregate_focal: Counter[str] = Counter()
    aggregate_outside: Counter[str] = Counter()
    aggregate_audited: Counter[str] = Counter()
    for marker in completed.values():
        aggregate_focal.update(marker.get("focal_gt_by_size_bin", {}))
        aggregate_outside.update(marker.get("excluded_base_box_outside_image_by_size_bin", {}))
        aggregate_audited.update(marker.get("audited_focal_gt_by_size_bin", {}))
    complete_run = len(completed) == len(selected_rows)
    terminal_status = ("smoke_complete" if args.smoke else "complete") if complete_run else "partial"
    progress = progress_payload(selected_rows, completed, terminal_status)
    write_progress(args.output_dir, progress)
    summary = {
        "status": terminal_status,
        "selected_images": len(selected_rows),
        "processed_images": len(completed),
        "newly_completed_images": newly_completed_images,
        "focal_direction_rows": len(all_event_rows),
        "focal_gt_by_size_bin": dict(aggregate_focal),
        "excluded_base_box_outside_image_by_size_bin": dict(aggregate_outside),
        "audited_focal_gt_by_size_bin": dict(aggregate_audited),
        "radius_curve_rows": progress["curve_rows"],
        "replay_engine": args.replay_engine,
        "incremental_native_validation_scenarios": sum(
            int(marker.get("incremental_native_validation_scenarios", 0)) for marker in completed.values()
        ),
        "event_counts": event_counts,
        "o2o_first_divergence_counts": dict(sorted(first_divergence_counts.items())),
        "elapsed_sec": progress["elapsed_sec"],
        "invocation_elapsed_sec": time.monotonic() - started,
        "eta_sec": progress["eta_sec"],
    }
    if device.type == "cuda":
        summary["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        summary["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    write_json_atomic(args.output_dir / "summary.json", summary)
    manifest.update({
        "status": summary["status"],
        "elapsed_sec": summary["elapsed_sec"],
        "outputs": [
            "selected_images.csv", "per_direction_boundary_events.csv",
            "o2m_rank_set_radius_curve.csv", "summary.json", "progress.json",
            "progress_history.jsonl", "image_shards/",
        ],
    })
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
