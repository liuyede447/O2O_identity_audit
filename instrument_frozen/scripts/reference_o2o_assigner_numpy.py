"""Independent NumPy reference for the frozen YOLO26 TAL assignment contract.

This module intentionally imports neither PyTorch, Ultralytics, nor any
production audit helper.  It is used only for differential validation and
counterfactual instrument tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceState:
    eligible: np.ndarray
    align: np.ndarray
    overlaps: np.ndarray
    topk: np.ndarray
    conflict: np.ndarray
    post: np.ndarray
    target_gt_idx: np.ndarray
    fg_mask: np.ndarray


def ciou_xyxy(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Vectorized CIoU matching the frozen horizontal-box implementation."""
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    b1_x1, b1_y1, b1_x2, b1_y2 = np.moveaxis(gt, -1, 0)
    b2_x1, b2_y1, b2_x2, b2_y2 = np.moveaxis(pred, -1, 0)
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1 + eps
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1 + eps
    inter = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0.0)
    inter *= np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0.0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)
    ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0
    v = np.float32(4.0 / np.pi**2) * (np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2
    alpha = v / (v - iou + (1.0 + eps))
    return iou - (rho2 / c2 + v * alpha)


def stal_eligibility(
    anchors: np.ndarray,
    gt_boxes: np.ndarray,
    valid_gt: np.ndarray,
    minimum_stride: float = 8.0,
    reference_extent: float = 16.0,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return GT-by-candidate STAL eligibility with strict interior bounds."""
    anchors = np.asarray(anchors, dtype=np.float32)
    boxes = np.asarray(gt_boxes, dtype=np.float32).copy()
    valid_gt = np.asarray(valid_gt, dtype=bool).reshape(-1)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    wh = boxes[:, 2:] - boxes[:, :2]
    wh = np.where(wh < minimum_stride, reference_extent, wh)
    boxes[:, :2] = centres - wh / 2.0
    boxes[:, 2:] = centres + wh / 2.0
    left_top = anchors[None, :, :] - boxes[:, None, :2]
    right_bottom = boxes[:, None, 2:] - anchors[None, :, :]
    return (np.min(np.concatenate([left_top, right_bottom], axis=-1), axis=-1) > eps) & valid_gt[:, None]


def _topk_mask(metrics: np.ndarray, topk: int, valid_gt: np.ndarray) -> np.ndarray:
    n_gt, n_candidates = metrics.shape
    if n_candidates < topk:
        raise ValueError("candidate count must be at least topk")
    output = np.zeros_like(metrics, dtype=bool)
    for gt in range(n_gt):
        if not valid_gt[gt]:
            continue
        order = np.argsort(-metrics[gt], kind="stable")[:topk]
        output[gt, order] = True
    return output


def assign_reference(
    class_scores: np.ndarray,
    pred_boxes: np.ndarray,
    anchors: np.ndarray,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    valid_gt: np.ndarray,
    *,
    topk: int = 7,
    topk2: int = 1,
    alpha: float = 0.5,
    beta: float = 6.0,
    eligibility_override: np.ndarray | None = None,
    conflict_enabled: bool = True,
) -> ReferenceState:
    """Execute the reference TAL path and return every instrumented stage."""
    class_scores = np.asarray(class_scores, dtype=np.float32)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    valid_gt = np.asarray(valid_gt, dtype=bool).reshape(-1)
    n_gt = len(gt_boxes)
    n_candidates = len(pred_boxes)
    if class_scores.shape[0] != n_candidates or anchors.shape != (n_candidates, 2):
        raise ValueError("candidate arrays have incompatible shapes")

    eligible = stal_eligibility(anchors, gt_boxes, valid_gt)
    if eligibility_override is not None:
        eligible = np.asarray(eligibility_override, dtype=bool).copy()
    overlaps = np.zeros((n_gt, n_candidates), dtype=np.float32)
    for gt in range(n_gt):
        if not valid_gt[gt]:
            continue
        tiled = np.repeat(gt_boxes[gt][None, :], n_candidates, axis=0)
        overlaps[gt] = np.maximum(ciou_xyxy(tiled, pred_boxes), 0.0)
    overlaps = np.where(eligible, overlaps, 0.0)
    selected_scores = class_scores[:, gt_labels].T
    selected_scores = np.where(eligible, selected_scores, 0.0)
    align = (selected_scores**np.float32(alpha) * overlaps**np.float32(beta)).astype(np.float32)
    topk_mask = _topk_mask(align, topk, valid_gt) & eligible

    conflict = topk_mask.copy()
    if conflict_enabled:
        multi = conflict.sum(axis=0) > 1
        if np.any(multi):
            winners = np.argmax(overlaps[:, multi], axis=0)
            conflict[:, multi] = False
            conflict[winners, np.where(multi)[0]] = True

    post = conflict.copy()
    if topk2 != topk:
        reduced = align * post
        keep = np.zeros_like(post)
        for gt in range(n_gt):
            if valid_gt[gt]:
                keep[gt, np.argmax(reduced[gt])] = True
        post &= keep
    fg_mask = post.sum(axis=0).astype(bool)
    target_gt_idx = np.argmax(post, axis=0).astype(np.int64)
    return ReferenceState(eligible, align, overlaps, topk_mask, conflict, post, target_gt_idx, fg_mask)


def active_by_gt(state: ReferenceState) -> list[int | None]:
    output: list[int | None] = []
    for gt in range(state.post.shape[0]):
        candidates = np.where(state.fg_mask & (state.target_gt_idx == gt))[0]
        output.append(int(candidates[0]) if len(candidates) == 1 else None)
    return output


def first_divergence_reference(base: ReferenceState, shifted: ReferenceState, gt: int) -> str:
    base_active = active_by_gt(base)[gt]
    shifted_active = active_by_gt(shifted)[gt]
    if base_active == shifted_active:
        return "stable"
    pair = [candidate for candidate in (base_active, shifted_active) if candidate is not None]
    if any(base.eligible[gt, candidate] != shifted.eligible[gt, candidate] for candidate in pair):
        return "eligibility_boundary"
    if any(base.topk[gt, candidate] != shifted.topk[gt, candidate] for candidate in pair):
        return "topk_membership_transition"
    if any(base.conflict[gt, candidate] != shifted.conflict[gt, candidate] for candidate in pair):
        return "conflict_reassignment"
    if len(pair) == 2:
        a, b = pair
        if base.align[gt, a] > base.align[gt, b] and shifted.align[gt, a] < shifted.align[gt, b]:
            return "within_set_geometry_rank_reversal"
    if shifted_active is None:
        return "active_disappearance"
    return "compound_or_other"
