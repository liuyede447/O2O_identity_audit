"""Shared exact O2O replay primitives for the audit and Margin-Gated pilot.

The functions intentionally use ``TaskAlignedAssigner.get_pos_mask`` and
``select_highest_overlaps`` rather than reproducing TAL arithmetic.  They are the
same two native stages used by ``assignment_stability_audit_v2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ultralytics.utils.tal import make_anchors

PERTURBATIONS = (("left_1px", -1.0, 0.0), ("right_1px", 1.0, 0.0), ("up_1px", 0.0, -1.0), ("down_1px", 0.0, 1.0))


@dataclass
class O2OState:
    eligible: torch.Tensor
    pre: torch.Tensor
    conflict: torch.Tensor
    post: torch.Tensor
    align: torch.Tensor
    overlaps: torch.Tensor
    target_gt_idx: torch.Tensor
    fg_mask: torch.Tensor
    anchors: torch.Tensor
    stride: torch.Tensor
    decoded: torch.Tensor


@torch.no_grad()
def trace_o2o(criterion, raw: dict[str, torch.Tensor], labels: torch.Tensor, boxes: torch.Tensor, mask: torch.Tensor) -> O2OState:
    """Replay native O2O TAL up to the post-collision/top-1 state; no forward pass."""
    pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = raw["scores"].permute(0, 2, 1).contiguous()
    anchors, stride = make_anchors(raw["feats"], criterion.stride, 0.5)
    decoded = criterion.bbox_decode(anchors, pred_distri)
    assigner = criterion.assigner
    assigner.bs, assigner.n_max_boxes = pred_scores.shape[0], boxes.shape[1]
    eligible = assigner.select_candidates_in_gts(anchors * stride, boxes, mask).bool()
    pre, align, overlaps = assigner.get_pos_mask(
        pred_scores.detach().sigmoid(), (decoded.detach() * stride).type(boxes.dtype), labels, boxes, anchors * stride, mask
    )
    conflict = pre.clone()
    conflict_fg = conflict.sum(-2)
    if conflict_fg.max() > 1:
        multi = (conflict_fg.unsqueeze(1) > 1).expand(-1, boxes.shape[1], -1)
        max_overlap_gt = overlaps.argmax(1)
        winners = torch.zeros_like(conflict)
        winners.scatter_(1, max_overlap_gt.unsqueeze(1), 1)
        conflict = torch.where(multi, winners, conflict).float()
    # select_highest_overlaps applies the O2O top-k2 filter in place when no
    # conflict creates a replacement tensor. Preserve the pre-collision Top-k
    # mask so its audit meaning is identical in conflict and no-conflict cases.
    target_gt_idx, fg_mask, post = assigner.select_highest_overlaps(pre.clone(), overlaps, assigner.n_max_boxes, align)
    return O2OState(eligible, pre.bool(), conflict.bool(), post.bool(), align, overlaps, target_gt_idx, fg_mask.bool(), anchors, stride, decoded)


def candidate_and_margin(state: O2OState, batch_index: int, gt_index: int) -> tuple[int | None, int | None, float | None]:
    """Return the audit-v2-equivalent native candidate, Top-2 alternative, and margin."""
    scores = state.align[batch_index, gt_index]
    post_ids = torch.where(state.post[batch_index, gt_index])[0]
    candidate = int(post_ids[scores[post_ids].argmax()].item()) if post_ids.numel() else None
    eligible = torch.where(scores > 0)[0]
    if eligible.numel() < 2:
        return candidate, None, None
    ordered = eligible[torch.argsort(scores[eligible], descending=True)]
    first, second = float(scores[ordered[0]].item()), float(scores[ordered[1]].item())
    return candidate, int(ordered[1].item()), (first - second) / (first + 1e-12)


def pre_alignment_margin(state: O2OState, batch_index: int, gt_index: int) -> float | None:
    """Pre-resolution Top-1/Top-2 alignment margin used only as an ambiguity indicator."""
    scores = state.align[batch_index, gt_index]
    eligible = torch.where(scores > 0)[0]
    if eligible.numel() < 2:
        return None
    ordered = eligible[torch.argsort(scores[eligible], descending=True)]
    first, second = float(scores[ordered[0]].item()), float(scores[ordered[1]].item())
    return (first - second) / (first + 1e-12)


def active_candidate_and_margin(state: O2OState, batch_index: int, gt_index: int) -> tuple[int | None, int | None, float | None]:
    """Return the actual loss-target identity and its native-relative Top-2 margin.

    ``state.post`` is an assignment probe mask and can contain duplicate claims after
    O2O's top-k2 stage.  The native loss instead consumes ``fg_mask`` together with
    ``target_gt_idx``.  This function is therefore the only valid identity source
    for a training-time selector.
    """
    active = torch.where(state.fg_mask[batch_index] & state.target_gt_idx[batch_index].eq(gt_index))[0]
    if active.numel() != 1:
        return None, None, None
    native = int(active.item())
    scores = state.align[batch_index, gt_index]
    if float(scores[native].item()) <= 1e-12:
        # A collision-resolved zero-alignment target has no meaningful relative margin.
        return native, None, None
    eligible = torch.where(scores > 0)[0]
    alternatives = eligible[eligible.ne(native)]
    if alternatives.numel() == 0:
        return native, None, None
    alternative = int(alternatives[scores[alternatives].argmax()].item())
    first, second = float(scores[native].item()), float(scores[alternative].item())
    return native, alternative, (first - second) / (first + 1e-12)


def active_normalized_score(state: O2OState, batch_index: int, gt_index: int, candidate: int | None) -> float:
    """Score a fixed candidate from a full replay's native alignment landscape.

    Active identity determines the native winner and flip semantics, while Minimax
    compares a baseline Top-2 counterfactually.  Requiring an alternative to already
    be the active winner would force its original-state score to zero and make every
    rewrite impossible.  A score is zero only when the candidate is not eligible.
    """
    if candidate is None:
        return 0.0
    scores = state.align[batch_index, gt_index]
    if float(scores[candidate].item()) <= 0.0:
        return 0.0
    return float(scores[candidate].item()) / max(float(scores.amax().item()), 1e-12)


def legal_shift(box: torch.Tensor, dx: float, dy: float, image_width: float, image_height: float) -> torch.Tensor | None:
    """Shift only a center; never clip or resize an invalid perturbation."""
    shifted = box.clone()
    shifted[[0, 2]] += dx
    shifted[[1, 3]] += dy
    if shifted[0] < 0 or shifted[1] < 0 or shifted[2] > image_width or shifted[3] > image_height:
        return None
    return shifted
