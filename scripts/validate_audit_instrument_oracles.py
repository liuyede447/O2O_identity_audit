"""Analytical oracle suite and metamorphic tests for the O2O audit instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ultralytics_local"))

from reference_o2o_assigner_numpy import (  # noqa: E402
    ReferenceState,
    active_by_gt,
    assign_reference,
    first_divergence_reference,
)
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_payload(name: str) -> dict:
    score = np.full((8, 1), 0.8, dtype=float)
    if name == "O1_within_set":
        anchors = np.asarray([[10, 10], [12, 10], [8, 10], [10, 12], [10, 8], [11, 11], [9, 9], [10.5, 9.5]])
        boxes = np.asarray([[0, 0, 20, 20], [2, 0, 22, 20], [1, 1, 19, 19], [-1, -1, 18, 18], [3, 1, 21, 19], [0, 2, 20, 22], [0, -2, 20, 18], [5, 5, 15, 15]])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0]), valid=np.asarray([1]), base=np.asarray([[0, 0, 20, 20]]), shifted=np.asarray([[2, 0, 22, 20]]), expected="within_set_geometry_rank_reversal", focal=0)
    if name == "O2_eligibility":
        anchors = np.asarray([[4, 10], [12, 10], [10, 10], [14, 10], [8, 10], [16, 10], [10, 12], [10, 8]])
        boxes = np.asarray([[5, 5, 15, 15], [9, 5, 19, 15], [7, 5, 17, 15], [10, 5, 20, 15], [6, 5, 16, 15], [11, 5, 21, 15], [7, 7, 17, 17], [7, 3, 17, 13]])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0]), valid=np.asarray([1]), base=np.asarray([[5, 5, 15, 15]]), shifted=np.asarray([[9, 5, 19, 15]]), expected="eligibility_boundary", focal=0, expected_clipping=True)
    if name == "O3_top7":
        xs = [0, 4, 0.5, 1, 1.5, 2, 2.5, 3]
        boxes = np.asarray([[x, 0, x + 20, 20] for x in xs])
        anchors = np.asarray([[10 + index * 0.05, 10] for index in range(8)])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0]), valid=np.asarray([1]), base=np.asarray([[0, 0, 20, 20]]), shifted=np.asarray([[4, 0, 24, 20]]), expected="topk_membership_transition", focal=0)
    if name == "O4_conflict":
        xs = [2, 8, 4, 5, 6, 7, 3, 30]
        boxes = np.asarray([[x, 0, x + 20, 20] for x in xs])
        anchors = np.asarray([[10 + index * 0.5, 10] for index in range(8)])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0, 0]), valid=np.asarray([1, 1]), base=np.asarray([[0, 0, 20, 20], [4, 0, 24, 20]]), shifted=np.asarray([[8, 0, 28, 20], [4, 0, 24, 20]]), expected="conflict_reassignment", focal=0)
    if name == "O5_clipping":
        anchors = np.asarray([[18.2 + index * 0.2, 10] for index in range(8)])
        xs = [0, 16, 2, 4, 6, 8, 10, 40]
        boxes = np.asarray([[x, 0, x + 20, 20] for x in xs])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0]), valid=np.asarray([1]), base=np.asarray([[0, 0, 20, 20]]), shifted=np.asarray([[16, 0, 36, 20]]), expected="within_set_geometry_rank_reversal", focal=0, expected_clipping=True)
    if name == "O6_disappearance":
        anchors = np.asarray([[4 + index, 10] for index in range(8)])
        boxes = np.asarray([[index, 5, index + 10, 15] for index in range(8)])
        return dict(name=name, anchors=anchors, boxes=boxes, scores=score, labels=np.asarray([0]), valid=np.asarray([1]), base=np.asarray([[5, 5, 15, 15]]), shifted=np.asarray([[40, 5, 50, 15]]), expected="eligibility_boundary", focal=0, expected_shift_active=None)
    raise KeyError(name)


def native_stages(case: dict, gt_boxes: np.ndarray, *, conflict_enabled: bool = True) -> ReferenceState:
    scores = torch.tensor(case["scores"], dtype=torch.float32).unsqueeze(0)
    boxes = torch.tensor(case["boxes"], dtype=torch.float32).unsqueeze(0)
    anchors = torch.tensor(case["anchors"], dtype=torch.float32)
    labels = torch.tensor(case["labels"], dtype=torch.float32).view(1, -1, 1)
    gt = torch.tensor(gt_boxes, dtype=torch.float32).unsqueeze(0)
    mask = torch.tensor(case["valid"], dtype=torch.float32).view(1, -1, 1)
    assigner = TaskAlignedAssigner(topk=7, topk2=1, num_classes=1, alpha=0.5, beta=6.0, stride=[8, 16, 32])
    assigner.bs, assigner.n_max_boxes = 1, len(case["labels"])
    eligible = assigner.select_candidates_in_gts(anchors, gt, mask).bool()
    pre, align, overlaps = assigner.get_pos_mask(scores, boxes, labels, gt, anchors, mask)
    conflict = pre.bool().clone()
    if conflict_enabled and conflict.sum(-2).max() > 1:
        multi = (conflict.sum(-2).unsqueeze(1) > 1).expand_as(conflict)
        winner = overlaps.argmax(1)
        selected = torch.zeros_like(conflict)
        selected.scatter_(1, winner.unsqueeze(1), 1)
        conflict = torch.where(multi, selected, conflict).bool()
    final_input = pre if conflict_enabled else conflict.float()
    target, fg, post = assigner.select_highest_overlaps(final_input.clone(), overlaps, assigner.n_max_boxes, align)
    return ReferenceState(
        eligible[0].cpu().numpy(), align[0].cpu().numpy(), overlaps[0].cpu().numpy(),
        pre[0].bool().cpu().numpy(), conflict[0].cpu().numpy(), post[0].bool().cpu().numpy(),
        target[0].cpu().numpy(), fg[0].bool().cpu().numpy(),
    )


def compare_states(native: ReferenceState, reference: ReferenceState) -> dict:
    exact = {}
    for key in ("eligible", "topk", "conflict", "post", "target_gt_idx", "fg_mask"):
        exact[key] = bool(np.array_equal(getattr(native, key), getattr(reference, key)))
    exact["align"] = bool(np.allclose(native.align, reference.align, rtol=2e-5, atol=1e-8))
    exact["overlaps"] = bool(np.allclose(native.overlaps, reference.overlaps, rtol=2e-5, atol=1e-7))
    return exact


def run_case(case: dict) -> dict:
    ref0 = assign_reference(case["scores"], case["boxes"], case["anchors"], case["labels"], case["base"], case["valid"])
    ref1 = assign_reference(case["scores"], case["boxes"], case["anchors"], case["labels"], case["shifted"], case["valid"])
    nat0 = native_stages(case, case["base"])
    nat1 = native_stages(case, case["shifted"])
    label = first_divergence_reference(ref0, ref1, case["focal"])
    base_active = active_by_gt(ref0)[case["focal"]]
    shifted_active = active_by_gt(ref1)[case["focal"]]
    clipping = False
    if base_active is not None and shifted_active is not None:
        pair = [base_active, shifted_active]
        clipping = bool(np.min(np.r_[ref0.overlaps[case["focal"], pair], ref1.overlaps[case["focal"], pair]]) <= 0)
    checks = {
        "expected_label": label == case["expected"],
        "expected_clipping": clipping == bool(case.get("expected_clipping", False)),
        "expected_shift_active": shifted_active == case.get("expected_shift_active", shifted_active),
        "native_reference_base": all(compare_states(nat0, ref0).values()),
        "native_reference_shift": all(compare_states(nat1, ref1).values()),
    }
    return {
        "name": case["name"], "expected": case["expected"], "observed": label,
        "base_active": base_active, "shifted_active": shifted_active,
        "geometry_clipping": clipping, "checks": checks, "pass": all(checks.values()),
    }


def permutation_invariance(case: dict) -> bool:
    rng = np.random.default_rng(20260831)
    permutation = rng.permutation(len(case["boxes"]))
    inverse = np.argsort(permutation)
    original = assign_reference(case["scores"], case["boxes"], case["anchors"], case["labels"], case["base"], case["valid"])
    permuted = assign_reference(case["scores"][permutation], case["boxes"][permutation], case["anchors"][permutation], case["labels"], case["base"], case["valid"])
    active_original = active_by_gt(original)
    active_permuted = [None if value is None else int(permutation[value]) for value in active_by_gt(permuted)]
    return active_original == active_permuted and np.array_equal(original.eligible, permuted.eligible[:, inverse])


def gt_order_invariance(case: dict) -> bool:
    original = assign_reference(case["scores"], case["boxes"], case["anchors"], case["labels"], case["base"], case["valid"])
    reverse = np.arange(len(case["labels"]))[::-1]
    reordered = assign_reference(case["scores"], case["boxes"], case["anchors"], case["labels"][reverse], case["base"][reverse], case["valid"][reverse])
    return active_by_gt(original) == [active_by_gt(reordered)[index] for index in reverse]


def separated_gt_order_case() -> dict:
    anchors = np.asarray([[8, 8], [10, 10], [12, 12], [9, 11], [48, 8], [50, 10], [52, 12], [49, 11]], dtype=float)
    boxes = np.asarray([
        [0, 0, 20, 20], [1, 0, 21, 20], [0, 1, 20, 21], [2, 0, 22, 20],
        [40, 0, 60, 20], [41, 0, 61, 20], [40, 1, 60, 21], [42, 0, 62, 20],
    ], dtype=float)
    return {
        "anchors": anchors, "boxes": boxes, "scores": np.full((8, 1), 0.8),
        "labels": np.asarray([0, 0]), "valid": np.asarray([1, 1]),
        "base": np.asarray([[0, 0, 20, 20], [40, 0, 60, 20]], dtype=float),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cases = [case_payload(name) for name in ("O1_within_set", "O2_eligibility", "O3_top7", "O4_conflict", "O5_clipping", "O6_disappearance")]
    oracle_results = [run_case(case) for case in cases]

    o1 = cases[0]
    equal_p = assign_reference(o1["scores"], o1["boxes"], o1["anchors"], o1["labels"], o1["base"], o1["valid"])
    equal_p_order = np.argsort(-equal_p.align[0]).tolist() == np.argsort(-equal_p.overlaps[0]).tolist()
    same_box = np.repeat(np.asarray([[0, 0, 20, 20]], dtype=float), 8, axis=0)
    varying_scores = np.linspace(0.1, 0.8, 8)[:, None]
    equal_ciou = assign_reference(varying_scores, same_box, o1["anchors"], o1["labels"], o1["base"], o1["valid"])
    equal_ciou_order = np.argsort(-equal_ciou.align[0]).tolist() == np.argsort(-varying_scores[:, 0]).tolist()

    o4 = cases[3]
    conflict_base = assign_reference(o4["scores"], o4["boxes"], o4["anchors"], o4["labels"], o4["base"], o4["valid"], conflict_enabled=False)
    conflict_shift = assign_reference(o4["scores"], o4["boxes"], o4["anchors"], o4["labels"], o4["shifted"], o4["valid"], conflict_enabled=False)
    conflict_off = first_divergence_reference(conflict_base, conflict_shift, 0) != "conflict_reassignment"

    o2 = cases[1]
    eligibility_base = assign_reference(o2["scores"], o2["boxes"], o2["anchors"], o2["labels"], o2["base"], o2["valid"])
    eligibility_shift = assign_reference(o2["scores"], o2["boxes"], o2["anchors"], o2["labels"], o2["shifted"], o2["valid"], eligibility_override=eligibility_base.eligible)
    eligibility_lock = first_divergence_reference(eligibility_base, eligibility_shift, 0) != "eligibility_boundary"

    labels = [item["observed"] for item in oracle_results]
    direction_order = Counter(labels) == Counter(reversed(labels))
    properties = {
        "candidate_permutation_invariant": permutation_invariance(o1),
        "gt_order_invariant": gt_order_invariance(separated_gt_order_case()),
        "direction_processing_order_invariant": direction_order,
        "equal_p_ranking_is_ciou_ranking": equal_p_order,
        "equal_ciou_ranking_is_p_ranking": equal_ciou_order,
        "conflict_off_eliminates_conflict_label": conflict_off,
        "eligibility_lock_eliminates_eligibility_label": eligibility_lock,
    }
    payload = {
        "status": "PASS" if all(item["pass"] for item in oracle_results) and all(properties.values()) else "FAIL",
        "protocol": "analytical_oracle_and_metamorphic_validation_v1",
        "oracle_cases": oracle_results,
        "properties": properties,
        "source_sha256": {
            "reference": sha256(ROOT / "scripts" / "reference_o2o_assigner_numpy.py"),
            "validator": sha256(Path(__file__)),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "oracle_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
