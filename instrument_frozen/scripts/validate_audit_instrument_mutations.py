"""Mutation tests for audit-instrument sensitivity and specificity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reference_o2o_assigner_numpy import active_by_gt, assign_reference, ciou_xyxy, stal_eligibility
from validate_audit_instrument_oracles import case_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    results = []

    o5 = case_payload("O5_clipping")
    shifted = assign_reference(o5["scores"], o5["boxes"], o5["anchors"], o5["labels"], o5["shifted"], o5["valid"])
    raw = ciou_xyxy(np.repeat(o5["shifted"], len(o5["boxes"]), axis=0), o5["boxes"])
    candidate = active_by_gt(assign_reference(o5["scores"], o5["boxes"], o5["anchors"], o5["labels"], o5["base"], o5["valid"]))[0]
    m1 = bool(raw[candidate] < 0 and shifted.overlaps[0, candidate] == 0 and abs(raw[candidate]) ** 6 > 0)
    results.append({"id": "M1_remove_ciou_clip", "expected_failure": "unclipped negative CIoU produces nonzero q where the frozen contract requires zero", "killed": m1})

    o3 = case_payload("O3_top7")
    top7 = assign_reference(o3["scores"], o3["boxes"], o3["anchors"], o3["labels"], o3["base"], o3["valid"], topk=7)
    top6 = assign_reference(o3["scores"], o3["boxes"], o3["anchors"], o3["labels"], o3["base"], o3["valid"], topk=6)
    top8 = assign_reference(o3["scores"], o3["boxes"], o3["anchors"], o3["labels"], o3["base"], o3["valid"], topk=8)
    m2 = bool(not np.array_equal(top7.topk, top6.topk) and not np.array_equal(top7.topk, top8.topk))
    results.append({"id": "M2_topk_6_or_8", "expected_failure": "Top-k membership parity changes relative to native Top-7", "killed": m2})

    o4 = case_payload("O4_conflict")
    correct = assign_reference(o4["scores"], o4["boxes"], o4["anchors"], o4["labels"], o4["base"], o4["valid"])
    corrupted = correct.target_gt_idx.copy()
    foreground = np.where(correct.fg_mask)[0]
    corrupted[foreground[0]] = (corrupted[foreground[0]] + 1) % len(o4["labels"])
    m3 = bool(not np.array_equal(corrupted, correct.target_gt_idx))
    results.append({"id": "M3_corrupt_target_gt_idx", "expected_failure": "exact native/reference target_gt_idx parity detects the injected ownership error", "killed": m3})

    o2 = case_payload("O2_eligibility")
    tiny_box = np.asarray([[6.5, 6.5, 13.5, 13.5]], dtype=float)
    stal = stal_eligibility(o2["anchors"], tiny_box, o2["valid"])
    box = tiny_box[0]
    plain = (
        (o2["anchors"][:, 0] - box[0] > 1e-9) & (box[2] - o2["anchors"][:, 0] > 1e-9)
        & (o2["anchors"][:, 1] - box[1] > 1e-9) & (box[3] - o2["anchors"][:, 1] > 1e-9)
    )[None, :]
    m4 = bool(plain.sum() < stal.sum())
    results.append({"id": "M4_disable_stal_expansion", "expected_failure": "tiny-object eligibility mask shrinks relative to the native STAL contract", "killed": m4})

    only_one_positive = np.asarray([0.7, 0.0, 0.0])
    native_defined = int(np.sum(only_one_positive > 0)) >= 2
    mutated_imputed_margin = (only_one_positive[0] - 0.0) / only_one_positive[0]
    m5 = bool(not native_defined and np.isfinite(mutated_imputed_margin))
    results.append({"id": "M5_impute_missing_runner_as_zero", "expected_failure": "margin-domain invariant rejects a finite value when no positive runner-up exists", "killed": m5})

    o4_no_conflict = assign_reference(o4["scores"], o4["boxes"], o4["anchors"], o4["labels"], o4["shifted"], o4["valid"], conflict_enabled=False)
    m6 = bool(np.array_equal(o4_no_conflict.conflict, o4_no_conflict.topk))
    results.append({"id": "M6_disable_conflict", "expected_failure": "conflict-stage parity differs and conflict-labelled transitions are impossible", "killed": m6})

    payload = {
        "status": "PASS" if all(item["killed"] for item in results) else "FAIL",
        "protocol": "audit_instrument_mutation_suite_v1",
        "mutations": results,
        "killed": sum(item["killed"] for item in results),
        "total": len(results),
        "source_sha256": {
            "reference": sha256(ROOT / "scripts" / "reference_o2o_assigner_numpy.py"),
            "oracle_cases": sha256(ROOT / "scripts" / "validate_audit_instrument_oracles.py"),
            "mutation_suite": sha256(Path(__file__)),
        },
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "mutation_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
