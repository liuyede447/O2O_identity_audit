"""Class-specific sensitivity models for a frozen random-stratified O2O audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from weighted_random_stratified_models import model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-valid-gt", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_class_ids(rows: list[dict], minimum: int) -> list[int]:
    counts = Counter(int(row["class_id"]) for row in rows)
    return sorted(class_id for class_id, count in counts.items() if count >= minimum)


def main() -> None:
    args = parse_args()
    for path in (args.selected_images, args.audit, args.outcomes, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.min_valid_gt <= 0 or args.replicates <= 0:
        raise ValueError("invalid or pre-existing sensitivity target")

    with args.selected_images.open(newline="", encoding="utf-8") as handle:
        selected = {
            row["image_id"]: {
                "stratum": row["stratum"],
                "sampling_weight": float(row["sampling_weight"]),
            }
            for row in csv.DictReader(handle)
        }
    with args.audit.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    audit_by_gt = {}
    for row in audit_rows:
        audit_by_gt.setdefault((row["image_id"], row["gt_id"]), row)
    with args.outcomes.open(newline="", encoding="utf-8") as handle:
        outcome_by_gt = {(row["image_id"], row["gt_id"]): row for row in csv.DictReader(handle)}
    if set(audit_by_gt) != set(outcome_by_gt):
        raise RuntimeError("audit/outcome GT keys differ")

    with args.data.open(encoding="utf-8") as handle:
        names = (yaml.safe_load(handle) or {}).get("names", {})
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    names = {int(key): str(value) for key, value in names.items()}

    rows = []
    invalid_by_class = Counter()
    total_by_class = Counter()
    for key, audit in audit_by_gt.items():
        class_id = int(audit["class_id"])
        total_by_class[class_id] += 1
        margin_text = audit["o2o_margin_0"]
        valid = audit["o2o_candidate_0"] not in {"", "None"} and margin_text not in {"", "None"}
        if not valid:
            invalid_by_class[class_id] += 1
            continue
        outcome = outcome_by_gt[key]
        sampling = selected[audit["image_id"]]
        rows.append({
            "image_id": audit["image_id"],
            "stratum": sampling["stratum"],
            "sampling_weight": sampling["sampling_weight"],
            "class_id": class_id,
            "margin": float(margin_text),
            "log_area": math.log(max(float(audit["area"]), 1.0)),
            "nplus": float(outcome["o2m_positive_count"]),
            "flip": float(float(outcome["o2o_flip_rate"]) > 0.0),
            "fn": float(outcome["fn_at_iou50"]),
        })

    groups = defaultdict(list)
    for row in rows:
        groups[int(row["class_id"])].append(row)
    selected_classes = eligible_class_ids(rows, args.min_valid_gt)
    results = []
    for class_id in selected_classes:
        group = groups[class_id]
        item = {
            "class_id": class_id,
            "class_name": names.get(class_id, f"class_{class_id}"),
            "raw_total_gt": total_by_class[class_id],
            "raw_valid_gt": len(group),
            "raw_invalid_gt": invalid_by_class[class_id],
            "valid_fraction_of_all_valid_gt": len(group) / len(rows),
            "models": {},
        }
        for offset, outcome in enumerate(("flip", "fn")):
            try:
                item["models"][outcome] = model(group, outcome, args.replicates, args.seed + class_id * 10 + offset)
            except RuntimeError as exc:
                item["models"][outcome] = {"status": "not_estimable", "reason": str(exc)}
        results.append(item)

    if not results:
        raise RuntimeError("no class meets the predeclared minimum valid-GT count")
    args.output_dir.mkdir(parents=True)
    payload = {
        "status": "complete",
        "protocol": "class_specific_ipw_stratified_image_cluster_sensitivity_v1",
        "selection_rule": f"classes with raw valid-margin GT >= {args.min_valid_gt}; fixed before model fitting",
        "interpretation": "Sensitivity analysis against class-composition confounding; not a claim of universality for rare classes.",
        "replicates": args.replicates,
        "sources": {
            "selected_images": {"path": str(args.selected_images), "sha256": sha256(args.selected_images)},
            "audit": {"path": str(args.audit), "sha256": sha256(args.audit)},
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
            "data": {"path": str(args.data), "sha256": sha256(args.data)},
        },
        "all_valid_gt": len(rows),
        "selected_class_ids": selected_classes,
        "classes": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RANDOM_STRATIFIED_CLASS_SENSITIVITY_PASS", args.output_dir)


if __name__ == "__main__":
    main()
