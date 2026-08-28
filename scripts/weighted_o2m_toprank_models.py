"""IPW models for O2M top-ranked-candidate fragility and post-NMS FN."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from weighted_random_stratified_models import model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.selected_images, args.audit, args.outcomes):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.replicates <= 0:
        raise ValueError("invalid or pre-existing output target")
    with args.selected_images.open(newline="", encoding="utf-8") as handle:
        selected = {row["image_id"]: {"stratum": row["stratum"], "sampling_weight": float(row["sampling_weight"])} for row in csv.DictReader(handle)}
    with args.outcomes.open(newline="", encoding="utf-8") as handle:
        outcome_rows = list(csv.DictReader(handle))
    rows = []
    for row in outcome_rows:
        margin_text = row["o2m_rank_margin_0"]
        if margin_text in ("", "None"):
            continue
        sampling = selected[row["image_id"]]
        rows.append({
            "image_id": row["image_id"], "stratum": sampling["stratum"], "sampling_weight": sampling["sampling_weight"],
            "margin": float(margin_text), "log_area": math.log(max(float(row["area"]), 1.0)), "nplus": float(row["o2m_positive_count"]),
            "flip": float(float(row["o2m_rank_flip_rate"]) > 0.0), "fn": float(row["post_nms_fn_at_iou50"]),
        })
    if not rows:
        raise RuntimeError("no valid O2M rank-margin outcomes")

    unique_audit = {}
    with args.audit.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unique_audit.setdefault((row["image_id"], row["gt_id"]), row)
    boundary = defaultdict(lambda: {"raw_total": 0, "raw_valid": 0, "weighted_total": 0.0, "weighted_valid": 0.0})
    for (image_id, _), row in unique_audit.items():
        weight = selected[image_id]["sampling_weight"]
        valid = row["o2m_rank_margin_0"] not in ("", "None")
        for key in ("all", row["size_bin"]):
            boundary[key]["raw_total"] += 1; boundary[key]["raw_valid"] += int(valid)
            boundary[key]["weighted_total"] += weight; boundary[key]["weighted_valid"] += weight * int(valid)
    boundary_summary = {}
    for key, value in boundary.items():
        boundary_summary[key] = {**value, "raw_invalid_rate": 1.0 - value["raw_valid"] / value["raw_total"], "weighted_invalid_rate": 1.0 - value["weighted_valid"] / value["weighted_total"]}

    args.output_dir.mkdir(parents=True)
    payload = {
        "status": "complete",
        "protocol": "o2m_toprank_ipw_stratified_cluster_bootstrap_v1",
        "estimand_warning": "O2M top-ranked alignment candidate is not an O2O loss-active identity.",
        "sources": {
            "selected_images": {"path": str(args.selected_images), "sha256": sha256(args.selected_images)},
            "audit": {"path": str(args.audit), "sha256": sha256(args.audit)},
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
        },
        "rank_margin_boundary": boundary_summary,
        "rank_margin_to_flip_any": model(rows, "flip", args.replicates, args.seed),
        "rank_margin_to_post_nms_fn": model(rows, "fn", args.replicates, args.seed + 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WEIGHTED_O2M_TOPRANK_MODELS_PASS", args.output_dir)


if __name__ == "__main__":
    main()
