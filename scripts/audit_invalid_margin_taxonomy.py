"""Read-only taxonomy for focal GTs without a valid margin/score proxy."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "margin_score_audit_corrected_20260822" / "O2O_AMBIGUITY_STATS_FROZEN_V1_1"
SOURCES = {
    "AI-TOD-v2": (ROOT / "runs/audit/aitod_p3_margin_confirmation_v3/outcomes/per_gt_outcomes.csv", ROOT / "runs/audit/aitod_p3_margin_confirmation_v3/per_gt.csv"),
    "VisDrone": (ROOT / "runs/audit/visdrone_p3_margin_confirmation_v3/outcomes/per_gt_outcomes.csv", ROOT / "runs/audit/visdrone_p3_margin_confirmation_v3/per_gt.csv"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "images": len({r["image_id"] for r in rows}), "mean_side": sum(r["area"] ** .5 for r in rows) / len(rows), "mean_native_o2m_Nplus": sum(r["nplus"] for r in rows) / len(rows), "fn_rate": sum(r["fn"] for r in rows) / len(rows), "size_bins": dict(Counter(r["size_bin"] for r in rows))}


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=OUT);a=p.parse_args()
    if not a.output_dir.exists():
        raise FileNotFoundError("run frozen bundle first")
    result = {}
    for dataset, (outcome_path, audit_path) in SOURCES.items():
        audit = defaultdict(list)
        for r in csv.DictReader(audit_path.open(newline="", encoding="utf-8")):
            audit[(r["image_id"], r["gt_id"])].append(r)
        focal, valid, invalid, details = [], [], [], []
        for r in csv.DictReader(outcome_path.open(newline="", encoding="utf-8")):
            if r["size_bin"] not in ("t_8_16", "s_16_32"):
                continue
            item = {"image_id": r["image_id"], "gt_id": r["gt_id"], "size_bin": r["size_bin"], "area": float(r["area"]), "nplus": float(r["o2m_positive_count"]), "fn": float(r["fn_at_iou50"]), "margin_present": bool(r["o2o_margin_0"]), "score_present": bool(r["best_prediction_score"])}
            focal.append(item)
            if item["margin_present"] and item["score_present"]:
                valid.append(item); continue
            traces = audit.get((item["image_id"], item["gt_id"]), [])
            candidate_present = any(bool(x.get("o2o_candidate_0")) for x in traces)
            # audit_v2 summary emits margin=None exactly when fewer than two eligible alignment scores exist.
            if not candidate_present:
                reason = "no_native_o2o_candidate_recorded"
            elif not item["margin_present"]:
                reason = "native_candidate_but_no_defined_top1_top2_margin"
            else:
                reason = "detection_score_proxy_missing"
            item["reason"] = reason; item["audit_records"] = len(traces); invalid.append(item); details.append(item)
        result[dataset] = {"outcome_csv_sha256": sha(outcome_path), "audit_csv_sha256": sha(audit_path), "focal_before_validity": describe(focal), "valid": describe(valid), "invalid": describe(invalid), "invalid_rate": len(invalid) / len(focal), "reasons": dict(Counter(r["reason"] for r in invalid))}
        with (a.output_dir / f"invalid_margin_taxonomy_{dataset.replace('-','_')}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(details[0]) if details else ["image_id"]); w.writeheader(); w.writerows(details)
    (a.output_dir / "09_invalid_margin_taxonomy.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("INVALID_MARGIN_TAXONOMY_PASS")


if __name__ == "__main__": main()
