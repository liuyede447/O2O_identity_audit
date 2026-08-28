"""Create the frozen YOLO26/YOLOv10 phenomenon comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_row(root: Path, label: str, contract: str, audit: Path, validation: Path) -> tuple[dict, dict]:
    manifest_path = audit / "sampling_manifest.json"
    selected_path = audit / "selected_images.csv"
    per_gt_path = audit / "per_gt.csv"
    models_path = audit / "weighted_models_v1" / "summary.json"
    desc_path = audit / "descriptive_v1" / "summary.json"
    for path in (manifest_path, selected_path, per_gt_path, models_path, desc_path, validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text())
    models = json.loads(models_path.read_text())
    desc = json.loads(desc_path.read_text())
    val = json.loads(validation.read_text())
    scale = {row["size_bin"]: row for row in desc["scale_rows"]}
    taxonomy = Counter()
    unique = {}
    with per_gt_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            unique.setdefault((item["image_id"], item["gt_id"]), item)
    for item in unique.values():
        state = "active_missing" if item["o2o_candidate_0"] in ("", "None") else "margin_undefined_with_active" if item["o2o_margin_0"] in ("", "None") else "valid_margin"
        taxonomy[state] += 1
    frag = models["margin_to_flip_any"]["bootstrap"]
    fn = models["margin_to_fn"]["bootstrap"]
    metrics = val["metrics"]
    row = {
        "architecture": label,
        "assignment_contract": contract,
        "mAP50_95": metrics["metrics/mAP50-95(B)"],
        "mAP50": metrics["metrics/mAP50(B)"],
        "precision": metrics["metrics/precision(B)"],
        "recall": metrics["metrics/recall(B)"],
        "selected_images": manifest["selected_images"],
        "focal_gt": sum(manifest["focal_gt_by_bin"].values()),
        "audited_gt": sum(manifest["audited_gt_by_bin"].values()),
        "valid_margin_gt": models["margin_to_flip_any"]["n_gt"],
        "active_missing": taxonomy["active_missing"],
        "margin_undefined_with_active": taxonomy["margin_undefined_with_active"],
        "fragility_8_16": scale["t_8_16"]["weighted_fragility_rate_valid"],
        "fragility_16_32": scale["s_16_32"]["weighted_fragility_rate_valid"],
        "invalid_8_16": scale["t_8_16"]["weighted_invalid_rate"],
        "invalid_16_32": scale["s_16_32"]["weighted_invalid_rate"],
        "margin_fragility_or": frag["margin_OR_median"],
        "margin_fragility_ci_low": frag["margin_OR_percentile_CI_95"][0],
        "margin_fragility_ci_high": frag["margin_OR_percentile_CI_95"][1],
        "margin_fn_or": fn["margin_OR_median"],
        "margin_fn_ci_low": fn["margin_OR_percentile_CI_95"][0],
        "margin_fn_ci_high": fn["margin_OR_percentile_CI_95"][1],
    }
    source = {"manifest": sha256(manifest_path), "selected_images": sha256(selected_path), "audit": sha256(per_gt_path), "models": sha256(models_path), "descriptive": sha256(desc_path), "validation": sha256(validation)}
    return row, source


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    y26_audit = root / "runs/audit/aitod_random_stratified_active_v2"
    y10_audit = root / "runs/audit/yolov10s_aitod_phenomenon_audit_v1_formal300"
    y26_val = root / "outputs/rechecks/yolo26s_aitodv2_p3baseline_native_b4_e100_s0/recheck_report.json"
    y10_val = root / "outputs/second_detector/yolov10s_aitod_phenomenon_audit_v1/validation.json"

    # Convert the YOLO26 recheck to the same compact validation schema in memory.
    report = json.loads(y26_val.read_text())
    compact = output.parent / ".yolo26_validation_tmp.json"
    compact.write_text(json.dumps({"metrics": report["checkpoints"]["best"]["ultralytics_metrics"]}))
    try:
        y26, source26 = audit_row(root, "YOLO26s", "topk10 / topk7→1 / direct-LTRB", y26_audit, compact)
        y10, source10 = audit_row(root, "YOLOv10-S", "topk10 / topk1 / DFL", y10_audit, y10_val)
    finally:
        compact.unlink(missing_ok=True)
    rows = [y26, y10]
    selected_hashes = {source26["selected_images"], source10["selected_images"]}
    if len(selected_hashes) != 1:
        raise RuntimeError("architectures did not use identical selected images")
    for row in rows:
        if not (row["fragility_8_16"] > row["fragility_16_32"] and row["invalid_8_16"] > row["invalid_16_32"] and row["margin_fragility_ci_high"] < 1 and row["margin_fn_ci_high"] < 1):
            raise RuntimeError(f"phenomenon contract failed for {row['architecture']}")

    output.mkdir(parents=True)
    with (output / "cross_architecture.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "status": "complete",
        "protocol": "cross_architecture_native_assignment_phenomenon_v1",
        "same_300_selected_images": True,
        "primary_verdict": "replicated_across_yolo26_and_yolov10",
        "all_primary_and_secondary_ci_below_one": True,
        "ap_interpretation": "Architecture baselines are reported descriptively; AP differences are not method-effect estimates.",
        "rows": rows,
        "sources": {"YOLO26s": source26, "YOLOv10-S": source10},
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CROSS_ARCHITECTURE_DIAGNOSTIC_SYNTHESIS_PASS")


if __name__ == "__main__":
    main()
