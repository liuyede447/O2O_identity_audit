"""Aggregate the frozen AI-TOD b4 and b8 random-stratified checkpoint audits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    {
        "label": "b4_s0_main",
        "batch": 4,
        "seed": 0,
        "run": "runs/detect/experiments/p2_protocol_aitod_b4/yolo26s_aitodv2_p3baseline_native_b4_e100_s0",
        "recheck": "outputs/rechecks/yolo26s_aitodv2_p3baseline_native_b4_e100_s0/recheck_report.json",
        "audit": "runs/audit/aitod_random_stratified_active_v2",
    },
    *[
        {
            "label": f"b8_s{seed}",
            "batch": 8,
            "seed": seed,
            "run": f"runs/detect/experiments/tiny_geometry/yolo26s_aitodv2_baseline_posthoc20_e100_s{seed}",
            "recheck": f"outputs/rechecks/yolo26s_aitodv2_baseline_posthoc20_e100_s{seed}/recheck_report.json",
            "audit": f"runs/audit/aitod_random_stratified_active_s{seed}_b8_v2",
        }
        for seed in range(4)
    ],
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rows = []
    selection_hashes = set()
    sources = []
    for spec in RUNS:
        run = ROOT / spec["run"]
        recheck_path = ROOT / spec["recheck"]
        audit = ROOT / spec["audit"]
        manifest_path = audit / "sampling_manifest.json"
        selected_path = audit / "selected_images.csv"
        audit_path = audit / "per_gt.csv"
        model_path = audit / "weighted_models_v1" / "summary.json"
        for path in (run / "args.yaml", recheck_path, manifest_path, selected_path, audit_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        train_args = yaml.safe_load((run / "args.yaml").read_text()) or {}
        if int(train_args["batch"]) != spec["batch"] or int(train_args["seed"]) != spec["seed"]:
            raise RuntimeError(f"training protocol mismatch for {spec['label']}")
        recheck = json.loads(recheck_path.read_text())
        if recheck.get("warnings"):
            raise RuntimeError(f"formal recheck warning for {spec['label']}: {recheck['warnings']}")
        metrics = recheck["checkpoints"]["best"]["ultralytics_metrics"]
        manifest = json.loads(manifest_path.read_text())
        models = json.loads(model_path.read_text())
        selected_sha = sha256(selected_path)
        selection_hashes.add(selected_sha)
        with selected_path.open(newline="", encoding="utf-8") as handle:
            sampling_weights = {row["image_id"]: float(row["sampling_weight"]) for row in csv.DictReader(handle)}
        with audit_path.open(newline="", encoding="utf-8") as handle:
            unique_gt = {}
            for audit_row in csv.DictReader(handle):
                unique_gt.setdefault((audit_row["image_id"], audit_row["gt_id"]), audit_row)
        taxonomy_raw = Counter()
        taxonomy_weighted = Counter()
        for audit_row in unique_gt.values():
            state = (
                "active_missing"
                if audit_row["o2o_candidate_0"] in ("", "None")
                else "margin_undefined_with_active"
                if audit_row["o2o_margin_0"] in ("", "None")
                else "valid_margin"
            )
            weight = sampling_weights[audit_row["image_id"]]
            for size_bin in ("all", audit_row["size_bin"]):
                taxonomy_raw[(size_bin, state)] += 1
                taxonomy_weighted[(size_bin, state)] += weight
        if sum(taxonomy_raw[("all", state)] for state in ("active_missing", "margin_undefined_with_active", "valid_margin")) != len(unique_gt):
            raise RuntimeError(f"candidate-deficiency taxonomy count mismatch for {spec['label']}")
        flip = models["margin_to_flip_any"]["bootstrap"]
        fn = models["margin_to_fn"]["bootstrap"]
        boundary = models["valid_invalid_boundary"]
        rows.append({
            "label": spec["label"],
            "batch": spec["batch"],
            "seed": spec["seed"],
            "mAP50_95": metrics["metrics/mAP50-95(B)"],
            "mAP50": metrics["metrics/mAP50(B)"],
            "precision": metrics["metrics/precision(B)"],
            "recall": metrics["metrics/recall(B)"],
            "selected_images": manifest["selected_images"],
            "audited_gt": sum(manifest["audited_gt_by_bin"].values()),
            "excluded_no_legal_replay": sum(manifest["excluded_no_legal_replay_by_bin"].values()),
            "valid_margin_gt": models["margin_to_flip_any"]["n_gt"],
            "weighted_invalid_all": boundary["all"]["weighted_invalid_rate"],
            "weighted_invalid_8_16": boundary["t_8_16"]["weighted_invalid_rate"],
            "weighted_invalid_16_32": boundary["s_16_32"]["weighted_invalid_rate"],
            "raw_active_missing_all": taxonomy_raw[("all", "active_missing")],
            "raw_margin_undefined_with_active_all": taxonomy_raw[("all", "margin_undefined_with_active")],
            "raw_active_missing_8_16": taxonomy_raw[("t_8_16", "active_missing")],
            "raw_margin_undefined_with_active_8_16": taxonomy_raw[("t_8_16", "margin_undefined_with_active")],
            "raw_active_missing_16_32": taxonomy_raw[("s_16_32", "active_missing")],
            "raw_margin_undefined_with_active_16_32": taxonomy_raw[("s_16_32", "margin_undefined_with_active")],
            "weighted_active_missing_all": taxonomy_weighted[("all", "active_missing")],
            "weighted_margin_undefined_with_active_all": taxonomy_weighted[("all", "margin_undefined_with_active")],
            "margin_fragility_or_median": flip["margin_OR_median"],
            "margin_fragility_ci_low": flip["margin_OR_percentile_CI_95"][0],
            "margin_fragility_ci_high": flip["margin_OR_percentile_CI_95"][1],
            "margin_fn_or_median": fn["margin_OR_median"],
            "margin_fn_ci_low": fn["margin_OR_percentile_CI_95"][0],
            "margin_fn_ci_high": fn["margin_OR_percentile_CI_95"][1],
        })
        sources.append({
            "label": spec["label"],
            "recheck": {"path": str(recheck_path), "sha256": sha256(recheck_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "models": {"path": str(model_path), "sha256": sha256(model_path)},
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
            "selected_images_sha256": selected_sha,
        })
    if len(selection_hashes) != 1:
        raise RuntimeError("checkpoint audits did not use the exact same selected images")
    if not all(row["margin_fragility_ci_high"] < 1 and row["margin_fn_ci_high"] < 1 for row in rows):
        raise RuntimeError("association direction did not replicate across every checkpoint")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "checkpoint_stress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "protocol": "aitod_random_stratified_checkpoint_stress_v3",
        "same_selected_images": True,
        "selected_images_sha256": next(iter(selection_hashes)),
        "strict_b8_seed_replication": "b8_s0 through b8_s3 share batch, epochs, data, input size, and audit selection; only seed differs.",
        "b4_comparison_note": "b4_s0_main versus b8 runs is a batch/checkpoint stress comparison, not a strict paired seed comparison.",
        "all_fragility_ci_below_one": True,
        "all_fn_ci_below_one": True,
        "rows": rows,
        "sources": sources,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AITOD_CHECKPOINT_STRESS_SUMMARY_PASS", args.output_dir)


if __name__ == "__main__":
    main()
