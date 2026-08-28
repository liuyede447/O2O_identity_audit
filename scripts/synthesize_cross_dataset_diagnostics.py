"""Combine frozen AI-TOD and VisDrone random-stratified checkpoint evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VIS_SPECS = [
    ("b4_s0", 4, 0, "89ec3b6dd85fa2503d7b1d7a6f2ae788679d71c395af165fbdc601db3800254e"),
    ("b8_s1", 8, 1, "340897a05cb9f883d4cc89e791b181a3e782526454d58d95d39e88a0e41467fa"),
    ("b8_s2", 8, 2, "8471e79bb3c7ec9397033f8c29f060e5e7a683f97ecafa4c4fe63fef7e666c33"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-summary", type=Path, required=True)
    parser.add_argument("--vis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_fields(models: dict) -> dict:
    flip = models["margin_to_flip_any"]
    fn = models["margin_to_fn"]
    boundary = models["valid_invalid_boundary"]
    return {
        "valid_margin_gt": flip["n_gt"],
        "valid_margin_images": flip["n_images"],
        "weighted_invalid_all": boundary["all"]["weighted_invalid_rate"],
        "weighted_invalid_8_16": boundary["t_8_16"]["weighted_invalid_rate"],
        "weighted_invalid_16_32": boundary["s_16_32"]["weighted_invalid_rate"],
        "fragility_or": flip["bootstrap"]["margin_OR_median"],
        "fragility_ci_low": flip["bootstrap"]["margin_OR_percentile_CI_95"][0],
        "fragility_ci_high": flip["bootstrap"]["margin_OR_percentile_CI_95"][1],
        "fn_or": fn["bootstrap"]["margin_OR_median"],
        "fn_ci_low": fn["bootstrap"]["margin_OR_percentile_CI_95"][0],
        "fn_ci_high": fn["bootstrap"]["margin_OR_percentile_CI_95"][1],
        "bootstrap_flip_ok": flip["bootstrap"]["replicates_ok"],
        "bootstrap_fn_ok": fn["bootstrap"]["replicates_ok"],
    }


def main() -> None:
    args = parse_args()
    if not args.ai_summary.is_file() or not args.vis_dir.is_dir():
        raise FileNotFoundError("frozen cross-dataset sources missing")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    ai = json.loads(args.ai_summary.read_text(encoding="utf-8"))
    if ai["protocol"] != "aitod_random_stratified_checkpoint_stress_v3":
        raise RuntimeError("unexpected AI checkpoint-stress protocol")
    rows = []
    ai_sources_by_label = {source["label"]: source for source in ai["sources"]}
    for item in ai["rows"]:
        source = ai_sources_by_label[item["label"]]
        models_path = Path(source["models"]["path"])
        if not models_path.is_file() or sha256(models_path) != source["models"]["sha256"]:
            raise RuntimeError(f"AI model source hash mismatch: {item['label']}")
        fields = model_fields(json.loads(models_path.read_text(encoding="utf-8")))
        rows.append({
            "dataset": "AI-TOD-v2",
            "checkpoint_label": item["label"],
            "batch": item["batch"],
            "seed": item["seed"],
            **fields,
        })

    vis_sources = []
    for label, batch, seed, expected_sha in VIS_SPECS:
        path = args.vis_dir / f"{label}_weighted_summary.json"
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"VisDrone source hash mismatch: {label}")
        models = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": "VisDrone", "checkpoint_label": label, "batch": batch, "seed": seed, **model_fields(models)})
        vis_sources.append({"label": label, "path": str(path), "sha256": expected_sha})

    if not all(row["fragility_ci_high"] < 1.0 and row["fn_ci_high"] < 1.0 for row in rows):
        raise RuntimeError("cross-dataset direction contract failed")
    if not all(row["bootstrap_flip_ok"] == 5000 and row["bootstrap_fn_ok"] == 5000 for row in rows):
        raise RuntimeError("incomplete bootstrap source")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "cross_dataset_checkpoint_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "protocol": "cross_dataset_random_stratified_checkpoint_matrix_v1",
        "rows": rows,
        "datasets": sorted({row["dataset"] for row in rows}),
        "checkpoints": len(rows),
        "all_fragility_ci_below_one": True,
        "all_fn_ci_below_one": True,
        "interpretation": "Direction and structural boundary are compared; OR magnitudes are not ranked across checkpoints or datasets.",
        "batch_caveat": "b4 versus b8 rows are checkpoint/batch stress, whereas same-batch multi-seed subsets are stricter robustness evidence.",
        "sources": {
            "ai": {"path": str(args.ai_summary), "sha256": sha256(args.ai_summary)},
            "visdrone": vis_sources,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CROSS_DATASET_DIAGNOSTIC_SYNTHESIS_PASS", len(rows), "checkpoints")


if __name__ == "__main__":
    main()
