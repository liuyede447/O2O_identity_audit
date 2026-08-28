"""Stratified image-cluster bootstrap for the YOLOv8 pre/post-NMS control."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def rate(rows: list[dict], key: str) -> float:
    numerator = sum(float(row["sampling_weight"]) * float(row[key]) for row in rows)
    denominator = sum(float(row["sampling_weight"]) for row in rows)
    return numerator / denominator


def delta(rows: list[dict], pre: str, post: str) -> float:
    return rate(rows, post) - rate(rows, pre)


def interval(values: list[float]) -> dict:
    array = np.asarray(values)
    return {"median": float(np.median(array)), "CI_95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]}


def main() -> None:
    args = parse_args()
    if not args.per_gt.is_file() or args.output_dir.exists() or args.replicates <= 0:
        raise ValueError("invalid bootstrap inputs")
    with args.per_gt.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["stratum"]][row["image_id"]].append(row)
    rng = random.Random(args.seed)
    metrics = {"fn_rate_post_minus_pre": [], "duplicate_gt_rate_post_minus_pre": [], "mean_excess_matches_post_minus_pre": []}
    for _ in range(args.replicates):
        sample = []
        for stratum in sorted(groups):
            keys = list(groups[stratum])
            for key in (rng.choice(keys) for _ in keys):
                sample.extend(groups[stratum][key])
        metrics["fn_rate_post_minus_pre"].append(delta(sample, "pre_fn", "post_fn"))
        metrics["duplicate_gt_rate_post_minus_pre"].append(delta(sample, "pre_duplicate", "post_duplicate"))
        metrics["mean_excess_matches_post_minus_pre"].append(delta(sample, "pre_excess_matches", "post_excess_matches"))
    payload = {
        "status": "complete", "protocol": "stratified_image_cluster_bootstrap_nms_effect_v1",
        "replicates": args.replicates, "seed": args.seed, "clusters": sum(len(items) for items in groups.values()),
        "metric_definition": "post-NMS minus pre-top300-proxy; negative duplicate deltas indicate suppression",
        "metrics": {key: interval(value) for key, value in metrics.items()},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("YOLOV8_NMS_EFFECT_BOOTSTRAP_PASS", json.dumps(payload))


if __name__ == "__main__":
    main()
