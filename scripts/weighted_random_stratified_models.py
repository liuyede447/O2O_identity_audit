"""Inverse-probability weighted models for a random image-stratified audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def fit_weighted(x: np.ndarray, y: np.ndarray, sampling_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    for _ in range(150):
        probability = sigmoid(x @ beta)
        variance_weight = np.clip(probability * (1.0 - probability), 1e-8, None) * sampling_weight
        hessian = (x.T * variance_weight) @ x
        step = np.linalg.pinv(hessian) @ (x.T @ (sampling_weight * (y - probability)))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    else:
        raise RuntimeError("weighted IRLS did not converge")
    standard_error = np.sqrt(np.maximum(np.diag(np.linalg.pinv(hessian)), 0.0))
    return beta, standard_error


def design(rows: list[dict], outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack([
        np.ones(len(rows)),
        [row["margin"] for row in rows],
        [row["log_area"] for row in rows],
        [row["nplus"] for row in rows],
    ])
    y = np.asarray([row[outcome] for row in rows], dtype=float)
    weights = np.asarray([row["sampling_weight"] for row in rows], dtype=float)
    return x, y, weights


def model(rows: list[dict], outcome: str, replicates: int, seed: int) -> dict:
    names = ["intercept", "margin", "log_area", "native_o2m_positive_count"]
    x, y, weights = design(rows, outcome)
    beta, standard_error = fit_weighted(x, y, weights)
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["stratum"]][row["image_id"]].append(row)
    rng = random.Random(seed)
    bootstrap_margin = []
    for _ in range(replicates):
        sample = []
        for stratum in sorted(groups):
            keys = list(groups[stratum])
            for key in (rng.choice(keys) for _ in keys):
                sample.extend(groups[stratum][key])
        try:
            sample_beta, _ = fit_weighted(*design(sample, outcome))
            bootstrap_margin.append(float(sample_beta[1]))
        except RuntimeError:
            continue
    if not bootstrap_margin:
        raise RuntimeError("all stratified cluster bootstrap replicates failed")
    values = np.asarray(bootstrap_margin)
    terms = []
    for name, coefficient, error in zip(names, beta, standard_error):
        terms.append({
            "term": name,
            "coef": float(coefficient),
            "OR": float(np.exp(coefficient)),
            "wald_ci_95": [float(np.exp(coefficient - 1.96 * error)), float(np.exp(coefficient + 1.96 * error))],
        })
    return {
        "outcome": outcome,
        "n_gt": len(rows),
        "n_images": len({row["image_id"] for row in rows}),
        "sampling_weight": "inverse image inclusion probability",
        "bootstrap": {
            "unit": "image_id within image stratum",
            "replicates_requested": replicates,
            "replicates_ok": len(values),
            "seed": seed,
            "margin_OR_median": float(np.exp(np.median(values))),
            "margin_OR_percentile_CI_95": [float(np.exp(np.quantile(values, 0.025))), float(np.exp(np.quantile(values, 0.975)))],
        },
        "terms": terms,
    }


def main() -> None:
    args = parse_args()
    for path in (args.selected_images, args.audit, args.outcomes):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.replicates <= 0:
        raise ValueError("invalid or pre-existing weighted analysis target")

    selected = {}
    with args.selected_images.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected[row["image_id"]] = {
                "stratum": row["stratum"],
                "sampling_weight": float(row["sampling_weight"]),
            }

    rows = []
    with args.outcomes.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["size_bin"] not in {"t_8_16", "s_16_32"} or row["o2o_margin_0"] in {"", "None"}:
                continue
            sampling = selected[row["image_id"]]
            rows.append({
                "image_id": row["image_id"],
                "stratum": sampling["stratum"],
                "sampling_weight": sampling["sampling_weight"],
                "margin": float(row["o2o_margin_0"]),
                "log_area": math.log(max(float(row["area"]), 1.0)),
                "nplus": float(row["o2m_positive_count"]),
                "flip": float(float(row["o2o_flip_rate"]) > 0.0),
                "fn": float(row["fn_at_iou50"]),
            })
    if not rows:
        raise RuntimeError("no valid-margin outcomes")

    unique_gt = {}
    with args.audit.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            unique_gt.setdefault((row["image_id"], row["gt_id"]), row)
    boundary = defaultdict(lambda: {"raw_total": 0, "raw_valid": 0, "weighted_total": 0.0, "weighted_valid": 0.0})
    for (image_id, _), row in unique_gt.items():
        if row["size_bin"] not in {"t_8_16", "s_16_32"}:
            continue
        weight = selected[image_id]["sampling_weight"]
        valid = row["o2o_candidate_0"] not in {"", "None"} and row["o2o_margin_0"] not in {"", "None"}
        for key in ("all", row["size_bin"]):
            boundary[key]["raw_total"] += 1
            boundary[key]["raw_valid"] += int(valid)
            boundary[key]["weighted_total"] += weight
            boundary[key]["weighted_valid"] += weight * int(valid)
    boundary_summary = {}
    for key, value in boundary.items():
        boundary_summary[key] = {
            **value,
            "raw_invalid_rate": 1.0 - value["raw_valid"] / value["raw_total"],
            "weighted_invalid_rate": 1.0 - value["weighted_valid"] / value["weighted_total"],
        }

    args.output_dir.mkdir(parents=True)
    result = {
        "status": "complete",
        "protocol": "inverse_probability_weighted_stratified_cluster_bootstrap_v1",
        "sources": {
            "selected_images": {"path": str(args.selected_images), "sha256": sha256(args.selected_images)},
            "audit": {"path": str(args.audit), "sha256": sha256(args.audit)},
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
        },
        "valid_invalid_boundary": boundary_summary,
        "margin_to_flip_any": model(rows, "flip", args.replicates, args.seed),
        "margin_to_fn": model(rows, "fn", args.replicates, args.seed),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("WEIGHTED_RANDOM_STRATIFIED_MODELS_PASS", args.output_dir)


if __name__ == "__main__":
    main()
