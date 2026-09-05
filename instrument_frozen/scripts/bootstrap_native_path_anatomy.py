"""Cluster-bootstrap pathway fractions from frozen anatomy traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PATHWAYS = [
    "eligibility_boundary",
    "topk_membership_transition",
    "conflict_reassignment",
    "within_set_geometry_rank_reversal",
    "active_disappearance",
    "compound_or_other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def summarize(directory: Path, reps: int, seed: int) -> dict:
    rows = pd.read_csv(directory / "per_direction_anatomy.csv")
    q_rows = pd.read_csv(directory / "pairwise_q_gap.csv")
    result = {}
    rng = np.random.default_rng(seed)
    for bucket in ("all", "t_8_16", "s_16_32"):
        subset = rows[rows["o2o_flip"].eq(1)].copy()
        if bucket != "all":
            subset = subset[subset["size_bin"].eq(bucket)].copy()
        subset["flip_weight"] = subset["sampling_weight"].astype(float)
        for pathway in PATHWAYS:
            subset[f"p_{pathway}"] = subset["flip_weight"] * subset["first_divergence"].eq(pathway)
        subset["p_clipping"] = subset["flip_weight"] * subset["geometry_clipping_transition"].astype(int)
        columns = ["flip_weight"] + [f"p_{pathway}" for pathway in PATHWAYS] + ["p_clipping"]
        per_image = subset.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
        total = per_image[columns].sum().to_numpy(float)
        point = total[1:] / total[0]
        samples = []
        groups = {stratum: group[columns].to_numpy(float) for stratum, group in per_image.groupby("stratum")}
        for _ in range(reps):
            vector = np.zeros(len(columns), dtype=float)
            for values in groups.values():
                vector += values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
            samples.append(vector[1:] / vector[0])
        samples = np.asarray(samples)
        labels = PATHWAYS + ["geometry_clipping_transition"]
        result[bucket] = {
            label: {
                "point": float(point[index]),
                "ci_95": [float(value) for value in np.percentile(samples[:, index], [2.5, 97.5])],
            }
            for index, label in enumerate(labels)
        }
        result[bucket]["flip_rows"] = int(len(subset))
        result[bucket]["contributing_images"] = int(per_image["image_id"].nunique())

    finite = q_rows[q_rows["delta_pairwise_log_q_gap"].notna()].copy()
    q_summary = {
        "n_simple_reversals": int(len(finite)),
        "static_class_offset_median": float(finite["static_class_offset"].median()),
        "static_class_offset_iqr": [float(value) for value in finite["static_class_offset"].quantile([0.25, 0.75])],
        "delta_geometry_gap_median": float(finite["delta_geometry_gap"].median()),
        "delta_geometry_gap_iqr": [float(value) for value in finite["delta_geometry_gap"].quantile([0.25, 0.75])],
        "delta_classification_term_unique": sorted(float(value) for value in finite["delta_classification_term"].dropna().unique()),
        "max_abs_exact_residual": float(max(finite["exact_residual_base"].abs().max(), finite["exact_residual_shift"].abs().max())),
    }
    return {"pathways": result, "q_gap": q_summary}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = {
        "status": "complete",
        "protocol": "stratified image-cluster bootstrap of prespecified first-divergence taxonomy",
        "bootstrap_replicates": args.reps,
        "fixed_1px": summarize(args.fixed_dir, args.reps, args.seed),
        "normalized_k00625": summarize(args.normalized_dir, args.reps, args.seed + 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
