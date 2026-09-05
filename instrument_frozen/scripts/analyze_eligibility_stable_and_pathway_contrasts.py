"""Read-only mechanism sensitivities from frozen O2O anatomy traces.

This script computes two prespecified post hoc summaries without rerunning a
detector:

1. object-level and direction-level O2O fragility after retaining only replay
   comparisons whose candidate-pair eligibility state is unchanged; and
2. paired fixed-versus-equivalent-side differences in first-divergence pathway
   composition using the same stratified image-cluster bootstrap draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PATHWAYS = (
    "eligibility_boundary",
    "topk_membership_transition",
    "conflict_reassignment",
    "within_set_geometry_rank_reversal",
    "active_disappearance",
    "compound_or_other",
)
SIZES = ("t_8_16", "s_16_32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {
        "image_id", "gt_id", "direction", "size_bin", "stratum",
        "sampling_weight", "o2o_flip", "eligibility_pair_changed",
        "first_divergence",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    rows = rows[rows["size_bin"].isin(SIZES)].copy()
    rows["sampling_weight"] = rows["sampling_weight"].astype(float)
    return rows


def weighted_rate(frame: pd.DataFrame, outcome: str) -> float:
    return float(np.average(frame[outcome].astype(float), weights=frame["sampling_weight"]))


def object_level(rows: pd.DataFrame) -> pd.DataFrame:
    objects = (
        rows.groupby(["stratum", "image_id", "gt_id", "size_bin"], as_index=False)
        .agg(
            sampling_weight=("sampling_weight", "first"),
            legal_directions=("direction", "count"),
            eligibility_stable=("eligibility_pair_changed", lambda s: int((s.astype(int) == 0).all())),
            any_o2o_flip=("o2o_flip", "max"),
        )
    )
    return objects


def image_vectors(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    working = frame.copy()
    for size in SIZES:
        mask = working["size_bin"].eq(size)
        working[f"num_{size}"] = working["sampling_weight"] * working[outcome].astype(float) * mask
        working[f"den_{size}"] = working["sampling_weight"] * mask
    columns = [item for size in SIZES for item in (f"num_{size}", f"den_{size}")]
    return working.groupby(["stratum", "image_id"], as_index=False)[columns].sum()


def scale_contrast(vector: np.ndarray) -> tuple[float, float, float]:
    tiny = vector[0] / vector[1]
    small = vector[2] / vector[3]
    return float(tiny), float(small), float(tiny - small)


def bootstrap_scale(frame: pd.DataFrame, outcome: str, reps: int, seed: int) -> dict:
    per_image = image_vectors(frame, outcome)
    columns = [item for size in SIZES for item in (f"num_{size}", f"den_{size}")]
    point = scale_contrast(per_image[columns].sum().to_numpy(float))
    groups = {
        stratum: group[columns].to_numpy(float)
        for stratum, group in per_image.groupby("stratum")
    }
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, 3), dtype=float)
    for rep in range(reps):
        vector = np.zeros(4, dtype=float)
        for values in groups.values():
            vector += values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        samples[rep] = scale_contrast(vector)
    labels = ("fragility_8_16", "fragility_16_32", "scale_contrast")
    return {
        "point": dict(zip(labels, point)),
        "ci_95": {
            label: [float(value) for value in np.percentile(samples[:, index], [2.5, 97.5])]
            for index, label in enumerate(labels)
        },
        "object_or_direction_rows": int(len(frame)),
        "contributing_images": int(per_image["image_id"].nunique()),
    }


def eligibility_stable_summary(rows: pd.DataFrame, reps: int, seed: int) -> dict:
    objects = object_level(rows)
    stable_objects = objects[objects["eligibility_stable"].eq(1)].copy()
    stable_directions = rows[rows["eligibility_pair_changed"].eq(0)].copy()
    total_weight = float(objects["sampling_weight"].sum())
    retained_weight = float(stable_objects["sampling_weight"].sum())
    return {
        "definition": (
            "object-level: retain a focal object only when candidate-pair eligibility is unchanged "
            "for every legal replay direction; direction-level: retain each unchanged-eligibility direction"
        ),
        "object_level": {
            "retained_objects": int(len(stable_objects)),
            "all_objects": int(len(objects)),
            "weighted_retention": retained_weight / total_weight,
            **bootstrap_scale(stable_objects, "any_o2o_flip", reps, seed),
        },
        "direction_level": {
            "retained_directions": int(len(stable_directions)),
            "all_directions": int(len(rows)),
            **bootstrap_scale(stable_directions, "o2o_flip", reps, seed + 1),
        },
    }


def pathway_image_vectors(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    flips = rows[rows["o2o_flip"].eq(1)].copy()
    flips[f"{prefix}_den"] = flips["sampling_weight"]
    for pathway in PATHWAYS:
        flips[f"{prefix}_{pathway}"] = (
            flips["sampling_weight"] * flips["first_divergence"].eq(pathway)
        )
    columns = [f"{prefix}_den"] + [f"{prefix}_{pathway}" for pathway in PATHWAYS]
    return flips.groupby(["stratum", "image_id"], as_index=False)[columns].sum()


def pathway_contrasts(fixed: pd.DataFrame, normalized: pd.DataFrame, reps: int, seed: int) -> dict:
    left = pathway_image_vectors(fixed, "fixed")
    right = pathway_image_vectors(normalized, "normalized")
    per_image = left.merge(right, on=["stratum", "image_id"], how="outer").fillna(0.0)
    columns = [column for column in per_image.columns if column not in {"stratum", "image_id"}]

    def contrast(vector: np.ndarray) -> np.ndarray:
        values = dict(zip(columns, vector))
        fixed_den = values["fixed_den"]
        normalized_den = values["normalized_den"]
        return np.asarray([
            values[f"normalized_{pathway}"] / normalized_den
            - values[f"fixed_{pathway}"] / fixed_den
            for pathway in PATHWAYS
        ])

    point = contrast(per_image[columns].sum().to_numpy(float))
    groups = {
        stratum: group[columns].to_numpy(float)
        for stratum, group in per_image.groupby("stratum")
    }
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, len(PATHWAYS)), dtype=float)
    for rep in range(reps):
        vector = np.zeros(len(columns), dtype=float)
        for values in groups.values():
            vector += values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        samples[rep] = contrast(vector)
    return {
        "definition": "equivalent-side minus fixed-pixel pathway fraction using paired stratified image-cluster resampling",
        "point": {pathway: float(point[index]) for index, pathway in enumerate(PATHWAYS)},
        "ci_95": {
            pathway: [float(value) for value in np.percentile(samples[:, index], [2.5, 97.5])]
            for index, pathway in enumerate(PATHWAYS)
        },
        "paired_images": int(per_image["image_id"].nunique()),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.reps < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    fixed_path = args.fixed / "per_direction_anatomy.csv"
    normalized_path = args.normalized / "per_direction_anatomy.csv"
    fixed = load_rows(fixed_path)
    normalized = load_rows(normalized_path)

    payload = {
        "status": "complete",
        "protocol": "eligibility_stable_sensitivity_and_paired_pathway_contrast_v1",
        "read_only": True,
        "bootstrap_replicates": args.reps,
        "bootstrap_seed": args.seed,
        "source_sha256": {
            str(fixed_path): sha256(fixed_path),
            str(normalized_path): sha256(normalized_path),
        },
        "fixed_1px": eligibility_stable_summary(fixed, args.reps, args.seed),
        "equivalent_side_k00625": eligibility_stable_summary(normalized, args.reps, args.seed + 100),
        "pathway_fraction_difference_equivalent_minus_fixed": pathway_contrasts(
            fixed, normalized, args.reps, args.seed + 200
        ),
    }
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = []
    for contract in ("fixed_1px", "equivalent_side_k00625"):
        summary = payload[contract]["object_level"]
        rows.append({
            "contract": contract,
            "retained_objects": summary["retained_objects"],
            "all_objects": summary["all_objects"],
            "weighted_retention": summary["weighted_retention"],
            "fragility_8_16": summary["point"]["fragility_8_16"],
            "fragility_16_32": summary["point"]["fragility_16_32"],
            "scale_contrast": summary["point"]["scale_contrast"],
            "scale_contrast_ci_low": summary["ci_95"]["scale_contrast"][0],
            "scale_contrast_ci_high": summary["ci_95"]["scale_contrast"][1],
        })
    pd.DataFrame(rows).to_csv(args.output_dir / "eligibility_stable_object_summary.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
