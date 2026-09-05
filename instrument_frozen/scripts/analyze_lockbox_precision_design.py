"""Prospective lockbox precision planning from frozen discovery outputs.

This script estimates expected precision for the prespecified H1--H4 lockbox
estimands.  It is not a post-hoc power analysis and deliberately does not
report power, minimum detectable effects, or sample-size claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SIZES = ("t_8_16", "s_16_32")
Z_975 = 1.959963984540054
ESTIMANDS = (
    ("H1", "normalized_paired_gap", "positive"),
    ("H2", "normalized_o2m_gap", "negative"),
    ("H3", "fixed_minus_normalized_o2o_gap", "positive"),
    ("H4a", "fixed_eligibility_minus_normalized", "positive"),
    ("H4b", "normalized_within_minus_fixed", "positive"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-per-gt", type=Path)
    parser.add_argument("--normalized-per-gt", type=Path)
    parser.add_argument("--fixed-anatomy", type=Path)
    parser.add_argument("--normalized-anatomy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--expected-images", type=int, default=300)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run only the synthetic self-test; no discovery files are read.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_roster(*frames: pd.DataFrame, expected_images: int | None) -> pd.DataFrame:
    rosters = []
    for index, frame in enumerate(frames):
        roster = frame[["stratum", "image_id"]].drop_duplicates()
        if roster["image_id"].duplicated().any():
            raise ValueError(f"input {index} assigns an image to multiple strata")
        rosters.append(roster.sort_values("image_id").reset_index(drop=True))
    reference = rosters[0]
    for index, roster in enumerate(rosters[1:], start=1):
        if not reference.equals(roster):
            raise ValueError(f"input {index} does not share the same image/stratum roster")
    if expected_images is not None and len(reference) != expected_images:
        raise ValueError(f"expected {expected_images} images, observed {len(reference)}")
    return reference


def branch_image_aggregate(
    fixed: pd.DataFrame, normalized: pd.DataFrame, roster: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "image_id", "gt_id", "size_bin", "stratum", "sampling_weight",
        "common_valid_margin", "o2o_fragile", "o2m_fragile",
    }
    require_columns(fixed, required, "fixed per-GT CSV")
    require_columns(normalized, required, "normalized per-GT CSV")
    keys = ["image_id", "gt_id"]
    if fixed.duplicated(keys).any() or normalized.duplicated(keys).any():
        raise ValueError("per-GT inputs must be unique on image_id and gt_id")
    left = fixed[fixed["common_valid_margin"].eq(1)].copy()
    right = normalized[normalized["common_valid_margin"].eq(1)].copy()
    merged = left.merge(right, on=keys, suffixes=("_fixed", "_norm"), validate="one_to_one")
    for column in ("size_bin", "stratum"):
        if not merged[f"{column}_fixed"].equals(merged[f"{column}_norm"]):
            raise ValueError(f"fixed and normalized {column} values differ")
    if not np.allclose(
        merged["sampling_weight_fixed"], merged["sampling_weight_norm"], rtol=0, atol=1e-12
    ):
        raise ValueError("fixed and normalized sampling weights differ")
    merged["size_bin"] = merged["size_bin_fixed"]
    merged["stratum"] = merged["stratum_fixed"]
    merged["sampling_weight"] = merged["sampling_weight_fixed"].astype(float)
    columns: list[str] = []
    for contract, suffix in (("fixed", "fixed"), ("norm", "norm")):
        for branch in ("o2o", "o2m"):
            for size in SIZES:
                den = f"{contract}_{branch}_{size}_den"
                num = f"{contract}_{branch}_{size}_num"
                mask = merged["size_bin"].eq(size)
                merged[den] = merged["sampling_weight"] * mask
                merged[num] = merged["sampling_weight"] * merged[f"{branch}_fragile_{suffix}"] * mask
                columns.extend((num, den))
    aggregate = merged.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    return roster.merge(aggregate, on=["stratum", "image_id"], how="left").fillna(0.0)


def branch_stats(frame: pd.DataFrame, contract: str) -> dict[str, float]:
    vector = []
    for branch in ("o2o", "o2m"):
        for size in SIZES:
            denominator = float(frame[f"{contract}_{branch}_{size}_den"].sum())
            if denominator <= 0:
                raise ValueError(f"zero denominator for {contract}/{branch}/{size}")
            vector.append(float(frame[f"{contract}_{branch}_{size}_num"].sum()) / denominator)
    o2o_gap = vector[0] - vector[1]
    o2m_gap = vector[2] - vector[3]
    return {"o2o_gap": o2o_gap, "o2m_gap": o2m_gap, "paired_gap": o2o_gap - o2m_gap}


def pathway_image_aggregate(
    fixed: pd.DataFrame, normalized: pd.DataFrame, roster: pd.DataFrame
) -> pd.DataFrame:
    required = {"image_id", "stratum", "sampling_weight", "o2o_flip", "first_divergence"}
    require_columns(fixed, required, "fixed anatomy CSV")
    require_columns(normalized, required, "normalized anatomy CSV")
    pieces = []
    for contract, frame in (("fixed", fixed), ("norm", normalized)):
        subset = frame[frame["o2o_flip"].eq(1)].copy()
        subset[f"{contract}_den"] = subset["sampling_weight"].astype(float)
        subset[f"{contract}_eligibility"] = (
            subset["sampling_weight"] * subset["first_divergence"].eq("eligibility_boundary")
        )
        subset[f"{contract}_within"] = (
            subset["sampling_weight"]
            * subset["first_divergence"].eq("within_set_geometry_rank_reversal")
        )
        columns = [f"{contract}_den", f"{contract}_eligibility", f"{contract}_within"]
        aggregate = subset.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
        pieces.append(roster.merge(aggregate, on=["stratum", "image_id"], how="left").fillna(0.0))
    return pieces[0].merge(pieces[1], on=["stratum", "image_id"], validate="one_to_one")


def pathway_stats(frame: pd.DataFrame) -> dict[str, float]:
    fixed_den = float(frame["fixed_den"].sum())
    norm_den = float(frame["norm_den"].sum())
    if fixed_den <= 0 or norm_den <= 0:
        raise ValueError("pathway fractions require at least one weighted flip per contract")
    fixed_eligibility = float(frame["fixed_eligibility"].sum()) / fixed_den
    norm_eligibility = float(frame["norm_eligibility"].sum()) / norm_den
    fixed_within = float(frame["fixed_within"].sum()) / fixed_den
    norm_within = float(frame["norm_within"].sum()) / norm_den
    return {
        "fixed_eligibility_minus_normalized": fixed_eligibility - norm_eligibility,
        "normalized_within_minus_fixed": norm_within - fixed_within,
    }


def resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("stratum", sort=True):
        draw = rng.integers(0, len(group), size=len(group))
        pieces.append(group.iloc[draw])
    return pd.concat(pieces, ignore_index=True)


def point_estimands(branch: pd.DataFrame, pathway: pd.DataFrame) -> dict[str, float]:
    fixed = branch_stats(branch, "fixed")
    normalized = branch_stats(branch, "norm")
    return {
        "normalized_paired_gap": normalized["paired_gap"],
        "normalized_o2m_gap": normalized["o2m_gap"],
        "fixed_minus_normalized_o2o_gap": fixed["o2o_gap"] - normalized["o2o_gap"],
        **pathway_stats(pathway),
    }


def bootstrap_estimands(
    branch: pd.DataFrame, pathway: pd.DataFrame, reps: int, seed: int
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if reps < 2:
        raise ValueError("at least two bootstrap replicates are required")
    point = point_estimands(branch, pathway)
    samples = {name: np.empty(reps, dtype=float) for _, name, _ in ESTIMANDS}
    rng = np.random.default_rng(seed)
    for index in range(reps):
        branch_draw = resample(branch, rng)
        pathway_draw = resample(pathway, rng)
        values = point_estimands(branch_draw, pathway_draw)
        for name in samples:
            samples[name][index] = values[name]
    return point, samples


def summarize_precision(
    point: dict[str, float], samples: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    rows = []
    for hypothesis, name, direction in ESTIMANDS:
        values = samples[name]
        low, high = np.quantile(values, [0.025, 0.975])
        sd = float(np.std(values, ddof=1))
        recovery = float(np.mean(values > 0)) if direction == "positive" else float(np.mean(values < 0))
        rows.append({
            "hypothesis": hypothesis,
            "estimand": name,
            "expected_direction": direction,
            "discovery_point_estimate": float(point[name]),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_sd": sd,
            "normal_approx_95_expected_half_width": Z_975 * sd,
            "percentile_95_low": float(low),
            "percentile_95_high": float(high),
            "percentile_95_expected_half_width": float((high - low) / 2.0),
            "direction_recovery_probability": recovery,
        })
    return rows


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branch_fixed, branch_norm, anatomy_fixed, anatomy_norm = [], [], [], []
    for image_index in range(4):
        image_id = f"self_{image_index}"
        stratum = "a" if image_index < 2 else "b"
        weight = 2.0 if stratum == "a" else 3.0
        for gt_id, size in enumerate(SIZES):
            common = {
                "image_id": image_id, "gt_id": gt_id, "size_bin": size,
                "stratum": stratum, "sampling_weight": weight, "common_valid_margin": 1,
            }
            branch_fixed.append({**common, "o2o_fragile": int(size == SIZES[0]), "o2m_fragile": int(size == SIZES[1])})
            norm_o2o = int(size == SIZES[0] and image_index % 2 == 0)
            branch_norm.append({**common, "o2o_fragile": norm_o2o, "o2m_fragile": int(size == SIZES[1])})
        anatomy_fixed.append({
            "image_id": image_id, "stratum": stratum, "sampling_weight": weight,
            "o2o_flip": 1, "first_divergence": "eligibility_boundary",
        })
        anatomy_norm.append({
            "image_id": image_id, "stratum": stratum, "sampling_weight": weight,
            "o2o_flip": 1, "first_divergence": "within_set_geometry_rank_reversal",
        })
    return tuple(pd.DataFrame(rows) for rows in (branch_fixed, branch_norm, anatomy_fixed, anatomy_norm))


def run_self_test() -> dict[str, object]:
    fixed, normalized, fixed_a, normalized_a = synthetic_frames()
    roster = image_roster(fixed, normalized, fixed_a, normalized_a, expected_images=4)
    branch = branch_image_aggregate(fixed, normalized, roster)
    pathway = pathway_image_aggregate(fixed_a, normalized_a, roster)
    point, samples = bootstrap_estimands(branch, pathway, reps=50, seed=17)
    expected = {
        "normalized_paired_gap": 1.5,
        "normalized_o2m_gap": -1.0,
        "fixed_minus_normalized_o2o_gap": 0.5,
        "fixed_eligibility_minus_normalized": 1.0,
        "normalized_within_minus_fixed": 1.0,
    }
    checks = {
        f"point_{name}": bool(np.isclose(point[name], value, atol=1e-12))
        for name, value in expected.items()
    }
    checks["all_bootstrap_values_finite"] = all(np.isfinite(values).all() for values in samples.values())
    checks["all_estimands_sampled"] = set(samples) == set(expected)
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "points": point}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    self_test = run_self_test()
    write_json(args.output_dir / "self_test.json", self_test)
    if self_test["status"] != "pass":
        raise RuntimeError("synthetic self-test failed")
    if args.self_test:
        print(json.dumps(self_test, indent=2, sort_keys=True))
        return
    inputs = {
        "fixed_per_gt": args.fixed_per_gt,
        "normalized_per_gt": args.normalized_per_gt,
        "fixed_anatomy": args.fixed_anatomy,
        "normalized_anatomy": args.normalized_anatomy,
    }
    missing = [name for name, path in inputs.items() if path is None or not path.is_file()]
    if missing:
        raise ValueError(f"missing required input files: {missing}")
    fixed = pd.read_csv(inputs["fixed_per_gt"])
    normalized = pd.read_csv(inputs["normalized_per_gt"])
    fixed_a = pd.read_csv(inputs["fixed_anatomy"])
    normalized_a = pd.read_csv(inputs["normalized_anatomy"])
    roster = image_roster(fixed, normalized, fixed_a, normalized_a, expected_images=args.expected_images)
    branch = branch_image_aggregate(fixed, normalized, roster)
    pathway = pathway_image_aggregate(fixed_a, normalized_a, roster)
    point, samples = bootstrap_estimands(branch, pathway, args.reps, args.seed)
    rows = summarize_precision(point, samples)
    summary = {
        "status": "complete",
        "analysis_role": "prospective precision planning from discovery data; not post-hoc power",
        "protocol": "IPW stratified image-cluster bootstrap for lockbox H1--H4 estimands",
        "bootstrap_replicates": args.reps,
        "bootstrap_seed": args.seed,
        "cluster_unit": "image",
        "stratification": "outcome-blind image sampling stratum",
        "image_count": int(len(roster)),
        "stratum_image_counts": {str(key): int(value) for key, value in roster.groupby("stratum").size().items()},
        "estimands": rows,
    }
    summary_path = args.output_dir / "precision_planning_summary.json"
    csv_path = args.output_dir / "precision_planning_estimands.csv"
    write_json(summary_path, summary)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    script_path = Path(__file__).resolve()
    input_manifest = {
        name: {"path": str(path.resolve()), "sha256": sha256(path)}
        for name, path in inputs.items()
    }
    output_manifest = {
        path.name: sha256(path)
        for path in (summary_path, csv_path, args.output_dir / "self_test.json")
    }
    manifest = {
        "status": "complete",
        "analysis_role": summary["analysis_role"],
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "inputs": input_manifest,
        "outputs": output_manifest,
        "parameters": {"reps": args.reps, "seed": args.seed, "expected_images": args.expected_images},
        "software": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
