"""Bootstrap object-level branch contrasts with all legal vs four valid shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_rows(per_direction: Path, per_gt: Path) -> pd.DataFrame:
    directions = pd.read_csv(per_direction)
    gt = pd.read_csv(per_gt)
    required = {"image_id", "gt_id", "size_bin", "stratum", "sampling_weight", "o2o_flip", "o2m_flip"}
    if required - set(directions.columns):
        raise RuntimeError(f"missing direction columns: {sorted(required - set(directions.columns))}")
    valid = gt.loc[gt["common_valid_margin"].eq(1), ["image_id", "gt_id"]].drop_duplicates()
    directions = directions.merge(valid, on=["image_id", "gt_id"], how="inner", validate="many_to_one")
    return directions.groupby(
        ["image_id", "gt_id", "size_bin", "stratum", "sampling_weight"], as_index=False
    ).agg(
        legal_directions=("direction", "nunique"),
        o2o_fragile=("o2o_flip", "max"),
        o2m_fragile=("o2m_flip", "max"),
    )


def image_aggregate(objects: pd.DataFrame, roster: pd.DataFrame, four_valid: bool) -> pd.DataFrame:
    rows = objects.loc[objects["legal_directions"].eq(4)].copy() if four_valid else objects.copy()
    columns = []
    for branch in ("o2o", "o2m"):
        for size in ("t_8_16", "s_16_32"):
            den = f"{branch}_{size}_den"
            num = f"{branch}_{size}_num"
            mask = rows["size_bin"].eq(size)
            rows[den] = rows["sampling_weight"] * mask
            rows[num] = rows["sampling_weight"] * rows[f"{branch}_fragile"] * mask
            columns.extend([num, den])
    aggregate = rows.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    out = roster.merge(aggregate, on=["stratum", "image_id"], how="left", validate="one_to_one")
    out[columns] = out[columns].fillna(0.0)
    return out


def estimate(frame: pd.DataFrame) -> dict[str, float]:
    rates = {}
    for branch in ("o2o", "o2m"):
        for size in ("t_8_16", "s_16_32"):
            den = float(frame[f"{branch}_{size}_den"].sum())
            if den <= 0:
                raise RuntimeError(f"zero denominator for {branch}/{size}")
            rates[f"{branch}_{size}"] = float(frame[f"{branch}_{size}_num"].sum() / den)
    o2o_gap = rates["o2o_t_8_16"] - rates["o2o_s_16_32"]
    o2m_gap = rates["o2m_t_8_16"] - rates["o2m_s_16_32"]
    return {**rates, "o2o_gap": o2o_gap, "o2m_gap": o2m_gap, "paired_gap": o2o_gap - o2m_gap}


def paired_bootstrap(all_legal: pd.DataFrame, four_valid: pd.DataFrame, reps: int, seed: int) -> dict:
    point_all, point_four = estimate(all_legal), estimate(four_valid)
    keys = list(point_all)
    samples = {name: {key: np.empty(reps) for key in keys} for name in ("all_legal", "four_valid", "four_minus_all")}
    rng = np.random.default_rng(seed)
    groups = [np.asarray(index, dtype=int) for index in all_legal.groupby("stratum", sort=True).indices.values()]
    for rep in range(reps):
        take = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
        a, f = estimate(all_legal.iloc[take]), estimate(four_valid.iloc[take])
        for key in keys:
            samples["all_legal"][key][rep] = a[key]
            samples["four_valid"][key][rep] = f[key]
            samples["four_minus_all"][key][rep] = f[key] - a[key]
    return {
        "point": {
            "all_legal": point_all,
            "four_valid": point_four,
            "four_minus_all": {key: point_four[key] - point_all[key] for key in keys},
        },
        "ci_95": {
            name: {
                key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                for key, values in metrics.items()
            }
            for name, metrics in samples.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--equivalent-area-dir", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    roster = pd.read_csv(args.selected_images)[["stratum", "image_id"]].drop_duplicates()
    payload = {
        "status": "complete",
        "estimand": "IPW any-direction object fragility among common-valid O2O/O2M objects",
        "bootstrap": "paired stratified image-cluster percentile bootstrap",
        "bootstrap_reps": args.reps,
        "contracts": {},
        "source_sha256": {str(args.selected_images): sha256(args.selected_images)},
    }
    for offset, (name, directory) in enumerate(
        (("fixed_1px", args.fixed_dir), ("equivalent_area_side", args.equivalent_area_dir))
    ):
        per_direction, per_gt = directory / "per_direction.csv", directory / "per_gt.csv"
        objects = object_rows(per_direction, per_gt)
        all_table = image_aggregate(objects, roster, False)
        four_table = image_aggregate(objects, roster, True)
        result = paired_bootstrap(all_table, four_table, args.reps, args.seed + offset)
        result["objects_all_legal"] = int(len(objects))
        result["objects_four_valid"] = int(objects["legal_directions"].eq(4).sum())
        result["four_valid_retention_by_size"] = {
            size: float(np.average(part["legal_directions"].eq(4), weights=part["sampling_weight"]))
            for size, part in objects.groupby("size_bin")
        }
        payload["contracts"][name] = result
        payload["source_sha256"][str(per_direction)] = sha256(per_direction)
        payload["source_sha256"][str(per_gt)] = sha256(per_gt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
