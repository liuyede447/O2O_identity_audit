"""Multi-label co-occurrence audit for O2O directional identity exchanges.

First-divergence labels are order dependent. This analysis reports every
instrumented stage flag and their pairwise co-occurrence among O2O flips, so an
upstream first label is not misread as an exclusive contribution share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


FLAGS = {
    "eligibility": "eligibility_pair_changed",
    "topk": "topk_pair_changed",
    "conflict": "conflict_pair_changed",
    "within_set_rank": "pair_rank_reversal",
    "active_missing": "active_shift_missing",
    "geometry_clipping": "geometry_clipping_transition",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_table(rows: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    flips = rows.loc[rows["o2o_flip"].eq(1)].copy()
    flips["den"] = flips["sampling_weight"].astype(float)
    columns = ["den"]
    for name, column in FLAGS.items():
        out = f"flag__{name}"
        flips[out] = flips["den"] * flips[column].astype(float)
        columns.append(out)
    for left, right in combinations(FLAGS, 2):
        out = f"pair__{left}__{right}"
        flips[out] = flips["den"] * (
            flips[FLAGS[left]].astype(bool) & flips[FLAGS[right]].astype(bool)
        ).astype(float)
        columns.append(out)
    aggregate = flips.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    result = roster.merge(aggregate, on=["stratum", "image_id"], how="left", validate="one_to_one")
    result[columns] = result[columns].fillna(0.0)
    return result


def estimate(frame: pd.DataFrame) -> dict[str, float]:
    den = float(frame["den"].sum())
    if den <= 0:
        raise RuntimeError("zero directional-flip denominator")
    return {column: float(frame[column].sum() / den) for column in frame if column.startswith(("flag__", "pair__"))}


def bootstrap(frame: pd.DataFrame, reps: int, seed: int) -> dict[str, list[float]]:
    point = estimate(frame)
    values = {key: np.empty(reps, dtype=float) for key in point}
    rng = np.random.default_rng(seed)
    groups = [np.asarray(index, dtype=int) for index in frame.groupby("stratum", sort=True).indices.values()]
    for rep in range(reps):
        take = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in groups])
        current = estimate(frame.iloc[take])
        for key, value in current.items():
            values[key][rep] = value
    return {
        key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for key, value in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--equivalent-area", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    roster = pd.read_csv(args.selected_images)[["stratum", "image_id"]].drop_duplicates()
    if roster.empty or roster["image_id"].duplicated().any():
        raise RuntimeError("invalid selected-image roster")
    payload = {
        "status": "complete",
        "estimand": "IPW multi-label stage occurrence among directional O2O identity exchanges",
        "interpretation": "descriptive co-occurrence, not causal attribution",
        "bootstrap": "stratified image-cluster percentile bootstrap",
        "bootstrap_reps": args.reps,
        "contracts": {},
        "source_sha256": {
            str(args.fixed): sha256(args.fixed),
            str(args.equivalent_area): sha256(args.equivalent_area),
            str(args.selected_images): sha256(args.selected_images),
        },
    }
    rows_out = []
    for offset, (name, path) in enumerate(
        (("fixed_1px", args.fixed), ("equivalent_area_side", args.equivalent_area))
    ):
        source = pd.read_csv(path)
        required = {"stratum", "image_id", "o2o_flip", "sampling_weight", *FLAGS.values()}
        missing = required - set(source.columns)
        if missing:
            raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
        table = image_table(source, roster)
        point = estimate(table)
        interval = bootstrap(table, args.reps, args.seed + offset)
        payload["contracts"][name] = {
            "directional_flip_rows": int(source["o2o_flip"].eq(1).sum()),
            "point": point,
            "ci_95": interval,
        }
        for key, value in point.items():
            rows_out.append(
                {
                    "contract": name,
                    "quantity": key,
                    "point_pct": 100.0 * value,
                    "ci_low_pct": 100.0 * interval[key][0],
                    "ci_high_pct": 100.0 * interval[key][1],
                }
            )
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows_out).to_csv(args.output_dir / "cooccurrence.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
