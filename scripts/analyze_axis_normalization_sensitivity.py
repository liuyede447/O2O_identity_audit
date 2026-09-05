"""Compare equivalent-area-side and axis-normalised replay on matched GTs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACTS = ("equivalent_area_side", "axis_normalised")
BRANCHES = ("o2o", "o2m")
SIZES = ("t_8_16", "s_16_32")
ASPECTS = ("near_square", "moderate", "elongated")
DIR_AXES = ("horizontal", "vertical")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aspect_bin(frame: pd.DataFrame) -> pd.Series:
    ratio = np.maximum(frame["width"] / frame["height"], frame["height"] / frame["width"])
    return pd.cut(
        ratio,
        bins=[1.0, 1.5, 2.5, np.inf],
        labels=ASPECTS,
        include_lowest=True,
        right=True,
    ).astype(str)


def matched_inputs(eq_dir: Path, axis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eq_gt, ax_gt = pd.read_csv(eq_dir / "per_gt.csv"), pd.read_csv(axis_dir / "per_gt.csv")
    keys = ["image_id", "gt_id"]
    for label, frame in (("equivalent", eq_gt), ("axis", ax_gt)):
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{label} per_gt has duplicate keys")
    eq_valid = eq_gt.loc[eq_gt["common_valid_margin"].eq(1)].copy()
    ax_valid = ax_gt.loc[ax_gt["common_valid_margin"].eq(1)].copy()
    eq_keys, ax_keys = set(map(tuple, eq_valid[keys].to_numpy())), set(map(tuple, ax_valid[keys].to_numpy()))
    common = eq_keys & ax_keys
    if not common:
        raise RuntimeError("no cross-contract common-valid GTs")
    eq_valid = eq_valid[eq_valid[keys].apply(tuple, axis=1).isin(common)].copy()
    ax_valid = ax_valid[ax_valid[keys].apply(tuple, axis=1).isin(common)].copy()
    keep = keys + ["class_id", "width", "height", "area", "size_bin", "stratum", "sampling_weight", "legal_directions"]
    merged = eq_valid[keep + ["o2o_fragile", "o2m_fragile"]].merge(
        ax_valid[keys + ["o2o_fragile", "o2m_fragile"]],
        on=keys,
        suffixes=("__equivalent_area_side", "__axis_normalised"),
        validate="one_to_one",
    )
    merged["aspect_bin"] = aspect_bin(merged)

    eq_dir_rows, ax_dir_rows = pd.read_csv(eq_dir / "per_direction.csv"), pd.read_csv(axis_dir / "per_direction.csv")
    dir_keys = keys + ["direction"]
    common_frame = pd.DataFrame(list(common), columns=keys)
    eq_dir_rows = eq_dir_rows.merge(common_frame, on=keys, how="inner", validate="many_to_one")
    ax_dir_rows = ax_dir_rows.merge(common_frame, on=keys, how="inner", validate="many_to_one")
    direction = eq_dir_rows[
        dir_keys + ["size_bin", "stratum", "sampling_weight", "o2o_flip", "o2m_flip"]
    ].merge(
        ax_dir_rows[dir_keys + ["o2o_flip", "o2m_flip"]],
        on=dir_keys,
        suffixes=("__equivalent_area_side", "__axis_normalised"),
        validate="one_to_one",
    )
    direction["direction_axis"] = np.where(direction["direction"].isin(["left", "right"]), "horizontal", "vertical")
    audit = {
        "equivalent_common_valid": len(eq_keys),
        "axis_common_valid": len(ax_keys),
        "intersection": len(common),
        "equivalent_only": len(eq_keys - ax_keys),
        "axis_only": len(ax_keys - eq_keys),
        "matched_direction_rows": len(direction),
    }
    return merged, direction, audit


def aggregate_cells(rows: pd.DataFrame, roster: pd.DataFrame, strata: tuple[str, ...], outcome: str) -> pd.DataFrame:
    data = rows.copy()
    columns = []
    for contract in CONTRACTS:
        for branch in BRANCHES:
            y = f"{branch}_{outcome}__{contract}"
            for size in SIZES:
                for level in strata:
                    den, num = f"{contract}|{branch}|{size}|{level}|den", f"{contract}|{branch}|{size}|{level}|num"
                    take = data["size_bin"].eq(size) & data["analysis_stratum"].eq(level)
                    data[den] = data["sampling_weight"] * take
                    data[num] = data["sampling_weight"] * data[y] * take
                    columns.extend([num, den])
    aggregate = data.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    result = roster.merge(aggregate, on=["stratum", "image_id"], how="left", validate="one_to_one")
    result[columns] = result[columns].fillna(0.0)
    return result


def metrics(sums: pd.Series, levels: tuple[str, ...], standardise: bool) -> dict[str, float]:
    out = {}
    target = {}
    if standardise:
        for level in levels:
            target[level] = sum(
                sums[f"equivalent_area_side|o2o|{size}|{level}|den"] for size in SIZES
            )
        total = sum(target.values())
        target = {key: value / total for key, value in target.items()}
    for contract in CONTRACTS:
        branch_gaps = {}
        for branch in BRANCHES:
            for level in levels:
                rates = {}
                for size in SIZES:
                    den = sums[f"{contract}|{branch}|{size}|{level}|den"]
                    rates[size] = sums[f"{contract}|{branch}|{size}|{level}|num"] / den if den > 0 else np.nan
                out[f"{contract}|{branch}|gap|{level}"] = rates["t_8_16"] - rates["s_16_32"]
            rates_all = {}
            for size in SIZES:
                num = sum(sums[f"{contract}|{branch}|{size}|{level}|num"] for level in levels)
                den = sum(sums[f"{contract}|{branch}|{size}|{level}|den"] for level in levels)
                rates_all[size] = num / den
            branch_gaps[branch] = rates_all["t_8_16"] - rates_all["s_16_32"]
            out[f"{contract}|{branch}|gap|overall"] = branch_gaps[branch]
            if standardise:
                standard_rates = {}
                for size in SIZES:
                    standard_rates[size] = sum(
                        target[level]
                        * sums[f"{contract}|{branch}|{size}|{level}|num"]
                        / sums[f"{contract}|{branch}|{size}|{level}|den"]
                        for level in levels
                    )
                out[f"{contract}|{branch}|gap|aspect_standardised"] = (
                    standard_rates["t_8_16"] - standard_rates["s_16_32"]
                )
        out[f"{contract}|paired|gap|overall"] = branch_gaps["o2o"] - branch_gaps["o2m"]
        if standardise:
            out[f"{contract}|paired|gap|aspect_standardised"] = (
                out[f"{contract}|o2o|gap|aspect_standardised"]
                - out[f"{contract}|o2m|gap|aspect_standardised"]
            )
    for key in [key for key in out if key.startswith("axis_normalised|")]:
        eq_key = key.replace("axis_normalised|", "equivalent_area_side|", 1)
        out[f"axis_minus_equivalent|{key.split('|', 1)[1]}"] = out[key] - out[eq_key]
    return out


def multiplicities(roster: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((reps, len(roster)), dtype=np.int16)
    for indices in roster.groupby("stratum", sort=True).indices.values():
        positions = np.asarray(indices, dtype=int)
        result[:, positions] = rng.multinomial(
            len(positions), np.full(len(positions), 1.0 / len(positions)), size=reps
        )
    return result


def evaluate_table(table: pd.DataFrame, levels: tuple[str, ...], standardise: bool, mult: np.ndarray) -> dict:
    value_columns = [column for column in table if "|" in column]
    matrix = table[value_columns].to_numpy(float)
    point = metrics(pd.Series(matrix.sum(axis=0), index=value_columns), levels, standardise)
    boot_sums = mult @ matrix
    samples = {key: np.empty(len(mult)) for key in point}
    for index, row in enumerate(boot_sums):
        current = metrics(pd.Series(row, index=value_columns), levels, standardise)
        for key, value in current.items():
            samples[key][index] = value
    return {
        "point": point,
        "ci_95": {
            key: [float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))]
            for key, values in samples.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--equivalent-area-dir", type=Path, required=True)
    parser.add_argument("--axis-dir", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    roster = pd.read_csv(args.selected_images)[["stratum", "image_id"]].drop_duplicates().sort_values(
        ["stratum", "image_id"]
    ).reset_index(drop=True)
    objects, directions, audit = matched_inputs(args.equivalent_area_dir, args.axis_dir)
    objects["analysis_stratum"] = objects["aspect_bin"]
    directions["analysis_stratum"] = directions["direction_axis"]
    object_table = aggregate_cells(objects, roster, ASPECTS, "fragile")
    direction_table = aggregate_cells(directions, roster, DIR_AXES, "flip")
    mult = multiplicities(roster, args.reps, args.seed)
    payload = {
        "status": "complete",
        "estimand": "matched common-valid IPW branch contrasts under two displacement parameterisations",
        "interpretation": "descriptive stress-contract and composition sensitivity, not a causal effect",
        "bootstrap": "paired stratified image-cluster percentile bootstrap",
        "bootstrap_reps": args.reps,
        "support_audit": audit,
        "object_aspect_analysis": evaluate_table(object_table, ASPECTS, True, mult),
        "direction_axis_analysis": evaluate_table(direction_table, DIR_AXES, False, mult),
        "source_sha256": {
            str(args.equivalent_area_dir / "per_gt.csv"): sha256(args.equivalent_area_dir / "per_gt.csv"),
            str(args.equivalent_area_dir / "per_direction.csv"): sha256(args.equivalent_area_dir / "per_direction.csv"),
            str(args.axis_dir / "per_gt.csv"): sha256(args.axis_dir / "per_gt.csv"),
            str(args.axis_dir / "per_direction.csv"): sha256(args.axis_dir / "per_direction.csv"),
            str(args.selected_images): sha256(args.selected_images),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
