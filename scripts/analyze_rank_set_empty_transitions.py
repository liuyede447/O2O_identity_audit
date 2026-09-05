"""Bootstrap O2M empty-set structural transition rates by size group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ("empty_empty", "empty_to_nonempty", "nonempty_to_empty")
SIZES = ("t_8_16", "s_16_32")


def image_table(rows: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    data = rows.copy()
    columns = []
    for size in SIZES:
        for state in STATES:
            den, num = f"{size}|{state}|den", f"{size}|{state}|num"
            take = data["size_bin"].eq(size)
            data[den] = data["sampling_weight"] * take
            data[num] = data["sampling_weight"] * take * data["set_structural_state"].eq(state)
            columns.extend([num, den])
    agg = data.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    out = roster.merge(agg, on=["stratum", "image_id"], how="left", validate="one_to_one")
    out[columns] = out[columns].fillna(0.0)
    return out


def estimate(frame: pd.DataFrame) -> dict[str, float]:
    out = {}
    for state in STATES:
        rates = {}
        for size in SIZES:
            den = frame[f"{size}|{state}|den"].sum()
            rates[size] = frame[f"{size}|{state}|num"].sum() / den
            out[f"{size}|{state}"] = float(rates[size])
        out[f"tiny_minus_small|{state}"] = float(rates["t_8_16"] - rates["s_16_32"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--equivalent-area", type=Path, required=True)
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
    rng = np.random.default_rng(args.seed)
    groups = [np.asarray(indices, dtype=int) for indices in roster.groupby("stratum", sort=True).indices.values()]
    multiplicities = np.zeros((args.reps, len(roster)), dtype=np.int16)
    for positions in groups:
        multiplicities[:, positions] = rng.multinomial(
            len(positions), np.full(len(positions), 1.0 / len(positions)), size=args.reps
        )
    payload = {"status": "complete", "bootstrap_reps": args.reps, "contracts": {}}
    for name, path in (("fixed_1px", args.fixed), ("equivalent_area_side", args.equivalent_area)):
        table = image_table(pd.read_csv(path), roster)
        value_columns = [column for column in table if "|" in column]
        matrix = table[value_columns].to_numpy(float)
        point = estimate(pd.DataFrame([matrix.sum(axis=0)], columns=value_columns))
        sums = multiplicities @ matrix
        samples = {key: np.empty(args.reps) for key in point}
        for index, row in enumerate(sums):
            current = estimate(pd.DataFrame([row], columns=value_columns))
            for key, value in current.items():
                samples[key][index] = value
        payload["contracts"][name] = {
            "point": point,
            "ci_95": {
                key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
                for key, values in samples.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
