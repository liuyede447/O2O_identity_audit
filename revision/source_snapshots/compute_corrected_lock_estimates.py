"""Recompute the eligibility-lock sensitivity on the base-O2O-defined population.

Inputs are copied frozen CSVs. The estimator is inverse-inclusion weighted and
the bootstrap resamples image clusters within the recorded outcome-blind strata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
SEED = 20260823 + 901  # Frozen lock-analysis bootstrap seed.
REPLICATES = 5000


def rate_from_totals(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("empty effective denominator")
    return numerator / denominator


def estimate(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for arm in ("native", "eligibility_lock"):
        for label, size in (("tiny", "t_8_16"), ("small", "s_16_32")):
            cell = frame.loc[frame["size_bin"].eq(size)]
            result[f"{arm}_{label}"] = rate_from_totals(
                float((cell["sampling_weight"] * cell[f"{arm}_o2o_fragile"]).sum()),
                float(cell["sampling_weight"].sum()),
            )
        result[f"{arm}_gap"] = result[f"{arm}_tiny"] - result[f"{arm}_small"]
    result["reduction_pp"] = result["native_gap"] - result["eligibility_lock_gap"]
    result["reduction_fraction"] = result["reduction_pp"] / result["native_gap"]
    return result


def image_sufficient_statistics(frame: pd.DataFrame, stress: str) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    subset = frame.loc[frame["stress"].eq(stress)].copy()
    images = subset[["image_id", "stratum"]].drop_duplicates().sort_values("image_id")
    image_ids = images["image_id"].tolist()
    strata = images["stratum"].tolist()
    stats: dict[str, np.ndarray] = {}
    for arm in ("native", "eligibility_lock"):
        for label, size in (("tiny", "t_8_16"), ("small", "s_16_32")):
            cell = subset.loc[subset["size_bin"].eq(size), ["image_id", "sampling_weight", f"{arm}_o2o_fragile"]]
            grouped = cell.assign(weighted_event=cell["sampling_weight"] * cell[f"{arm}_o2o_fragile"]).groupby("image_id", sort=False).agg(
                numerator=("weighted_event", "sum"), denominator=("sampling_weight", "sum")
            ).reindex(image_ids, fill_value=0.0)
            stats[f"{arm}_{label}"] = grouped[["numerator", "denominator"]].to_numpy(float)
    return image_ids, strata, stats


def bootstrap(valid: pd.DataFrame) -> pd.DataFrame:
    per_stress = {stress: image_sufficient_statistics(valid, stress) for stress in ("fixed_1px", "equivalent_side")}
    ids = per_stress["fixed_1px"][0]
    strata = per_stress["fixed_1px"][1]
    if per_stress["equivalent_side"][0] != ids or per_stress["equivalent_side"][1] != strata:
        raise RuntimeError("stress contracts do not share the same effective image clusters")
    by_stratum: dict[str, np.ndarray] = {}
    for index, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(index)
    by_stratum = {name: np.asarray(indices, dtype=int) for name, indices in by_stratum.items()}
    rng = np.random.default_rng(SEED)
    records = []
    for replicate in range(REPLICATES):
        counts = np.zeros(len(ids), dtype=float)
        for indices in by_stratum.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            counts += np.bincount(sampled, minlength=len(ids))
        row: dict[str, float | int] = {"replicate": replicate}
        for stress, (_, _, stats) in per_stress.items():
            values: dict[str, float] = {}
            for arm in ("native", "eligibility_lock"):
                for label in ("tiny", "small"):
                    numerator, denominator = (counts[:, None] * stats[f"{arm}_{label}"]).sum(axis=0)
                    values[f"{arm}_{label}"] = rate_from_totals(float(numerator), float(denominator))
                values[f"{arm}_gap"] = values[f"{arm}_tiny"] - values[f"{arm}_small"]
            values["reduction_pp"] = values["native_gap"] - values["eligibility_lock_gap"]
            values["reduction_fraction"] = values["reduction_pp"] / values["native_gap"]
            for name, value in values.items():
                row[f"{stress}_{name}"] = value
        records.append(row)
    return pd.DataFrame.from_records(records)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    lock = pd.read_csv(INPUTS / "eligibility_lock_per_gt.csv")
    fixed = pd.read_csv(INPUTS / "fixed_branch_per_gt.csv")
    gate = fixed[["image_id", "gt_id", "o2o_active"]].copy()
    gate["base_o2o_defined"] = gate["o2o_active"].notna()
    merged = lock.merge(gate[["image_id", "gt_id", "base_o2o_defined"]], on=["image_id", "gt_id"], how="left", validate="many_to_one")
    if merged["base_o2o_defined"].isna().any():
        raise RuntimeError("missing frozen base-state gate")
    valid = merged.loc[merged["base_o2o_defined"]].copy()
    boot = bootstrap(valid)
    boot.to_csv(OUTPUTS / "LOCK_CORRECTED_BOOTSTRAP.csv", index=False)

    output: dict[str, object] = {
        "audit": "LOCK_CORRECTED_ESTIMATES_V1",
        "read_only": True,
        "estimator": "inverse-inclusion weighted object-level fragility; base O2O identity must be defined; 5,000 image-cluster bootstrap replicates resampled within frozen outcome-blind strata",
        "effective_population": {},
        "estimates": {},
    }
    table_rows = []
    for stress in ("fixed_1px", "equivalent_side"):
        frame = valid.loc[valid["stress"].eq(stress)]
        point = estimate(frame)
        prefix = f"{stress}_"
        ci = {name: [float(value) for value in np.percentile(boot[prefix + name], [2.5, 97.5])] for name in point}
        output["effective_population"][stress] = {
            "gt": int(len(frame)),
            "images": int(frame["image_id"].nunique()),
            "tiny_gt": int(frame["size_bin"].eq("t_8_16").sum()),
            "small_gt": int(frame["size_bin"].eq("s_16_32").sum()),
            "base_undefined_excluded": int((merged["stress"].eq(stress) & ~merged["base_o2o_defined"]).sum()),
        }
        output["estimates"][stress] = {"point": point, "ci_95": ci}
        for name, value in point.items():
            table_rows.append({"stress": stress, "estimand": name, "point": value, "ci_low": ci[name][0], "ci_high": ci[name][1]})
    pd.DataFrame(table_rows).to_csv(OUTPUTS / "LOCK_CORRECTED_ESTIMATES.csv", index=False)
    (OUTPUTS / "LOCK_CORRECTED_ESTIMATES.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
