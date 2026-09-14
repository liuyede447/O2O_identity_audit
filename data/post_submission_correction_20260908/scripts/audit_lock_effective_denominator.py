"""Audit eligibility-lock denominators using copied, frozen row-level CSVs only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"


def weighted_rate(frame: pd.DataFrame, column: str, size_bin: str) -> float:
    subset = frame.loc[frame["size_bin"].eq(size_bin)]
    return float(np.average(subset[column].astype(float), weights=subset["sampling_weight"].astype(float)))


def arm_summary(frame: pd.DataFrame, stress: str, arm: str) -> dict[str, float]:
    subset = frame.loc[frame["stress"].eq(stress)]
    tiny = weighted_rate(subset, f"{arm}_o2o_fragile", "t_8_16")
    small = weighted_rate(subset, f"{arm}_o2o_fragile", "s_16_32")
    return {"tiny_rate": tiny, "small_rate": small, "tiny_minus_small": tiny - small}


def bootstrap_ci(frame: pd.DataFrame, stress: str, arm: str, replicates: int = 5000) -> list[float]:
    """Mirror the frozen script's cluster-within-stratum resampling for one contrast."""
    subset = frame.loc[frame["stress"].eq(stress)].copy()
    image_strata = subset[["image_id", "stratum"]].drop_duplicates()
    groups = {name: part["image_id"].to_numpy() for name, part in image_strata.groupby("stratum")}
    rows_by_image = {name: part.index.to_numpy() for name, part in subset.groupby("image_id", sort=False)}
    rng = np.random.default_rng(20260823 + 901)
    estimates = []
    for _ in range(replicates):
        indices: list[int] = []
        for images in groups.values():
            for image in rng.choice(images, size=len(images), replace=True):
                indices.extend(rows_by_image[image])
        estimates.append(arm_summary(subset.loc[np.asarray(indices)], stress, arm)["tiny_minus_small"])
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    lock = pd.read_csv(INPUTS / "eligibility_lock_per_gt.csv")
    fixed = pd.read_csv(INPUTS / "fixed_branch_per_gt.csv")

    key = ["image_id", "gt_id"]
    source = fixed[key + ["o2o_active"]].copy()
    source["base_o2o_defined"] = source["o2o_active"].notna()
    merged = lock.merge(source[key + ["base_o2o_defined"]], on=key, how="left", validate="many_to_one")
    if merged["base_o2o_defined"].isna().any():
        raise RuntimeError("could not map every lock row to its frozen base-state row")

    rows = []
    summaries: dict[str, dict[str, object]] = {}
    for stress in ("fixed_1px", "equivalent_side"):
        full = merged.loc[merged["stress"].eq(stress)].copy()
        valid = full.loc[full["base_o2o_defined"]].copy()
        undefined = full.loc[~full["base_o2o_defined"]].copy()
        summaries[stress] = {
            "input_rows": int(len(full)),
            "base_o2o_defined_rows": int(len(valid)),
            "base_o2o_undefined_rows": int(len(undefined)),
            "undefined_by_size_bin": undefined.groupby("size_bin").size().to_dict(),
            "published_like_unfiltered": {
                "native": arm_summary(full, stress, "native"),
                "eligibility_lock": arm_summary(full, stress, "eligibility_lock"),
            },
            "base_defined_only": {
                "native": arm_summary(valid, stress, "native"),
                "eligibility_lock": arm_summary(valid, stress, "eligibility_lock"),
            },
            "base_defined_only_ci_95": {
                "native_tiny_minus_small": bootstrap_ci(valid, stress, "native"),
                "eligibility_lock_tiny_minus_small": bootstrap_ci(valid, stress, "eligibility_lock"),
            },
        }
        for population, frame in (("all_input_rows", full), ("base_o2o_defined_only", valid), ("base_o2o_undefined_only", undefined)):
            for size_bin in ("t_8_16", "s_16_32"):
                cell = frame.loc[frame["size_bin"].eq(size_bin)]
                if cell.empty:
                    continue
                rows.append({
                    "stress": stress,
                    "population": population,
                    "size_bin": size_bin,
                    "n": len(cell),
                    "weighted_native_o2o_fragility": weighted_rate(cell, "native_o2o_fragile", size_bin),
                    "weighted_locked_o2o_fragility": weighted_rate(cell, "eligibility_lock_o2o_fragile", size_bin),
                    "raw_native_events": int(cell["native_o2o_fragile"].sum()),
                    "raw_locked_events": int(cell["eligibility_lock_o2o_fragile"].sum()),
                })

    result = {
        "audit": "LOCK_EFFECTIVE_DENOMINATOR_AUDIT_V1",
        "read_only": True,
        "finding": "The original lock output contains every sampled in-scope row; the source script does not filter base O2O-unavailable rows before calculating fragility.",
        "interpretation": "This audit establishes a denominator defect in the derived lock summary. It does not alter submitted files or establish the corrected bootstrap interval.",
        "summaries": summaries,
    }
    pd.DataFrame(rows).to_csv(OUTPUTS / "lock_effective_denominator_cells.csv", index=False)
    (OUTPUTS / "lock_effective_denominator_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Lock effective-denominator audit", "", "Status: **finding requiring corrected derived analysis**.", ""]
    for stress, values in summaries.items():
        corrected = values["base_defined_only"]
        lines += [
            f"## {stress}", "",
            f"- Input rows: {values['input_rows']}",
            f"- Base-O2O-defined rows: {values['base_o2o_defined_rows']}",
            f"- Base-O2O-undefined rows: {values['base_o2o_undefined_rows']} ({values['undefined_by_size_bin']})",
            f"- Base-defined native tiny-minus-small: {corrected['native']['tiny_minus_small'] * 100:.2f} pp",
            f"- Base-defined locked tiny-minus-small: {corrected['eligibility_lock']['tiny_minus_small'] * 100:.2f} pp",
            "",
        ]
    (OUTPUTS / "LOCK_EFFECTIVE_DENOMINATOR_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
