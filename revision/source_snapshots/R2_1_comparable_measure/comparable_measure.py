"""R2.1 comparable O2O/O2M stability measure, four YOLO26s seeds, equivalent-area kappa=0.0625.

Read-only over publication_reproducible. Writes only into this script's folder.

Gate: reproduce Table 1 (seed 0) o2o_8_16=0.09114546, o2m_8_16=0.57925765 from per_gt.csv
using the same common-valid filter, weights and stratified image-cluster bootstrap as
scripts/run_reviewer_killer_controls.py (functions copied verbatim below).

Measures (object level = any change over the object's joined legal directions):
  O2O change            : assigned O2O identity changes. O2O assigns one candidate, so this is
                          simultaneously an assigned-set change and an assigned Top-1 change.
  O2M Top-1 (paper)     : o2m_flip from per_direction.csv (Table 1 definition).
  O2M assigned Top-1    : assigned_set_q_top1_flip from the rank-set file.
  O2M set change        : exact_set_equal == 0 from the rank-set file.
Like-for-like pairs: (O2O vs O2M set change) and (O2O vs O2M assigned Top-1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
OUT = Path(__file__).resolve().parent
REPS = 5000
SEED = 20261120

SEEDS = {
    0: ("results/reviewer_controls_20260829/formal_s0e300_k00625", "runs/20260831T162551_rank_set_normalized_v1/artifact"),
    1: ("runs/20260905_multiseed_equivalent_s1_v1", "runs/20260905_multiseed_rank_set_s1_v1/artifact"),
    2: ("results/reviewer_controls_20260829/ai_s2e300_equiv_k00625", "runs/20260905_multiseed_rank_set_s2_v1/artifact"),
    3: ("results/reviewer_controls_20260829/ai_s3e300_equiv_k00625", "runs/20260905_multiseed_rank_set_s3_v1/artifact"),
}
KEY = ["image_id", "gt_id", "direction"]


# ---- copied verbatim from scripts/run_reviewer_killer_controls.py ----
def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_rate(df: pd.DataFrame, column: str, bucket: str) -> float:
    part = df[df["size_bin"].eq(bucket)]
    return weighted_mean(part[column].to_numpy(float), part["sampling_weight"].to_numpy(float))


def stratified_cluster_samples(df: pd.DataFrame, reps: int, seed: int):
    image_strata = df[["image_id", "stratum"]].drop_duplicates()
    by_stratum = {key: values["image_id"].tolist() for key, values in image_strata.groupby("stratum")}
    row_groups = {key: values.index.to_numpy() for key, values in df.groupby("image_id", sort=False)}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        indices = []
        for images in by_stratum.values():
            chosen = rng.choice(images, size=len(images), replace=True)
            indices.extend(row_groups[image] for image in chosen)
        yield np.concatenate(indices)


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
# ---- end copy ----


MEASURES = ["o2o_change", "o2m_top1_paper", "o2m_assigned_top1", "o2m_set_change"]
BINS = ["t_8_16", "s_16_32"]


def stats(df: pd.DataFrame, level_col_prefix: str = "") -> dict:
    out = {}
    for m in MEASURES:
        col = level_col_prefix + m
        t, s = weighted_rate(df, col, "t_8_16"), weighted_rate(df, col, "s_16_32")
        out[f"{m}|8_16"] = t
        out[f"{m}|16_32"] = s
        out[f"{m}|gap_8_16_minus_16_32"] = t - s
    for b in ["8_16", "16_32", "gap_8_16_minus_16_32"]:
        out[f"set_minus_o2o|{b}"] = out[f"o2m_set_change|{b}"] - out[f"o2o_change|{b}"]
        out[f"top1_minus_o2o|{b}"] = out[f"o2m_assigned_top1|{b}"] - out[f"o2o_change|{b}"]
    return out


def boot(df: pd.DataFrame, fn, seed: int) -> tuple[dict, dict]:
    point = fn(df)
    vals = {k: [] for k in point}
    for idx in stratified_cluster_samples(df, REPS, seed):
        st = fn(df.iloc[idx])
        for k, v in st.items():
            vals[k].append(v)
    return point, {k: percentile_ci(v) for k, v in vals.items()}


def run_seed(seed: int) -> dict:
    a, b = SEEDS[seed]
    gt = pd.read_csv(PR / a / "per_gt.csv")
    pdir = pd.read_csv(PR / a / "per_direction.csv")
    rs = pd.read_csv(PR / b / "per_direction_rank_set.csv")
    checks = {}

    # --- gate (Table 1 definition) ---
    common = gt[gt["common_valid_margin"].eq(1)].copy().reset_index(drop=True)
    gate = {
        "o2o_8_16": weighted_rate(common, "o2o_fragile", "t_8_16"),
        "o2m_8_16": weighted_rate(common, "o2m_fragile", "t_8_16"),
        "n_gt": len(common),
        "n_images": int(common["image_id"].nunique()),
    }
    ref = json.loads((PR / a / "normalized_perturbation_summary.json").read_text(encoding="utf-8"))
    gate["ref_o2o_8_16"] = ref["point"]["o2o_8_16"]
    gate["ref_o2m_8_16"] = ref["point"]["o2m_8_16"]
    gate["ref_n_gt"] = ref["n_gt"]
    gate["pass"] = bool(
        abs(gate["o2o_8_16"] - gate["ref_o2o_8_16"]) < 1e-12
        and abs(gate["o2m_8_16"] - gate["ref_o2m_8_16"]) < 1e-12
        and gate["n_gt"] == gate["ref_n_gt"]
    )

    # --- join checks ---
    for name, d in [("per_direction", pdir), ("rank_set", rs)]:
        checks[f"{name}_rows"] = len(d)
        checks[f"{name}_dup_keys"] = int(d.duplicated(KEY).sum())
    j = pdir.merge(rs[KEY + ["assigned_set_q_top1_flip", "assigned_top1_defined_both", "exact_set_equal",
                             "union_size", "base_set_size", "shift_set_size", "set_structural_state",
                             "sampling_weight", "size_bin"]],
                   on=KEY, how="outer", indicator=True, suffixes=("", "_rs"))
    checks["join_both"] = int((j["_merge"] == "both").sum())
    checks["join_left_only"] = int((j["_merge"] == "left_only").sum())
    checks["join_right_only"] = int((j["_merge"] == "right_only").sum())
    j = j[j["_merge"] == "both"].copy()
    checks["weight_mismatch"] = int((~np.isclose(j["sampling_weight"], j["sampling_weight_rs"])).sum())
    checks["size_bin_mismatch"] = int((j["size_bin"] != j["size_bin_rs"]).sum())
    checks["empty_union_rows"] = int((j["union_size"] == 0).sum())
    checks["empty_union_exact_equal_values"] = j.loc[j["union_size"] == 0, "exact_set_equal"].value_counts(dropna=False).to_dict()
    checks["exact_set_equal_na"] = int(j["exact_set_equal"].isna().sum())
    checks["assigned_top1_defined_both_false"] = int((j["assigned_top1_defined_both"].astype(str).str.lower().isin(["false", "0"])).sum())
    checks["o2o_flip_na"] = int(j["o2o_flip"].isna().sum())
    checks["o2m_flip_na"] = int(j["o2m_flip"].isna().sum())
    checks["set_structural_state_counts"] = j["set_structural_state"].value_counts(dropna=False).to_dict()

    # direction-level indicators (rows where all four are defined)
    j["o2o_change"] = j["o2o_flip"].astype(float)
    j["o2m_top1_paper"] = j["o2m_flip"].astype(float)
    j["o2m_assigned_top1"] = j["assigned_set_q_top1_flip"].astype(float)
    j["o2m_set_change"] = 1.0 - j["exact_set_equal"].astype(float)

    # restrict to the Table 1 common-valid objects
    cv = common[["image_id", "gt_id"]]
    j = j.merge(cv, on=["image_id", "gt_id"], how="inner")
    defined = j[MEASURES].notna().all(axis=1)
    checks["direction_rows_common_valid"] = len(j)
    checks["direction_rows_all_defined"] = int(defined.sum())
    jd = j[defined].copy()

    obj = (jd.groupby(["image_id", "gt_id"], sort=False)
             .agg(**{m: (m, "max") for m in MEASURES},
                  sampling_weight=("sampling_weight", "first"),
                  size_bin=("size_bin", "first"),
                  stratum=("stratum", "first"),
                  n_dir=("direction", "size"))
             .reset_index())
    checks["objects_used"] = len(obj)
    # consistency: object-level O2O over joined directions vs per_gt o2o_fragile
    chk = obj.merge(common[["image_id", "gt_id", "o2o_fragile", "o2m_fragile"]], on=["image_id", "gt_id"])
    checks["obj_o2o_vs_per_gt_mismatch"] = int((chk["o2o_change"] != chk["o2o_fragile"]).sum())
    checks["obj_o2m_vs_per_gt_mismatch"] = int((chk["o2m_top1_paper"] != chk["o2m_fragile"]).sum())

    obj_point, obj_ci = boot(obj.reset_index(drop=True), stats, SEED + seed)
    jd = jd.reset_index(drop=True)
    dir_point, dir_ci = boot(jd, stats, SEED + 100 + seed)

    return {"seed": seed, "gate": gate, "checks": checks,
            "object_level": {"point": obj_point, "ci_95": obj_ci},
            "direction_level": {"point": dir_point, "ci_95": dir_ci}}


def main() -> None:
    results = [run_seed(s) for s in SEEDS]
    summary = {"protocol": "R2.1 comparable O2O/O2M measure, equivalent-area kappa=0.0625",
               "bootstrap_replicates": REPS,
               "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
               "seeds": results}
    for level in ["object_level", "direction_level"]:
        keys = results[0][level]["point"].keys()
        summary[f"{level}_across_seeds"] = {
            k: {"per_seed": [r[level]["point"][k] for r in results],
                "mean": float(np.mean([r[level]["point"][k] for r in results])),
                "sd": float(np.std([r[level]["point"][k] for r in results], ddof=1))}
            for k in keys}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    rows = []
    for r in results:
        for level in ["object_level", "direction_level"]:
            for k, v in r[level]["point"].items():
                m, b = k.split("|")
                lo, hi = r[level]["ci_95"][k]
                rows.append({"seed": r["seed"], "level": level, "measure": m, "bin": b,
                             "point_pp": 100 * v, "ci_lo_pp": 100 * lo, "ci_hi_pp": 100 * hi})
    pd.DataFrame(rows).to_csv(OUT / "comparable_measure_long.csv", index=False)
    for r in results:
        print(f"seed {r['seed']} gate pass={r['gate']['pass']} "
              f"o2o={r['gate']['o2o_8_16']:.8f} (ref {r['gate']['ref_o2o_8_16']:.8f}) "
              f"o2m={r['gate']['o2m_8_16']:.8f} (ref {r['gate']['ref_o2m_8_16']:.8f}) n={r['gate']['n_gt']}")
        print("   checks:", json.dumps(r["checks"], default=str))


if __name__ == "__main__":
    main()
