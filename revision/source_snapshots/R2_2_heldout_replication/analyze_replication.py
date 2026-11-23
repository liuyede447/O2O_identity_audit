"""Primary analysis for the pre-specified replication (and its discovery rehearsal).

Inputs: one eligibility-lock replay output directory (per_gt.csv, selected_images.csv) and a
base-gate CSV (image_id, gt_id, o2o_active) computed for the same checkpoint and images.

Estimator: the frozen corrected lock estimator (inverse-inclusion weighted object-level O2O
fragility, 8-16 px minus 16-32 px), restricted to GTs whose base O2O identity is defined.
The replay script's own summary.json is NOT the reported estimate.

Bootstrap (5,000 replicates, frozen lock seed 20260823+901): stratified image-cluster resampling.
Cluster index = sorted union of images with at least one gated row under either stress; an image
missing under one stress contributes zero to that stress. Strata come from the selection manifest.
With no single-stress images this reproduces corr.bootstrap() exactly (same RNG stream).
Images in the selection with no gated row are listed and contribute nothing (as in the published
estimator). Replicates with an empty denominator are counted and excluded from the percentile.

Decision (fixed 1 px): R1 native-gap lower bound > 0; R2 reduction lower bound > 0.
S1: fixed-minus-equivalent native O2O gap on GTs gated under both stresses (same resampling counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260823 + 901
REPLICATES = 5000
STRESSES = ("fixed_1px", "equivalent_side")
ARMS = ("native", "eligibility_lock")
SIZES = (("tiny", "t_8_16"), ("small", "s_16_32"))
KEY = ["image_id", "gt_id", "stress"]
RATE_KEYS = ("native_tiny", "native_small", "eligibility_lock_tiny", "eligibility_lock_small", "native_gap",
             "eligibility_lock_gap", "reduction_pp", "tiny_drop", "small_drop")


class AnalysisError(RuntimeError):
    pass


def load(lock_dir: Path, gate_csv: Path, expected_selection: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per_gt = pd.read_csv(lock_dir / "per_gt.csv")
    sel = pd.read_csv(lock_dir / "selected_images.csv")
    gate = pd.read_csv(gate_csv, usecols=["image_id", "gt_id", "o2o_active"])
    return validate(per_gt, sel, gate, expected_selection)


def validate(per_gt: pd.DataFrame, sel: pd.DataFrame, gate: pd.DataFrame, expected_selection: Path | None):
    info: dict = {}
    if expected_selection is not None:
        exp = pd.read_csv(expected_selection)
        a = sel[["dataset_index", "image_id", "stratum"]].astype(str).sort_values("image_id").reset_index(drop=True)
        b = exp[["dataset_index", "image_id", "stratum"]].astype(str).sort_values("image_id").reset_index(drop=True)
        w_ok = np.allclose(sel.sort_values("image_id")["sampling_weight"].to_numpy(), exp.sort_values("image_id")["sampling_weight"].to_numpy(), rtol=0, atol=1e-9)
        if not a.equals(b) or not w_ok:
            raise AnalysisError("selection differs from the preregistered list: run invalid")
        info["selection_matches_preregistered"] = True
    for name, frame, key in (("per_gt", per_gt, KEY), ("gate", gate, ["image_id", "gt_id"]), ("selection", sel, ["image_id"])):
        if frame.duplicated(key).any():
            raise AnalysisError(f"duplicate keys in {name}: {int(frame.duplicated(key).sum())}")
    if not set(per_gt["image_id"]).issubset(set(sel["image_id"])):
        raise AnalysisError("per_gt contains images outside the selection")
    m = per_gt.merge(gate, on=["image_id", "gt_id"], how="left", validate="many_to_one", indicator=True)
    if (m["_merge"] != "both").any():
        raise AnalysisError(f"{int((m['_merge'] != 'both').sum())} per_gt rows lack a base-gate row")
    m = m.drop(columns="_merge")
    strat = sel.set_index("image_id")["stratum"]
    if (m["stratum"] != m["image_id"].map(strat)).any():
        raise AnalysisError("per_gt stratum disagrees with the selection manifest")
    m["base_defined"] = m["o2o_active"].notna()
    v = m.loc[m["base_defined"]].copy()
    for arm in ARMS:
        col = f"{arm}_o2o_fragile"
        if v[col].isna().any() or not v[col].isin([0, 1]).all():
            raise AnalysisError(f"{col} not binary on gated rows")
    if not v["size_bin"].isin(["t_8_16", "s_16_32"]).all():
        raise AnalysisError("unexpected size_bin")
    info["rows"] = {s: int(per_gt["stress"].eq(s).sum()) for s in STRESSES}
    info["gated_rows"] = {s: int(v["stress"].eq(s).sum()) for s in STRESSES}
    info["base_undefined_excluded"] = {s: int((m["stress"].eq(s) & ~m["base_defined"]).sum()) for s in STRESSES}
    per_gt_images = set(v["image_id"])
    info["selected_images"] = int(len(sel))
    info["zero_row_images"] = sorted(set(sel["image_id"]) - per_gt_images)
    by_stress = {s: set(v.loc[v["stress"].eq(s), "image_id"]) for s in STRESSES}
    info["single_stress_images"] = sorted(by_stress[STRESSES[0]] ^ by_stress[STRESSES[1]])
    keys = v.groupby(["image_id", "gt_id"])["stress"].nunique()
    info["single_stress_gts"] = int(keys.eq(1).sum())
    return v, sel, info


def suff(frame: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """Array [cell, image, (num, den)] for the 4 arm x size cells."""
    out = []
    for arm in ARMS:
        for _, size in SIZES:
            c = frame.loc[frame["size_bin"].eq(size)]
            g = pd.DataFrame({"image_id": c["image_id"], "num": c["sampling_weight"] * c[f"{arm}_o2o_fragile"],
                              "den": c["sampling_weight"]}).groupby("image_id", sort=False).sum()
            out.append(g.reindex(ids, fill_value=0.0)[["num", "den"]].to_numpy(float))
    return np.stack(out)


def rates(arr: np.ndarray, counts: np.ndarray | None) -> dict[str, float]:
    tot = arr.sum(1) if counts is None else np.einsum("i,cij->cj", counts, arr)
    if (tot[:, 1] <= 0).any():
        raise ZeroDivisionError
    r = tot[:, 0] / tot[:, 1]
    v = {"native_tiny": r[0], "native_small": r[1], "eligibility_lock_tiny": r[2], "eligibility_lock_small": r[3]}
    v["native_gap"] = v["native_tiny"] - v["native_small"]
    v["eligibility_lock_gap"] = v["eligibility_lock_tiny"] - v["eligibility_lock_small"]
    v["reduction_pp"] = v["native_gap"] - v["eligibility_lock_gap"]
    v["tiny_drop"] = v["native_tiny"] - v["eligibility_lock_tiny"]
    v["small_drop"] = v["native_small"] - v["eligibility_lock_small"]
    return {k: float(x) for k, x in v.items()}


def analyze(v: pd.DataFrame, sel: pd.DataFrame, replicates: int = REPLICATES) -> dict:
    ids = sorted(set(v["image_id"]))
    strat = sel.set_index("image_id")["stratum"]
    strata = [strat[i] for i in ids]
    arrays = {s: suff(v.loc[v["stress"].eq(s)], ids) for s in STRESSES}
    both = v.groupby(["image_id", "gt_id"])["stress"].transform("nunique").eq(2)
    arrays_s1 = {s: suff(v.loc[both & v["stress"].eq(s)], ids) for s in STRESSES}

    keys_fixed = [f"fixed_1px|{k}" for k in RATE_KEYS]
    keys_eq = [f"equivalent_side|{k}" for k in RATE_KEYS]

    def all_values(counts):
        """Primary (fixed 1 px) and each secondary block are computed independently; an undefined
        block becomes NaN without affecting the others."""
        out = {}
        try:
            out.update({f"fixed_1px|{k}": x for k, x in rates(arrays["fixed_1px"], counts).items()})
        except ZeroDivisionError:
            out.update({k: np.nan for k in keys_fixed})
        try:
            out.update({f"equivalent_side|{k}": x for k, x in rates(arrays["equivalent_side"], counts).items()})
        except ZeroDivisionError:
            out.update({k: np.nan for k in keys_eq})
        try:
            f, e = rates(arrays_s1["fixed_1px"], counts), rates(arrays_s1["equivalent_side"], counts)
            out["S1|fixed_minus_equivalent_native_gap"] = f["native_gap"] - e["native_gap"]
        except ZeroDivisionError:
            out["S1|fixed_minus_equivalent_native_gap"] = np.nan
        return out

    point = all_values(None)
    if any(np.isnan(point[k]) for k in keys_fixed):
        raise AnalysisError("structural failure: primary (fixed 1 px) point estimate undefined (empty size cell)")
    by_stratum: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by_stratum.setdefault(s, []).append(i)
    by_stratum = {k: np.asarray(x, dtype=int) for k, x in by_stratum.items()}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(replicates):
        counts = np.zeros(len(ids), dtype=float)
        for idx in by_stratum.values():
            counts += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(ids))
        boot.append(all_values(counts))
    b = pd.DataFrame(boot)
    undefined = {k: int(b[k].isna().sum()) for k in point}
    ci = {k: ([float(x) for x in np.nanpercentile(b[k], [2.5, 97.5])] if undefined[k] < len(b) else [np.nan, np.nan])
          for k in point}
    n = {s: {"gt": int(v["stress"].eq(s).sum()), "tiny": int((v["stress"].eq(s) & v["size_bin"].eq("t_8_16")).sum()),
             "small": int((v["stress"].eq(s) & v["size_bin"].eq("s_16_32")).sum()),
             "images": int(v.loc[v["stress"].eq(s), "image_id"].nunique())} for s in STRESSES}
    n["S1_common_gt"] = int(both.sum() // 2)
    n["clusters"] = len(ids)
    n["clusters_by_stratum"] = {k: int(len(x)) for k, x in by_stratum.items()}
    return {"point": point, "ci_95": ci, "counts": n, "undefined_replicates": undefined, "replicates": replicates}


def decide(res: dict) -> dict:
    lo_native = res["ci_95"]["fixed_1px|native_gap"][0]
    lo_red = res["ci_95"]["fixed_1px|reduction_pp"][0]
    r1, r2 = lo_native > 0, lo_red > 0
    if r1 and r2:
        verdict = "replicated: native tiny-object disadvantage and its attenuation under eligibility locking"
    elif r2:
        verdict = "attenuation supported, but the native disadvantage was not clearly reproduced"
    else:
        verdict = "not replicated under the pre-specified rule"
    return {"R1_native_gap_lower_gt_0": r1, "R2_reduction_lower_gt_0": r2, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock-dir", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--expected-selection", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    v, sel, info = load(a.lock_dir, a.gate, a.expected_selection)
    res = analyze(v, sel)
    res["decision"] = decide(res)
    res["validation"] = info
    res["inputs"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                     (str(a.lock_dir / "per_gt.csv"), str(a.lock_dir / "selected_images.csv"), str(a.gate))}
    # The primary result is written before any secondary analysis is attempted.
    a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    # Secondary, descriptive only: Table-2 (competition-defined) support, when the gate carries the flag.
    flags = pd.read_csv(a.gate)
    if "common_valid_margin" in flags.columns:
        try:
            t2 = v.merge(flags[["image_id", "gt_id", "common_valid_margin"]], on=["image_id", "gt_id"], how="left", validate="many_to_one")
            t2 = t2.loc[t2["common_valid_margin"].eq(1)].drop(columns="common_valid_margin")
            sec = analyze(t2, sel)
            res["secondary_table2_support"] = {"point": sec["point"], "ci_95": sec["ci_95"], "counts": sec["counts"],
                                               "undefined_replicates": sec["undefined_replicates"]}
        except Exception as e:  # reported, never blocks the primary result
            res["secondary_table2_support"] = {"unavailable": f"{type(e).__name__}: {e}"}
        a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    show = {k: f"{100 * res['point'][k]:.2f} [{100 * res['ci_95'][k][0]:.2f}, {100 * res['ci_95'][k][1]:.2f}]" for k in res["point"]}
    print(json.dumps({"counts": res["counts"], "estimates_pp": show, "decision": res["decision"],
                      "zero_row_images": info["zero_row_images"], "single_stress_images": info["single_stress_images"],
                      "single_stress_gts": info["single_stress_gts"], "undefined_replicates": res["undefined_replicates"]}, indent=1))


if __name__ == "__main__":
    main()
