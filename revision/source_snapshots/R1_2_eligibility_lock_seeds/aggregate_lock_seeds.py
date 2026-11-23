"""R1.2: corrected (base-O2O-defined) eligibility-lock estimates for YOLO26s seeds 0-3.

Reuses estimate() and bootstrap() from the frozen post-submission correction script
unchanged (same SEED, same strata resampling). Only the inputs differ per seed:
  lock per_gt  : eligibility-lock replay output for that seed
  base gate    : o2o_active.notna() from that seed's normalized-perturbation per_gt.csv
Seed 0 uses the frozen inputs and must reproduce LOCK_CORRECTED_ESTIMATES.json exactly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BEIFEN = Path(r"E:\two_paper\beifen")
PR = BEIFEN / "publication_reproducible"
CORR = BEIFEN / "修复" / "post_submission_audit_20260908"
LOCK_OUT = Path(r"G:\revision_20261120\eligibility_lock")
HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("corr", CORR / "scripts" / "compute_corrected_lock_estimates.py")
corr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corr)

SEEDS = {
    0: (CORR / "inputs" / "eligibility_lock_per_gt.csv", CORR / "inputs" / "fixed_branch_per_gt.csv"),
    1: (LOCK_OUT / "seed1" / "per_gt.csv", PR / "runs" / "20260905_multiseed_equivalent_s1_v1" / "per_gt.csv"),
    2: (LOCK_OUT / "seed2" / "per_gt.csv", PR / "results" / "reviewer_controls_20260829" / "ai_s2e300_equiv_k00625" / "per_gt.csv"),
    3: (LOCK_OUT / "seed3" / "per_gt.csv", PR / "results" / "reviewer_controls_20260829" / "ai_s3e300_equiv_k00625" / "per_gt.csv"),
}
STRESSES = ("fixed_1px", "equivalent_side")


def run_seed(seed: int) -> dict:
    lock_path, gate_path = SEEDS[seed]
    lock = pd.read_csv(lock_path)
    gate = pd.read_csv(gate_path, usecols=["image_id", "gt_id", "o2o_active"])
    if seed != 0:
        # GTs present only in fixed_1px rows: base state recomputed by base_gate_fixed_only.py.
        extra = pd.read_csv(HERE / "base_gate_fixed_only.csv")
        extra = extra.loc[extra["seed"].eq(seed), ["image_id", "gt_id", "o2o_active"]]
        gate = pd.concat([gate, extra], ignore_index=True)
    gate = gate.drop_duplicates(["image_id", "gt_id"])
    gate["base_o2o_defined"] = gate["o2o_active"].notna()
    merged = lock.merge(gate[["image_id", "gt_id", "base_o2o_defined"]], on=["image_id", "gt_id"],
                        how="left", validate="many_to_one")
    if merged["base_o2o_defined"].isna().any():
        raise RuntimeError(f"seed {seed}: missing base gate for {int(merged['base_o2o_defined'].isna().sum())} rows")
    valid = merged.loc[merged["base_o2o_defined"].astype(bool)].copy()
    for arm in ("native", "eligibility_lock"):
        n_nan = int(valid[f"{arm}_o2o_fragile"].isna().sum())
        if n_nan:
            raise RuntimeError(f"seed {seed}: {n_nan} NaN {arm}_o2o_fragile after base gate")
    boot = corr.bootstrap(valid)
    out = {"lock_input": str(lock_path), "gate_input": str(gate_path), "population": {}, "estimates": {}}
    for stress in STRESSES:
        frame = valid.loc[valid["stress"].eq(stress)]
        point = corr.estimate(frame)
        ci = {k: [float(v) for v in np.percentile(boot[f"{stress}_{k}"], [2.5, 97.5])] for k in point}
        out["population"][stress] = {
            "gt": int(len(frame)), "images": int(frame["image_id"].nunique()),
            "tiny_gt": int(frame["size_bin"].eq("t_8_16").sum()),
            "small_gt": int(frame["size_bin"].eq("s_16_32").sum()),
            "base_undefined_excluded": int((merged["stress"].eq(stress) & ~merged["base_o2o_defined"].astype(bool)).sum()),
        }
        out["estimates"][stress] = {"point": point, "ci_95": ci}
    return out


def main() -> None:
    frozen = json.loads((CORR / "outputs" / "LOCK_CORRECTED_ESTIMATES.json").read_text(encoding="utf-8"))
    results = {}
    for seed in SEEDS:
        results[seed] = run_seed(seed)
        print(f"seed {seed} done", flush=True)
    # Gate: seed 0 must equal the frozen corrected estimates.
    for stress in STRESSES:
        for k, v in frozen["estimates"][stress]["point"].items():
            got = results[0]["estimates"][stress]["point"][k]
            if abs(got - v) > 1e-12:
                raise RuntimeError(f"seed0 gate failed {stress}/{k}: {got} vs {v}")
        for k, v in frozen["estimates"][stress]["ci_95"].items():
            got = results[0]["estimates"][stress]["ci_95"][k]
            if max(abs(a - b) for a, b in zip(got, v)) > 1e-12:
                raise RuntimeError(f"seed0 CI gate failed {stress}/{k}")
    rows = []
    for seed, r in results.items():
        for stress in STRESSES:
            p, ci = r["estimates"][stress]["point"], r["estimates"][stress]["ci_95"]
            rows.append({"seed": seed, "stress": stress, "gt": r["population"][stress]["gt"],
                         **{k: p[k] for k in ("native_tiny", "native_small", "native_gap",
                                              "eligibility_lock_tiny", "eligibility_lock_small",
                                              "eligibility_lock_gap", "reduction_pp", "reduction_fraction")},
                         "native_gap_ci": ci["native_gap"], "lock_gap_ci": ci["eligibility_lock_gap"],
                         "reduction_pp_ci": ci["reduction_pp"]})
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "lock_by_seed.csv", index=False)
    summ = {"gate_seed0_reproduces_frozen": True, "bootstrap_seed": corr.SEED, "replicates": corr.REPLICATES,
            "seeds": {str(k): v for k, v in results.items()}}
    for stress in STRESSES:
        sub = table[table["stress"].eq(stress)]
        summ[f"across_seed_{stress}"] = {c: {"mean": float(sub[c].mean()), "sd": float(sub[c].std(ddof=1)),
                                             "min": float(sub[c].min()), "max": float(sub[c].max())}
                                         for c in ("native_gap", "eligibility_lock_gap", "reduction_pp", "reduction_fraction")}
    (HERE / "summary.json").write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table[["seed", "stress", "gt", "native_gap", "eligibility_lock_gap", "reduction_pp", "reduction_fraction"]].round(4))


if __name__ == "__main__":
    sys.exit(main())
