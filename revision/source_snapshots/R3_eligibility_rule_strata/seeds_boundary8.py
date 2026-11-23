"""Discovery sample, seeds 0-3 (epoch 280): short-side strata of the fixed 1 px eligibility lock.

Gate and inputs follow R1_2_eligibility_lock_seeds/aggregate_lock_seeds.py run_seed(). Box sizes are
seed-independent and come from the seed-0 fixed branch output; they are checked against each seed's
own size-scaled per_gt. Strata, estimator and bootstrap follow sensitivity_rule_strata.py.
Recombination check: weighted strata reproduce each seed's published native/locked tiny rates.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sensitivity_rule_strata as sens

HERE = Path(__file__).resolve().parent
SEEDS_DIR = HERE.parent / "R1_2_eligibility_lock_seeds"
spec = importlib.util.spec_from_file_location("agg", SEEDS_DIR / "aggregate_lock_seeds.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)
SIZES = pd.read_csv(agg.CORR / "inputs" / "fixed_branch_per_gt.csv", usecols=["image_id", "gt_id", "width", "height"])


def valid_fixed(seed: int) -> pd.DataFrame:
    lock_path, gate_path = agg.SEEDS[seed]
    lock = pd.read_csv(lock_path)
    gate = pd.read_csv(gate_path, usecols=["image_id", "gt_id", "o2o_active"])
    if seed != 0:
        extra = pd.read_csv(SEEDS_DIR / "base_gate_fixed_only.csv")
        gate = pd.concat([gate, extra.loc[extra["seed"].eq(seed), ["image_id", "gt_id", "o2o_active"]]], ignore_index=True)
    gate = gate.drop_duplicates(["image_id", "gt_id"])
    m = lock.merge(gate, on=["image_id", "gt_id"], how="left", validate="many_to_one")
    m = m[m["o2o_active"].notna() & m["stress"].eq("fixed_1px")]
    m = m.merge(SIZES, on=["image_id", "gt_id"], how="left", validate="one_to_one")
    if m["width"].isna().any():
        raise RuntimeError(f"seed {seed}: {int(m['width'].isna().sum())} rows lack size")
    own = pd.read_csv(gate_path, usecols=["image_id", "gt_id", "width", "height"]).drop_duplicates(["image_id", "gt_id"])
    chk = m.merge(own, on=["image_id", "gt_id"], suffixes=("", "_own"))
    err = float(np.abs(chk[["width", "height"]].to_numpy() - chk[["width_own", "height_own"]].to_numpy()).max())
    if err > 1e-3:
        raise RuntimeError(f"seed {seed}: size mismatch {err}")
    return m


def main() -> None:
    pub = pd.read_csv(SEEDS_DIR / "lock_by_seed.csv")
    res = {"note": "post hoc exploratory; discovery sample, epoch-280 checkpoints, fixed 1 px"}
    for seed in range(4):
        m = valid_fixed(seed)
        sel = m.drop_duplicates("image_id")[["image_id", "stratum"]]
        r = sens.run_groups(m, sel, sens.boundary_labels, ("rule_clear", "boundary_8", "norule_clear"))
        # recombination: all tiny bands vs published tiny rates
        t = m[m["size_bin"].eq("t_8_16")]
        p = pub[(pub["seed"] == seed) & (pub["stress"] == "fixed_1px")].iloc[0]
        rec = {arm: float((t["sampling_weight"] * t[f"{arm}_o2o_fragile"]).sum() / t["sampling_weight"].sum())
               for arm in ("native", "eligibility_lock")}
        r["recombination_abs_diff"] = {arm: abs(rec[arm] - float(p[f"{arm}_tiny"])) for arm in rec}
        lab = sens.boundary_labels(m)
        tw = t["sampling_weight"].sum()
        r["contribution_to_tiny_drop_pp"] = {
            g: float(100 * (m.loc[lab == g, "sampling_weight"] * (m.loc[lab == g, "native_o2o_fragile"]
                     - m.loc[lab == g, "eligibility_lock_o2o_fragile"])).sum() / tw)
            for g in ("rule_clear", "boundary_8", "norule_clear")}
        r["tiny_weight_share"] = {g: float(100 * m.loc[lab == g, "sampling_weight"].sum() / tw)
                                  for g in ("rule_clear", "boundary_8", "norule_clear")}
        res[f"seed{seed}"] = r
    (HERE / "seeds_boundary8.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for seed in range(4):
        r = res[f"seed{seed}"]
        print(f"== seed {seed}  n={r['n_gt']}  recomb diff={r['recombination_abs_diff']}")
        print("   weight share:", {k: round(v, 1) for k, v in r["tiny_weight_share"].items()},
              " drop contribution:", {k: round(v, 2) for k, v in r["contribution_to_tiny_drop_pp"].items()})
        for k, v in r["point"].items():
            print(f"   {k:26s} {v:6.2f} [{r['ci_95'][k][0]:6.2f}, {r['ci_95'][k][1]:6.2f}]")


if __name__ == "__main__":
    main()
