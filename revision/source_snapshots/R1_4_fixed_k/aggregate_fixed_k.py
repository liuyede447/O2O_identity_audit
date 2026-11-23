"""R1.4: four-seed summary of the fixed-K O2M counterfactual (equivalent-area kappa=0.0625).

K is imposed after native O2M conflict resolution: the K highest-q members of the native
assigned set. An object is usable for K only when base and shifted sets hold >= K members in
every legal direction (k{K}_strict_eligible). Two populations are reported per seed:
  own    : each K on its own usable sample (as in fixed_k_summary.json), with coverage
  common : every K, and native O2O / O2M Top-1, on the objects usable for K = 5, so that
           differences between K are not differences in sample
Bootstrap: stratified image clusters, 5000 replicates (summarise() from the R2.2 audit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(r"G:\revision_20261120\fixed_k")
PR = Path(r"E:\two_paper\beifen\publication_reproducible")
NATIVE = {  # equivalent-area native per_gt (same 300 images, same checkpoint) per seed
    0: PR / "results/reviewer_controls_20260829/formal_s0e300_k00625/per_gt.csv",
    1: PR / "runs/20260905_multiseed_equivalent_s1_v1/per_gt.csv",
    2: PR / "results/reviewer_controls_20260829/ai_s2e300_equiv_k00625/per_gt.csv",
    3: PR / "results/reviewer_controls_20260829/ai_s3e300_equiv_k00625/per_gt.csv",
}
spec = importlib.util.spec_from_file_location("aud", HERE.parent / "R2_2_independent_assigners" / "assigner_audit.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)
KEYS = ["image_id", "gt_id"]
KS = (1, 3, 5)


def coverage(obj: pd.DataFrame, flag: str) -> dict:
    res = {}
    for b in ("t_8_16", "s_16_32"):
        p = obj[obj.size_bin.eq(b)]
        res[b] = float((p.sampling_weight * p[flag]).sum() / p.sampling_weight.sum())
    return res


def main() -> None:
    result, rows = {}, []
    for seed in range(4):
        obj = pd.read_csv(OUT / f"seed{seed}" / "per_gt_fixed_k.csv")
        obj = obj[obj.common_valid_margin.eq(1)].copy()
        r = {"n_common_valid": len(obj), "own": {}, "coverage": {}}
        for k in KS:
            flag = f"k{k}_strict_eligible"
            r["coverage"][k] = coverage(obj, flag)
            sub = obj[obj[flag].eq(1)]
            r["own"][k] = aud.summarise(sub, [f"k{k}_top1_fragile", f"k{k}_set_turnover_any"], 5000, 20260901 + k)
        nat = pd.read_csv(NATIVE[seed], usecols=KEYS + ["o2o_fragile", "o2m_fragile"])
        com = obj[obj.k5_strict_eligible.eq(1)].merge(nat, on=KEYS, how="left", validate="one_to_one")
        assert com[["o2o_fragile", "o2m_fragile"]].notna().all().all()
        cols = ["o2o_fragile", "o2m_fragile"] + [f"k{k}_{m}" for k in KS for m in ("top1_fragile", "set_turnover_any")]
        r["common_k5"] = aud.summarise(com, cols, 5000, 20260911)
        result[seed] = r
        for pop, d in [("own_k%d" % k, r["own"][k]) for k in KS] + [("common_k5", r["common_k5"])]:
            for key, v in d["point"].items():
                if key.endswith("gap"):
                    ci = d["ci_95"][key]
                    rows.append({"seed": seed, "population": pop, "n_gt": d["n_gt"], "measure": key[:-4],
                                 "gap_pp": 100 * v, "ci_low": 100 * ci[0], "ci_high": 100 * ci[1],
                                 "rate_8_16": 100 * d["point"][key[:-4] + "|8_16"],
                                 "rate_16_32": 100 * d["point"][key[:-4] + "|16_32"]})
        print("seed", seed, "done", flush=True)
    # Binding truncation: direction rows where base AND shifted native sets hold MORE than K
    # members, so the K-truncation actually removes members. Where both sets hold exactly K,
    # top-K is the whole native set and turnover is near zero by construction.
    binding = {}
    for seed in range(4):
        d = pd.read_csv(OUT / f"seed{seed}" / "per_direction_fixed_k.csv")
        d = d[d.common_valid_margin.eq(1)].reset_index(drop=True)
        binding[seed] = {}
        for k in (3, 5):
            e = d[d[f"k{k}_eligible"].eq(1)]
            exact = e.base_native_assigned_count.eq(k) & e.shift_native_assigned_count.eq(k)
            share = {b: float(e[e.size_bin.eq(b)].pipe(lambda x: (x.sampling_weight * exact[x.index]).sum() / x.sampling_weight.sum()))
                     for b in ("t_8_16", "s_16_32")}
            bnd = e[e.base_native_assigned_count.gt(k) & e.shift_native_assigned_count.gt(k)]
            # native O2O identity flip on exactly the same direction rows
            nat = pd.read_csv(NATIVE[seed].with_name("per_direction.csv"), usecols=KEYS + ["direction", "o2o_flip"])
            bnd = bnd.merge(nat, on=KEYS + ["direction"], how="left", validate="one_to_one")
            assert bnd["o2o_flip"].notna().all()
            s = aud.summarise(bnd.reset_index(drop=True), [f"k{k}_set_turnover", "o2o_flip"], 5000, 20260921 + k)
            binding[seed][k] = {"share_exactly_k": share, "binding": s}
            for m in (f"k{k}_set_turnover", "o2o_flip"):
                key = f"{m}|gap"
                rows.append({"seed": seed, "population": f"direction_binding_k{k}", "n_gt": s["n_gt"], "measure": m,
                             "gap_pp": 100 * s["point"][key], "ci_low": 100 * s["ci_95"][key][0], "ci_high": 100 * s["ci_95"][key][1],
                             "rate_8_16": 100 * s["point"][f"{m}|8_16"], "rate_16_32": 100 * s["point"][f"{m}|16_32"]})
        result[seed]["binding_direction_level"] = binding[seed]
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "fixed_k_by_seed.csv", index=False)
    (HERE / "summary.json").write_text(json.dumps({str(k): v for k, v in result.items()}, indent=1) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
    for seed in range(4):
        print(seed, {k: {b: round(100 * c, 1) for b, c in v.items()} for k, v in result[seed]["coverage"].items()})


if __name__ == "__main__":
    main()
