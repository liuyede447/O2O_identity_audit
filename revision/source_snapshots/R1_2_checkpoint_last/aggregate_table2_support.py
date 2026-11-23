"""Epoch-280 eligibility-lock estimates on the Table-2 (competition-defined) support, seeds 0-3.

Support flag common_valid_margin is a base-state property (verified stress-independent on seeds 0-1:
0 disagreements). Sources: seed 0 frozen fixed per_gt; seed 1 fixed per_gt; seeds 2-3 equivalent-area
per_gt plus support_flag_fixed_only.csv for the two fixed-only GTs.
Estimator: frozen corrected lock estimator (corr.estimate / rate_from_totals). Bootstrap: stratified
image-cluster resampling, frozen lock seed, 5,000 replicates; cluster index is the union of images
with any row under either stress (missing cells contribute zero).
Gate: seed-0 native fixed rates equal Table 2 (22.14 / 7.77 on 6,378 GTs).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "R1_2_eligibility_lock_seeds"
PR = Path(r"E:\two_paper\beifen\publication_reproducible")
CORR_IN = Path(r"E:\two_paper\beifen\修复\post_submission_audit_20260908\inputs")
spec = importlib.util.spec_from_file_location("prev", PREV / "aggregate_lock_seeds.py")
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
corr = prev.corr
FLAGS = {
    0: CORR_IN / "fixed_branch_per_gt.csv",
    1: PR / "runs" / "20260905_multiseed_fixed_s1_v1" / "per_gt.csv",
    2: PR / "results" / "reviewer_controls_20260829" / "ai_s2e300_equiv_k00625" / "per_gt.csv",
    3: PR / "results" / "reviewer_controls_20260829" / "ai_s3e300_equiv_k00625" / "per_gt.csv",
}
STRESSES = ("fixed_1px", "equivalent_side")
CELLS = [f"{a}_{s}" for a in ("native", "eligibility_lock") for s in ("tiny", "small")]


def flags(seed: int) -> pd.DataFrame:
    f = pd.read_csv(FLAGS[seed], usecols=["image_id", "gt_id", "common_valid_margin"])
    if seed in (2, 3):
        extra = pd.read_csv(HERE / "support_flag_fixed_only.csv")
        f = pd.concat([f, extra.loc[extra["seed"].eq(seed), ["image_id", "gt_id", "common_valid_margin"]]])
    return f.drop_duplicates(["image_id", "gt_id"])


def stats(frame: pd.DataFrame, ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for arm in ("native", "eligibility_lock"):
        for label, size in (("tiny", "t_8_16"), ("small", "s_16_32")):
            c = frame.loc[frame["size_bin"].eq(size)]
            g = pd.DataFrame({"image_id": c["image_id"], "num": c["sampling_weight"] * c[f"{arm}_o2o_fragile"],
                              "den": c["sampling_weight"]}).groupby("image_id").sum()
            out[f"{arm}_{label}"] = g.reindex(ids, fill_value=0.0)[["num", "den"]].to_numpy(float)
    return out


def values(st: dict[str, np.ndarray], counts: np.ndarray | None) -> dict[str, float]:
    v = {}
    for k, arr in st.items():
        num, den = arr.sum(0) if counts is None else (counts[:, None] * arr).sum(0)
        v[k] = corr.rate_from_totals(float(num), float(den))
    for arm in ("native", "eligibility_lock"):
        v[f"{arm}_gap"] = v[f"{arm}_tiny"] - v[f"{arm}_small"]
    v["reduction_pp"] = v["native_gap"] - v["eligibility_lock_gap"]
    v["tiny_drop"] = v["native_tiny"] - v["eligibility_lock_tiny"]
    v["small_drop"] = v["native_small"] - v["eligibility_lock_small"]
    return v


def run_seed(seed: int) -> dict:
    lock = pd.read_csv(prev.SEEDS[seed][0])
    m = lock.merge(flags(seed), on=["image_id", "gt_id"], how="left", validate="many_to_one")
    if m["common_valid_margin"].isna().any():
        raise RuntimeError(f"seed {seed}: {int(m['common_valid_margin'].isna().sum())} rows without support flag")
    v = m.loc[m["common_valid_margin"].eq(1)].copy()
    for arm in ("native", "eligibility_lock"):
        if v[f"{arm}_o2o_fragile"].isna().any():
            raise RuntimeError(f"seed {seed}: NaN {arm} outcome on support")
    ids_df = v[["image_id", "stratum"]].drop_duplicates().sort_values("image_id")
    ids, strata = ids_df["image_id"].tolist(), ids_df["stratum"].tolist()
    st = {s: stats(v.loc[v["stress"].eq(s)], ids) for s in STRESSES}
    point = {s: values(st[s], None) for s in STRESSES}
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    rng = np.random.default_rng(corr.SEED)
    boot = {s: [] for s in STRESSES}
    for _ in range(corr.REPLICATES):
        c = np.zeros(len(ids))
        for idx in groups.values():
            c += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(ids))
        for s in STRESSES:
            boot[s].append(values(st[s], c))
    ci = {s: {k: [float(x) for x in np.percentile([b[k] for b in boot[s]], [2.5, 97.5])] for k in point[s]} for s in STRESSES}
    counts = {s: {"gt": int(v["stress"].eq(s).sum()), "images": int(v.loc[v["stress"].eq(s), "image_id"].nunique()),
                  "tiny": int((v["stress"].eq(s) & v["size_bin"].eq("t_8_16")).sum())} for s in STRESSES}
    return {"counts": counts, "point": point, "ci_95": ci}


def main() -> None:
    res = {s: run_seed(s) for s in range(4)}
    p0 = res[0]["point"]["fixed_1px"]
    if res[0]["counts"]["fixed_1px"]["gt"] != 6378 or round(100 * p0["native_tiny"], 2) != 22.14 or round(100 * p0["native_small"], 2) != 7.77:
        raise RuntimeError("seed-0 gate against Table 2 failed")
    rows = []
    for s, r in res.items():
        for stress in STRESSES:
            p, c = r["point"][stress], r["ci_95"][stress]
            rows.append({"seed": s, "stress": stress, **r["counts"][stress],
                         **{k: round(100 * p[k], 2) for k in ("native_tiny", "native_small", "native_gap", "eligibility_lock_tiny",
                                                               "eligibility_lock_small", "eligibility_lock_gap", "reduction_pp",
                                                               "tiny_drop", "small_drop")},
                         **{f"{k}_ci": [round(100 * x, 2) for x in c[k]] for k in ("native_gap", "eligibility_lock_gap", "reduction_pp")}})
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "table2_support_by_seed.csv", index=False)
    (HERE / "table2_support_summary.json").write_text(json.dumps({"gate_seed0_table2": True, "seeds": {str(k): v for k, v in res.items()}}, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    for stress in STRESSES:
        sub = t[t["stress"].eq(stress)]
        print(stress, {k: (round(sub[k].mean(), 2), round(sub[k].std(ddof=1), 2)) for k in ("native_gap", "eligibility_lock_gap", "reduction_pp", "tiny_drop", "small_drop")})


if __name__ == "__main__":
    main()
