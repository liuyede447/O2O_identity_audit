"""Post hoc, exploratory (not part of the preregistered replication).

1) Eligibility-rule strata. YOLO26s replaces any GT dimension < 8 px by 16 px for the eligibility
   test. Split 8-16 px targets by whether that rule fires (min(w, h) < 8 at 800 px input) and report
   native, locked and drop rates per subgroup against the 16-32 px targets that do not trigger the rule.
2) Small-only influence on the validation sample: drop the s_only stratum, and leave each s_only image out.

Estimator and bootstrap follow analyze_replication.py (inverse-inclusion weights, stratified
image-cluster bootstrap, 5,000 replicates, seed 20260823 + 901). Fixed 1 px, base-defined O2O support.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REP = HERE.parent / "R2_2_heldout_replication"
sys.path.insert(0, str(REP))
import analyze_replication as ar  # noqa: E402

CORR = Path(r"E:\two_paper\beifen\修复\post_submission_audit_20260908\inputs")
FORMAL = Path(r"G:\revision_20261120\heldout_replication\formal")
GROUPS = ("tiny_rule", "tiny_norule", "small_norule")
ARMS = ("native", "eligibility_lock")


def discovery() -> tuple[pd.DataFrame, pd.DataFrame]:
    lock = pd.read_csv(CORR / "eligibility_lock_per_gt.csv")
    size = pd.read_csv(CORR / "fixed_branch_per_gt.csv", usecols=["image_id", "gt_id", "width", "height", "o2o_active"])
    m = lock[lock["stress"].eq("fixed_1px")].merge(size, on=["image_id", "gt_id"], how="left", validate="one_to_one")
    m = m[m["o2o_active"].notna()]
    sel = m.drop_duplicates("image_id")[["image_id", "stratum"]]
    return m, sel


def validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    v, sel, _ = ar.load(FORMAL / "lock", FORMAL / "base_gate.csv", REP / "selected_images_preregistered.csv")
    v = v[v["stress"].eq("fixed_1px")]
    size = pd.read_csv(FORMAL / "normalized" / "per_gt.csv", usecols=["image_id", "gt_id", "width", "height"])
    m = v.merge(size, on=["image_id", "gt_id"], how="left", validate="one_to_one")
    # Four fixed-only GTs have no size-scaled row; read their size from the YOLO label line gt_id.
    # The label-line mapping is checked on every GT that has both sources.
    labels = FORMAL.parent / "AI-TOD-v2-officialval-YOLO" / "val" / "labels"

    def from_label(row):
        line = (labels / f"{row.image_id}.txt").read_text().split("\n")[int(row.gt_id)].split()
        return float(line[3]) * 800, float(line[4]) * 800

    known = m[m["width"].notna()].sample(2000, random_state=0)
    lw = np.array([from_label(r) for r in known.itertuples()])
    err = np.abs(lw - known[["width", "height"]].to_numpy()).max()
    if err > 1e-3:
        raise RuntimeError(f"label-line mapping check failed: max |diff| {err}")
    miss = m["width"].isna()
    m.loc[miss, ["width", "height"]] = np.array([from_label(r) for r in m[miss].itertuples()])
    print(f"label mapping checked on 2,000 GTs (max |diff| {err:.2e}); filled {int(miss.sum())} sizes")
    return m, sel


def label(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    rule = np.minimum(m["width"], m["height"]) < 8
    tiny = m["size_bin"].eq("t_8_16")
    m["group"] = np.select([tiny & rule, tiny & ~rule, ~tiny & ~rule], list(GROUPS), default="small_rule")
    return m


def suff(m: pd.DataFrame, ids: list[str]) -> np.ndarray:
    out = []
    for arm in ARMS:
        for g in GROUPS:
            c = m[m["group"].eq(g)]
            t = pd.DataFrame({"image_id": c["image_id"], "num": c["sampling_weight"] * c[f"{arm}_o2o_fragile"],
                              "den": c["sampling_weight"]}).groupby("image_id").sum()
            out.append(t.reindex(ids, fill_value=0.0)[["num", "den"]].to_numpy(float))
    return np.stack(out)


def values(arr: np.ndarray, counts: np.ndarray | None) -> dict[str, float]:
    tot = arr.sum(1) if counts is None else np.einsum("i,cij->cj", counts, arr)
    r = np.where(tot[:, 1] > 0, tot[:, 0] / np.where(tot[:, 1] > 0, tot[:, 1], 1), np.nan)
    v = {f"{a}|{g}": r[i * len(GROUPS) + j] for i, a in enumerate(ARMS) for j, g in enumerate(GROUPS)}
    ref_n, ref_l = v["native|small_norule"], v["eligibility_lock|small_norule"]
    for g in ("tiny_rule", "tiny_norule"):
        v[f"native_gap|{g}"] = v[f"native|{g}"] - ref_n
        v[f"locked_gap|{g}"] = v[f"eligibility_lock|{g}"] - ref_l
        v[f"reduction|{g}"] = v[f"native_gap|{g}"] - v[f"locked_gap|{g}"]
        v[f"drop|{g}"] = v[f"native|{g}"] - v[f"eligibility_lock|{g}"]
    return {k: 100 * float(x) for k, x in v.items()}


def run(m: pd.DataFrame, sel: pd.DataFrame) -> dict:
    m = label(m)
    if m["width"].isna().any():
        raise RuntimeError(f"{int(m['width'].isna().sum())} rows lack box size")
    ids = sorted(set(m["image_id"]))
    strat = sel.set_index("image_id")["stratum"]
    by = {}
    for i, s in enumerate(strat[x] for x in ids):
        by.setdefault(s, []).append(i)
    arr = suff(m, ids)
    point = values(arr, None)
    rng = np.random.default_rng(ar.SEED)
    boot = []
    for _ in range(ar.REPLICATES):
        c = np.zeros(len(ids))
        for idx in by.values():
            c += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(ids))
        boot.append(values(arr, c))
    b = pd.DataFrame(boot)
    ci = {k: [float(x) for x in np.nanpercentile(b[k], [2.5, 97.5])] for k in point}
    n = {g: {"gt": int(m["group"].eq(g).sum()), "images": int(m.loc[m["group"].eq(g), "image_id"].nunique()),
             "weight_share_within_size_bin": float(m.loc[m["group"].eq(g), "sampling_weight"].sum()
                                                   / m.loc[m["size_bin"].eq("t_8_16" if g.startswith("tiny") else "s_16_32"), "sampling_weight"].sum())}
         for g in (*GROUPS, "small_rule")}
    return {"point": point, "ci_95": ci, "counts": n, "undefined_replicates": {k: int(b[k].isna().sum()) for k in point}}


def small_only_influence() -> dict:
    v, sel, _ = ar.load(FORMAL / "lock", FORMAL / "base_gate.csv", REP / "selected_images_preregistered.csv")
    s_only = sorted(sel.loc[sel["stratum"].eq("s_only"), "image_id"])
    out = {"s_only_images": s_only}
    full = ar.analyze(v, sel, replicates=ar.REPLICATES)
    keep = sel[~sel["stratum"].eq("s_only")]
    drop = ar.analyze(v[v["image_id"].isin(keep["image_id"])], keep, replicates=ar.REPLICATES)
    keys = ["fixed_1px|native_gap", "fixed_1px|eligibility_lock_gap", "fixed_1px|reduction_pp", "fixed_1px|native_small"]
    out["full"] = {k: [100 * full["point"][k], *[100 * x for x in full["ci_95"][k]]] for k in keys}
    out["without_s_only"] = {k: [100 * drop["point"][k], *[100 * x for x in drop["ci_95"][k]]] for k in keys}
    out["leave_one_out_point"] = {}
    for img in s_only:
        k_sel = sel[sel["image_id"].ne(img)]
        r = ar.analyze(v[v["image_id"].ne(img)], k_sel, replicates=1)
        out["leave_one_out_point"][img] = {k: 100 * r["point"][k] for k in keys}
    out["gts_in_s_only_images"] = int(v[v["image_id"].isin(s_only) & v["stress"].eq("fixed_1px")].shape[0])
    return out


def main() -> None:
    res = {"note": "post hoc exploratory; not part of the preregistered decision rule",
           "discovery_seed0_epoch280": run(*discovery()),
           "validation_seed0_final_epoch": run(*validation()),
           "validation_small_only_influence": small_only_influence()}
    (HERE / "rule_strata_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for name in ("discovery_seed0_epoch280", "validation_seed0_final_epoch"):
        r = res[name]
        print(f"== {name}")
        for g, c in r["counts"].items():
            print(f"  {g}: n={c['gt']} images={c['images']} weight share={100 * c['weight_share_within_size_bin']:.1f}%")
        for k in r["point"]:
            print(f"  {k}: {r['point'][k]:.2f} [{r['ci_95'][k][0]:.2f}, {r['ci_95'][k][1]:.2f}]")
    print("== small-only influence")
    print(json.dumps(res["validation_small_only_influence"], indent=1))


if __name__ == "__main__":
    main()
