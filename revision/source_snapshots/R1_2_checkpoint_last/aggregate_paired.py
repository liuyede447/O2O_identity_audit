"""Checkpoint sensitivity: epoch-280 vs final-epoch (last.pt) eligibility-lock estimates, seeds 0-3.

Estimator is identical to the frozen corrected lock estimator (inverse-inclusion weighted
object-level O2O fragility, 8-16 minus 16-32 px). Pairing:
  * common support: (image, GT, stress) rows present in both lock outputs whose base O2O
    identity is defined under BOTH checkpoints (base_gate_all.csv);
  * paired bootstrap: one set of stratified image-cluster resampling counts per replicate,
    shared by both checkpoints and both stresses (5,000 replicates, frozen lock seed).
Gates:
  G1  recomputed epoch-280 base gate equals the gate used in aggregate_lock_seeds.py;
  G2  epoch-280 point estimates on the original epoch-280 support reproduce lock_by_seed.csv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "R1_2_eligibility_lock_seeds"
LOCK_E280 = Path(r"G:\revision_20261120\eligibility_lock")
LOCK_LAST = Path(r"G:\revision_20261120\checkpoint_last")
STRESSES = ("fixed_1px", "equivalent_side")
ROLES = ("e280", "last")
QTY = ("native_gap", "eligibility_lock_gap", "reduction_pp")

spec = importlib.util.spec_from_file_location("prev", PREV / "aggregate_lock_seeds.py")
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
corr = prev.corr
SEED, REPLICATES = corr.SEED, corr.REPLICATES


def lock_path(seed: int, role: str) -> Path:
    if role == "e280":
        return prev.SEEDS[seed][0]
    return LOCK_LAST / f"seed{seed}" / "per_gt.csv"


def old_gate(seed: int) -> pd.DataFrame:
    gate = pd.read_csv(prev.SEEDS[seed][1], usecols=["image_id", "gt_id", "o2o_active"])
    if seed != 0:
        extra = pd.read_csv(PREV / "base_gate_fixed_only.csv")
        gate = pd.concat([gate, extra.loc[extra["seed"].eq(seed), ["image_id", "gt_id", "o2o_active"]]])
    gate = gate.drop_duplicates(["image_id", "gt_id"])
    return gate.assign(defined=gate["o2o_active"].notna())[["image_id", "gt_id", "defined"]]


def sufficient(frame: pd.DataFrame, role: str, ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for arm in ("native", "eligibility_lock"):
        for label, size in (("tiny", "t_8_16"), ("small", "s_16_32")):
            cell = frame.loc[frame["size_bin"].eq(size)]
            ev = cell["sampling_weight"] * cell[f"{role}_{arm}_o2o_fragile"]
            g = pd.DataFrame({"image_id": cell["image_id"], "num": ev, "den": cell["sampling_weight"]}).groupby("image_id").sum()
            out[f"{arm}_{label}"] = g.reindex(ids, fill_value=0.0)[["num", "den"]].to_numpy(float)
    return out


def rates(stats: dict[str, np.ndarray], counts: np.ndarray | None) -> dict[str, float]:
    v = {}
    for arm in ("native", "eligibility_lock"):
        for label in ("tiny", "small"):
            s = stats[f"{arm}_{label}"]
            num, den = (s.sum(axis=0) if counts is None else (counts[:, None] * s).sum(axis=0))
            v[f"{arm}_{label}"] = corr.rate_from_totals(float(num), float(den))
        v[f"{arm}_gap"] = v[f"{arm}_tiny"] - v[f"{arm}_small"]
    v["reduction_pp"] = v["native_gap"] - v["eligibility_lock_gap"]
    return v


def run_seed(seed: int) -> dict:
    gates = pd.read_csv(HERE / "base_gate_all.csv")
    gates = gates.loc[gates["seed"].eq(seed)]
    g = {r: gates.loc[gates["checkpoint"].eq(r)].assign(defined=lambda d: d["o2o_active"].notna())[["image_id", "gt_id", "defined"]]
         for r in ROLES}

    # G1: recomputed epoch-280 gate equals the gate used for the published lock table.
    old = old_gate(seed)
    lock280 = pd.read_csv(lock_path(seed, "e280"))
    keys = lock280[["image_id", "gt_id"]].drop_duplicates()
    cmp = keys.merge(old, on=["image_id", "gt_id"], how="left").merge(g["e280"], on=["image_id", "gt_id"], how="left", suffixes=("_old", "_new"))
    if cmp[["defined_old", "defined_new"]].isna().any().any() or not cmp["defined_old"].astype(bool).equals(cmp["defined_new"].astype(bool)):
        raise RuntimeError(f"G1 failed for seed {seed}: {int((cmp['defined_old'] != cmp['defined_new']).sum())} mismatches")

    # G2: epoch-280 point estimates on its own support reproduce lock_by_seed.csv.
    own = lock280.merge(g["e280"], on=["image_id", "gt_id"], how="left", validate="many_to_one")
    own = own.loc[own["defined"].astype(bool)]
    ref = pd.read_csv(PREV / "lock_by_seed.csv")
    for stress in STRESSES:
        p = corr.estimate(own.loc[own["stress"].eq(stress)])
        r = ref.loc[ref["seed"].eq(seed) & ref["stress"].eq(stress)].iloc[0]
        for k in QTY:
            if abs(p[k] - r[k]) > 1e-12:
                raise RuntimeError(f"G2 failed seed {seed} {stress} {k}: {p[k]} vs {r[k]}")

    # Common support.
    cols = ["native_o2o_fragile", "eligibility_lock_o2o_fragile"]
    base_cols = ["image_id", "gt_id", "stress", "size_bin", "stratum", "sampling_weight"]
    a = lock280[base_cols + cols].rename(columns={c: f"e280_{c}" for c in cols})
    lastdf = pd.read_csv(lock_path(seed, "last"))
    b = lastdf[base_cols + cols].rename(columns={c: f"last_{c}" for c in cols})
    m = a.merge(b, on=base_cols, how="inner", validate="one_to_one")
    for r in ROLES:
        m = m.merge(g[r].rename(columns={"defined": f"def_{r}"}), on=["image_id", "gt_id"], how="left", validate="many_to_one")
    m["both"] = m["def_e280"].astype(bool) & m["def_last"].astype(bool)
    counts_info = {}
    for stress in STRESSES:
        s = m["stress"].eq(stress)
        counts_info[stress] = {
            "rows_e280": int(lock280["stress"].eq(stress).sum()), "rows_last": int(lastdf["stress"].eq(stress).sum()),
            "rows_in_both_outputs": int(s.sum()),
            "excluded_base_undefined_e280_only": int((s & ~m["def_e280"].astype(bool) & m["def_last"].astype(bool)).sum()),
            "excluded_base_undefined_last_only": int((s & m["def_e280"].astype(bool) & ~m["def_last"].astype(bool)).sum()),
            "excluded_base_undefined_both": int((s & ~m["def_e280"].astype(bool) & ~m["def_last"].astype(bool)).sum()),
            "common_gt": int((s & m["both"]).sum()),
            "common_tiny": int((s & m["both"] & m["size_bin"].eq("t_8_16")).sum()),
            "common_small": int((s & m["both"] & m["size_bin"].eq("s_16_32")).sum()),
            "common_images": int(m.loc[s & m["both"], "image_id"].nunique()),
        }
    v = m.loc[m["both"]].copy()

    ids_df = v[["image_id", "stratum"]].drop_duplicates().sort_values("image_id")
    if ids_df["image_id"].duplicated().any():
        raise RuntimeError("image with multiple strata")
    ids, strata = ids_df["image_id"].tolist(), ids_df["stratum"].tolist()
    stats = {(st, r): sufficient(v.loc[v["stress"].eq(st)], r, ids) for st in STRESSES for r in ROLES}

    def all_values(counts):
        out = {}
        for st in STRESSES:
            per = {r: rates(stats[(st, r)], counts) for r in ROLES}
            for r in ROLES:
                for k in QTY:
                    out[f"{st}|{r}|{k}"] = per[r][k]
            for k in QTY:
                out[f"{st}|last_minus_e280|{k}"] = per["last"][k] - per["e280"][k]
        return out

    point = all_values(None)
    by_stratum: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by_stratum.setdefault(s, []).append(i)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(REPLICATES):
        c = np.zeros(len(ids))
        for idx in by_stratum.values():
            c += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(ids))
        boot.append(all_values(c))
    boot = pd.DataFrame(boot)
    ci = {k: [float(x) for x in np.percentile(boot[k], [2.5, 97.5])] for k in point}
    return {"counts": counts_info, "point": point, "ci_95": ci, "lock_inputs": {r: str(lock_path(seed, r)) for r in ROLES}}


def main() -> None:
    results = {s: run_seed(s) for s in range(4)}
    rows = []
    for s, r in results.items():
        for st in STRESSES:
            for role in ("e280", "last", "last_minus_e280"):
                row = {"seed": s, "stress": st, "checkpoint": role, "common_gt": r["counts"][st]["common_gt"],
                       "common_images": r["counts"][st]["common_images"]}
                for k in QTY:
                    key = f"{st}|{role}|{k}"
                    row[k] = r["point"][key]
                    row[f"{k}_ci"] = r["ci_95"][key]
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "paired_by_seed.csv", index=False)
    summ = {"gates": {"G1_e280_gate_recomputed_equal": True, "G2_e280_points_reproduce_lock_by_seed": True},
            "bootstrap_seed": SEED, "replicates": REPLICATES, "seeds": {str(k): v for k, v in results.items()}}
    for st in STRESSES:
        for role in ("e280", "last", "last_minus_e280"):
            sub = table[table["stress"].eq(st) & table["checkpoint"].eq(role)]
            summ[f"across_seed|{st}|{role}"] = {k: {"mean": float(sub[k].mean()), "sd": float(sub[k].std(ddof=1)),
                                                    "min": float(sub[k].min()), "max": float(sub[k].max())} for k in QTY}
    (HERE / "summary.json").write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    show = table.copy()
    for k in QTY:
        show[k] = (100 * show[k]).round(2)
        show[f"{k}_ci"] = show[f"{k}_ci"].map(lambda x: f"[{100 * x[0]:.2f}, {100 * x[1]:.2f}]")
    print(show.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
