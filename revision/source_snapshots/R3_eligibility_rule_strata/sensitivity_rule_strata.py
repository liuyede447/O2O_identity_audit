"""Post hoc sensitivity for analyze_rule_strata.py.

A) Boundary: targets with min(w, h) within 1e-3 of 8 px form their own group, because the rule
   (tal.py:505, w or h < 8) is strict and these boxes sit on the threshold in float arithmetic.
B) Rule versus elongation: compare triggered and non-triggered 8-16 px targets within a shared
   aspect-ratio band (long/short side in [2, 4)), excluding boundary boxes.
Same estimator, weights and bootstrap as analyze_rule_strata.py; reference = 16-32 px, no rule.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import analyze_rule_strata as base

EPS = 1e-3


def run_groups(m: pd.DataFrame, sel: pd.DataFrame, labeller, groups: tuple[str, ...]) -> dict:
    m = m.copy()
    m["group"] = labeller(m)
    m = m[m["group"].isin(groups + ("ref",))]
    ids = sorted(set(m["image_id"]))
    strat = sel.set_index("image_id")["stratum"]
    by: dict[str, list[int]] = {}
    for i, s in enumerate(strat[x] for x in ids):
        by.setdefault(s, []).append(i)
    cells = [(a, g) for a in base.ARMS for g in groups + ("ref",)]
    arr = []
    for a, g in cells:
        c = m[m["group"].eq(g)]
        t = pd.DataFrame({"image_id": c["image_id"], "num": c["sampling_weight"] * c[f"{a}_o2o_fragile"],
                          "den": c["sampling_weight"]}).groupby("image_id").sum()
        arr.append(t.reindex(ids, fill_value=0.0)[["num", "den"]].to_numpy(float))
    arr = np.stack(arr)

    def values(counts):
        tot = arr.sum(1) if counts is None else np.einsum("i,cij->cj", counts, arr)
        r = {cells[i]: (tot[i, 0] / tot[i, 1] if tot[i, 1] > 0 else np.nan) for i in range(len(cells))}
        out = {}
        for g in groups:
            ng = r[("native", g)] - r[("native", "ref")]
            lg = r[("eligibility_lock", g)] - r[("eligibility_lock", "ref")]
            out[f"native_gap|{g}"], out[f"locked_gap|{g}"], out[f"reduction|{g}"] = 100 * ng, 100 * lg, 100 * (ng - lg)
        return out

    point = values(None)
    rng = np.random.default_rng(base.ar.SEED)
    boot = []
    for _ in range(base.ar.REPLICATES):
        c = np.zeros(len(ids))
        for idx in by.values():
            c += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=len(ids))
        boot.append(values(c))
    b = pd.DataFrame(boot)
    ci = {k: [float(x) for x in np.nanpercentile(b[k], [2.5, 97.5])] for k in point}
    n = {g: int(m["group"].eq(g).sum()) for g in groups + ("ref",)}
    return {"point": point, "ci_95": ci, "n_gt": n}


def boundary_labels(m):
    lo = np.minimum(m["width"], m["height"])
    tiny = m["size_bin"].eq("t_8_16")
    return np.select([~tiny & (lo >= 8 + EPS), tiny & (lo < 8 - EPS), tiny & (np.abs(lo - 8) < EPS), tiny & (lo >= 8 + EPS)],
                     ["ref", "rule_clear", "boundary_8", "norule_clear"], default="other")


def aspect_labels(m):
    lo = np.minimum(m["width"], m["height"])
    hi = np.maximum(m["width"], m["height"])
    ar_ = hi / lo
    tiny = m["size_bin"].eq("t_8_16")
    band = (ar_ >= 2) & (ar_ < 4)
    return np.select([~tiny & (lo >= 8 + EPS), tiny & band & (lo < 8 - EPS), tiny & band & (lo >= 8 + EPS)],
                     ["ref", "rule_ar2to4", "norule_ar2to4"], default="other")


def main() -> None:
    res = {"note": "post hoc exploratory sensitivity"}
    for name, loader in (("discovery_seed0_epoch280", base.discovery), ("validation_seed0_final_epoch", base.validation)):
        m, sel = loader()
        res[name] = {"boundary": run_groups(m, sel, boundary_labels, ("rule_clear", "boundary_8", "norule_clear")),
                     "aspect_band_2_4": run_groups(m, sel, aspect_labels, ("rule_ar2to4", "norule_ar2to4"))}
    (base.HERE / "rule_strata_sensitivity.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for name in ("discovery_seed0_epoch280", "validation_seed0_final_epoch"):
        for block, r in res[name].items():
            print(f"== {name} / {block}  n={r['n_gt']}")
            for k, v in r["point"].items():
                print(f"  {k:28s} {v:6.2f} [{r['ci_95'][k][0]:6.2f}, {r['ci_95'][k][1]:6.2f}]")


if __name__ == "__main__":
    main()
