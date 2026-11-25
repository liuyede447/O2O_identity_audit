"""Prespecified analysis. Usage: analyze.py <root holding seed0..seed3> <out dir> <images.csv|rehearsal> [B]

Frozen discovery coefficients are applied unchanged to the new images (weight 1 per image).
Main intervals: source-scene cluster bootstrap crossed with equal-probability resampling of the four seeds.
The same scene draw is used for every seed, endpoint, model and metric of a frame set.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import BUDGETS, HERE, MODELS, OUTCOMES, Ranker, analysis_frame, base_frame, ci, load_json, predict

SEED, SEED_RESAMPLE, CHUNK = 20261123, 20261124, 50
SUP = {"clip": ("full", OUTCOMES),
       "ordered": ("ordered", ["scaled_any", "scaled_lock_persistent", "scaled_lock_removed", "fixed_any"]),
       "ranklast": ("full", ["scaled_any"])}


def scene_weights(codes, n, reps):
    rng = np.random.default_rng(SEED)
    for s in range(0, reps, CHUNK):
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=min(CHUNK, reps - s)).astype(np.float32)[:, codes]


def image_weights(codes, strata_of_image, reps):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(strata_of_image == st) for st in sorted(set(strata_of_image))]
    for s in range(0, reps, CHUNK):
        k = min(CHUNK, reps - s)
        W = np.zeros((k, len(strata_of_image)), np.float32)
        for g in groups:
            W[:, g] = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=k)
        yield W[:, codes]


def load(root, s, images, subset):
    g = base_frame(root / f"seed{s}")
    if images is not None:
        if set(g.image_id) - set(images.image_id):
            raise RuntimeError("images outside the frozen list")
        g = g.drop(columns=["scene"]).merge(images[["image_id", "scene", "primary", "fully_disjoint"]], on="image_id", validate="many_to_one")
        if subset != "all_eligible":
            g = g[g[subset].eq(1)]
    return g.reset_index(drop=True)


def rank_last(g):
    """All objects with an assigned candidate; objects without a runner-up are ranked last by margin."""
    h = g[g.o2o_active.notna() & g.scaled_any.notna()].copy()
    h["log_area"] = np.log(h.area)
    h["log1p_o2m_count"] = np.log1p(h.o2m_positive_count)
    h["log_q_active"] = np.log(np.maximum(h.q_active, 1e-12))
    h["margin_clip"] = h.o2o_margin.astype(float).clip(0, 1).fillna(2.0)
    h["scaled_any"] = h["scaled_any"].astype(int)
    return h.reset_index(drop=True)


def crossed(V, pick):
    return ci(np.take_along_axis(V.T, pick, 1).mean(1))


def main():
    root, out, arg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    images = None if arg == "rehearsal" else pd.read_csv(arg, dtype={"image_id": str})
    out.mkdir(parents=True, exist_ok=True)
    t0, frozen = time.time(), load_json(HERE / "frozen_models.json")
    res, contrast, points, counts = {}, {}, [], []
    for subset in (["primary", "all_eligible", "fully_disjoint"] if images is not None else ["primary"]):
        frames = {}
        for s in range(4):
            g = load(root, s, images, subset)
            f = analysis_frame(g)
            frames["clip", s], frames["ordered", s], frames["ranklast", s] = f, f[f.margin >= 0].reset_index(drop=True), rank_last(g)
            counts.append({"subset": subset, "seed": s, "images": g.image_id.nunique(), "scenes": g.scene.nunique(), "focal_gt": len(g),
                           "no_assigned": int(g.o2o_active.isna().sum()),
                           "assigned_no_runner": int((g.o2o_active.notna() & g.o2o_runner.isna()).sum()), "analysed": len(f),
                           "negative_margin": int((f.margin < 0).sum()), "min_margin": float(f.margin.min()),
                           **{"positives_" + o: int(f[o].sum()) for o in OUTCOMES}})
        scenes = np.unique(np.concatenate([frames["ranklast", s].scene.to_numpy(str) for s in range(4)] +
                                          [frames["clip", s].scene.to_numpy(str) for s in range(4)]))
        im = pd.concat([frames[k, s][["image_id", "stratum"]] for k in ("clip", "ranklast") for s in range(4)]).drop_duplicates().sort_values("image_id")
        if im.image_id.duplicated().any():
            raise RuntimeError("image with two strata")
        si, ii = {v: i for i, v in enumerate(scenes)}, {v: i for i, v in enumerate(im.image_id)}
        for sup, (modelsup, outs) in SUP.items():
            dr = {}
            for outcome in outs:
                R = {}
                for s in range(4):
                    f = frames[sup, s]
                    y = f[outcome].to_numpy(int)
                    P = {m: predict(f, frozen[str(s)]["models"][f"{modelsup}|{outcome}|{m}"]) for m in MODELS
                         if f"{modelsup}|{outcome}|{m}" in frozen[str(s)]["models"] and (sup != "ranklast" or m == "M2")}
                    P["margin"] = -f.margin_clip.to_numpy(float)
                    R[s] = {m: Ranker(y, v) for m, v in P.items()}
                    one = np.ones((1, len(f)), np.float32)
                    points.append({"subset": subset, "support": sup, "outcome": outcome, "seed": s, "n": len(f), "positives": int(y.sum()),
                                   **{f"{m}_{k}": float(v[0]) for m, r in R[s].items() for k, v in r.all(one).items()}})
                for design in ("scene", "image"):
                    reps = B if (subset, sup, design) == ("primary", "clip", "scene") else max(200, B // 5)
                    pick = np.random.default_rng(SEED_RESAMPLE).integers(0, 4, size=(reps, 4))

                    def run(s):
                        f, acc1 = frames[sup, s], {}
                        if design == "scene":
                            gen = scene_weights(f.scene.map(si).to_numpy(int), len(scenes), reps)
                        else:
                            gen = image_weights(f.image_id.map(ii).to_numpy(int), im.stratum.to_numpy(str), reps)
                        for W in gen:
                            for m, r in R[s].items():
                                for k, v in r.all(W).items():
                                    acc1.setdefault((m, k), []).append(v)
                        return {mk: np.concatenate(v) for mk, v in acc1.items()}

                    with ThreadPoolExecutor(int(os.environ.get("MARGIN_THREADS", "4"))) as ex:
                        per = list(ex.map(run, range(4)))
                    acc = {mk: [per[s][mk] for s in range(4)] for mk in per[0]}
                    V = {mk: np.stack(v) for mk, v in acc.items()}  # seeds x reps
                    key = f"{subset}|{sup}|{outcome}|{design}"
                    e = {}
                    for (m, k), v in V.items():
                        e[f"{m}_{k}"] = {"crossed": crossed(v, pick), "fixed_seeds": ci(v.mean(0))}
                        if m != "M2":
                            d = v - V["M2", k]
                            e[f"{m}_minus_M2_{k}"] = {"crossed": crossed(d, pick), "fixed_seeds": ci(d.mean(0)), "per_seed": [ci(x) for x in d]}
                    res[key] = e
                    for m in ("M3clip", "M3raw"):
                        if (m, "auc") in V:
                            dr[outcome, design, m] = V[m, "auc"] - V["M2", "auc"]
                    print(f"{time.time() - t0:.0f}s {key}", flush=True)
            for st in ("scaled", "fixed"):
                for design in ("scene", "image"):
                    for m in ("M3clip", "M3raw"):
                        a, b = (f"{st}_lock_persistent", design, m), (f"{st}_lock_removed", design, m)
                        if a in dr and b in dr:
                            d = dr[a] - dr[b]
                            pick = np.random.default_rng(SEED_RESAMPLE).integers(0, 4, size=(d.shape[1], 4))
                            contrast[f"{subset}|{sup}|{st}|{design}|{m}"] = {"crossed": crossed(d, pick), "fixed_seeds": ci(d.mean(0)),
                                                                              "per_seed": [ci(x) for x in d]}
    pt = pd.DataFrame(points)
    pt.to_csv(out / "per_seed_points.csv", index=False)
    vals = [c for c in pt.columns if c not in ("subset", "support", "outcome", "seed", "n", "positives")]
    pt.groupby(["subset", "support", "outcome"], sort=False)[vals].mean().reset_index().to_csv(out / "seed_mean_points.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "counts.csv", index=False)
    (out / "intervals.json").write_text(json.dumps({"intervals": res, "lock_contrast": contrast, "B": B, "seed": SEED, "seed_resample": SEED_RESAMPLE}, indent=1))
    decide(out, res, contrast, pt)
    print(f"finished {time.time() - t0:.0f}s")


def decide(out, res, contrast, pt):
    """Fixed-sequence tests P1-P4 on the primary subset, full support with clipped margin, scene bootstrap crossed with seeds."""
    P = "primary|clip|{}|scene"
    mean = pt[pt.subset.eq("primary") & pt.support.eq("clip")].groupby("outcome").mean(numeric_only=True)
    miss_gate = load_json(HERE / "expected" / "gate_results.json").get("miss_usable", False)

    def row(i, name, point, interval, rule, ok):
        return {"test": i, "name": name, "point": float(point), "ci": interval, "rule": rule, "met": bool(ok)}

    d = mean.M3clip_auc - mean.M2_auc
    i1 = res[P.format("scaled_any")]["M3clip_minus_M2_auc"]["crossed"]
    i2 = contrast["primary|clip|scaled|scene|M3clip"]["crossed"]
    i3 = res[P.format("scaled_any")]["margin_minus_M2_recall20"]["crossed"]
    i4 = res[P.format("miss")]["M3clip_minus_M2_auc"]["crossed"]
    T = [row("P1", "AUC gain, native change, scaled", d["scaled_any"], i1, "lower bound > 0", i1[0] > 0),
         row("P2", "AUC gain lock-persistent minus lock-removed, scaled", d["scaled_lock_persistent"] - d["scaled_lock_removed"], i2, "lower bound > 0", i2[0] > 0),
         row("P3", "recall at 20% budget, margin minus M2, native change, scaled", mean.margin_recall20["scaled_any"] - mean.M2_recall20["scaled_any"], i3, "lower bound > 0", i3[0] > 0)]
    if miss_gate:
        T.append(row("P4", "AUC gain, missed detection", d["miss"], i4, "interval inside (-0.01, 0.01)", i4[0] > -0.01 and i4[1] < 0.01))
    alive = True
    for t in T:
        t["formally_tested"] = alive
        t["confirmed"] = bool(alive and t["met"])
        alive = t["confirmed"]
    (out / "decisions.json").write_text(json.dumps(T, indent=1))
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
