"""Scale gaps of each condition (8-16 minus 16-32 px, unweighted) with image-cluster bootstrap, plus size-standardised gaps."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "inputs" / "half_pixel"
TAG = sys.argv[1] if len(sys.argv) > 1 else "seed0"
CONDS = ["train_clean", "val_translate", "val_affine", "val_mosaic", "train_affine", "train_mosaic"]
REP = HERE / "inputs" / "margin" / "formal"
SIDES = [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 32]


def prep(g):
    f = g[g.o2o_active.notna() & g.fixed_native.notna() & g.scaled_native.notna()].copy()
    f["side"] = np.sqrt(f.area)
    f["ar"] = f[["width", "height"]].max(axis=1) / f[["width", "height"]].min(axis=1)
    f["touch"] = 0
    return f


def val_clean(s=0):
    g = pd.read_csv(REP / f"seed{s}" / "per_gt.csv", dtype={"image_id": str})
    p = pd.read_csv(REP / f"seed{s}" / "per_direction.csv", dtype={"image_id": str})
    for st, n in (("fixed_1px", "fixed"), ("equivalent_side", "scaled")):
        a = p[p.stress == st].groupby(["image_id", "gt_id"])[["native_o2o_flip", "eligibility_lock_o2o_flip"]].max()
        a.columns = [n + "_native", n + "_lock"]
        g = g.merge(a.reset_index(), on=["image_id", "gt_id"], how="left")
    g["sample_id"] = g.image_id
    return prep(g)


def stats(f, B=1000):
    cols = ["fixed_native", "fixed_lock", "scaled_native", "scaled_lock"]
    ids = np.array(sorted(f.sample_id.unique()))
    t = f.groupby(["sample_id", "size_bin"])[cols + ["side"]].agg({**{c: "sum" for c in cols}, "side": "size"}).unstack(fill_value=0).reindex(ids)

    def gap(w):
        return [100 * ((t[(c, "t_8_16")] * w).sum() / (t[("side", "t_8_16")] * w).sum() - (t[(c, "s_16_32")] * w).sum() / (t[("side", "s_16_32")] * w).sum()) for c in cols]
    pt = gap(np.ones(len(ids)))
    rng = np.random.default_rng(3)
    Bs = np.array([gap(rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))) for _ in range(B)])
    lo, hi = np.percentile(Bs, [2.5, 97.5], axis=0)
    removed = 100 * (pt[0] - pt[1]) / pt[0]
    return ids, pt, lo, hi, removed


def rates(f):
    return {(b, c): 100 * f[f.size_bin == b][c].mean() for b in ("t_8_16", "s_16_32") for c in ("fixed_native", "scaled_native")}


ref = val_clean()
frames = {"val_clean(2013)": ref}
for c in CONDS:
    p = D / f"{TAG}_{c}.csv"
    if p.exists():
        frames[c] = prep(pd.read_csv(p))
rows = []
for nm, f in frames.items():
    ids, pt, lo, hi, rem = stats(f)
    r = rates(f)
    # standardise within size_bin to the (side x aspect) distribution of val_clean
    std = {}
    for c in ("fixed_native", "scaled_native"):
        tot = {}
        for b in ("t_8_16", "s_16_32"):
            q, z = f[f.size_bin == b], ref[ref.size_bin == b]
            kq = pd.cut(q.side, SIDES, right=False).astype(str) + "|" + pd.cut(q.ar, [0, 1.25, 1.5, 2, 3, 99], right=False).astype(str)
            kz = pd.cut(z.side, SIDES, right=False).astype(str) + "|" + pd.cut(z.ar, [0, 1.25, 1.5, 2, 3, 99], right=False).astype(str)
            pw = kz.value_counts(normalize=True)
            m = q.groupby(kq)[c].mean()
            common = pw.index.intersection(m.index)
            tot[b] = 100 * float((pw[common] * m[common]).sum() / pw[common].sum())
        std[c] = tot["t_8_16"] - tot["s_16_32"]
    rows.append({"condition": nm, "images": len(ids), "n": len(f), "fixed_gap": round(pt[0], 2), "fixed_CI": f"[{lo[0]:.2f},{hi[0]:.2f}]", "locked_gap": round(pt[1], 2),
                 "removed_%": round(rem, 1), "scaled_gap": round(pt[2], 2), "scaled_CI": f"[{lo[2]:.2f},{hi[2]:.2f}]", "scaled_locked": round(pt[3], 2),
                 "tiny_scaled": round(r["t_8_16", "scaled_native"], 2), "small_scaled": round(r["s_16_32", "scaled_native"], 2),
                 "std_fixed": round(std["fixed_native"], 2), "std_scaled": round(std["scaled_native"], 2),
                 "short8_%": round(100 * (f[f.size_bin == "t_8_16"].short if "short" in f else f[f.size_bin == "t_8_16"][["width", "height"]].min(axis=1)).sub(8).abs().lt(1e-3).mean(), 1)})
out = pd.DataFrame(rows)
pd.set_option("display.width", 260, "display.max_columns", 30)
print(out.to_string(index=False))
out_dir = HERE / "out"
out_dir.mkdir(exist_ok=True)
out.to_csv(out_dir / f"{TAG}_controls_summary.csv", index=False)
