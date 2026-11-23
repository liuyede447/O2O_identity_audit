"""R1.4 follow-up: fixed-K set turnover stratified by base native O2M set size.

Fixing K fixes the size of the truncated output set but not of the pool being truncated:
small objects with more native positives have more members that can cross the K boundary.
Here the tiny-minus-small gap is computed within bins of base native assigned count and
standardised to the tiny-object bin distribution (direct standardisation).

Gates (direction rows, common-valid, K-eligible):
  binding   : base > K and shifted > K   (as in RESULTS.md main table)
  base_only : base > K                   (shifted >= K already required by eligibility;
                                           no further conditioning on the post-shift set)
Uncertainty: stratified image-cluster bootstrap, 5000 replicates, as cluster multiplicities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(r"G:\revision_20261120\fixed_k")
REPS = 5000
BINS = {3: [4, 5, 6, 7, 8], 5: [6, 7, 8, 9, 10]}  # last bin is "this value or more"
SIZES = ("t_8_16", "s_16_32")


def bin_of(count: pd.Series, k: int) -> pd.Series:
    edges = BINS[k]
    return count.clip(upper=edges[-1]).astype(int)


def multiplicities(images: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((REPS + 1, len(images)), dtype=np.int32)
    m[0] = 1
    for _, idx in images.groupby("stratum").groups.items():
        pos = np.asarray(list(idx))
        m[1:, pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=REPS)
    return m


def analyse(d: pd.DataFrame, k: int, outcome: str, seed: int) -> dict:
    d = d.copy()
    d["bin"] = bin_of(d["base_native_assigned_count"], k)
    images = d[["image_id", "stratum"]].drop_duplicates("image_id").sort_values(["stratum", "image_id"]).reset_index(drop=True)
    pos = {im: i for i, im in enumerate(images["image_id"])}
    ci = d["image_id"].map(pos).to_numpy()
    w = d["sampling_weight"].to_numpy(float)
    y = d[outcome].to_numpy(float)
    bins = BINS[k]
    num = {}
    den = {}
    for s in SIZES:
        for b in bins:
            sel = (d["size_bin"].eq(s) & d["bin"].eq(b)).to_numpy()
            n_ = np.zeros(len(images))
            d_ = np.zeros(len(images))
            np.add.at(n_, ci[sel], w[sel] * y[sel])
            np.add.at(d_, ci[sel], w[sel])
            num[s, b], den[s, b] = n_, d_
    M = multiplicities(images, seed)
    N = {key: M @ v for key, v in num.items()}
    D = {key: M @ v for key, v in den.items()}
    # bins usable only where both sizes have support in the point estimate
    usable = [b for b in bins if D["t_8_16", b][0] > 0 and D["s_16_32", b][0] > 0]
    tiny_w = np.stack([D["t_8_16", b] for b in usable], axis=1)
    tiny_w = tiny_w / tiny_w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = np.stack([N["t_8_16", b] / D["t_8_16", b] - N["s_16_32", b] / D["s_16_32", b] for b in usable], axis=1)
    std_gap = np.nansum(tiny_w * diffs, axis=1)
    raw_t = sum(N["t_8_16", b] for b in bins) / sum(D["t_8_16", b] for b in bins)
    raw_s = sum(N["s_16_32", b] for b in bins) / sum(D["s_16_32", b] for b in bins)
    raw = raw_t - raw_s
    q = lambda a: [float(np.nanquantile(a[1:], 0.025)), float(np.nanquantile(a[1:], 0.975))]
    per_bin = {}
    for j, b in enumerate(usable):
        per_bin[str(b)] = {
            "rows_tiny": int((d["size_bin"].eq("t_8_16") & d["bin"].eq(b)).sum()),
            "rows_small": int((d["size_bin"].eq("s_16_32") & d["bin"].eq(b)).sum()),
            "tiny_weight_share": float(tiny_w[0, j]),
            "rate_tiny": float(N["t_8_16", b][0] / D["t_8_16", b][0]),
            "rate_small": float(N["s_16_32", b][0] / D["s_16_32", b][0]),
            "gap": float(diffs[0, j]), "gap_ci": q(diffs[:, j]),
        }
    dropped = [b for b in bins if b not in usable]
    return {"rows": len(d), "raw_gap": float(raw[0]), "raw_gap_ci": q(raw),
            "standardised_gap": float(std_gap[0]), "standardised_gap_ci": q(std_gap),
            "bins_dropped_no_support": dropped, "per_bin": per_bin}


def main() -> None:
    res, rows = {}, []
    for seed in range(4):
        d = pd.read_csv(OUT / f"seed{seed}" / "per_direction_fixed_k.csv")
        d = d[d.common_valid_margin.eq(1)].reset_index(drop=True)
        res[seed] = {}
        for k in (3, 5):
            e = d[d[f"k{k}_eligible"].eq(1)]
            gates = {
                "binding": e[e.base_native_assigned_count.gt(k) & e.shift_native_assigned_count.gt(k)],
                "base_only": e[e.base_native_assigned_count.gt(k)],
            }
            res[seed][k] = {}
            for g, sub in gates.items():
                r = analyse(sub.reset_index(drop=True), k, f"k{k}_set_turnover", 20261001 + 10 * seed + k)
                res[seed][k][g] = r
                rows.append({"seed": seed, "K": k, "gate": g, "rows": r["rows"],
                             "raw_gap_pp": 100 * r["raw_gap"], "raw_ci": [round(100 * x, 2) for x in r["raw_gap_ci"]],
                             "std_gap_pp": 100 * r["standardised_gap"], "std_ci": [round(100 * x, 2) for x in r["standardised_gap_ci"]],
                             "dropped_bins": r["bins_dropped_no_support"]})
            print("seed", seed, "K", k, "done", flush=True)
    (HERE / "stratified_summary.json").write_text(json.dumps({str(s): {str(k): v for k, v in r.items()} for s, r in res.items()}, indent=1) + "\n", encoding="utf-8")
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "stratified_by_seed.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(2).to_string(index=False))
    for k in (3, 5):
        print(f"\nseed 0 K={k} binding per-bin:")
        for b, v in res[0][k]["binding"]["per_bin"].items():
            print(f"  bin {b}: tiny rows {v['rows_tiny']}, small rows {v['rows_small']}, tiny share {v['tiny_weight_share']:.3f}, "
                  f"rates {100*v['rate_tiny']:.1f}/{100*v['rate_small']:.1f}, gap {100*v['gap']:.2f} [{100*v['gap_ci'][0]:.2f}, {100*v['gap_ci'][1]:.2f}]")


if __name__ == "__main__":
    main()
