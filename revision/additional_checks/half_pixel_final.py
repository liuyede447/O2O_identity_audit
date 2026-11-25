"""Single deterministic analysis for Supplementary Section 20.3 / Table S55 (replaces the earlier per-table scripts).
Strata are visited in sorted order; size groups are always those of the whole-pixel annotation; every comparison uses the
targets defined in both settings. Writes half_pixel_final.json next to this file."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "inputs" / "half_pixel"
FR = {seed: HERE / "inputs" / "discovery" / f"seed{seed}" for seed in range(4)}
EXP = HERE / "inputs" / "first_divergence"
BINS = ("t_8_16", "s_16_32")
C = ["fixed_native", "fixed_lock", "scaled_native", "scaled_lock"]
B = 2000
KEY = ["image_id", "gt_id"]


def ci(a):
    return [float(x) for x in np.percentile(a, [2.5, 97.5], axis=0)]


class Boot:
    """Stratified image-cluster multinomial weights, one stream per frame."""

    def __init__(self, frame, seed=20261124):
        a = frame[["image_id", "stratum"]].drop_duplicates().sort_values("image_id")
        self.ids, st = a.image_id.to_numpy(), a.stratum.to_numpy()
        self.ix = {v: i for i, v in enumerate(self.ids)}
        rng = np.random.default_rng(seed)
        self.W = np.zeros((B, len(a)))
        for s in sorted(set(st)):
            m = np.flatnonzero(st == s)
            self.W[:, m] = rng.multinomial(len(m), np.full(len(m), 1 / len(m)), size=B)

    def rates(self, f, cols, mask):
        q = f[mask]
        S = np.zeros((len(self.ids), len(cols) + 1))
        i = q.image_id.map(self.ix).to_numpy()
        w = q.sampling_weight.to_numpy(float)
        for j, c in enumerate(cols):
            np.add.at(S[:, j], i, w * q[c].to_numpy(float))
        np.add.at(S[:, -1], i, w)
        pt, bt = S.sum(0), self.W @ S
        return 100 * pt[:-1] / pt[-1], 100 * bt[:, :-1] / bt[:, -1:]


def base(s):
    return pd.read_csv(FR[s] / "per_gt.csv", dtype={"image_id": str})[KEY + ["sampling_weight", "stratum", "common_valid_margin"]]


def load(p):
    g = pd.read_csv(p)
    g["image_id"] = g.sample_id.str.replace(r"^disc_\d{4}_", "", regex=True)
    return g[g.o2o_active.notna() & g.fixed_native.notna() & g.scaled_native.notna()][KEY + ["size_bin"] + C]


def readouts(p, b, o):
    return {"fixed_gap": (p[o], b[:, o]), "fixed_locked": (p[o + 1], b[:, o + 1]),
            "lock_reduction": (p[o] - p[o + 1], b[:, o] - b[:, o + 1]),
            "scaled_gap": (p[o + 2], b[:, o + 2]), "scaled_locked": (p[o + 3], b[:, o + 3]),
            "scaled_lock_reduction": (p[o + 2] - p[o + 3], b[:, o + 2] - b[:, o + 3]),
            "fixed_minus_scaled": (p[o] - p[o + 2], b[:, o] - b[:, o + 2])}


def o2o(s, alt):
    a, b = load(D / f"dither_disc_ck{s}_h0.0_d0.csv"), load(alt)
    m = a.merge(b, on=KEY, suffixes=("_i", "_d"))
    out = {"n_int": len(a), "n_alt": len(b), "n_common": len(m), "size_group_changes": int((m.size_bin_i != m.size_bin_d).sum())}
    m = m.merge(base(s), on=KEY)
    for sup, f in (("table5", m), ("table2", m[m.common_valid_margin.astype(bool)])):
        bt = Boot(f)
        cols = [c + x for x in ("_i", "_d") for c in C]
        R = {k: bt.rates(f, cols, f.size_bin_i.eq(k).to_numpy()) for k in BINS}
        pt, dr = R[BINS[0]][0] - R[BINS[1]][0], R[BINS[0]][1] - R[BINS[1]][1]
        g0, g1 = readouts(pt, dr, 0), readouts(pt, dr, 4)
        q = {"n": len(f)}
        for k in g0:
            q["int_" + k] = [float(g0[k][0]), *ci(g0[k][1])]
            q["alt_" + k] = [float(g1[k][0]), *ci(g1[k][1])]
            q["change_" + k] = [float(g1[k][0] - g0[k][0]), *ci(g1[k][1] - g0[k][1])]
        for k in BINS:
            q["rates_" + k] = {f"{x}_{c}": float(R[k][0][o + j]) for x, o in (("int", 0), ("alt", 4)) for j, c in enumerate(C)}
        out[sup] = q
    new = b.merge(a[KEY], on=KEY, how="left", indicator=True).query("_merge=='left_only'")
    out["added_targets"], out["added_fixed_rate"] = len(new), float(100 * new.fixed_native.mean())
    out["lost_targets"] = len(a) - len(m)
    return out


LAB = {"elig": "eligibility_boundary", "rank": "within_set_geometry_rank_reversal", "conf": "conflict_reassignment", "topk": "topk_membership_transition"}


def fd(s, pa=None):
    pi = D / "fd_dither_h0.0.csv" if s == 0 else EXP / f"discovery_seed{s}_per_direction_anatomy.csv"
    pa = pa or D / ("fd_dither_h0.5.csv" if s == 0 else f"fd_dither_h0.5_seed{s}.csv")
    X = {k: pd.read_csv(p, dtype={"image_id": str}) for k, p in (("int", pi), ("alt", pa))}
    dfn = {k: set(map(tuple, d[d.active_base.notna() & d.size_bin.isin(BINS)][KEY].drop_duplicates().to_numpy())) for k, d in X.items()}
    common = dfn["int"] & dfn["alt"]
    out = {"n_common_targets": len(common)}
    bt = Boot(X["int"])
    for sup in ("common", "all"):
        for k, d in X.items():
            f = d[(d.o2o_flip == 1) & d.active_base.notna() & d.size_bin.isin(BINS)].copy()
            if sup == "common":
                f = f[[t in common for t in map(tuple, f[KEY].to_numpy())]].copy()
            for nm, lab in LAB.items():
                f[nm] = f.first_divergence.eq(lab).astype(int)
            r = {}
            for st in ("fixed_1px", "equivalent_side"):
                m = f.stress.eq(st).to_numpy()
                p, b = bt.rates(f, list(LAB), m)
                r[st] = (p, b)
                out[f"{sup}_{k}_{st}"] = {"exchanges": int(m.sum()), "shares": [float(v) for v in p], "elig_ci": ci(b[:, 0]), "rank_ci": ci(b[:, 1]),
                                          "elig_minus_rank": [float(p[0] - p[1]), *ci(b[:, 0] - b[:, 1])]}
            a0, a1 = r["fixed_1px"], r["equivalent_side"]
            out[f"{sup}_{k}_elig_fixed_minus_scaled"] = [float(a0[0][0] - a1[0][0]), *ci(a0[1][:, 0] - a1[1][:, 0])]
    return out


def parity(p, q):
    """Any-direction exchange label per target: first-divergence replay against the joint-shift replay."""
    a = pd.read_csv(p, dtype={"image_id": str})
    a = a[a.active_base.notna()]
    b = load(q)
    n = bad = 0
    for st, c in (("fixed_1px", "fixed_native"), ("equivalent_side", "scaled_native")):
        f = a[a.stress == st].groupby(KEY).o2o_flip.max().rename("flip").reset_index()
        m = f.merge(b[KEY + [c]], on=KEY)
        n += len(m)
        bad += int((m.flip.astype(int) != m[c].astype(int)).sum())
    return {"n": n, "mismatch": bad}


def o2m(s):
    root = D if s == 0 else D / f"o2m_seed{s}"
    if not (root / "o2m_dither_h0.5.csv").exists():
        return None
    Z = {}
    for k, h in (("i", "0.0"), ("d", "0.5")):
        g = pd.read_csv(root / f"o2m_dither_h{h}.csv", dtype={"image_id": str})
        g = g[g.stress == "equivalent_side"].groupby(KEY).agg(size_bin=("size_bin", "first"), setc=("set_change", "max"), top1=("top1_change", "max")).reset_index()
        Z[k] = g.merge(load(D / f"dither_disc_ck{s}_h{h}_d0.csv")[KEY + ["scaled_native"]], on=KEY)
    m = Z["i"].merge(Z["d"], on=KEY, suffixes=("_i", "_d")).merge(base(s), on=KEY)
    m = m[m.common_valid_margin.astype(bool)]
    bt, cols = Boot(m), ["setc_i", "top1_i", "scaled_native_i", "setc_d", "top1_d", "scaled_native_d"]
    T, S = (bt.rates(m, cols, m.size_bin_i.eq(k).to_numpy()) for k in BINS)
    out = {"n": len(m)}
    for x, o in (("int", 0), ("alt", 3)):
        t, tb = T[0][o:o + 3], T[1][:, o:o + 3]
        g, gb = t - S[0][o:o + 3], tb - S[1][:, o:o + 3]
        for k, p, b in (("tiny_set_minus_o2o", t[0] - t[2], tb[:, 0] - tb[:, 2]), ("set_gap", g[0], gb[:, 0]), ("top1_gap", g[1], gb[:, 1]),
                        ("paired_o2o_minus_top1", g[2] - g[1], gb[:, 2] - gb[:, 1]), ("set_gap_minus_o2o_gap", g[0] - g[2], gb[:, 0] - gb[:, 2])):
            out[f"{x}_{k}"] = [float(p), *ci(b)]
    return out


def main():
    J = {"B": B, "o2o": {}, "fd": {}, "o2m": {}, "draws": {}, "joint": {}, "fd_joint": {}, "fd_joint_parity": {}}
    for s in range(4):
        J["o2o"][s] = o2o(s, D / f"dither_disc_ck{s}_h0.5_d0.csv")
        J["fd"][s] = fd(s)
        J["o2m"][s] = o2m(s)
        for d in (1, 2):
            p = D / f"dither_disc_ck{s}_h0.5_d{d}.csv"
            if p.exists():
                J["draws"][f"{s}_{d}"] = o2o(s, p)
        p = D / f"imgshift_joint_ck{s}.csv"
        if p.exists():
            J["joint"][s] = o2o(s, p)
        p = D / f"fd_joint_seed{s}.csv"
        if p.exists():
            J["fd_joint"][s] = fd(s, p)
            J["fd_joint_parity"][s] = parity(p, D / f"imgshift_joint_ck{s}.csv")
    z = D / "imgshift_zero_ck0.csv"
    if z.exists():
        m = load(z).merge(load(D / "dither_disc_ck0_h0.0_d0.csv"), on=KEY, suffixes=("_a", "_b"))
        J["zero_shift_check"] = {"n": len(m), "mismatch": int(sum((m[c + "_a"] != m[c + "_b"]).sum() for c in C))}
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "half_pixel_final.json").write_text(json.dumps(J, indent=1), encoding="utf-8")

    def f(v):
        return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for grp in ("o2o", "draws", "joint"):
        for k, r in J[grp].items():
            print(grp, k, "n", r["n_common"], "moved", r["size_group_changes"], "added", r["added_targets"], round(r["added_fixed_rate"], 1), "lost", r["lost_targets"])
            for sup in ("table2", "table5"):
                q = r[sup]
                print("  ", sup, q["n"], "| fix", f(q["int_fixed_gap"]), "->", f(q["alt_fixed_gap"]), "| scl", f(q["int_scaled_gap"]), "->", f(q["alt_scaled_gap"]),
                      "| dscl", f(q["change_scaled_gap"]), "| f-s", f(q["int_fixed_minus_scaled"]), "->", f(q["alt_fixed_minus_scaled"]),
                      "| red", f(q["int_lock_reduction"]), "->", f(q["alt_lock_reduction"]), "| fl", f(q["alt_fixed_locked"]), "| sl", f(q["alt_scaled_locked"]))
    for s, r in [(s, r) for s, r in J["fd"].items()] + [(f"joint{s}", r) for s, r in J["fd_joint"].items()]:
        for sup in ("common", "all"):
            for st in ("fixed_1px", "equivalent_side"):
                q, i = r[f"{sup}_alt_{st}"], r[f"{sup}_int_{st}"]
                print("fd", s, sup, st, i["exchanges"], [round(v, 2) for v in i["shares"]], "->", q["exchanges"], [round(v, 2) for v in q["shares"]],
                      "e-r", f(i["elig_minus_rank"]), "->", f(q["elig_minus_rank"]))
            print("   elig f-s", f(r[f"{sup}_int_elig_fixed_minus_scaled"]), "->", f(r[f"{sup}_alt_elig_fixed_minus_scaled"]))
    for s, r in J["o2m"].items():
        if r:
            print("o2m", s, r["n"], {k: f(v) for k, v in r.items() if k != "n"})
    print("fd_joint_parity", J["fd_joint_parity"])
    print("zero", J.get("zero_shift_check"))


if __name__ == "__main__":
    main()
