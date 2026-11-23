"""Evaluate assigned-relative margin as a screen for native assignment changes.

Part A (trajectory level, seed 0): competing-risk cumulative incidence by margin quintile on
all trajectories with a defined margin, for (1) the fixed base pair and (2) the replayed O2O
identity. Reuses the frozen Aalen-Johansen code, bootstrap and seed.
Part B (GT level, kappa=0.0625 and fixed 1 px endpoints): out-of-fold M2 vs M3 on the frozen
image folds, four seeds plus VisDrone. Outcomes: any identity change, change that persists when
eligibility is locked, change removed by the lock, coverage-based miss.
Read-only on all inputs. Post hoc: outcomes and cuts were looked at before this script existed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
EVENTS = INPUTS / "cardinal.csv.gz"
DIAG = INPUTS / "diagonal.csv.gz"
FROZEN_CIF = INPUTS / "frozen_cif.csv.gz"
FIXED0 = INPUTS / "fixed_seed0.csv.gz"
OOF0 = INPUTS / "oof_seed0.csv.gz"
OOFV = INPUTS / "oof_visdrone.csv.gz"
PER_GT = {s: INPUTS / f"endpoints_seed{s}.csv.gz" for s in range(4)}
LOCK = {s: INPUTS / f"lock_seed{s}.csv.gz" for s in range(4)}
SEED = 20260831
TAUS = (0.0625, 0.125, 0.25)  # 0.125 is the RMCBD horizon already fixed in Methods
KEY = ["image_id", "gt_id"]
M2 = ["log_area", "log1p_o2m_count", "log_q_active", "active_ciou"]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


geo = load("analyze_assignment_boundary_geometry")
shape = load("analyze_margin_shape_sensitivity")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


# ------------------------------------------------------------------ Part A
PAIR_CAUSES = ("rank_crossing", "eligibility", "topk", "conflict")


def make_process(d, dur, obs, cause, names, lookup, n):
    """Same construction as the frozen process_matrices, for arbitrary durations/causes."""
    w = d["sampling_weight"].to_numpy(float)
    c = np.array([lookup[str(i)] for i in d["image_id"]], int)
    if np.any(~np.isfinite(dur)) or np.any(dur < 0):
        raise ValueError("invalid durations")
    times = np.unique(dur[obs & (dur <= max(TAUS) + 1e-12)])
    risk = np.zeros((n, len(times)))
    inc = {k: np.zeros_like(risk) for k in names}
    for j, t in enumerate(times):
        r = dur >= t - 1e-12
        np.add.at(risk[:, j], c[r], w[r])
        at = obs & np.isclose(dur, t, atol=1e-12, rtol=0.0)
        for k in names:
            m = at & (cause == k)
            np.add.at(inc[k][:, j], c[m], w[m])
    if set(np.unique(cause[obs])) - set(names):
        raise ValueError("unknown cause")
    return {"times": times, "risk": risk, "increments": inc, "cause_names": names}


def pair_arrays(d):
    ev = d["rho_R_event_observed"].eq(1).to_numpy()
    comp = d["rho_R_competing_censored"].eq(1).to_numpy()
    dur = np.where(ev, d["rho_R_radius"], d["rho_R_censor_radius"]).astype(float)
    m = {"eligibility_boundary": "eligibility", "topk_membership_transition": "topk", "conflict_reassignment": "conflict"}
    cause = np.where(ev, "rank_crossing", d["rho_R_censor_reason"].map(m).fillna("none").to_numpy()).astype(object)
    return dur, ev | comp, cause


def replay_arrays(d):
    _, dur, obs, cause = geo.event_arrays(d, "o2o")
    return dur, obs, cause


def eval_groups(d, groups, arrays, names, lookup, mult):
    out = {}
    for g in groups:
        sub = d.loc[d["grp"].eq(g)].reset_index(drop=True)
        dur, obs, cause = arrays(sub)
        p = make_process(sub, dur, obs, cause, names, lookup, mult.shape[1])
        e = geo.evaluate_process(p, mult, max(TAUS), np.array(TAUS))
        out[g] = {k: e["cif"][k] for k in names}
        out[g]["_n"] = len(sub)
    return out


QS = [f"q{i}" for i in range(1, 6)]
UNDEF = ["base_runner_missing", "base_order_not_active_over_runner"]


def label(d, cuts=None):
    d = d.copy()
    u = d["rho_R_censor_reason"].isin(UNDEF + ["base_active_missing"])
    d["grp"] = np.where(u, d["rho_R_censor_reason"], "")
    v = d.loc[~u]
    mval = v["o2o_base_margin"].to_numpy(float)
    if np.isnan(mval).any():
        raise ValueError("margin NaN inside defined support")
    if cuts is None:
        cuts = shape.weighted_quantile(mval, v["sampling_weight"].to_numpy(float), np.array([0.2, 0.4, 0.6, 0.8]))
        if len(np.unique(cuts)) != 4:
            raise ValueError("non-unique cuts")
    d.loc[~u, "grp"] = [QS[i] for i in np.searchsorted(cuts, mval, side="right")]
    return d, cuts


def cif_rows(res, names, tag, scope):
    rows = []
    for g in res:
        for ti, t in enumerate(TAUS):
            for k in names:
                v = 100 * res[g][k][:, ti]
                rows.append({"analysis": tag, "scope": scope, "group": g, "n_traj": res[g]["_n"], "tau": t, "cause": k,
                             "cif_pct": v[0], "lo": ci(v[1:])[0], "hi": ci(v[1:])[1]})
    if all(q in res for q in QS):
        rank = "rank_crossing" if "rank_crossing" in names else "within_set"
        for ti, t in enumerate(TAUS):
            diffs = {}
            for k in (rank, "eligibility"):
                v = 100 * (res["q1"][k][:, ti] - res["q5"][k][:, ti])
                diffs[k] = v
                rows.append({"analysis": tag, "scope": scope, "group": "q1_minus_q5", "n_traj": None, "tau": t, "cause": k,
                             "cif_pct": v[0], "lo": ci(v[1:])[0], "hi": ci(v[1:])[1]})
            v = diffs[rank] - diffs["eligibility"]
            rows.append({"analysis": tag, "scope": scope, "group": "q1_minus_q5", "n_traj": None, "tau": t,
                         "cause": "rank_minus_eligibility", "cif_pct": v[0], "lo": ci(v[1:])[0], "hi": ci(v[1:])[1]})
    return rows


def part_a():
    ev = pd.read_csv(EVENTS, low_memory=False)
    meta, lookup = geo.cluster_index(ev, ev)
    boot = geo.bootstrap_multiplicities(meta, REPS, SEED)
    mult = np.vstack([np.ones((1, len(meta)), np.int32), boot]).astype(float)
    # Gate: pooled O2O incidence must reproduce the frozen survival_cif.csv, intervals included.
    d0 = ev.loc[geo.base_defined(ev["o2o_base_active"])].reset_index(drop=True)
    dur, obs, cause = replay_arrays(d0)
    p = make_process(d0, dur, obs, cause, geo.CAUSES, lookup, len(meta))
    e = geo.evaluate_process(p, mult, max(TAUS), np.array(TAUS))
    fz = pd.read_csv(FROZEN_CIF)
    fz = fz[fz["size_group"].eq("all") & fz["estimand"].eq("o2o") & fz["radius"].eq(0.125)].set_index("metric")
    gate = {}
    for k in geo.CAUSES:
        v = e["cif"][k][:, 1]
        f = fz.loc[f"cif_{k}"]
        gate[k] = {"got": float(v[0]), "frozen": float(f.estimate), "ci_got": ci(v[1:]), "ci_frozen": [float(f.ci_low), float(f.ci_high)]}
        if abs(v[0] - f.estimate) > 1e-9:
            raise RuntimeError(f"gate A point {k}")
        if REPS == 5000 and max(abs(a - b) for a, b in zip(ci(v[1:]), gate[k]["ci_frozen"])) > 1e-9:
            raise RuntimeError(f"gate A CI {k}")
    print("gate A ok", flush=True)

    d, cuts = label(ev)
    comp = d["grp"].value_counts().to_dict()
    states = {"rank_crossing": int(ev.rho_R_event_observed.sum()), "competing": int(ev.rho_R_competing_censored.sum()),
              "right_censored": int(ev.rho_R_right_censored.sum())}
    if states != {"rank_crossing": 1593, "competing": 7439, "right_censored": 16548} or len(ev) != 29724:
        raise RuntimeError("trajectory composition changed")
    rows = []
    for scope in ("all", "t_8_16", "s_16_32"):
        s = d if scope == "all" else d[d["size_bin"].eq(scope)]
        rows += cif_rows(eval_groups(s, QS, pair_arrays, PAIR_CAUSES, lookup, mult), PAIR_CAUSES, "fixed_pair", scope)
        g = QS + (UNDEF if scope == "all" else [])
        rows += cif_rows(eval_groups(s, g, replay_arrays, geo.CAUSES, lookup, mult), geo.CAUSES, "replayed_identity", scope)
        print("part A", scope, flush=True)
    # share of first rank crossings (weighted, observed at any radius <= 0.25) in the lowest quintile
    share = {}
    for nm, mask in (("pair_rank_crossing", d["rho_R_event_observed"].eq(1)),
                     ("replayed_within_set", d["o2o_first_divergence"].eq("within_set_geometry_rank_reversal"))):
        x = d[mask & d["grp"].isin(QS)]
        wq = x.groupby("grp")["sampling_weight"].sum()
        share[nm] = {q: float(100 * wq.get(q, 0) / wq.sum()) for q in QS}
    # diagonal rays: same GT, same cuts, legacy adaptive grid
    dg = pd.read_csv(DIAG, low_memory=False)
    dgl, _ = label(dg, cuts)
    base = d.drop_duplicates(KEY).set_index(KEY)["o2o_base_margin"]
    b2 = dg.drop_duplicates(KEY).set_index(KEY)["o2o_base_margin"].reindex(base.index)
    if not np.allclose(base, b2, equal_nan=True, rtol=0, atol=0):
        raise RuntimeError("diagonal base margins differ")
    rows += cif_rows(eval_groups(dgl, QS, pair_arrays, PAIR_CAUSES, lookup, mult), PAIR_CAUSES, "fixed_pair", "diagonal_rays")
    rows += cif_rows(eval_groups(dgl, QS, replay_arrays, geo.CAUSES, lookup, mult), geo.CAUSES, "replayed_identity", "diagonal_rays")
    pd.DataFrame(rows).to_csv(OUT / "cif_by_margin.csv", index=False)
    m = d.loc[d.grp.isin(QS)]
    return {"gate_frozen_cif": gate, "cuts": [float(c) for c in cuts], "group_counts": comp, "states": states,
            "n_images": int(ev.image_id.nunique()), "n_gt": int(ev.groupby(KEY).ngroups),
            "n_gt_margin_defined": int(m.groupby(KEY).ngroups), "share_by_quintile_pct": share,
            "diagonal_group_counts": dgl["grp"].value_counts().to_dict()}


# ------------------------------------------------------------------ Part B
def fit_oof(df, y, cols):
    p = np.full(len(df), np.nan)
    w = df["sampling_weight"].to_numpy(float)
    X = df[cols].to_numpy(float)
    f = df["oof_fold"].to_numpy(int)
    for k in range(5):
        tr, te = f != k, f == k
        a, b = shape.weighted_standardize(X[tr], X[te], w[tr])
        p[te] = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(a, y[tr], sample_weight=w[tr]).predict_proba(b)[:, 1]
    if np.isnan(p).any():
        raise RuntimeError("OOF incomplete")
    return p


class FoldMetric:
    """Weighted ROC-AUC / AP / log loss / Brier within one fold for many weight vectors at once."""

    def __init__(self, y, p):
        o = np.argsort(p, kind="mergesort")
        self.o, self.y, ps = o, y[o].astype(float), p[o]
        self.start = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
        q = np.clip(p, np.finfo(p.dtype).eps, 1 - np.finfo(p.dtype).eps)
        self.ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
        self.br = (p - y) ** 2

    def __call__(self, W):  # W: B x n
        T = W.sum(1)
        S = W[:, self.o]
        pos = np.add.reduceat(S * self.y, self.start, axis=1)
        neg = np.add.reduceat(S * (1 - self.y), self.start, axis=1)
        P, N = pos.sum(1), neg.sum(1)
        with np.errstate(all="ignore"):
            cn = np.cumsum(neg, axis=1) - neg
            auc = (pos * (cn + 0.5 * neg)).sum(1) / (P * N)
            tp = np.cumsum(pos[:, ::-1], axis=1)
            fp = np.cumsum(neg[:, ::-1], axis=1)
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            ap = (pos[:, ::-1] * prec).sum(1) / P
            return np.stack([auc, ap, (W @ self.ll) / T, (W @ self.br) / T], 1), T, (P > 0) & (N > 0)


NAMES = ("roc_auc", "pr_auc", "log_loss", "brier")


def cells_of(df):
    """Image positions and fold x stratum cells, in the frozen order (images by first appearance)."""
    m = df[["image_id", "stratum", "oof_fold"]].drop_duplicates()
    ims = {i: k for k, i in enumerate(m["image_id"])}
    return ims, {k: [ims[i] for i in v["image_id"]] for k, v in m.groupby(["oof_fold", "stratum"])}


def oof_multiplicity(df, reps, seed):
    ims, cells = cells_of(df)
    rng = np.random.default_rng(seed)
    M = np.zeros((reps, len(ims)))
    for b in range(reps):
        for members in cells.values():
            np.add.at(M[b], rng.choice(members, size=len(members), replace=True), 1)
    return ims, M


def fold_metrics(df, y, p, W):
    f = df["oof_fold"].to_numpy(int)
    tot, acc, ok = 0, 0, np.ones(len(W), bool)
    for k in range(5):
        idx = np.flatnonzero(f == k)
        met, T, good = FoldMetric(y[idx], p[idx])(W[:, idx])
        acc = acc + met * T[:, None]
        tot = tot + T
        ok &= good
    r = acc / tot[:, None]
    r[~ok] = np.nan
    return r


def selfcheck():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400) * 0.6 + 0.3 * y * rng.random(400), 2).clip(0.01, 0.99)
    w = rng.random(400) * 3
    got = FoldMetric(y, p)(w[None])[0][0]
    ref = [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w),
           log_loss(y, p, sample_weight=w, labels=[0, 1]), brier_score_loss(y, p, sample_weight=w)]
    if not np.allclose(got, ref, atol=1e-10):
        raise RuntimeError(f"metric selfcheck {got} {ref}")


def frames():
    z = pd.read_csv(OOF0)
    foldmap = z.drop_duplicates("image_id").set_index("image_id")["oof_fold"]
    out = {}
    for s, path in PER_GT.items():
        g = pd.read_csv(path)
        f = g[g["o2o_margin"].notna() & g["q_active"].notna() & g["active_ciou"].notna()].copy()
        f["log_area"] = np.log(f["area"])
        f["log1p_o2m_count"] = np.log1p(f["o2m_positive_count"])
        f["log_q_active"] = np.log(np.maximum(f["q_active"], 1e-12))
        f["margin"] = f["o2o_margin"].astype(float)
        f["oof_fold"] = f["image_id"].map(foldmap)
        if f["oof_fold"].isna().any():
            raise RuntimeError("image outside frozen folds")
        f["oof_fold"] = f["oof_fold"].astype(int)
        if s == 0:
            z2 = z.set_index(KEY).reindex(pd.MultiIndex.from_frame(f[KEY]))
            for c in M2 + ["margin"]:
                if not np.allclose(z2[c].to_numpy(float), f[c].to_numpy(float), rtol=1e-12):
                    raise RuntimeError(f"seed0 frame mismatch {c}")
            f["fn_at_iou50"] = z2["fn_at_iou50"].to_numpy(int)
        lk = pd.read_csv(LOCK[s])
        lk["removed"] = ((lk["native_o2o_flip"] == 1) & (lk["eligibility_lock_o2o_flip"] == 0)).astype(int)
        for st, nm in (("equivalent_side", "scaled"), ("fixed_1px", "fixed")):
            a = lk[lk["stress"].eq(st)].groupby(KEY)[["native_o2o_flip", "eligibility_lock_o2o_flip", "removed"]].max()
            a.columns = [f"{nm}_any", f"{nm}_lock_persistent", f"{nm}_lock_removed"]
            f = f.merge(a.reset_index(), on=KEY, how="left", validate="one_to_one")
        f["scaled_mismatch"] = (f["scaled_any"] != f["o2o_fragile"])
        if f["scaled_mismatch"].any() or f.filter(regex="^(scaled|fixed)_").isna().any().any():
            raise RuntimeError(f"seed {s}: lock/per_gt mismatch or missing")
        if s == 0:
            fx = pd.read_csv(FIXED0).set_index(KEY).reindex(pd.MultiIndex.from_frame(f[KEY]))
            for c in ("o2o_active", "o2o_runner"):
                if (fx[c].to_numpy(float) != f[c].to_numpy(float)).any():
                    raise RuntimeError("fixed/scaled base mismatch")
            if (fx["o2o_fragile"].to_numpy(int) != f["fixed_any"].to_numpy(int)).any():
                raise RuntimeError("fixed flag mismatch")
        out[f"seed{s}"] = f.reset_index(drop=True)
    v = pd.read_csv(OOFV)
    v["scaled_any"] = v["o2o_fragile"].astype(int)
    out["visdrone"] = v
    return out, z


def part_b():
    selfcheck()
    fr, z = frames()
    # Gate: refit coverage-based miss M2/M3 on frozen folds; must match frozen predictions, S29 points and CIs.
    z = z.copy()
    y = z["fn_at_iou50"].to_numpy(int)
    gate = {}
    pp = {}
    for nm, cols in (("M2_plus_active_ciou", M2), ("M3_plus_margin", M2 + ["margin"])):
        p = fit_oof(z, y, cols)
        gate[nm + "_max_abs_diff"] = float(np.abs(p - z[f"prob_{nm}"]).max())
        pp[nm] = z[f"prob_{nm}"].to_numpy(float)
        if gate[nm + "_max_abs_diff"] > 1e-5:
            raise RuntimeError(f"gate B refit {nm} {gate}")
    w = z["sampling_weight"].to_numpy(float)
    m2 = fold_metrics(z, y, pp["M2_plus_active_ciou"], w[None])[0]
    m3 = fold_metrics(z, y, pp["M3_plus_margin"], w[None])[0]
    gate["S29"] = {"M2": m2.tolist(), "delta": (m3 - m2).tolist()}
    if abs(m2[0] - 0.95354) > 6e-6 or abs((m3 - m2)[0] + 0.00100) > 6e-6:
        raise RuntimeError(f"gate B S29 {gate}")
    ims, M = oof_multiplicity(z, REPS, SEED)
    col = z["image_id"].map(ims).to_numpy()
    W = w[None] * M[:, col]
    dd = fold_metrics(z, y, pp["M3_plus_margin"], W) - fold_metrics(z, y, pp["M2_plus_active_ciou"], W)
    gate["S29"]["delta_ci"] = {n: ci(dd[:, i]) for i, n in enumerate(NAMES)}
    print("gate B ok", json.dumps(gate), flush=True)

    # "_ordered": drop negative margins (assigned candidate not the q leader), the support rule Part A already
    # inherits from rho_R ("base_order_not_active_over_runner"). Primary frames; full frames are sensitivity.
    for k in list(fr):
        fr[k + "_ordered"] = fr[k][fr[k]["margin"] >= 0].reset_index(drop=True)
    part_a_neg = pd.read_csv(EVENTS, usecols=KEY + ["rho_R_censor_reason"], low_memory=False)
    part_a_neg = set(map(tuple, part_a_neg[part_a_neg["rho_R_censor_reason"].eq("base_order_not_active_over_runner")][KEY].drop_duplicates().to_numpy().tolist()))
    neg0 = set(map(tuple, fr["seed0"][fr["seed0"]["margin"] < 0][KEY].to_numpy().tolist()))
    gate["negative_margin_gts"] = {"oof_frame": len(neg0), "part_a": len(part_a_neg), "oof_subset_of_part_a": neg0 <= part_a_neg}
    print(gate["negative_margin_gts"], flush=True)

    rows, qrows, boots, meta = [], [], {}, {}
    for name, f in fr.items():
        f = f.copy()
        f["margin_clip"] = f["margin"].clip(0, 1)
        w = f["sampling_weight"].to_numpy(float)
        # AI-TOD frames share one synchronous draw of the 288 frozen-fold images; VisDrone has its own.
        ref = z if name.startswith("seed") else fr[name.replace("_ordered", "")]
        ims, M = oof_multiplicity(ref, REPS, SEED)
        W = w[None] * M[:, f["image_id"].map(ims).to_numpy()]
        meta[name] = {"n_gt": len(f), "n_images": int(f["image_id"].nunique()), "negative_margin": int((f["margin"] < 0).sum()), "min_margin": float(f["margin"].min())}
        outs = [c for c in ("scaled_any", "scaled_lock_persistent", "scaled_lock_removed", "fixed_any", "fixed_lock_persistent",
                            "fixed_lock_removed", "fn_at_iou50") if c in f]
        # quintile rates (frozen cut rule; cuts fixed)
        cuts, qid = shape.margin_quintiles(f)
        mx = geo.bootstrap_multiplicities(f[["image_id", "stratum"]].drop_duplicates().sort_values(["stratum", "image_id"]).reset_index(drop=True), REPS, SEED)
        order = f[["image_id", "stratum"]].drop_duplicates().sort_values(["stratum", "image_id"])["image_id"].tolist()
        pos = {i: k for k, i in enumerate(order)}
        ic = f["image_id"].map(pos).to_numpy()
        for o in outs:
            y = f[o].to_numpy(int)
            rate = []
            for q in range(5):
                s = qid == q
                num = np.bincount(ic[s], weights=w[s] * y[s], minlength=len(order))
                den = np.bincount(ic[s], weights=w[s], minlength=len(order))
                rate.append(np.r_[num.sum() / den.sum(), (mx @ num) / np.maximum(mx @ den, 1e-300)])
            for q in range(5):
                qrows.append({"frame": name, "outcome": o, "quintile": q + 1, "n_gt": int((qid == q).sum()), "events": int(y[qid == q].sum()),
                              "rate_pct": 100 * rate[q][0], "lo": 100 * ci(rate[q][1:])[0], "hi": 100 * ci(rate[q][1:])[1]})
            dq = 100 * (rate[0] - rate[4])
            qrows.append({"frame": name, "outcome": o, "quintile": "q1_minus_q5", "n_gt": len(f), "events": int(y.sum()),
                          "rate_pct": dq[0], "lo": ci(dq[1:])[0], "hi": ci(dq[1:])[1]})
            # OOF models
            est, bt = {}, {}
            for m, cols in (("M2", M2), ("M3", M2 + ["margin"]), ("M3clip", M2 + ["margin_clip"]), ("margin_only", ["margin"])):
                p = fit_oof(f, y, cols)
                est[m] = fold_metrics(f, y, p, w[None])[0]
                bt[m] = np.concatenate([fold_metrics(f, y, p, W[i:i + 500]) for i in range(0, REPS, 500)])
            okn = int(np.isfinite(bt["M2"][:, 0]).sum())
            for m in est:
                for i, n in enumerate(NAMES):
                    rows.append({"frame": name, "outcome": o, "events": int(y.sum()), "n_gt": len(f), "model": m, "metric": n, "value": est[m][i],
                                 "lo": ci(bt[m][:, i])[0], "hi": ci(bt[m][:, i])[1], "reps_ok": okn})
            for a in ("M3", "M3clip"):
                for i, n in enumerate(NAMES):
                    dv = bt[a][:, i] - bt["M2"][:, i]
                    rows.append({"frame": name, "outcome": o, "events": int(y.sum()), "n_gt": len(f), "model": f"{a}-M2", "metric": n,
                                 "value": est[a][i] - est["M2"][i], "lo": ci(dv)[0], "hi": ci(dv)[1], "reps_ok": okn})
            boots[(name, o)] = {k: (est[k], bt[k], list(ims)) for k in ("M2", "M3", "M3clip")}
            print(name, o, okn, np.round(est["M2"][0], 3), np.round(est["M3"][0], 3), np.round(est["M3clip"][0], 3), flush=True)
    # paired contrast of AUC gains: lock-persistent minus lock-removed
    con = []
    for nm in [f"seed{s}{x}" for x in ("_ordered", "") for s in range(4)]:
        for st in ("scaled", "fixed"):
            for a in ("M3", "M3clip"):
                A, R = boots[(nm, f"{st}_lock_persistent")], boots[(nm, f"{st}_lock_removed")]
                pt = (A[a][0] - A["M2"][0] - R[a][0] + R["M2"][0])[0]
                bv = (A[a][1] - A["M2"][1] - R[a][1] + R["M2"][1])[:, 0]
                con.append({"frame": nm, "stress": st, "model": a, "dAUC_persistent_minus_removed": pt, "lo": ci(bv)[0], "hi": ci(bv)[1]})
    # four-seed equal mean, synchronous image draws (conditional on the four checkpoints)
    agg = []
    sync = {}
    seed_draw = np.random.default_rng(SEED + 1).integers(0, 4, size=(REPS, 4))
    for x in ("_ordered", ""):
        sync[x or "full"] = len({tuple(boots[(f"seed{s}{x}", "scaled_any")]["M2"][2]) for s in range(4)}) == 1
        if not sync[x or "full"]:
            continue
        for o in ("scaled_any", "scaled_lock_persistent", "scaled_lock_removed", "fixed_any", "fixed_lock_persistent", "fixed_lock_removed"):
            for a in ("M3", "M3clip"):
                d = [boots[(f"seed{s}{x}", o)] for s in range(4)]
                pt = np.array([(u[a][0] - u["M2"][0])[0] for u in d])
                per = np.stack([(u[a][1] - u["M2"][1])[:, 0] for u in d], 1)
                bv = per.mean(1)
                # crossed: equal-probability training-seed resampling on top of the synchronous image draws
                bs = np.take_along_axis(per, seed_draw, axis=1).mean(1)
                agg.append({"support": x.strip("_") or "full", "outcome": o, "model": f"{a}-M2", "mean_M2_auc": np.mean([u["M2"][0][0] for u in d]),
                            "mean_model_auc": np.mean([u[a][0][0] for u in d]), "mean_dAUC": pt.mean(), "lo": ci(bv)[0], "hi": ci(bv)[1], "seed_lo": ci(bs)[0], "seed_hi": ci(bs)[1],
                            "seed_min": pt.min(), "seed_max": pt.max()})
    pd.DataFrame(rows).to_csv(OUT / "oof_models.csv", index=False)
    pd.DataFrame(qrows).to_csv(OUT / "rates_by_margin_quintile.csv", index=False)
    pd.DataFrame(con).to_csv(OUT / "auc_gain_contrast.csv", index=False)
    pd.DataFrame(agg).to_csv(OUT / "four_seed_auc_gain.csv", index=False)
    return {"gate": gate, "frames": meta, "four_seed_synchronous": sync}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    REPS = args.reps
    OUT = args.output_dir.resolve()
    if OUT.is_relative_to(HERE):
        parser.error("Choose an output directory outside the archived margin_screen folder.")
    OUT.mkdir(parents=True, exist_ok=True)
    for item in json.loads((HERE / "inputs_manifest.json").read_text("utf-8")):
        if sha(HERE / item["file"]) != item["sha256"]:
            raise RuntimeError(f"Input checksum mismatch: {item['file']}")
    t0 = time.time()
    summary = {"reps": REPS, "seed": SEED, "taus": TAUS,
               "inputs_sha256": {p.relative_to(HERE).as_posix(): sha(p) for p in [EVENTS, DIAG, FROZEN_CIF, FIXED0, OOF0, OOFV, *PER_GT.values(), *LOCK.values()]},
               "script_sha256": sha(Path(__file__))}
    summary["part_b"] = part_b()
    summary["part_a"] = part_a()
    summary["seconds"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("done", summary["seconds"], flush=True)
