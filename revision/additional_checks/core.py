"""Shared frame construction, model fitting and vectorised metrics (frozen with the protocol)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
G = HERE / "inputs" / "margin"
PR = HERE.parents[1]
DISC_SEL = PR / "results" / "reviewer_controls_20260829" / "formal_s0e300_k00625" / "selected_images.csv"
KEY = ["image_id", "gt_id"]
M2 = ["log_area", "log1p_o2m_count", "log_q_active", "active_ciou"]
BUDGETS = (0.1, 0.2, 0.3)
MODELS = {"M2": M2, "M3clip": M2 + ["margin_clip"], "M3raw": M2 + ["margin"]}
OUTCOMES = ["scaled_any", "scaled_lock_persistent", "scaled_lock_removed",
            "fixed_any", "fixed_lock_persistent", "fixed_lock_removed", "miss"]


def base_frame(directory: Path) -> pd.DataFrame:
    """All focal GTs with legal directions under both stresses; outcomes are any-direction flags."""
    g = pd.read_csv(directory / "per_gt.csv", dtype={"image_id": str})
    d = pd.read_csv(directory / "per_direction.csv", dtype={"image_id": str})
    if g.duplicated(KEY).any() or d.duplicated(KEY + ["stress", "direction"]).any():
        raise RuntimeError("duplicate keys")
    d["removed"] = ((d.native_o2o_flip == 1) & (d.eligibility_lock_o2o_flip == 0)).astype(int)
    for st, nm in (("equivalent_side", "scaled"), ("fixed_1px", "fixed")):
        a = d[d.stress.eq(st)].groupby(KEY)[["native_o2o_flip", "eligibility_lock_o2o_flip", "removed"]].max()
        a.columns = [f"{nm}_any", f"{nm}_lock_persistent", f"{nm}_lock_removed"]
        g = g.merge(a.reset_index(), on=KEY, how="left", validate="one_to_one")
    g["miss"] = (g.best_iou < 0.5).astype(int)
    g["scene"] = g.image_id.str.split("__").str[0]
    return g


def analysis_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Primary support: defined margin (assigned candidate and runner-up) and legal shifts under both stresses."""
    f = g[g.o2o_margin.notna() & g.q_active.notna() & g.active_ciou.notna() & g.scaled_any.notna() & g.fixed_any.notna()].copy()
    f["log_area"] = np.log(f.area)
    f["log1p_o2m_count"] = np.log1p(f.o2m_positive_count)
    f["log_q_active"] = np.log(np.maximum(f.q_active, 1e-12))
    f["margin"] = f.o2o_margin.astype(float)
    f["margin_clip"] = f.margin.clip(0, 1)
    for o in OUTCOMES:
        f[o] = f[o].astype(int)
    if not np.isfinite(f[M2 + ["margin"]].to_numpy(float)).all():
        raise RuntimeError("non-finite feature")
    return f.reset_index(drop=True)


def fit(f: pd.DataFrame, y: str, cols: list[str], w: np.ndarray) -> dict:
    X = f[cols].to_numpy(float)
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.maximum(np.average((X - mu) ** 2, axis=0, weights=w), 1e-12))
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit((X - mu) / sd, f[y].to_numpy(int), sample_weight=w)
    if m.n_iter_[0] >= 5000:
        raise RuntimeError("no convergence")
    return {"cols": cols, "mean": mu.tolist(), "sd": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def predict(f: pd.DataFrame, p: dict) -> np.ndarray:
    z = (f[p["cols"]].to_numpy(float) - np.array(p["mean"])) / np.array(p["sd"])
    e = z @ np.array(p["coef"]) + p["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(e, -700, 700)))


class Ranker:
    """Weighted ROC-AUC, average precision and recall at a budget for many weight vectors (score descending)."""

    def __init__(self, y, score):
        o = np.argsort(-score, kind="mergesort")
        s = score[o]
        self.o, self.y = o, y[o].astype(np.float64)
        self.start = np.r_[0, np.flatnonzero(np.diff(s)) + 1]

    def groups(self, W):
        S = W[:, self.o]
        pos = np.add.reduceat(S * self.y, self.start, axis=1)
        return pos, np.add.reduceat(S, self.start, axis=1) - pos

    def all(self, W):
        """auc, ap, and recall of positives in the top 10/20/30% of weight (a tie group at the cut enters fractionally)."""
        pos, neg = self.groups(W)
        tot = pos + neg
        tp, fp, c = np.cumsum(pos, 1), np.cumsum(neg, 1), np.cumsum(tot, 1)
        P, N = tp[:, -1], fp[:, -1]
        out = {}
        with np.errstate(all="ignore"):
            out["auc"] = np.where((P > 0) & (N > 0), (neg * (tp - 0.5 * pos)).sum(1) / (P * N), np.nan)
            out["ap"] = np.where(P > 0, (pos * np.where(c > 0, tp / c, 0.0)).sum(1) / P, np.nan)
            for b in BUDGETS:
                take = np.clip((b * c[:, -1:] - (c - tot)) / np.where(tot > 0, tot, 1.0), 0.0, 1.0)
                out[f"recall{round(100 * b)}"] = np.where(P > 0, (pos * take).sum(1) / P, np.nan)
        return out

    def auc(self, W):
        return self.all(W)["auc"]


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))
