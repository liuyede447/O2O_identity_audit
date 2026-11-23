"""Build event/curve inputs for the frozen boundary-geometry analyzer.

cardinal : discovery v2 rays (left/right/up/down), the paper's population
diagonal : new rays at 45 degrees, same protocol (rmax 0.25, coarse 1/64, fine 1/512)
eight    : cardinal + diagonal pooled, one row per focal-GT ray
min8     : one row per focal GT; for each estimand the duration is the minimum over the
           eight ray durations (event radius if observed, else censor radius) and the row
           is an event iff the minimising ray is an event (ties resolved toward the event).
           This is the first exit from the GT-centre neighbourhood sampled on eight rays,
           an inner approximation of the 2-D boundary radius.
Only GTs audited on all eight rays enter eight/min8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
CARD = PR / "runs" / "20260831T174200_boundary_discovery_v2" / "artifact"
DIAG = Path(r"G:\revision_20261120\boundary_diagonal_v1")
OUT = Path(r"G:\revision_20261120\boundary_sets")
KEYS = ["image_id", "gt_id"]
ESTIMANDS = {
    "o2o": ("o2o_base_active", "o2o_event_observed", "o2o_radius", "o2o_censor_radius"),
    "o2m_legacy": ("o2m_legacy_base", "o2m_legacy_event_observed", "o2m_legacy_radius", "o2m_legacy_censor_radius"),
    "o2m_pre_topk": ("o2m_pre_topk_base", "o2m_pre_topk_event_observed", "o2m_pre_topk_radius", "o2m_pre_topk_censor_radius"),
    "o2m_assigned_positive": ("o2m_assigned_positive_base", "o2m_assigned_positive_event_observed",
                              "o2m_assigned_positive_radius", "o2m_assigned_positive_censor_radius"),
    "rho_R": (None, "rho_R_event_observed", "rho_R_radius", "rho_R_censor_radius"),
}


def write(name: str, events: pd.DataFrame, curves: pd.DataFrame) -> None:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    events.to_csv(d / "events.csv", index=False)
    curves.to_csv(d / "curves.csv", index=False)
    print(name, len(events), events[KEYS].drop_duplicates().shape[0], len(curves), flush=True)


def min_over_rays(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(KEYS + ["direction"]).reset_index(drop=True)
    first = ev.groupby(KEYS, sort=False).head(1).set_index(KEYS)
    out = first.copy()
    out["direction"] = "min8"
    for name, (_, event_col, radius_col, censor_col) in ESTIMANDS.items():
        obs = pd.to_numeric(ev[event_col], errors="coerce").fillna(0).astype(int).astype(bool)
        dur = np.where(obs, pd.to_numeric(ev[radius_col], errors="coerce"), pd.to_numeric(ev[censor_col], errors="coerce"))
        tmp = ev[KEYS].copy()
        tmp["dur"] = dur
        tmp["obs"] = obs.to_numpy()
        tmp["row"] = np.arange(len(ev))
        # sort: smallest duration first, events before censorings at ties
        tmp = tmp.sort_values(KEYS + ["dur", "obs"], ascending=[True, True, True, False], na_position="last")
        pick = tmp.groupby(KEYS, sort=False).head(1).set_index(KEYS)
        chosen = ev.iloc[pick["row"].to_numpy()].set_index(KEYS)
        # Copy every per-ray field of this estimand as a block from the chosen ray, so
        # event/censor flags, radii, reasons and brackets stay mutually consistent.
        prefix = f"{name}_"
        block = [c for c in ev.columns if c.startswith(prefix) and not c.startswith(prefix + "base")
                 and not (name == "o2o" and c.startswith("o2o_base"))]
        if name == "o2o":
            block = [c for c in ev.columns if c.startswith("o2o_") and not c.startswith("o2o_base")]
        out.loc[pick.index, block] = chosen.loc[pick.index, block].to_numpy()
        if name == "o2o":
            out.loc[pick.index, "o2o_min8_direction"] = chosen.loc[pick.index, "direction"].to_numpy()
    # consistency: no row may be both event and right-censored
    for name in ESTIMANDS:
        ec, rc = f"{name}_event_observed", f"{name}_right_censored"
        if rc in out.columns:
            both = (pd.to_numeric(out[ec]) == 1) & (pd.to_numeric(out[rc]) == 1)
            assert not both.any(), f"{name}: {int(both.sum())} rows both event and censored"
    return out.reset_index()


def main() -> None:
    ce = pd.read_csv(CARD / "per_direction_boundary_events.csv")
    cc = pd.read_csv(CARD / "o2m_rank_set_radius_curve.csv")
    de = pd.read_csv(DIAG / "per_direction_boundary_events.csv")
    dc = pd.read_csv(DIAG / "o2m_rank_set_radius_curve.csv")
    assert list(ce.columns) == list(de.columns) and list(cc.columns) == list(dc.columns)
    # base state must agree across the two runs (same model, same unperturbed image)
    base_cols = ["o2o_base_active", "o2m_legacy_base", "o2m_pre_topk_base", "o2m_assigned_positive_base", "sampling_weight", "stratum", "size_bin"]
    b1 = ce.groupby(KEYS)[base_cols].first()
    b2 = de.groupby(KEYS)[base_cols].first()
    common = b1.index.intersection(b2.index)
    mism = (b1.loc[common].fillna(-1) != b2.loc[common].fillna(-1)).any(axis=1).sum()
    print("base-state mismatches on common GTs:", int(mism))
    assert mism == 0
    write("cardinal", ce, cc)
    write("diagonal", de, dc)
    ev8 = pd.concat([ce, de], ignore_index=True)
    n_dir = ev8.groupby(KEYS)["direction"].nunique()
    full = n_dir[n_dir.eq(8)].index
    print("GTs with all 8 rays:", len(full), "of", len(n_dir))
    ev8 = ev8.set_index(KEYS).loc[full].reset_index()
    cv8 = pd.concat([cc, dc], ignore_index=True).set_index(KEYS)
    cv8 = cv8.loc[cv8.index.isin(full)].reset_index()
    write("eight", ev8, cv8)
    m8 = min_over_rays(ev8)
    # The analyzer joins curve rows to event rows by (image, gt, direction). For min8 only
    # survival/CIF/RMSR are interpretable; supply the curve of the O2O-minimising ray,
    # relabelled, so the join succeeds. min8 endpoint tables are not reported.
    pick = m8[KEYS + ["o2o_min8_direction"]].rename(columns={"o2o_min8_direction": "direction"})
    cm8 = cv8.merge(pick, on=KEYS + ["direction"], how="inner")
    cm8["direction"] = "min8"
    write("min8", m8, cm8)
    # Sensitivity population: every one of the eight rays can be scanned legally to tau,
    # so no ray is censored by the image edge before tau.
    tau = 0.125
    lim = ev8.groupby(KEYS)["scan_limit"].min()
    legal = lim[lim >= tau - 1e-12].index
    print("GTs legal to tau on all 8 rays:", len(legal))
    for name, ev_, cv_ in (("cardinal_legal", ce, cc), ("diagonal_legal", de, dc), ("min8_legal", m8, cm8)):
        e2 = ev_.set_index(KEYS)
        e2 = e2.loc[e2.index.isin(legal)].reset_index()
        c2 = cv_.set_index(KEYS)
        c2 = c2.loc[c2.index.isin(legal)].reset_index()
        write(name, e2, c2)


if __name__ == "__main__":
    main()
