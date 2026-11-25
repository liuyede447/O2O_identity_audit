"""Post-hoc evaluation of the two-track screening workflow fixing the ranklast whole-pool shortfall.

Strictly read-only on all frozen inputs and prespecified formal/results outputs.
Evaluates:
  (A) The assigned_no_runner subgroup (size composition, native vs eligibility-locked flip rates,
      share of all O2O flips, and within-subgroup ranking by frozen M2 and by -q_active).
  (B) Whole-pool O2O-assigned screening across all assigned targets (no runner-up exclusions),
      comparing:
        - M2_all        : frozen M2 applied to all assigned targets
        - ranklast      : -clip(margin) with missing runner-up ranked last (2.0)
        - rankfirst     : -clip(margin) with missing runner-up ranked first (-1.0)
        - joint_M3_M2   : frozen M3clip when runner-up exists, frozen M2 when runner-up is absent
        - strat_M2_marg : proportional budget b in each track (M2 inside no-runner, -clip(margin) inside runner-valid)
        - strat_q_marg  : score-free proportional budget b (-q_active inside no-runner, -clip(margin) inside runner-valid)
        - strat_M2_M3   : proportional budget b (M2 inside no-runner, M3clip inside runner-valid)

Intervals use the exact bootstrap seeds and crossed scene x seed design from analyze.py
(B=5000 for primary|scene; max(200, B//5)=1000 otherwise).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import image_weights, scene_weights  # noqa: E402
from core import BUDGETS, M2, Ranker, base_frame, ci, load_json, predict  # noqa: E402

FORMAL_ROOT = HERE / "inputs" / "margin" / "formal"
OUT_DIR = HERE / "out" / "two_track"
SEED_RESAMPLE = 20261124


def crossed(V: np.ndarray, pick: np.ndarray) -> list[float]:
    return ci(np.take_along_axis(V.T, pick, 1).mean(1))


def load_assigned_pool(root: Path, s: int, images: pd.DataFrame, subset: str) -> pd.DataFrame:
    """All focal GTs with a defined base O2O identity and legal shifts under both stresses."""
    g = base_frame(root / f"seed{s}")
    g = g.drop(columns=["scene"]).merge(
        images[["image_id", "scene", "primary", "fully_disjoint"]],
        on="image_id",
        validate="many_to_one",
    )
    if subset != "all_eligible":
        g = g[g[subset].eq(1)]
    p = g[g.o2o_active.notna() & g.scaled_any.notna() & g.fixed_any.notna()].copy()
    p["log_area"] = np.log(p.area)
    p["log1p_o2m_count"] = np.log1p(p.o2m_positive_count)
    p["log_q_active"] = np.log(np.maximum(p.q_active, 1e-12))
    p["has_runner"] = p.o2o_margin.notna().astype(int)
    p["margin_clip"] = p.o2o_margin.astype(float).clip(0.0, 1.0)
    for col in ("scaled_any", "scaled_lock_persistent", "fixed_any", "fixed_lock_persistent", "miss"):
        p[col] = p[col].astype(int)
    if not np.isfinite(p[M2 + ["q_active", "area"]].to_numpy(float)).all():
        raise RuntimeError(f"seed {s} {subset}: non-finite features in assigned pool")
    return p.reset_index(drop=True)


def rank_subgroup_stratified(
    y: np.ndarray,
    has_runner: np.ndarray,
    nr_score: np.ndarray,
    ru_score: np.ndarray,
    W: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute exact weighted recall@b when taking top b weight in Track 1 (no-runner) + top b weight in Track 2."""
    nr = ~has_runner
    ru = has_runner
    r_nr = Ranker(y[nr], nr_score[nr]).all(W[:, nr])
    r_ru = Ranker(y[ru], ru_score[ru]).all(W[:, ru])
    p_nr = W[:, nr] @ y[nr].astype(np.float64)
    p_ru = W[:, ru] @ y[ru].astype(np.float64)
    p_tot = p_nr + p_ru
    out = {}
    for b in BUDGETS:
        kb = f"recall{round(100 * b)}"
        with np.errstate(all="ignore"):
            out[kb] = np.where(
                p_tot > 0,
                (np.where(p_nr > 0, p_nr * r_nr[kb], 0.0) + np.where(p_ru > 0, p_ru * r_ru[kb], 0.0)) / p_tot,
                np.nan,
            )
    return out


def eval_subgroup_rates(df_seeds: list[pd.DataFrame], images: pd.DataFrame, subset: str, reps: int) -> dict:
    """Subgroup descriptive stats & 95% crossed CIs for assigned_no_runner."""
    scenes = np.unique(np.concatenate([f.scene.to_numpy(str) for f in df_seeds]))
    si = {v: i for i, v in enumerate(scenes)}
    pick = np.random.default_rng(SEED_RESAMPLE).integers(0, 4, size=(reps, 4))

    def one_seed(s: int):
        f = df_seeds[s]
        nr = f[f.has_runner == 0].reset_index(drop=True)
        sc_codes = nr.scene.map(si).to_numpy(int)
        metrics = {
            "tiny_pct": 100.0 * nr.size_bin.eq("t_8_16").to_numpy(float),
            "fixed_native_pct": 100.0 * nr.fixed_any.to_numpy(float),
            "fixed_lock_pct": 100.0 * nr.fixed_lock_persistent.to_numpy(float),
            "fixed_drop_pct": 100.0 * (nr.fixed_any.to_numpy(float) - nr.fixed_lock_persistent.to_numpy(float)),
            "scaled_native_pct": 100.0 * nr.scaled_any.to_numpy(float),
            "scaled_lock_pct": 100.0 * nr.scaled_lock_persistent.to_numpy(float),
            "scaled_drop_pct": 100.0 * (nr.scaled_any.to_numpy(float) - nr.scaled_lock_persistent.to_numpy(float)),
            "miss_pct": 100.0 * nr.miss.to_numpy(float),
        }
        # Also share of ALL flips in the full assigned pool coming from assigned_no_runner
        is_nr = (f.has_runner == 0).to_numpy(float)
        f_sc = f.scene.map(si).to_numpy(int)

        pt = {k: float(v.mean()) for k, v in metrics.items()}
        pt["share_assigned_pct"] = 100.0 * float(is_nr.mean())
        pt["share_scaled_flips_pct"] = 100.0 * float((is_nr * f.scaled_any.to_numpy(float)).sum() / f.scaled_any.sum())
        pt["share_fixed_flips_pct"] = 100.0 * float((is_nr * f.fixed_any.to_numpy(float)).sum() / f.fixed_any.sum())

        arr = {k: [] for k in list(metrics) + ["share_assigned_pct", "share_scaled_flips_pct", "share_fixed_flips_pct"]}
        mat_nr = np.vstack([metrics[k] for k in metrics])
        y_sc_nr = is_nr * f.scaled_any.to_numpy(float)
        y_fx_nr = is_nr * f.fixed_any.to_numpy(float)
        y_sc_all = f.scaled_any.to_numpy(float)
        y_fx_all = f.fixed_any.to_numpy(float)

        for W in scene_weights(sc_codes, len(scenes), reps):
            den = np.maximum(W.sum(1), 1e-12)
            vals = (W @ mat_nr.T) / den[:, None]
            for j, k in enumerate(metrics):
                arr[k].append(vals[:, j])
        for W in scene_weights(f_sc, len(scenes), reps):
            den = np.maximum(W.sum(1), 1e-12)
            arr["share_assigned_pct"].append(100.0 * (W @ is_nr) / den)
            arr["share_scaled_flips_pct"].append(100.0 * (W @ y_sc_nr) / np.maximum(W @ y_sc_all, 1e-12))
            arr["share_fixed_flips_pct"].append(100.0 * (W @ y_fx_nr) / np.maximum(W @ y_fx_all, 1e-12))
        return pt, {k: np.concatenate(v) for k, v in arr.items()}

    with ThreadPoolExecutor(4) as ex:
        per = list(ex.map(one_seed, range(4)))
    out = {}
    for k in per[0][0]:
        pts = [per[s][0][k] for s in range(4)]
        V = np.stack([per[s][1][k] for s in range(4)])
        out[k] = {
            "mean": float(np.mean(pts)),
            "seed_range": [float(np.min(pts)), float(np.max(pts))],
            "per_seed": pts,
            "crossed": crossed(V, pick),
            "fixed_seeds": ci(V.mean(0)),
        }
    return out


def main():
    t0 = time.time()
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(HERE / "images.csv", dtype={"image_id": str})
    frozen = load_json(HERE / "frozen_models.json")

    point_rows = []
    intervals = {}
    subgroup_rates = {}

    for subset in ("primary", "all_eligible", "fully_disjoint"):
        frames = [load_assigned_pool(FORMAL_ROOT, s, images, subset) for s in range(4)]
        reps_rate = B if subset == "primary" else max(200, B // 5)
        subgroup_rates[subset] = eval_subgroup_rates(frames, images, subset, reps_rate)
        print(f"[{time.time()-t0:.0f}s] subgroup rates {subset} done", flush=True)

        scenes = np.unique(np.concatenate([f.scene.to_numpy(str) for f in frames]))
        im = pd.concat([f[["image_id", "stratum"]] for f in frames]).drop_duplicates().sort_values("image_id")
        si = {v: i for i, v in enumerate(scenes)}
        ii = {v: i for i, v in enumerate(im.image_id)}

        for outcome in ("scaled_any", "fixed_any"):
            # Prebuild rankers for whole-pool scorers and no-runner subgroup scorers
            whole_rankers = [{} for _ in range(4)]
            nr_rankers = [{} for _ in range(4)]
            ru_rankers = [{} for _ in range(4)]
            raw_arrays = [{} for _ in range(4)]

            for s in range(4):
                f = frames[s]
                y = f[outcome].to_numpy(int)
                hm = f.has_runner.to_numpy(bool)
                m2_all = predict(f, frozen[str(s)]["models"][f"full|{outcome}|M2"])
                m3_sub = predict(f[hm], frozen[str(s)]["models"][f"full|{outcome}|M3clip"])
                joint = m2_all.copy()
                joint[hm] = m3_sub
                margin_score = -f.margin_clip.to_numpy(float)
                ranklast = np.where(hm, margin_score, -2.0)
                rankfirst = np.where(hm, margin_score, 1.0)  # missing runner placed before [ -1, 0 ] margin scores
                q_nr_score = -f.q_active.to_numpy(float)

                scorers = {
                    "M2_all": m2_all,
                    "ranklast": ranklast,
                    "rankfirst": rankfirst,
                    "joint_M3_M2": joint,
                }
                for nm, sc in scorers.items():
                    whole_rankers[s][nm] = Ranker(y, sc)
                nr_rankers[s]["M2"] = Ranker(y[~hm], m2_all[~hm])
                nr_rankers[s]["neg_q"] = Ranker(y[~hm], q_nr_score[~hm])
                ru_rankers[s]["margin"] = Ranker(y[hm], margin_score[hm])
                ru_rankers[s]["M3clip"] = Ranker(y[hm], m3_sub)
                raw_arrays[s] = {"y": y, "hm": hm, "m2": m2_all, "m3": m3_sub, "marg": margin_score, "q": q_nr_score}

                ones = np.ones((1, len(f)), np.float32)
                ones_nr = np.ones((1, int((~hm).sum())), np.float32)
                for nm, rk in whole_rankers[s].items():
                    m = rk.all(ones)
                    point_rows.append({
                        "subset": subset, "pool": "whole_assigned", "outcome": outcome, "seed": s,
                        "n": len(f), "positives": int(y.sum()), "method": nm,
                        **{k: float(v[0]) for k, v in m.items()},
                    })
                for nr_nm, rk in nr_rankers[s].items():
                    m = rk.all(ones_nr)
                    point_rows.append({
                        "subset": subset, "pool": "assigned_no_runner", "outcome": outcome, "seed": s,
                        "n": int((~hm).sum()), "positives": int(y[~hm].sum()), "method": f"nr_{nr_nm}",
                        **{k: float(v[0]) for k, v in m.items()},
                    })
                for strat_nm, nr_sc, ru_sc in (
                    ("strat_M2_marg", q_nr_score * 0 + m2_all, margin_score),
                    ("strat_q_marg", q_nr_score, margin_score),
                    ("strat_M2_M3", m2_all, np.where(hm, 0.0, 0.0)),
                ):
                    if strat_nm == "strat_M2_M3":
                        ru_full = np.zeros(len(f), float)
                        ru_full[hm] = m3_sub
                        ru_sc = ru_full
                    m = rank_subgroup_stratified(y, hm, nr_sc, ru_sc, ones)
                    point_rows.append({
                        "subset": subset, "pool": "whole_assigned", "outcome": outcome, "seed": s,
                        "n": len(f), "positives": int(y.sum()), "method": strat_nm,
                        "auc": np.nan, "ap": np.nan, **{k: float(v[0]) for k, v in m.items()},
                    })

            for design in ("scene", "image"):
                reps = B if (subset, design) == ("primary", "scene") else max(200, B // 5)
                pick = np.random.default_rng(SEED_RESAMPLE).integers(0, 4, size=(reps, 4))

                def run_boot(s: int):
                    f = frames[s]
                    acc = {}
                    arr = raw_arrays[s]
                    y, hm = arr["y"], arr["hm"]
                    gen = (
                        scene_weights(f.scene.map(si).to_numpy(int), len(scenes), reps)
                        if design == "scene"
                        else image_weights(f.image_id.map(ii).to_numpy(int), im.stratum.to_numpy(str), reps)
                    )
                    for W in gen:
                        for nm, rk in whole_rankers[s].items():
                            for k, v in rk.all(W).items():
                                acc.setdefault((nm, k), []).append(v)
                        for nr_nm, rk in nr_rankers[s].items():
                            for k, v in rk.all(W[:, ~hm]).items():
                                acc.setdefault((f"nr_{nr_nm}", k), []).append(v)
                        ru_full_m3 = np.zeros(len(f), float)
                        ru_full_m3[hm] = arr["m3"]
                        for strat_nm, nr_sc, ru_sc in (
                            ("strat_M2_marg", arr["m2"], arr["marg"]),
                            ("strat_q_marg", arr["q"], arr["marg"]),
                            ("strat_M2_M3", arr["m2"], ru_full_m3),
                        ):
                            ms = rank_subgroup_stratified(y, hm, nr_sc, ru_sc, W)
                            for k, v in ms.items():
                                acc.setdefault((strat_nm, k), []).append(v)
                    return {mk: np.concatenate(v) for mk, v in acc.items()}

                with ThreadPoolExecutor(4) as ex:
                    per = list(ex.map(run_boot, range(4)))
                V = {mk: np.stack([per[s][mk] for s in range(4)]) for mk in per[0]}
                key = f"{subset}|{outcome}|{design}"
                e = {}
                for (m, k), v in V.items():
                    e[f"{m}_{k}"] = {"crossed": crossed(v, pick), "fixed_seeds": ci(v.mean(0))}
                    if m in ("ranklast", "rankfirst", "joint_M3_M2", "strat_M2_marg", "strat_q_marg", "strat_M2_M3"):
                        d = v - V["M2_all", k]
                        e[f"{m}_minus_M2_{k}"] = {
                            "crossed": crossed(d, pick),
                            "fixed_seeds": ci(d.mean(0)),
                            "per_seed": [ci(x) for x in d],
                        }
                        if m != "ranklast":
                            d_rl = v - V["ranklast", k]
                            e[f"{m}_minus_ranklast_{k}"] = {
                                "crossed": crossed(d_rl, pick),
                                "fixed_seeds": ci(d_rl.mean(0)),
                            }
                intervals[key] = e
                print(f"[{time.time()-t0:.0f}s] {key} done", flush=True)

    pt_df = pd.DataFrame(point_rows)
    pt_df.to_csv(OUT_DIR / "per_seed_points.csv", index=False)
    mean_df = pt_df.groupby(["subset", "pool", "outcome", "method"], sort=False).mean(numeric_only=True).reset_index()
    mean_df.drop(columns=["seed"], errors="ignore").to_csv(OUT_DIR / "seed_mean_points.csv", index=False)
    summary = {"B": B, "seed": 20261123, "seed_resample": SEED_RESAMPLE, "subgroup_rates": subgroup_rates, "intervals": intervals}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"finished in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
