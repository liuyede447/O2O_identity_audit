"""Read-only runner-free and dense common-risk-set sensitivity analysis.

This script consumes only frozen endpoint and dense-grid CSV artifacts.  It
does not load a detector, perform replay, train, or create new model outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SIZE_BINS = ("t_8_16", "s_16_32")
TAU = 0.125
ENDPOINT_REQUIRED = {
    "image_id", "gt_id", "size_bin", "stratum", "sampling_weight",
    "o2o_active", "o2o_runner", "o2o_fragile", "o2m_rank", "o2m_runner", "o2m_fragile",
    "common_valid_margin",
}
DENSE_REQUIRED = {
    "image_id", "gt_id", "direction", "size_bin", "stratum", "sampling_weight",
    "o2o_base_active", "o2o_event_observed", "o2o_radius", "o2o_right_censored",
    "o2o_censor_radius", "o2o_censor_reason", "o2o_first_divergence",
    "o2m_assigned_positive_base", "o2m_assigned_positive_event_observed",
    "o2m_assigned_positive_radius", "o2m_assigned_positive_right_censored",
    "o2m_assigned_positive_censor_radius", "o2m_assigned_positive_censor_reason",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def weighted_mean(frame: pd.DataFrame, value: str) -> float | None:
    vals = pd.to_numeric(frame[value], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame["sampling_weight"], errors="coerce").to_numpy(float)
    keep = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return None
    return float(np.average(vals[keep], weights=weights[keep]))


def endpoint_supports(frame: pd.DataFrame) -> dict[str, pd.Series]:
    o2o = frame["o2o_active"].notna()
    o2m = frame["o2m_rank"].notna()
    state_common = o2o & o2m
    competition = frame["common_valid_margin"].astype(int).eq(1)
    if not competition.equals(state_common & frame["o2o_runner"].notna() & frame["o2m_runner"].notna()):
        raise ValueError("common_valid_margin is not equal to both-state-and-runner rule")
    return {
        "o2o_natural": o2o,
        "o2m_natural": o2m,
        "state_common_runner_free": state_common,
        "competition_defined_common": competition,
    }


def endpoint_summary(frame: pd.DataFrame, contract: str) -> pd.DataFrame:
    supports = endpoint_supports(frame)
    rows: list[dict] = []
    for support_name, mask in supports.items():
        part = frame.loc[mask].copy()
        for size in ("all", *SIZE_BINS):
            sub = part if size == "all" else part.loc[part["size_bin"].eq(size)]
            o2o_defined = sub["o2o_active"].notna()
            o2m_defined = sub["o2m_rank"].notna()
            o2o = sub.loc[o2o_defined]
            o2m = sub.loc[o2m_defined]
            o2o_est = weighted_mean(o2o, "o2o_fragile")
            o2m_est = weighted_mean(o2m, "o2m_fragile")
            paired = None if o2o_est is None or o2m_est is None else o2o_est - o2m_est
            rows.append({
                "contract": contract,
                "support": support_name,
                "size_bin": size,
                "rows": int(len(sub)),
                "images": int(sub["image_id"].nunique()),
                "o2o_defined_rows": int(len(o2o)),
                "o2m_defined_rows": int(len(o2m)),
                "o2o_fragility_weighted": o2o_est,
                "o2m_top1_fragility_weighted": o2m_est,
                "o2o_minus_o2m_on_support": paired,
            })
    return pd.DataFrame(rows)


def dense_risk_counts(frame: pd.DataFrame) -> pd.DataFrame:
    o2o = frame["o2o_base_active"].notna()
    o2m = frame["o2m_assigned_positive_base"].notna()
    common = o2o & o2m
    rows: list[dict] = []
    for size in ("all", *SIZE_BINS):
        scope = frame if size == "all" else frame.loc[frame["size_bin"].eq(size)]
        masks = {
            "all_trajectories": pd.Series(True, index=scope.index),
            "o2o_initial_defined": scope["o2o_base_active"].notna(),
            "o2m_initial_defined": scope["o2m_assigned_positive_base"].notna(),
            "common_initial_defined": scope["o2o_base_active"].notna() & scope["o2m_assigned_positive_base"].notna(),
        }
        for name, mask in masks.items():
            rows.append({"size_bin": size, "metric": name, "count": int(mask.sum()), "tau": TAU})
        for branch, base_mask, event, radius, censor, censor_reason in (
            ("o2o", scope["o2o_base_active"].notna(), "o2o_event_observed", "o2o_radius", "o2o_censor_radius", "o2o_censor_reason"),
            ("o2m", scope["o2m_assigned_positive_base"].notna(), "o2m_assigned_positive_event_observed", "o2m_assigned_positive_radius", "o2m_assigned_positive_censor_radius", "o2m_assigned_positive_censor_reason"),
        ):
            observed = pd.to_numeric(scope[event], errors="coerce").fillna(0).astype(bool)
            r = pd.to_numeric(scope[radius], errors="coerce")
            cr = pd.to_numeric(scope[censor], errors="coerce")
            reason = scope[censor_reason].fillna("").astype(str)
            rows.extend([
                {"size_bin": size, "metric": f"{branch}_first_event_through_tau", "count": int((base_mask & observed & (r <= TAU + 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_geometric_censor_before_tau", "count": int((base_mask & ~observed & reason.eq("image_boundary") & (cr < TAU - 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_administrative_rmax_censor_before_tau", "count": int((base_mask & ~observed & reason.eq("rmax") & (cr < TAU - 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_observable_through_tau", "count": int((base_mask & ((observed & (r <= TAU + 1e-12)) | (~observed & (cr >= TAU - 1e-12)))).sum()), "tau": TAU},
            ])
        for branch, base_mask, event, radius, censor, censor_reason in (
            ("common_o2o", common.loc[scope.index], "o2o_event_observed", "o2o_radius", "o2o_censor_radius", "o2o_censor_reason"),
            ("common_o2m", common.loc[scope.index], "o2m_assigned_positive_event_observed", "o2m_assigned_positive_radius", "o2m_assigned_positive_censor_radius", "o2m_assigned_positive_censor_reason"),
        ):
            observed = pd.to_numeric(scope[event], errors="coerce").fillna(0).astype(bool)
            r = pd.to_numeric(scope[radius], errors="coerce")
            cr = pd.to_numeric(scope[censor], errors="coerce")
            reason = scope[censor_reason].fillna("").astype(str)
            rows.extend([
                {"size_bin": size, "metric": f"{branch}_first_event_through_tau", "count": int((base_mask & observed & (r <= TAU + 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_geometric_censor_before_tau", "count": int((base_mask & ~observed & reason.eq("image_boundary") & (cr < TAU - 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_administrative_rmax_censor_before_tau", "count": int((base_mask & ~observed & reason.eq("rmax") & (cr < TAU - 1e-12)).sum()), "tau": TAU},
                {"size_bin": size, "metric": f"{branch}_observable_through_tau", "count": int((base_mask & ((observed & (r <= TAU + 1e-12)) | (~observed & (cr >= TAU - 1e-12)))).sum()), "tau": TAU},
            ])
    return pd.DataFrame(rows)


def dense_rmsr(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    # Reuse the frozen production postprocessor's KM/RMCBD implementation.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_assignment_boundary_geometry import bootstrap_multiplicities, cluster_index, evaluate_process, process_matrices

    rows: list[dict] = []
    for size in ("all", *SIZE_BINS):
        scope = frame if size == "all" else frame.loc[frame["size_bin"].eq(size)]
        supports = {
            "o2o_natural": scope["o2o_base_active"].notna(),
            "o2m_natural": scope["o2m_assigned_positive_base"].notna(),
            "common_trajectory": scope["o2o_base_active"].notna() & scope["o2m_assigned_positive_base"].notna(),
        }
        for support_name, mask in supports.items():
            sub = scope.loc[mask].copy()
            if sub.empty:
                continue
            meta, lookup = cluster_index(sub, sub)
            # BUGFIX 2026-09-09: the production estimator (analyze_assignment_boundary_geometry.py)
            # prepends an identity (all-ones) weight row so index 0 of every evaluated array is the
            # true full-sample point estimate, with the 5,000 resampled rows starting at index 1. This
            # script previously omitted that identity row, so index 0 was the FIRST RANDOM BOOTSTRAP
            # REPLICATE mislabeled as the point estimate, and the reported CI silently dropped one of
            # the 5,000 replicates. Fixed to match the production convention exactly.
            boot = bootstrap_multiplicities(meta, 5000, 20260831)
            multipliers = np.vstack([np.ones((1, len(meta)), dtype=np.int32), boot])
            eval_grid = np.array([TAU], dtype=float)
            branch_values: dict[str, np.ndarray] = {}
            for estimand, label in (("o2o", "o2o"), ("o2m_assigned_positive", "o2m")):
                process = process_matrices(sub, estimand, TAU, lookup, len(meta))
                evaluated = evaluate_process(process, multipliers, TAU, eval_grid)
                branch_values[label] = evaluated["rmsr"]
                vals = evaluated["rmsr"]
                rows.append({
                    "size_bin": size,
                    "support": support_name,
                    "estimand": label,
                    "rows": int(len(sub)),
                    "images": int(sub["image_id"].nunique()),
                    "rmcbd_tau_0125": float(vals[0]),
                    "ci_low": float(np.quantile(vals[1:], 0.025)),
                    "ci_high": float(np.quantile(vals[1:], 0.975)),
                    "bootstrap_reps": 5000,
                    "bootstrap_seed": 20260831,
                })
            if support_name == "common_trajectory":
                diff = branch_values["o2o"] - branch_values["o2m"]
                rows.append({
                    "size_bin": size,
                    "support": support_name,
                    "estimand": "o2o_minus_o2m",
                    "rows": int(len(sub)),
                    "images": int(sub["image_id"].nunique()),
                    "rmcbd_tau_0125": float(diff[0]),
                    "ci_low": float(np.quantile(diff[1:], 0.025)),
                    "ci_high": float(np.quantile(diff[1:], 0.975)),
                    "bootstrap_reps": 5000,
                    "bootstrap_seed": 20260831,
                })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixed-endpoint", type=Path, required=True)
    p.add_argument("--equivalent-endpoint", type=Path, required=True)
    p.add_argument("--dense-events", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    fixed = pd.read_csv(args.fixed_endpoint)
    equiv = pd.read_csv(args.equivalent_endpoint)
    dense = pd.read_csv(args.dense_events)
    require(fixed, ENDPOINT_REQUIRED, "fixed endpoint")
    require(equiv, ENDPOINT_REQUIRED, "equivalent endpoint")
    require(dense, DENSE_REQUIRED, "dense events")
    if len(fixed) != 7434 or len(equiv) != 7432 or len(dense) != 29724:
        raise ValueError("frozen row counts do not match the recorded artifacts")
    args.output_dir.mkdir(parents=True)
    endpoint = pd.concat([endpoint_summary(fixed, "fixed_1px"), endpoint_summary(equiv, "equivalent_area_k00625")], ignore_index=True)
    endpoint.to_csv(args.output_dir / "RUNNER_FREE_ENDPOINT_SENSITIVITY.csv", index=False)
    counts = dense_risk_counts(dense)
    counts.to_csv(args.output_dir / "DENSE_COMMON_RISKSET_COUNTS.csv", index=False)
    rmsr = dense_rmsr(dense, args.output_dir)
    rmsr.to_csv(args.output_dir / "DENSE_COMMON_TRAJECTORY_RMCBD.csv", index=False)
    manifest = {
        "status": "PASS",
        "read_only": True,
        "new_model_outcome_access": False,
        "inputs": {
            "fixed_endpoint": {"path": str(args.fixed_endpoint.resolve()), "sha256": sha256(args.fixed_endpoint), "rows": len(fixed)},
            "equivalent_endpoint": {"path": str(args.equivalent_endpoint.resolve()), "sha256": sha256(args.equivalent_endpoint), "rows": len(equiv)},
            "dense_events": {"path": str(args.dense_events.resolve()), "sha256": sha256(args.dense_events), "rows": len(dense)},
        },
        "endpoint_support_definitions": {
            "o2o_natural": "o2o_active is defined; no runner requirement",
            "o2m_natural": "o2m_rank is defined; no runner requirement",
            "state_common_runner_free": "o2o_active and o2m_rank are both defined; no runner requirement",
            "competition_defined_common": "common_valid_margin == 1, equivalent to both states and both runners defined",
        },
        "dense_support_definitions": {
            "o2o_natural": "o2o_base_active is defined",
            "o2m_natural": "o2m_assigned_positive_base is defined",
            "common_trajectory": "both dense base states are defined on the same (image_id, gt_id, direction) row",
        },
        "dense_rmcbd": {"tau": TAU, "bootstrap_reps": 5000, "bootstrap_seed": 20260831, "estimator": "frozen weighted Kaplan-Meier integral"},
        "outputs": ["RUNNER_FREE_ENDPOINT_SENSITIVITY.csv", "DENSE_COMMON_RISKSET_COUNTS.csv", "DENSE_COMMON_TRAJECTORY_RMCBD.csv", "MANIFEST.json"],
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
