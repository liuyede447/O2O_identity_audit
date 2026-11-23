"""CPU-only summaries for the continuous assignment-boundary audit.

The input contract is produced by ``analyze_assignment_boundary_radius.py``.
This script deliberately separates first-boundary events from endpoint state:

* weighted Kaplan--Meier overall identity survival and Aalen--Johansen
  cause-specific cumulative incidence for the O2O boundary taxonomy;
* restricted mean stability radius (RMSR) without a proportional-hazards model;
* O2M rank/set endpoint summaries at prespecified radii; and
* observed-case-only margin construct checks with transparent rank-boundary
  censoring status.

All uncertainty uses inverse-probability weights from the frozen sampling
manifest and a stratified image-cluster bootstrap.  The conditional Spearman
analyses do not estimate a censoring-adjusted population correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SIZE_GROUPS = ("all", "t_8_16", "s_16_32")
ESTIMANDS = {
    "o2o": ("o2o_base_active", "o2o_event_observed", "o2o_radius", "o2o_censor_radius", "o2o_active"),
    "o2m_legacy": ("o2m_legacy_base", "o2m_legacy_event_observed", "o2m_legacy_radius", "o2m_legacy_censor_radius", "o2m_legacy_rank"),
    "o2m_pre_topk": ("o2m_pre_topk_base", "o2m_pre_topk_event_observed", "o2m_pre_topk_radius", "o2m_pre_topk_censor_radius", "o2m_pre_topk_rank"),
    "o2m_assigned_positive": (
        "o2m_assigned_positive_base",
        "o2m_assigned_positive_event_observed",
        "o2m_assigned_positive_radius",
        "o2m_assigned_positive_censor_radius",
        "o2m_assigned_positive_rank",
    ),
}
CAUSES = ("eligibility", "topk", "conflict", "within_set", "other")
CAUSE_MAP = {
    "eligibility_boundary": "eligibility",
    "topk_membership_transition": "topk",
    "conflict_reassignment": "conflict",
    "within_set_geometry_rank_reversal": "within_set",
    "active_disappearance": "other",
    "compound_or_other": "other",
}
DEFAULT_RADII = (0.03125, 0.0625, 0.125)
KEY_COLUMNS = ("image_id", "gt_id", "direction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-csv", "--events", dest="event_csv", type=Path)
    parser.add_argument("--curve-csv", "--curves", dest="curve_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tau", type=float, help="Prespecified RMSR truncation radius")
    parser.add_argument("--endpoint-radii", type=float, nargs="+", default=list(DEFAULT_RADII))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-report", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def group_frame(frame: pd.DataFrame, size_group: str) -> pd.DataFrame:
    return frame if size_group == "all" else frame.loc[frame["size_bin"].eq(size_group)]


def base_defined(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").notna().to_numpy()


def percentile_interval(values: np.ndarray, level: float) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def number(value: float | np.floating | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def bootstrap_multiplicities(cluster_meta: pd.DataFrame, reps: int, seed: int) -> np.ndarray:
    """Stratified image-cluster resampling represented as cluster frequencies."""
    clusters = cluster_meta.reset_index(drop=True)
    result = np.ones((1, len(clusters)), dtype=np.int32) if reps == 0 else np.zeros((reps, len(clusters)), dtype=np.int32)
    if reps == 0:
        return result
    rng = np.random.default_rng(seed)
    for _, indices in clusters.groupby("stratum", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        probabilities = np.full(len(positions), 1.0 / len(positions))
        result[:, positions] = rng.multinomial(len(positions), probabilities, size=reps)
    return result


def cluster_index(events: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    meta = pd.concat(
        [events[["image_id", "stratum"]], curves[["image_id", "stratum"]]], ignore_index=True
    ).drop_duplicates()
    conflicting = meta.groupby("image_id")["stratum"].nunique()
    if (conflicting > 1).any():
        raise ValueError("an image_id occurs in more than one sampling stratum")
    meta = meta.drop_duplicates("image_id").sort_values(["stratum", "image_id"]).reset_index(drop=True)
    return meta, {str(image_id): index for index, image_id in enumerate(meta["image_id"])}


def event_arrays(frame: pd.DataFrame, estimand: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_col, event_col, radius_col, censor_col, _ = ESTIMANDS[estimand]
    defined = base_defined(frame[base_col])
    sub = frame.loc[defined]
    observed = pd.to_numeric(sub[event_col], errors="raise").to_numpy(dtype=int).astype(bool)
    radii = pd.to_numeric(sub[radius_col], errors="coerce").to_numpy(dtype=float)
    censors = pd.to_numeric(sub[censor_col], errors="coerce").to_numpy(dtype=float)
    durations = np.where(observed, radii, censors)
    if np.any(~np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError(f"{estimand} has invalid event/censor durations among base-defined rows")
    causes = np.full(len(sub), "identity_change", dtype=object)
    if estimand == "o2o":
        raw = sub["o2o_first_divergence"].fillna("").astype(str).to_numpy()
        causes = np.array([CAUSE_MAP.get(item, "other") for item in raw], dtype=object)
    return defined, durations, observed, causes


def process_matrices(
    frame: pd.DataFrame,
    estimand: str,
    tau: float,
    cluster_lookup: dict[str, int],
    n_clusters: int,
) -> dict:
    defined, durations, observed, causes = event_arrays(frame, estimand)
    sub = frame.loc[defined].reset_index(drop=True)
    weights = pd.to_numeric(sub["sampling_weight"], errors="raise").to_numpy(dtype=float)
    cluster_ids = np.array([cluster_lookup[str(item)] for item in sub["image_id"]], dtype=int)
    event_times = np.unique(durations[observed & (durations <= tau + 1e-12)])
    event_times.sort()
    risk = np.zeros((n_clusters, len(event_times)), dtype=float)
    cause_names = CAUSES if estimand == "o2o" else ("identity_change",)
    increments = {cause: np.zeros_like(risk) for cause in cause_names}
    for time_index, event_time in enumerate(event_times):
        np.add.at(risk[:, time_index], cluster_ids[durations >= event_time - 1e-12], weights[durations >= event_time - 1e-12])
        at_time = observed & np.isclose(durations, event_time, atol=1e-12, rtol=0.0)
        for cause in cause_names:
            take = at_time & (causes == cause)
            np.add.at(increments[cause][:, time_index], cluster_ids[take], weights[take])
    return {
        "times": event_times,
        "risk": risk,
        "increments": increments,
        "cause_names": cause_names,
        "n_rows": len(sub),
        "excluded_base_undefined": int((~defined).sum()),
        "total_weight": float(weights.sum()),
    }


def evaluate_process(process: dict, multipliers: np.ndarray, tau: float, eval_grid: np.ndarray) -> dict:
    times = process["times"]
    boot_risk = multipliers @ process["risk"]
    boot_d = {cause: multipliers @ values for cause, values in process["increments"].items()}
    n_boot = len(multipliers)
    survival = np.ones(n_boot, dtype=float)
    cif = {cause: np.zeros(n_boot, dtype=float) for cause in process["cause_names"]}
    rmsr = np.zeros(n_boot, dtype=float)
    curve_s = np.ones((n_boot, len(eval_grid)), dtype=float)
    curve_f = {cause: np.zeros((n_boot, len(eval_grid)), dtype=float) for cause in process["cause_names"]}
    previous = 0.0
    grid_cursor = 0
    for time_index, event_time in enumerate(times):
        while grid_cursor < len(eval_grid) and eval_grid[grid_cursor] < event_time - 1e-12:
            curve_s[:, grid_cursor] = survival
            for cause in process["cause_names"]:
                curve_f[cause][:, grid_cursor] = cif[cause]
            grid_cursor += 1
        rmsr += survival * max(0.0, min(float(event_time), tau) - previous)
        y = boot_risk[:, time_index]
        total_d = np.zeros(n_boot, dtype=float)
        hazards = {}
        for cause in process["cause_names"]:
            hazards[cause] = np.divide(boot_d[cause][:, time_index], y, out=np.zeros(n_boot), where=y > 0)
            total_d += boot_d[cause][:, time_index]
        for cause in process["cause_names"]:
            cif[cause] += survival * hazards[cause]
        survival *= 1.0 - np.divide(total_d, y, out=np.zeros(n_boot), where=y > 0)
        survival = np.clip(survival, 0.0, 1.0)
        previous = min(float(event_time), tau)
        while grid_cursor < len(eval_grid) and eval_grid[grid_cursor] <= event_time + 1e-12:
            curve_s[:, grid_cursor] = survival
            for cause in process["cause_names"]:
                curve_f[cause][:, grid_cursor] = cif[cause]
            grid_cursor += 1
    rmsr += survival * max(0.0, tau - previous)
    while grid_cursor < len(eval_grid):
        curve_s[:, grid_cursor] = survival
        for cause in process["cause_names"]:
            curve_f[cause][:, grid_cursor] = cif[cause]
        grid_cursor += 1
    return {"survival": curve_s, "cif": curve_f, "rmsr": rmsr}


def append_estimate(rows: list[dict], base: dict, estimate: float, bootstrap: np.ndarray, level: float) -> None:
    low, high = percentile_interval(bootstrap[1:] if len(bootstrap) > 1 else np.array([]), level)
    rows.append({**base, "estimate": number(estimate), "ci_low": low, "ci_high": high})


def weighted_ratio_by_cluster(
    frame: pd.DataFrame,
    value: np.ndarray,
    cluster_lookup: dict[str, int],
    n_clusters: int,
    multipliers: np.ndarray,
    denominator_mask: np.ndarray | None = None,
) -> np.ndarray:
    weights = pd.to_numeric(frame["sampling_weight"], errors="raise").to_numpy(dtype=float)
    valid = np.isfinite(value)
    if denominator_mask is not None:
        valid &= denominator_mask
    numerator = np.zeros(n_clusters, dtype=float)
    denominator = np.zeros(n_clusters, dtype=float)
    cluster_ids = np.array([cluster_lookup[str(item)] for item in frame["image_id"]], dtype=int)
    np.add.at(numerator, cluster_ids[valid], weights[valid] * value[valid])
    np.add.at(denominator, cluster_ids[valid], weights[valid])
    num = multipliers @ numerator
    den = multipliers @ denominator
    return np.divide(num, den, out=np.full(len(multipliers), np.nan), where=den > 0)


def exact_endpoint(curves: pd.DataFrame, radius: float) -> pd.DataFrame:
    selected = curves.loc[np.isclose(curves["radius"].to_numpy(dtype=float), radius, atol=1e-12, rtol=0.0)].copy()
    duplicates = selected.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError(f"curve CSV has duplicate trajectory rows at radius {radius}")
    return selected


def same_identity(left: pd.Series, right: pd.Series) -> np.ndarray:
    sentinel = -9_223_372_036_854_775_000
    a = pd.to_numeric(left, errors="coerce").fillna(sentinel).to_numpy()
    b = pd.to_numeric(right, errors="coerce").fillna(sentinel).to_numpy()
    return a == b


def weighted_midranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values, sorted_weights = values[order], weights[order]
    ranks_sorted = np.empty(len(values), dtype=float)
    cumulative = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        group_weight = float(sorted_weights[start:stop].sum())
        ranks_sorted[start:stop] = cumulative + 0.5 * group_weight
        cumulative += group_weight
        start = stop
    result = np.empty(len(values), dtype=float)
    result[order] = ranks_sorted
    return result


def weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x, y, weights = x[valid], y[valid], weights[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx, ry = weighted_midranks(x, weights), weighted_midranks(y, weights)
    total = weights.sum()
    mx, my = np.dot(weights, rx) / total, np.dot(weights, ry) / total
    covariance = np.dot(weights, (rx - mx) * (ry - my)) / total
    vx = np.dot(weights, (rx - mx) ** 2) / total
    vy = np.dot(weights, (ry - my) ** 2) / total
    return float(covariance / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def bootstrap_spearman(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_lookup: dict[str, int],
    multipliers: np.ndarray,
) -> np.ndarray:
    x = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    base_weights = pd.to_numeric(frame["sampling_weight"], errors="raise").to_numpy(dtype=float)
    cluster_ids = np.array([cluster_lookup[str(item)] for item in frame["image_id"]], dtype=int)
    values = np.empty(len(multipliers), dtype=float)
    for index, cluster_counts in enumerate(multipliers):
        values[index] = weighted_spearman(x, y, base_weights * cluster_counts[cluster_ids])
    return values


def analyze(events: pd.DataFrame, curves: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame | dict]:
    if args.tau is None or args.tau <= 0:
        raise ValueError("--tau must be positive")
    if args.bootstrap_reps < 0 or not (0 < args.ci_level < 1):
        raise ValueError("invalid bootstrap-reps or ci-level")
    radii = sorted(set(float(value) for value in args.endpoint_radii))
    if any(value <= 0 or value > args.tau + 1e-12 for value in radii):
        raise ValueError("endpoint radii must be in (0, tau]")

    event_required = {"image_id", "gt_id", "direction", "size_bin", "stratum", "sampling_weight", "o2o_first_divergence", "o2o_base_margin"}
    for columns in ESTIMANDS.values():
        event_required.update(columns[:4])
    event_required.update({"rho_R_event_observed", "rho_R_radius", "rho_R_right_censored", "rho_R_competing_censored", "rho_R_censor_reason"})
    curve_required = {"image_id", "gt_id", "direction", "size_bin", "stratum", "sampling_weight", "radius", "o2m_positive_set_jaccard", "o2m_positive_set_retention", "o2m_positive_set_exact_change"}
    curve_required.update(item[4] for item in ESTIMANDS.values())
    require_columns(events, event_required, "event CSV")
    require_columns(curves, curve_required, "curve CSV")

    cluster_meta, cluster_lookup = cluster_index(events, curves)
    bootstrap = bootstrap_multiplicities(cluster_meta, args.bootstrap_reps, args.seed)
    multipliers = np.vstack([np.ones((1, len(cluster_meta)), dtype=np.int32), bootstrap]) if args.bootstrap_reps else bootstrap

    survival_rows: list[dict] = []
    rmsr_rows: list[dict] = []
    rmsr_boot: dict[tuple[str, str], np.ndarray] = {}
    for size_group in SIZE_GROUPS:
        subset = group_frame(events, size_group)
        for estimand in ESTIMANDS:
            process = process_matrices(subset, estimand, args.tau, cluster_lookup, len(cluster_meta))
            grid = np.unique(np.concatenate(([0.0, args.tau], np.asarray(radii), process["times"])))
            grid = grid[(grid >= 0) & (grid <= args.tau + 1e-12)]
            evaluated = evaluate_process(process, multipliers, args.tau, grid)
            rmsr_boot[(size_group, estimand)] = evaluated["rmsr"]
            for time_index, radius in enumerate(grid):
                append_estimate(
                    survival_rows,
                    {"size_group": size_group, "estimand": estimand, "metric": "survival", "radius": float(radius)},
                    evaluated["survival"][0, time_index],
                    evaluated["survival"][:, time_index],
                    args.ci_level,
                )
                for cause, values in evaluated["cif"].items():
                    append_estimate(
                        survival_rows,
                        {"size_group": size_group, "estimand": estimand, "metric": f"cif_{cause}", "radius": float(radius)},
                        values[0, time_index],
                        values[:, time_index],
                        args.ci_level,
                    )
            append_estimate(
                rmsr_rows,
                {
                    "size_group": size_group,
                    "estimand": estimand,
                    "tau": args.tau,
                    "n_base_defined": process["n_rows"],
                    "n_base_undefined_excluded": process["excluded_base_undefined"],
                    "weighted_n": process["total_weight"],
                },
                evaluated["rmsr"][0],
                evaluated["rmsr"],
                args.ci_level,
            )

    rmsr_contrast_rows: list[dict] = []
    for size_group in SIZE_GROUPS:
        for o2m_estimand in ("o2m_legacy", "o2m_pre_topk", "o2m_assigned_positive"):
            contrast = rmsr_boot[(size_group, "o2o")] - rmsr_boot[(size_group, o2m_estimand)]
            append_estimate(
                rmsr_contrast_rows,
                {"size_group": size_group, "contrast": f"o2o_minus_{o2m_estimand}", "tau": args.tau},
                contrast[0],
                contrast,
                args.ci_level,
            )

    endpoint_rows: list[dict] = []
    return_rows: list[dict] = []
    event_lookup = events.set_index(list(KEY_COLUMNS), verify_integrity=True)
    for radius in radii:
        endpoint = exact_endpoint(curves, radius)
        if endpoint.empty:
            raise ValueError(f"curve CSV contains no exact evaluations at prespecified radius {radius}")
        joined = endpoint.join(event_lookup, on=list(KEY_COLUMNS), rsuffix="_event", validate="one_to_one")
        if joined["size_bin_event"].isna().any():
            raise ValueError(f"endpoint rows at radius {radius} do not all match an event row")
        for size_group in SIZE_GROUPS:
            sub = group_frame(joined, size_group)
            jaccard = pd.to_numeric(sub["o2m_positive_set_jaccard"], errors="coerce").to_numpy(dtype=float)
            retention = pd.to_numeric(sub["o2m_positive_set_retention"], errors="coerce").to_numpy(dtype=float)
            set_changed = pd.to_numeric(sub["o2m_positive_set_exact_change"], errors="coerce").to_numpy(dtype=float)
            for estimand in ("o2m_legacy", "o2m_pre_topk", "o2m_assigned_positive"):
                base_col, _, _, _, endpoint_col = ESTIMANDS[estimand]
                defined = base_defined(sub[base_col])
                rank_changed = (~same_identity(sub[base_col], sub[endpoint_col])).astype(float)
                metrics = {
                    "rank_changed_rate": rank_changed,
                    "set_exact_changed_rate": set_changed,
                    "rank_changed_set_exact_same_rate": rank_changed * (1.0 - set_changed),
                    "rank_minus_set_exact_change_rate": rank_changed - set_changed,
                    "mean_jaccard": jaccard,
                    "mean_retention": retention,
                    "mean_jaccard_among_rank_changed": jaccard,
                    "mean_retention_among_rank_changed": retention,
                }
                for metric, values in metrics.items():
                    denominator = defined & (rank_changed.astype(bool) if metric.endswith("among_rank_changed") else True)
                    estimates = weighted_ratio_by_cluster(sub, values, cluster_lookup, len(cluster_meta), multipliers, denominator)
                    append_estimate(endpoint_rows, {"size_group": size_group, "estimand": estimand, "radius": radius, "metric": metric, "n_rows": int(denominator.sum())}, estimates[0], estimates, args.ci_level)

            for estimand, (base_col, event_col, event_radius_col, _, endpoint_col) in ESTIMANDS.items():
                defined = base_defined(sub[base_col])
                observed = pd.to_numeric(sub[event_col], errors="raise").to_numpy(dtype=int).astype(bool)
                event_radius = pd.to_numeric(sub[event_radius_col], errors="coerce").to_numpy(dtype=float)
                ever = (observed & (event_radius <= radius + 1e-12)).astype(float)
                endpoint_changed = (~same_identity(sub[base_col], sub[endpoint_col])).astype(float)
                diagnostic = {
                    "ever_boundary_rate": ever,
                    "endpoint_changed_rate": endpoint_changed,
                    "ever_and_endpoint_changed_rate": ever * endpoint_changed,
                    "returned_to_base_after_boundary_rate": ever * (1.0 - endpoint_changed),
                    "endpoint_changed_without_recorded_boundary_rate": (1.0 - ever) * endpoint_changed,
                }
                for metric, values in diagnostic.items():
                    estimates = weighted_ratio_by_cluster(sub, values, cluster_lookup, len(cluster_meta), multipliers, defined)
                    append_estimate(return_rows, {"size_group": size_group, "estimand": estimand, "radius": radius, "metric": metric, "n_base_defined": int(defined.sum())}, estimates[0], estimates, args.ci_level)

    status_rows: list[dict] = []
    correlation_rows: list[dict] = []
    for size_group in SIZE_GROUPS:
        sub = group_frame(events, size_group).copy()
        observed = pd.to_numeric(sub["rho_R_event_observed"], errors="raise").to_numpy(dtype=int).astype(bool)
        competing = pd.to_numeric(sub["rho_R_competing_censored"], errors="raise").to_numpy(dtype=int).astype(bool)
        right = pd.to_numeric(sub["rho_R_right_censored"], errors="raise").to_numpy(dtype=int).astype(bool)
        if np.any(observed.astype(int) + competing.astype(int) + right.astype(int) > 1):
            raise ValueError("rho_R status flags are not mutually exclusive")
        undefined = ~(observed | competing | right)
        for status, mask in {"observed": observed, "competing_censored": competing, "right_censored": right, "undefined": undefined}.items():
            estimates = weighted_ratio_by_cluster(sub, mask.astype(float), cluster_lookup, len(cluster_meta), multipliers)
            append_estimate(status_rows, {"size_group": size_group, "status": status, "n_rows": int(mask.sum())}, estimates[0], estimates, args.ci_level)
        reasons = sub.loc[undefined, "rho_R_censor_reason"].fillna("unspecified").astype(str)
        for reason, count in reasons.value_counts().sort_index().items():
            status_rows.append({"size_group": size_group, "status": f"undefined_reason:{reason}", "n_rows": int(count), "estimate": None, "ci_low": None, "ci_high": None})

        observed_frame = sub.loc[observed & pd.to_numeric(sub["o2o_base_margin"], errors="coerce").notna()].copy()
        eligibility_frame = sub.loc[
            pd.to_numeric(sub["o2o_event_observed"], errors="raise").astype(bool)
            & sub["o2o_first_divergence"].eq("eligibility_boundary")
            & pd.to_numeric(sub["o2o_base_margin"], errors="coerce").notna()
        ].copy()
        correlation_boot: dict[str, np.ndarray] = {}
        for analysis_name, corr_frame, y_col in (
            ("margin_vs_observed_rho_R", observed_frame, "rho_R_radius"),
            ("specificity_control_margin_vs_eligibility_cause_overall_radius", eligibility_frame, "o2o_radius"),
        ):
            values = bootstrap_spearman(corr_frame, "o2o_base_margin", y_col, cluster_lookup, multipliers)
            correlation_boot[analysis_name] = values
            append_estimate(
                correlation_rows,
                {"size_group": size_group, "analysis": analysis_name, "conditioning": "observed_cases_only", "n_rows": len(corr_frame), "n_images": corr_frame["image_id"].nunique(), "censoring_adjusted": False},
                values[0],
                values,
                args.ci_level,
            )
        difference = (
            correlation_boot["margin_vs_observed_rho_R"]
            - correlation_boot["specificity_control_margin_vs_eligibility_cause_overall_radius"]
        )
        append_estimate(
            correlation_rows,
            {
                "size_group": size_group,
                "analysis": "specificity_difference_rhoR_minus_eligibility_correlation",
                "conditioning": "two_separately_observed_case_sets",
                "n_rows": None,
                "n_images": None,
                "censoring_adjusted": False,
            },
            difference[0],
            difference,
            args.ci_level,
        )

    return {
        "survival_cif": pd.DataFrame(survival_rows),
        "rmsr": pd.DataFrame(rmsr_rows),
        "rmsr_contrasts": pd.DataFrame(rmsr_contrast_rows),
        "o2m_endpoint_rank_set": pd.DataFrame(endpoint_rows),
        "return_transition": pd.DataFrame(return_rows),
        "rho_R_status": pd.DataFrame(status_rows),
        "margin_construct": pd.DataFrame(correlation_rows),
        "metadata": {
            "analysis_unit": "focal-GT direction",
            "ipw": True,
            "bootstrap": "stratified image-cluster percentile bootstrap",
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
            "ci_level": args.ci_level,
            "tau": args.tau,
            "endpoint_radii": radii,
            "survival_estimator": "weighted Kaplan-Meier",
            "cumulative_incidence_estimator": "weighted Aalen-Johansen competing-risks estimator",
            "rmsr": "integral_0^tau S(r) dr",
            "rho_R_correlations": "IPW-weighted Spearman conditional on observed rho_R only; no censoring-adjusted population correlation is claimed",
            "eligibility_specificity_control": "IPW-weighted Spearman conditional on observed O2O events whose first divergence is eligibility",
            "endpoint_conditioning": "endpoint and return-transition summaries condition on an exact replay state being available at the prespecified radius; no interpolation is used",
            "event_censor_tie_rule": "events are processed before censoring at the same recorded radius",
            "set_turnover_threshold": None,
        },
    }


def synthetic_self_test() -> dict[str, object]:
    # Four equal-weight subjects: events at .25/.50/.75 and one censor at 1.
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "sampling_weight": [1.0] * 4,
            "o2o_base_active": [1, 1, 1, 1],
            "o2o_event_observed": [1, 1, 0, 1],
            "o2o_radius": [.25, .50, np.nan, .75],
            "o2o_censor_radius": [np.nan, np.nan, 1.0, np.nan],
            "o2o_first_divergence": ["eligibility_boundary", "within_set_geometry_rank_reversal", "right_censored", "eligibility_boundary"],
        }
    )
    lookup = {name: index for index, name in enumerate(frame["image_id"])}
    process = process_matrices(frame, "o2o", 1.0, lookup, 4)
    evaluated = evaluate_process(process, np.ones((1, 4), dtype=int), 1.0, np.array([.25, .5, .75, 1.0]))
    np.testing.assert_allclose(evaluated["survival"][0], [.75, .50, .25, .25])
    np.testing.assert_allclose(evaluated["cif"]["eligibility"][0], [.25, .25, .50, .50])
    np.testing.assert_allclose(evaluated["cif"]["within_set"][0], [0.0, .25, .25, .25])
    np.testing.assert_allclose(evaluated["rmsr"][0], .625)
    assert math.isclose(weighted_spearman(np.array([1., 2., 3.]), np.array([2., 4., 8.]), np.ones(3)), 1.0)
    assert same_identity(pd.Series([1, np.nan]), pd.Series([1, np.nan])).all()

    # Exercise the complete table contract, exact endpoint join, four mutually
    # exclusive rho_R states, and zero-replicate smoke path.
    rows = []
    for index, (size_bin, status) in enumerate(
        [("t_8_16", "observed"), ("t_8_16", "competing"), ("s_16_32", "right"), ("s_16_32", "undefined")]
    ):
        row = {
            "image_id": f"im{index}", "gt_id": 0, "direction": "left", "size_bin": size_bin,
            "stratum": "t_only" if size_bin == "t_8_16" else "s_only", "sampling_weight": 1.0,
            "o2o_first_divergence": "eligibility_boundary" if index == 0 else "right_censored",
            "o2o_base_margin": .1 + index * .1,
            "rho_R_event_observed": int(status == "observed"),
            "rho_R_radius": .0625 if status == "observed" else np.nan,
            "rho_R_right_censored": int(status == "right"),
            "rho_R_competing_censored": int(status == "competing"),
            "rho_R_censor_reason": {"observed": None, "competing": "eligibility_boundary", "right": "rmax", "undefined": "base_runner_missing"}[status],
        }
        for estimand, (base_col, event_col, radius_col, censor_col, _) in ESTIMANDS.items():
            row[base_col] = 10 + index
            row[event_col] = int(index == 0)
            row[radius_col] = .0625 if index == 0 else np.nan
            row[censor_col] = np.nan if index == 0 else .125
        rows.append(row)
    full_events = pd.DataFrame(rows)
    curve_rows = []
    for row in rows:
        for radius in (0.0, *DEFAULT_RADII):
            changed = row["image_id"] == "im0" and radius >= .0625
            curve_rows.append({
                "image_id": row["image_id"], "gt_id": 0, "direction": "left", "size_bin": row["size_bin"],
                "stratum": row["stratum"], "sampling_weight": 1.0, "radius": radius,
                "o2o_active": row["o2o_base_active"] + int(changed),
                "o2m_legacy_rank": row["o2m_legacy_base"] + int(changed),
                "o2m_pre_topk_rank": row["o2m_pre_topk_base"] + int(changed),
                "o2m_assigned_positive_rank": row["o2m_assigned_positive_base"] + int(changed),
                "o2m_positive_set_jaccard": 1.0 if not changed else .8,
                "o2m_positive_set_retention": 1.0 if not changed else .9,
                "o2m_positive_set_exact_change": int(changed),
            })
    smoke_args = argparse.Namespace(
        tau=.125, endpoint_radii=list(DEFAULT_RADII), bootstrap_reps=0, seed=7, ci_level=.95
    )
    complete = analyze(full_events, pd.DataFrame(curve_rows), smoke_args)
    assert len(complete["rmsr"]) == len(SIZE_GROUPS) * len(ESTIMANDS)
    assert set(complete["rho_R_status"]["status"]).issuperset(
        {"observed", "competing_censored", "right_censored", "undefined"}
    )
    return {"status": "PASS", "tests": 8}


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = synthetic_self_test()
        if args.self_test_report:
            if args.self_test_report.exists():
                raise FileExistsError(args.self_test_report)
            args.self_test_report.parent.mkdir(parents=True, exist_ok=True)
            args.self_test_report.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.event_csv is None or args.curve_csv is None or args.output_dir is None:
        raise ValueError("--event-csv, --curve-csv, and --output-dir are required")
    if not args.event_csv.is_file() or not args.curve_csv.is_file():
        raise FileNotFoundError("event or curve CSV not found")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Image identifiers are opaque strings.  In large mixed-ID CSVs, pandas'
    # chunk-wise inference can otherwise parse numeric-looking IDs (for example
    # ``06914``) as integers in only some chunks, destroying leading zeros and
    # breaking the event/curve trajectory join.
    events = pd.read_csv(args.event_csv, dtype={"image_id": str})
    curves = pd.read_csv(args.curve_csv, dtype={"image_id": str})
    results = analyze(events, curves, args)
    args.output_dir.mkdir(parents=True)
    outputs = []
    for name, table in results.items():
        if name == "metadata":
            continue
        output = args.output_dir / f"{name}.csv"
        table.to_csv(output, index=False)
        outputs.append(output.name)
    json_payload = {"metadata": results["metadata"]}
    for name, table in results.items():
        if name != "metadata":
            json_payload[name] = json.loads(table.to_json(orient="records"))
    results_json = args.output_dir / "analysis_results.json"
    results_json.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    outputs.append(results_json.name)
    summary = {
        "status": "complete",
        **results["metadata"],
        "event_csv": str(args.event_csv),
        "event_csv_sha256": sha256(args.event_csv),
        "curve_csv": str(args.curve_csv),
        "curve_csv_sha256": sha256(args.curve_csv),
        "script_sha256": sha256(Path(__file__)),
        "outputs": outputs,
        "output_sha256": {
            name: sha256(args.output_dir / name)
            for name in outputs
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
