"""Descriptive O2M positive-count and continuous-scale controls on frozen tables.

This script does not replay or modify the assigner.  It standardizes observed
O2M outcomes to a common exact positive-count distribution and fits weighted
linear-probability models with a continuous log-area term.  Uncertainty is an
outcome-blind-stratum image-cluster bootstrap and is conditional on the frozen
checkpoint and selected images.  The results are descriptive, not causal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TINY = "t_8_16"
SMALL = "s_16_32"
SIZE_LEVELS = [TINY, SMALL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-gt",
        type=Path,
        default=ROOT / "evidence/artifacts/results/reviewer_controls_20260829/formal_s0e300_k00625/per_gt.csv",
    )
    parser.add_argument(
        "--per-direction",
        type=Path,
        default=ROOT / "evidence/artifacts/results/reviewer_controls_20260829/formal_s0e300_k00625/per_direction.csv",
    )
    parser.add_argument(
        "--boundary-events",
        type=Path,
        default=ROOT / "evidence/artifacts/runs/20260831T174200_boundary_discovery_v2/artifact/per_direction_boundary_events.csv",
    )
    parser.add_argument(
        "--rank-set",
        type=Path,
        default=ROOT / "evidence/artifacts/runs/20260831T162551_rank_set_normalized_v1/artifact/per_direction_rank_set.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--min-cell-gt", type=int, default=20)
    parser.add_argument("--boundary-tau", type=float, default=0.125)
    parser.add_argument("--size-cutpoints", type=float, nargs="+", default=[14.0, 16.0, 18.0])
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_stride(value: object) -> str:
    if pd.isna(value):
        return "none"
    index = int(float(value))
    if 0 <= index < 10000:
        return "8"
    if index < 12500:
        return "16"
    if index < 13125:
        return "32"
    raise ValueError(f"candidate index outside frozen 800-pixel P3--P5 grid: {index}")


def weighted_rate(rows: pd.DataFrame, outcome: str) -> float:
    weights = rows["sampling_weight"].to_numpy(float)
    return float(np.average(rows[outcome].to_numpy(float), weights=weights))


def exact_count_table(
    objects: pd.DataFrame,
    directions: pd.DataFrame,
    rank_set: pd.DataFrame,
    min_cell_gt: int,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    gt_counts = objects.groupby(["o2m_positive_count", "size_bin"]).size().unstack(fill_value=0)
    common = [
        int(k)
        for k, row in gt_counts.iterrows()
        if all(int(row.get(size, 0)) > 0 for size in SIZE_LEVELS)
    ]
    adequate = [
        int(k)
        for k, row in gt_counts.iterrows()
        if all(int(row.get(size, 0)) >= min_cell_gt for size in SIZE_LEVELS)
    ]
    all_k = sorted(
        set(objects["o2m_positive_count"].astype(int))
        | set(directions["o2m_positive_count"].astype(int))
        | set(rank_set["base_set_size"].astype(int))
    )
    records: list[dict[str, object]] = []
    for k in all_k:
        record: dict[str, object] = {
            "positive_count": k,
            "full_common_support": int(k in common),
            "adequate_overlap_support": int(k in adequate),
        }
        for size in SIZE_LEVELS:
            o = objects[(objects["size_bin"] == size) & (objects["o2m_positive_count"] == k)]
            d = directions[(directions["size_bin"] == size) & (directions["o2m_positive_count"] == k)]
            r = rank_set[(rank_set["size_bin"] == size) & (rank_set["base_set_size"] == k)]
            suffix = "tiny" if size == TINY else "small"
            record[f"object_n_{suffix}"] = int(len(o))
            record[f"object_weight_{suffix}"] = float(o["sampling_weight"].sum())
            record[f"object_fragility_{suffix}"] = weighted_rate(o, "o2m_fragile") if len(o) else np.nan
            record[f"direction_n_{suffix}"] = int(len(d))
            record[f"direction_weight_{suffix}"] = float(d["sampling_weight"].sum())
            record[f"direction_flip_{suffix}"] = weighted_rate(d, "o2m_flip") if len(d) else np.nan
            record[f"rank_set_n_{suffix}"] = int(len(r))
            record[f"set_turnover_{suffix}"] = weighted_rate(r, "set_turnover") if len(r) else np.nan
            record[f"assigned_top1_flip_{suffix}"] = (
                weighted_rate(r, "assigned_set_q_top1_flip") if len(r) else np.nan
            )
        records.append(record)
    return pd.DataFrame(records), common, adequate


def image_frame(*frames: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([frame[["image_id", "stratum"]] for frame in frames], ignore_index=True)
    per_image = combined.drop_duplicates()
    conflicts = per_image.groupby("image_id")["stratum"].nunique()
    if int(conflicts.max()) != 1:
        raise ValueError("sampling stratum is inconsistent within image")
    return per_image.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)


def draw_cluster_multiplicity(images: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    multiplicity = np.zeros(len(images), dtype=float)
    for _, cell in images.groupby("stratum", sort=True):
        indices = cell.index.to_numpy()
        multiplicity[indices] = rng.multinomial(len(indices), np.full(len(indices), 1.0 / len(indices)))
    return multiplicity


def cell_tensors(
    frame: pd.DataFrame,
    images: pd.DataFrame,
    count_col: str,
    outcome: str,
    k_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    image_index = images.reset_index().set_index("image_id")["index"]
    k_index = {k: idx for idx, k in enumerate(k_values)}
    den = np.zeros((len(images), len(SIZE_LEVELS), len(k_values)), dtype=float)
    num = np.zeros_like(den)
    for (image_id, size, k), rows in frame.groupby(["image_id", "size_bin", count_col], sort=False):
        if size not in SIZE_LEVELS or int(k) not in k_index:
            continue
        i = int(image_index.loc[image_id])
        s = SIZE_LEVELS.index(size)
        j = k_index[int(k)]
        weights = rows["sampling_weight"].to_numpy(float)
        den[i, s, j] = float(weights.sum())
        num[i, s, j] = float(np.dot(weights, rows[outcome].to_numpy(float)))
    return den, num


def standardized_summary(
    den: np.ndarray,
    num: np.ndarray,
    support: list[int],
    k_values: list[int],
    multiplicity: np.ndarray | None = None,
) -> dict[str, float]:
    if multiplicity is None:
        d = den.sum(axis=0)
        n = num.sum(axis=0)
    else:
        d = np.tensordot(multiplicity, den, axes=(0, 0))
        n = np.tensordot(multiplicity, num, axes=(0, 0))
    indices = [k_values.index(k) for k in support]
    d_support = d[:, indices]
    n_support = n[:, indices]
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_rates = n_support / d_support
    pooled = d_support.sum(axis=0)
    q = pooled / pooled.sum()
    standardized = (cell_rates * q[None, :]).sum(axis=1)
    crude = n.sum(axis=1) / d.sum(axis=1)
    coverage = d_support.sum(axis=1) / d.sum(axis=1)
    return {
        "crude_tiny": float(crude[0]),
        "crude_small": float(crude[1]),
        "crude_gap_small_minus_tiny": float(crude[1] - crude[0]),
        "standardized_tiny": float(standardized[0]),
        "standardized_small": float(standardized[1]),
        "standardized_gap_small_minus_tiny": float(standardized[1] - standardized[0]),
        "coverage_tiny": float(coverage[0]),
        "coverage_small": float(coverage[1]),
    }


def cutpoint_tensors(
    objects: pd.DataFrame,
    images: pd.DataFrame,
    cutpoints: list[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image_index = images.reset_index().set_index("image_id")["index"]
    den = np.zeros((len(images), len(cutpoints), 2), dtype=float)
    numerators = {
        "o2m_fragility": np.zeros_like(den),
        "o2o_fragility": np.zeros_like(den),
        "paired_o2o_minus_o2m": np.zeros_like(den),
    }
    for row in objects.itertuples(index=False):
        side = float(row.equivalent_side)
        if not (8.0 <= side < 32.0):
            continue
        image = int(image_index.loc[row.image_id])
        weight = float(row.sampling_weight)
        for cut_idx, cutpoint in enumerate(cutpoints):
            group = 0 if side < cutpoint else 1
            den[image, cut_idx, group] += weight
            numerators["o2m_fragility"][image, cut_idx, group] += weight * float(row.o2m_fragile)
            numerators["o2o_fragility"][image, cut_idx, group] += weight * float(row.o2o_fragile)
            numerators["paired_o2o_minus_o2m"][image, cut_idx, group] += weight * (
                float(row.o2o_fragile) - float(row.o2m_fragile)
            )
    return den, numerators


def cutpoint_estimates(
    den: np.ndarray,
    numerators: dict[str, np.ndarray],
    cutpoints: list[float],
    multiplicity: np.ndarray | None = None,
) -> list[dict[str, float]]:
    if multiplicity is None:
        total_den = den.sum(axis=0)
        total_num = {name: values.sum(axis=0) for name, values in numerators.items()}
    else:
        total_den = np.tensordot(multiplicity, den, axes=(0, 0))
        total_num = {
            name: np.tensordot(multiplicity, values, axes=(0, 0))
            for name, values in numerators.items()
        }
    output: list[dict[str, float]] = []
    for cut_idx, cutpoint in enumerate(cutpoints):
        item: dict[str, float] = {
            "cutpoint_equivalent_side_px": float(cutpoint),
            "lower_weight": float(total_den[cut_idx, 0]),
            "upper_weight": float(total_den[cut_idx, 1]),
        }
        for name, values in total_num.items():
            rates = values[cut_idx] / total_den[cut_idx]
            item[f"{name}_lower"] = float(rates[0])
            item[f"{name}_upper"] = float(rates[1])
            item[f"{name}_gap_upper_minus_lower"] = float(rates[1] - rates[0])
        output.append(item)
    return output


def prepare_design(
    frame: pd.DataFrame,
    outcome: str,
    positive_count: str,
    candidate: str,
    include_direction: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    analysis = frame.copy()
    analysis = analysis[
        np.isfinite(analysis["border_scan_limit_min"]) & analysis[candidate].notna()
    ].copy()
    analysis["log_area"] = np.log(analysis["area"].to_numpy(float))
    analysis["log_aspect_signed"] = np.log(analysis["width"].to_numpy(float) / analysis["height"].to_numpy(float))
    analysis["log_aspect_abs"] = np.abs(analysis["log_aspect_signed"])
    analysis["base_stride"] = analysis[candidate].map(decode_stride)
    categorical = {
        "positive_count": analysis[positive_count].astype(int).astype(str),
        "class": analysis["class_id"].astype(int).astype(str),
        "base_stride": analysis["base_stride"],
        "legal_directions": analysis["legal_directions"].astype(int).astype(str),
        "sampling_stratum": analysis["stratum"].astype(str),
    }
    if include_direction:
        categorical["direction"] = analysis["direction"].astype(str)
    continuous = pd.DataFrame(
        {
            "log_area": analysis["log_area"],
            "log_aspect_signed": analysis["log_aspect_signed"],
            "log_aspect_abs": analysis["log_aspect_abs"],
            "border_scan_limit_min": analysis["border_scan_limit_min"].astype(float),
        },
        index=analysis.index,
    )
    if include_direction and "scan_limit" in analysis:
        continuous["direction_scan_limit"] = analysis["scan_limit"].astype(float)
    for column in continuous.columns:
        continuous[column] -= np.average(continuous[column], weights=analysis["sampling_weight"])
    dummy_parts = [pd.get_dummies(values, prefix=name, drop_first=True, dtype=float) for name, values in categorical.items()]
    design = pd.concat([continuous, *dummy_parts], axis=1)
    design.insert(0, "intercept", 1.0)
    design = design.loc[:, design.nunique(dropna=False) > 1].copy()
    if "intercept" not in design:
        design.insert(0, "intercept", 1.0)
    if design.isna().any().any():
        raise ValueError("non-finite design values after preparation")
    return analysis, design.to_numpy(float), analysis[outcome].to_numpy(float), list(design.columns)


def cluster_crossproducts(
    analysis: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    images: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    image_index = images.reset_index().set_index("image_id")["index"]
    p = x.shape[1]
    a = np.zeros((len(images), p, p), dtype=float)
    b = np.zeros((len(images), p), dtype=float)
    positions = pd.Series(np.arange(len(analysis)), index=analysis.index)
    for image_id, rows in analysis.groupby("image_id", sort=False):
        loc = positions.loc[rows.index].to_numpy(int)
        xi, yi = x[loc], y[loc]
        wi = rows["sampling_weight"].to_numpy(float)
        idx = int(image_index.loc[image_id])
        a[idx] = xi.T @ (wi[:, None] * xi)
        b[idx] = xi.T @ (wi * yi)
    return a, b


def solve_beta(a: np.ndarray, b: np.ndarray, multiplicity: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    if multiplicity is None:
        aa, bb = a.sum(axis=0), b.sum(axis=0)
    else:
        aa = np.tensordot(multiplicity, a, axes=(0, 0))
        bb = np.tensordot(multiplicity, b, axes=(0, 0))
    rank = int(np.linalg.matrix_rank(aa))
    beta = np.linalg.lstsq(aa, bb, rcond=1e-10)[0]
    return beta, rank


def model_bundle(
    name: str,
    frame: pd.DataFrame,
    outcome: str,
    positive_count: str,
    candidate: str,
    images: pd.DataFrame,
    include_direction: bool,
) -> dict[str, object]:
    analysis, x, y, names = prepare_design(frame, outcome, positive_count, candidate, include_direction)
    a, b = cluster_crossproducts(analysis, x, y, images)
    beta, rank = solve_beta(a, b)
    idx = names.index("log_area")
    return {
        "name": name,
        "analysis": analysis,
        "a": a,
        "b": b,
        "coefficient_index": idx,
        "coefficient_names": names,
        "point_beta_log_area": float(beta[idx]),
        "point_probability_change_per_area_doubling": float(beta[idx] * math.log(2.0)),
        "design_rank": rank,
        "design_columns": len(names),
        "n_rows": len(analysis),
        "n_images": int(analysis["image_id"].nunique()),
    }


def quantile_interval(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"low": None, "high": None, "valid_replicates": 0}
    low, high = np.quantile(finite, [0.025, 0.975])
    return {"low": float(low), "high": float(high), "valid_replicates": int(len(finite))}


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    objects = pd.read_csv(args.per_gt)
    directions = pd.read_csv(args.per_direction)
    boundary = pd.read_csv(args.boundary_events)
    rank_set = pd.read_csv(args.rank_set)
    require_columns(
        objects,
        ["image_id", "gt_id", "class_id", "width", "height", "area", "equivalent_side", "size_bin", "stratum",
         "sampling_weight", "o2m_rank", "o2m_positive_count", "legal_directions", "o2o_fragile", "o2m_fragile",
         "common_valid_margin"],
        "per_gt",
    )
    require_columns(
        directions,
        ["image_id", "gt_id", "direction", "size_bin", "stratum", "sampling_weight",
         "o2o_margin", "o2m_margin", "o2m_positive_count", "o2m_flip"],
        "per_direction",
    )
    require_columns(
        boundary,
        ["image_id", "gt_id", "class_id", "direction", "width", "height", "area", "size_bin",
         "stratum", "sampling_weight", "scan_limit", "o2m_assigned_positive_base",
         "o2m_positive_count_base", "o2m_assigned_positive_event_observed", "o2m_assigned_positive_radius"],
        "boundary_events",
    )
    require_columns(
        rank_set,
        ["image_id", "gt_id", "class_id", "direction", "width", "height", "area", "size_bin",
         "stratum", "sampling_weight", "assigned_set_q_top1_base", "base_set_size",
         "assigned_set_q_top1_flip", "set_turnover"],
        "rank_set",
    )
    if objects.duplicated(["image_id", "gt_id"]).any():
        raise ValueError("per_gt keys are not unique")
    for label, frame in (("per_direction", directions), ("boundary_events", boundary), ("rank_set", rank_set)):
        if frame.duplicated(["image_id", "gt_id", "direction"]).any():
            raise ValueError(f"{label} keys are not unique")

    primary_objects = objects[objects["common_valid_margin"].eq(1)].copy()
    primary_keys = primary_objects[["image_id", "gt_id"]]
    primary_directions = directions.merge(primary_keys, on=["image_id", "gt_id"], how="inner", validate="many_to_one")
    if not (
        primary_directions["o2o_margin"].notna().all()
        and primary_directions["o2m_margin"].notna().all()
    ):
        raise ValueError("common-valid object keys include invalid directional margins")
    primary_boundary = boundary.merge(primary_keys, on=["image_id", "gt_id"], how="inner", validate="many_to_one")
    primary_rank_set = rank_set.merge(primary_keys, on=["image_id", "gt_id"], how="inner", validate="many_to_one")

    boundary_object = primary_boundary.groupby(["image_id", "gt_id"], as_index=False).agg(
        border_scan_limit_min=("scan_limit", "min"),
        border_scan_limit_mean=("scan_limit", "mean"),
        boundary_direction_rows=("direction", "size"),
    )
    object_analysis = primary_objects.merge(boundary_object, on=["image_id", "gt_id"], how="left", validate="one_to_one")
    object_count_check = primary_objects.merge(
        primary_boundary.drop_duplicates(["image_id", "gt_id"])[["image_id", "gt_id", "o2m_positive_count_base"]],
        on=["image_id", "gt_id"], how="inner", validate="one_to_one",
    )
    count_mismatches = int(
        (object_count_check["o2m_positive_count"].astype(int)
         != object_count_check["o2m_positive_count_base"].astype(int)).sum()
    )
    if count_mismatches:
        raise ValueError(f"positive-count mismatch between per_gt and boundary table: {count_mismatches}")

    object_covariates = object_analysis[["image_id", "gt_id", "legal_directions", "border_scan_limit_min"]]
    boundary_analysis = primary_boundary.merge(object_covariates, on=["image_id", "gt_id"], how="inner", validate="many_to_one")
    rank_analysis = primary_rank_set.merge(object_covariates, on=["image_id", "gt_id"], how="inner", validate="many_to_one")
    rank_analysis = rank_analysis.merge(
        primary_boundary[["image_id", "gt_id", "direction", "scan_limit"]],
        on=["image_id", "gt_id", "direction"], how="inner", validate="one_to_one",
    )
    tau = float(args.boundary_tau)
    observed_by_tau = (
        boundary_analysis["o2m_assigned_positive_event_observed"].eq(1)
        & boundary_analysis["o2m_assigned_positive_radius"].le(tau)
    )
    evaluable_by_tau = boundary_analysis["scan_limit"].ge(tau) | observed_by_tau
    boundary_analysis = boundary_analysis[evaluable_by_tau].copy()
    boundary_analysis["event_by_tau"] = observed_by_tau[evaluable_by_tau].astype(int)

    exact, common_support, adequate_support = exact_count_table(
        primary_objects, primary_directions, primary_rank_set, args.min_cell_gt
    )
    if not adequate_support:
        raise ValueError("no adequate-overlap positive-count strata")
    k_values = sorted(set(exact["positive_count"].astype(int)))
    images = image_frame(objects, directions, boundary, rank_set)
    cutpoints = sorted(set(float(value) for value in args.size_cutpoints))
    if not cutpoints or any(not 8.0 < value < 32.0 for value in cutpoints):
        raise ValueError("all --size-cutpoints must lie strictly inside the 8--32 px support")
    cut_den, cut_num = cutpoint_tensors(primary_objects, images, cutpoints)
    cutpoint_points = cutpoint_estimates(cut_den, cut_num, cutpoints)

    analysis_specs = {
        "object_any_top1_fragility": (primary_objects, "o2m_positive_count", "o2m_fragile"),
        "direction_top1_flip": (primary_directions, "o2m_positive_count", "o2m_flip"),
        "rank_set_assigned_top1_flip": (primary_rank_set, "base_set_size", "assigned_set_q_top1_flip"),
        "rank_set_turnover": (primary_rank_set, "base_set_size", "set_turnover"),
    }
    tensors = {
        name: cell_tensors(frame, images, count, outcome, k_values)
        for name, (frame, count, outcome) in analysis_specs.items()
    }
    standardized: dict[str, dict[str, object]] = {}
    for name, (den, num) in tensors.items():
        standardized[name] = {
            "full_common_support": {
                "positive_counts": common_support,
                "point": standardized_summary(den, num, common_support, k_values),
            },
            "adequate_overlap_support": {
                "positive_counts": adequate_support,
                "minimum_raw_gt_per_size_cell": args.min_cell_gt,
                "point": standardized_summary(den, num, adequate_support, k_values),
            },
        }

    models = [
        model_bundle(
            "object_any_top1_fragility",
            object_analysis,
            "o2m_fragile",
            "o2m_positive_count",
            "o2m_rank",
            images,
            False,
        ),
        model_bundle(
            "rank_set_turnover",
            rank_analysis,
            "set_turnover",
            "base_set_size",
            "assigned_set_q_top1_base",
            images,
            True,
        ),
        model_bundle(
            f"assigned_positive_boundary_event_by_tau_{tau:g}",
            boundary_analysis,
            "event_by_tau",
            "o2m_positive_count_base",
            "o2m_assigned_positive_base",
            images,
            True,
        ),
    ]

    rng = np.random.default_rng(args.seed)
    bootstrap_values: dict[str, list[float]] = {}
    for name in standardized:
        for support_name in ("full_common_support", "adequate_overlap_support"):
            bootstrap_values[f"{name}.{support_name}.gap"] = []
    for model in models:
        bootstrap_values[f"model.{model['name']}.beta_log_area"] = []
        bootstrap_values[f"model.{model['name']}.per_doubling"] = []
    for cutpoint in cutpoints:
        for outcome in ("o2m_fragility", "o2o_fragility", "paired_o2o_minus_o2m"):
            bootstrap_values[f"cutpoint.{cutpoint:g}.{outcome}"] = []

    for _ in range(args.bootstrap_reps):
        multiplicity = draw_cluster_multiplicity(images, rng)
        for name, (den, num) in tensors.items():
            for support_name, support in (
                ("full_common_support", common_support),
                ("adequate_overlap_support", adequate_support),
            ):
                estimate = standardized_summary(den, num, support, k_values, multiplicity)
                bootstrap_values[f"{name}.{support_name}.gap"].append(
                    estimate["standardized_gap_small_minus_tiny"]
                )
        for model in models:
            beta, _ = solve_beta(model["a"], model["b"], multiplicity)
            value = float(beta[int(model["coefficient_index"])])
            bootstrap_values[f"model.{model['name']}.beta_log_area"].append(value)
            bootstrap_values[f"model.{model['name']}.per_doubling"].append(value * math.log(2.0))
        for estimate in cutpoint_estimates(cut_den, cut_num, cutpoints, multiplicity):
            cutpoint = estimate["cutpoint_equivalent_side_px"]
            for outcome in ("o2m_fragility", "o2o_fragility", "paired_o2o_minus_o2m"):
                bootstrap_values[f"cutpoint.{cutpoint:g}.{outcome}"].append(
                    estimate[f"{outcome}_gap_upper_minus_lower"]
                )

    for name in standardized:
        for support_name in ("full_common_support", "adequate_overlap_support"):
            standardized[name][support_name]["ci_95_gap_small_minus_tiny"] = quantile_interval(
                bootstrap_values[f"{name}.{support_name}.gap"]
            )

    model_rows: list[dict[str, object]] = []
    model_summaries: list[dict[str, object]] = []
    for model in models:
        name = str(model["name"])
        beta_ci = quantile_interval(bootstrap_values[f"model.{name}.beta_log_area"])
        doubling_ci = quantile_interval(bootstrap_values[f"model.{name}.per_doubling"])
        summary = {
            key: value for key, value in model.items() if key not in {"analysis", "a", "b", "coefficient_index"}
        }
        summary["ci_95_beta_log_area"] = beta_ci
        summary["ci_95_probability_change_per_area_doubling"] = doubling_ci
        summary["model_type"] = "sampling-weighted linear probability model"
        summary["interpretation"] = (
            "Descriptive adjusted association conditional on the frozen checkpoint; not a causal size effect."
        )
        model_summaries.append(summary)
        model_rows.append(
            {
                "model": name,
                "n_rows": model["n_rows"],
                "n_images": model["n_images"],
                "beta_log_area": model["point_beta_log_area"],
                "beta_log_area_ci_low": beta_ci["low"],
                "beta_log_area_ci_high": beta_ci["high"],
                "probability_change_per_area_doubling": model["point_probability_change_per_area_doubling"],
                "per_doubling_ci_low": doubling_ci["low"],
                "per_doubling_ci_high": doubling_ci["high"],
            }
        )

    cutpoint_rows: list[dict[str, object]] = []
    for item in cutpoint_points:
        cutpoint = item["cutpoint_equivalent_side_px"]
        row: dict[str, object] = dict(item)
        for outcome in ("o2m_fragility", "o2o_fragility", "paired_o2o_minus_o2m"):
            interval = quantile_interval(bootstrap_values[f"cutpoint.{cutpoint:g}.{outcome}"])
            row[f"{outcome}_gap_ci_low"] = interval["low"]
            row[f"{outcome}_gap_ci_high"] = interval["high"]
            row[f"{outcome}_gap_valid_replicates"] = interval["valid_replicates"]
        cutpoint_rows.append(row)

    count_means = {}
    for size in SIZE_LEVELS:
        rows = primary_objects[primary_objects["size_bin"] == size]
        weights = rows["sampling_weight"].to_numpy(float)
        count_means[size] = float(np.average(rows["o2m_positive_count"], weights=weights))

    input_paths = [args.per_gt, args.per_direction, args.boundary_events, args.rank_set]
    result = {
        "status": "complete",
        "protocol": "o2m_positive_count_and_continuous_scale_controls_v1",
        "read_only_inputs": True,
        "estimand_scope": (
            "Descriptive standardization/association on common-valid O2O/O2M margin support, conditional on one frozen seed-0 checkpoint and the selected image sample."
        ),
        "analysis_filter": "per_gt common_valid_margin == 1; direction, boundary, and rank-set rows joined by those frozen GT keys",
        "causal_claim": False,
        "gap_orientation": "small_16_32_minus_tiny_8_16",
        "stress": "equivalent-area-side shift (delta = kappa * sqrt(width * height), kappa=0.0625)",
        "positive_count_means": {
            **count_means,
            "small_minus_tiny": count_means[SMALL] - count_means[TINY],
        },
        "standardized_analyses": standardized,
        "size_cutpoint_sensitivity": {
            "support": "objects with equivalent-area side in [8, 32) px",
            "gap_orientation": "upper_at_or_above_cutpoint_minus_lower_below_cutpoint",
            "paired_definition": "within-GT O2O fragility minus O2M fragility, followed by upper-minus-lower contrast",
            "estimates": cutpoint_rows,
        },
        "continuous_log_area_models": model_summaries,
        "bootstrap": {
            "replicates": args.bootstrap_reps,
            "seed": args.seed,
            "unit": "image cluster",
            "within": "outcome-blind sampling stratum",
            "interval": "percentile 95%",
        },
        "border_distance_proxy": (
            "minimum direction-level scan_limit across each GT, capped at the scanner rmax=0.25; it is a censored proxy, not exact pixel distance"
        ),
        "fixed_k_limitation": (
            "Frozen rank-set tables contain set IDs and the selected set top-1 but not a complete ordered q-score list. "
            "They support exact base-count matching/standardization and set-turnover analysis, but not a new fixed-K replay. "
            "A fixed-K intervention still requires re-executing the assigner from frozen candidate scores."
        ),
        "input_validation": {
            "per_gt_rows": len(objects),
            "per_direction_rows": len(directions),
            "boundary_event_rows": len(boundary),
            "rank_set_rows": len(rank_set),
            "analysis_per_gt_rows": len(primary_objects),
            "analysis_per_direction_rows": len(primary_directions),
            "analysis_boundary_event_rows": len(primary_boundary),
            "analysis_rank_set_rows": len(primary_rank_set),
            "images": len(images),
            "per_gt_boundary_joined": len(object_count_check),
            "per_gt_missing_boundary": int(object_analysis["border_scan_limit_min"].isna().sum()),
            "positive_count_mismatches": count_mismatches,
            "boundary_tau": tau,
            "boundary_rows_evaluable_at_tau": len(boundary_analysis),
        },
        "input_files": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in input_paths
        ],
    }

    args.output_dir.mkdir(parents=True)
    exact.to_csv(args.output_dir / "exact_positive_count_strata.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "continuous_log_area_models.csv", index=False)
    pd.DataFrame(cutpoint_rows).to_csv(args.output_dir / "size_cutpoint_sensitivity.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    notes = f"""# O2M positive-count and scale controls

This is a descriptive, read-only analysis of frozen outputs. It does not identify a causal effect of size or positive-set cardinality and is conditional on one seed-0 checkpoint. The analysis sample is explicitly the `common_valid_margin == 1` GT support used by the reported 3.07-versus-6.54 positive-count comparison and 17.45 pp O2M contrast; directional, boundary, and rank-set rows are joined by those frozen GT keys.

- Exact-count standardization uses the pooled sampling-weighted count distribution on counts observed in both size groups. The adequate-overlap sensitivity requires at least {args.min_cell_gt} raw GTs from each size group per exact count.
- Confidence intervals resample image clusters within the outcome-blind sampling strata ({args.bootstrap_reps} replicates; seed {args.seed}).
- Continuous models are sampling-weighted linear-probability models. The log-area coefficient is also reported as a probability change per doubling of area. Controls are exact positive count, class, decoded base stride, signed and absolute log aspect ratio, legal-direction count, the capped minimum `scan_limit` border proxy, and sampling stratum; directional models additionally control direction and direction-specific `scan_limit`.
- The 14/16/18 px cutpoint sensitivity reports O2M, O2O, and within-GT paired O2O-minus-O2M contrasts on the common 8--32 px equivalent-area-side support; every gap is upper minus lower.
- The rank-set file contains exact member IDs but not the full q-score ordering. Therefore it permits exact base-count matching and set-turnover controls, but not a new fixed-K assigner replay.
"""
    (args.output_dir / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "output_dir": str(args.output_dir),
        "positive_count_means": result["positive_count_means"],
        "object_adequate_overlap": standardized["object_any_top1_fragility"]["adequate_overlap_support"],
        "models": model_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
