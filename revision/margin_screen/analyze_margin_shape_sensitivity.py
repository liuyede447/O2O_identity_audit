"""Check whether linear margin associations are driven by the lowest margins.

This is a read-only, CPU-only sensitivity analysis of an existing per-GT CSV.
It reports inverse-probability-weighted margin quintile outcome rates and
compares linear-margin with linear-tail cubic-spline logistic models using
image-grouped out-of-fold predictions.  It is diagnostic, not a search for a
best predictor and not a causal analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer


REQUIRED = {
    "image_id",
    "stratum",
    "sampling_weight",
    "margin",
    "o2o_fragile",
    "fn_at_iou50",
    "log_area",
    "log1p_o2m_count",
    "log_q_active",
    "active_ciou",
}
BASE_COVARIATES = ["log_area", "log1p_o2m_count", "log_q_active", "active_ciou"]
OUTCOMES = {
    "fragility": "o2o_fragile",
    "coverage_based_miss": "fn_at_iou50",
}
METRICS = ("roc_auc", "pr_auc", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Existing per-GT/incremental CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--spline-n-knots", type=int, default=4)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--self-test", action="store_true", help="Use deterministic synthetic data")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(frame: pd.DataFrame, folds: int, n_knots: int, degree: int) -> pd.DataFrame:
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    if folds < 2 or n_knots < 2 or degree < 1:
        raise ValueError("folds >= 2, spline-n-knots >= 2, and spline-degree >= 1 are required")
    use = frame[list(REQUIRED)].copy()
    numeric = sorted(REQUIRED.difference({"image_id", "stratum"}))
    for column in numeric:
        use[column] = pd.to_numeric(use[column], errors="raise")
    if use.isna().any().any() or not np.isfinite(use[numeric].to_numpy(float)).all():
        raise ValueError("analysis columns must be finite and complete")
    if (use["sampling_weight"] <= 0).any():
        raise ValueError("sampling_weight must be positive")
    for column in OUTCOMES.values():
        values = set(use[column].astype(int).unique())
        if not values.issubset({0, 1}) or len(values) != 2:
            raise ValueError(f"{column} must contain both binary classes")
        use[column] = use[column].astype(int)
    use["image_id"] = use["image_id"].astype(str)
    use["stratum"] = use["stratum"].astype(str)
    if (use["image_id"].eq("") | use["stratum"].eq("")).any():
        raise ValueError("image_id and stratum must be non-empty")
    if (use.groupby("image_id")["stratum"].nunique() > 1).any():
        raise ValueError("each image_id must belong to exactly one sampling stratum")
    if use["image_id"].nunique() < folds:
        raise ValueError("fewer image clusters than requested folds")
    if use["margin"].nunique() < max(5, n_knots):
        raise ValueError("too few unique margins for quintiles and the requested spline")
    return use.reset_index(drop=True)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    targets = probabilities * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return ordered[np.clip(indices, 0, len(ordered) - 1)]


def margin_quintiles(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    boundaries = weighted_quantile(
        frame["margin"].to_numpy(float),
        frame["sampling_weight"].to_numpy(float),
        np.array([0.2, 0.4, 0.6, 0.8]),
    )
    if len(np.unique(boundaries)) != 4:
        raise ValueError("weighted quintile boundaries are not unique because of margin ties")
    membership = np.searchsorted(boundaries, frame["margin"].to_numpy(float), side="right")
    return boundaries, membership.astype(int)


def cluster_resamples(frame: pd.DataFrame, cells: list[str], reps: int, seed: int):
    image_cells = frame[["image_id", *cells]].drop_duplicates()
    groups = {
        key: part["image_id"].tolist()
        for key, part in image_cells.groupby(cells, sort=True, dropna=False)
    }
    row_groups = {key: part.index.to_numpy() for key, part in frame.groupby("image_id", sort=False)}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        selected: list[np.ndarray] = []
        for images in groups.values():
            for image_id in rng.choice(images, size=len(images), replace=True):
                selected.append(row_groups[str(image_id)])
        yield np.concatenate(selected)


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def weighted_rate(frame: pd.DataFrame, outcome: str) -> float:
    return float(np.average(frame[outcome].to_numpy(float), weights=frame["sampling_weight"].to_numpy(float)))


def analyze_quintiles(frame: pd.DataFrame, reps: int, seed: int) -> tuple[list[dict], dict]:
    boundaries, membership = margin_quintiles(frame)
    work = frame.copy()
    work["margin_quintile"] = membership
    boot = {(name, quintile): [] for name in OUTCOMES for quintile in range(5)}
    for indices in cluster_resamples(work, ["stratum"], reps, seed):
        sample = work.iloc[indices]
        for name, outcome in OUTCOMES.items():
            for quintile in range(5):
                part = sample[sample["margin_quintile"].eq(quintile)]
                if len(part):
                    boot[(name, quintile)].append(weighted_rate(part, outcome))
    rows: list[dict] = []
    for name, outcome in OUTCOMES.items():
        for quintile in range(5):
            part = work[work["margin_quintile"].eq(quintile)]
            values = boot[(name, quintile)]
            if len(values) < max(1, int(0.99 * reps)):
                raise RuntimeError(f"too many empty bootstrap replicates for {name} quintile {quintile + 1}")
            rows.append(
                {
                    "outcome": name,
                    "source_column": outcome,
                    "margin_quintile": quintile + 1,
                    "n_gt": len(part),
                    "n_images": int(part["image_id"].nunique()),
                    "weighted_gt_mass": float(part["sampling_weight"].sum()),
                    "margin_min": float(part["margin"].min()),
                    "margin_max": float(part["margin"].max()),
                    "weighted_rate": weighted_rate(part, outcome),
                    "ci_95_low": percentile_ci(values)[0],
                    "ci_95_high": percentile_ci(values)[1],
                    "bootstrap_replicates_ok": len(values),
                }
            )
    contract = {
        "definition": "inverse-CDF weighted margin quintiles on the complete analysis frame",
        "tie_rule": "boundary values enter the higher-numbered quintile",
        "boundaries": [float(value) for value in boundaries],
        "bootstrap": "image clusters resampled within outcome-blind sampling stratum; point-estimate quintile boundaries held fixed",
    }
    return rows, contract


def weighted_standardize(train: np.ndarray, test: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(train, axis=0, weights=weights)
    variance = np.average((train - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return (train - mean) / scale, (test - mean) / scale


def spline_design(
    train_margin: np.ndarray,
    test_margin: np.ndarray,
    weights: np.ndarray,
    n_knots: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    probabilities = np.linspace(0.0, 1.0, n_knots)
    knots = weighted_quantile(train_margin, weights, probabilities)
    if len(np.unique(knots)) != n_knots:
        raise ValueError("a training fold has non-unique weighted spline knots")
    transformer = SplineTransformer(
        degree=degree,
        knots=knots.reshape(-1, 1),
        extrapolation="linear",
        include_bias=False,
    )
    train = transformer.fit_transform(train_margin.reshape(-1, 1))
    test = transformer.transform(test_margin.reshape(-1, 1))
    return train, test, [float(value) for value in knots]


def metric_set(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, probabilities, sample_weight=weights)),
        "pr_auc": float(average_precision_score(y, probabilities, sample_weight=weights)),
        "log_loss": float(log_loss(y, probabilities, sample_weight=weights, labels=[0, 1])),
    }


def fit_oof(
    frame: pd.DataFrame,
    outcome: str,
    folds: int,
    seed: int,
    n_knots: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    y = frame[outcome].to_numpy(int)
    weights = frame["sampling_weight"].to_numpy(float)
    groups = frame["image_id"].to_numpy(str)
    covariates = frame[BASE_COVARIATES].to_numpy(float)
    margin = frame["margin"].to_numpy(float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    linear_predictions = np.full(len(frame), np.nan)
    spline_predictions = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    fold_contracts: list[dict] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(np.zeros(len(frame)), y, groups)):
        if set(groups[train_indices]).intersection(groups[test_indices]):
            raise RuntimeError("image leakage detected in grouped OOF split")
        if len(np.unique(y[train_indices])) != 2 or len(np.unique(y[test_indices])) != 2:
            raise RuntimeError(f"fold {fold} does not contain both outcome classes")
        linear_train = np.column_stack([covariates[train_indices], margin[train_indices]])
        linear_test = np.column_stack([covariates[test_indices], margin[test_indices]])
        linear_train, linear_test = weighted_standardize(linear_train, linear_test, weights[train_indices])
        spline_train, spline_test, knots = spline_design(
            margin[train_indices], margin[test_indices], weights[train_indices], n_knots, degree
        )
        spline_train = np.column_stack([covariates[train_indices], spline_train])
        spline_test = np.column_stack([covariates[test_indices], spline_test])
        spline_train, spline_test = weighted_standardize(spline_train, spline_test, weights[train_indices])
        for design_train, design_test, destination in (
            (linear_train, linear_test, linear_predictions),
            (spline_train, spline_test, spline_predictions),
        ):
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
            model.fit(design_train, y[train_indices], sample_weight=weights[train_indices])
            destination[test_indices] = model.predict_proba(design_test)[:, 1]
        fold_ids[test_indices] = fold
        fold_contracts.append(
            {
                "fold": fold,
                "n_train_gt": len(train_indices),
                "n_test_gt": len(test_indices),
                "n_train_images": int(len(np.unique(groups[train_indices]))),
                "n_test_images": int(len(np.unique(groups[test_indices]))),
                "spline_knots_from_training_weighted_quantiles": knots,
            }
        )
    if np.isnan(linear_predictions).any() or np.isnan(spline_predictions).any() or (fold_ids < 0).any():
        raise RuntimeError("OOF prediction did not cover every row")
    if frame.assign(_fold=fold_ids).groupby("image_id")["_fold"].nunique().max() != 1:
        raise RuntimeError("an image cluster was assigned to more than one OOF fold")
    return linear_predictions, spline_predictions, fold_ids, fold_contracts


def foldwise_metrics(
    frame: pd.DataFrame,
    outcome: str,
    fold_column: str,
    linear_column: str,
    spline_column: str,
    indices: np.ndarray | None = None,
) -> tuple[dict, dict]:
    use = frame if indices is None else frame.iloc[indices]
    per_fold: dict[int, dict] = {}
    fold_weights: dict[int, float] = {}
    for fold, part in use.groupby(fold_column):
        y = part[outcome].to_numpy(int)
        if len(np.unique(y)) != 2:
            raise RuntimeError(f"resampled fold {fold} lost an outcome class")
        weights = part["sampling_weight"].to_numpy(float)
        per_fold[int(fold)] = {
            "linear_margin": metric_set(y, part[linear_column].to_numpy(float), weights),
            "restricted_cubic_margin": metric_set(y, part[spline_column].to_numpy(float), weights),
        }
        fold_weights[int(fold)] = float(weights.sum())
    total = sum(fold_weights.values())
    aggregate = {
        model: {
            metric: float(sum(fold_weights[fold] * per_fold[fold][model][metric] for fold in per_fold) / total)
            for metric in METRICS
        }
        for model in ("linear_margin", "restricted_cubic_margin")
    }
    return per_fold, aggregate


def analyze_models(
    frame: pd.DataFrame,
    outcome_name: str,
    outcome: str,
    folds: int,
    reps: int,
    seed: int,
    n_knots: int,
    degree: int,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    linear, spline, fold_ids, fold_contracts = fit_oof(frame, outcome, folds, seed, n_knots, degree)
    fold_column = f"oof_fold_{outcome_name}"
    linear_column = f"prob_linear_{outcome_name}"
    spline_column = f"prob_spline_{outcome_name}"
    work = frame.copy()
    work[fold_column] = fold_ids
    work[linear_column] = linear
    work[spline_column] = spline
    per_fold, aggregate = foldwise_metrics(work, outcome, fold_column, linear_column, spline_column)
    metric_values = {
        (model, metric): []
        for model in ("linear_margin", "restricted_cubic_margin")
        for metric in METRICS
    }
    deltas = {metric: [] for metric in METRICS}
    ok = 0
    for indices in cluster_resamples(work, [fold_column, "stratum"], reps, seed + 1000):
        try:
            _, boot_metrics = foldwise_metrics(work, outcome, fold_column, linear_column, spline_column, indices)
        except RuntimeError:
            continue
        for model in ("linear_margin", "restricted_cubic_margin"):
            for metric in METRICS:
                metric_values[(model, metric)].append(boot_metrics[model][metric])
        for metric in METRICS:
            deltas[metric].append(
                boot_metrics["restricted_cubic_margin"][metric] - boot_metrics["linear_margin"][metric]
            )
        ok += 1
    if ok < max(1, int(0.99 * reps)):
        raise RuntimeError(f"too many invalid model bootstrap replicates for {outcome_name}: {ok}/{reps}")
    cis = {
        model: {metric: percentile_ci(metric_values[(model, metric)]) for metric in METRICS}
        for model in ("linear_margin", "restricted_cubic_margin")
    }
    delta_point = {
        metric: aggregate["restricted_cubic_margin"][metric] - aggregate["linear_margin"][metric]
        for metric in METRICS
    }
    summary = {
        "source_column": outcome,
        "metric_protocol": "compute within each held-out image-grouped fold, then aggregate by fold test IPW mass",
        "bootstrap_protocol": "resample image clusters within OOF fold x outcome-blind sampling stratum",
        "fold_weighted_oof_metrics": aggregate,
        "fold_weighted_oof_metric_ci_95": cis,
        "spline_minus_linear_point": delta_point,
        "spline_minus_linear_ci_95": {metric: percentile_ci(values) for metric, values in deltas.items()},
        "bootstrap_replicates_requested": reps,
        "bootstrap_replicates_ok": ok,
        "fold_contracts": fold_contracts,
        "interpretation": "shape sensitivity only; neither model is claimed to be a best predictor or causal model",
    }
    metric_rows: list[dict] = []
    for fold, models in per_fold.items():
        for model, metrics in models.items():
            metric_rows.append({"outcome": outcome_name, "fold": fold, "model": model, **metrics})
    keep = work[["image_id", "stratum", outcome, fold_column, linear_column, spline_column]].copy()
    return summary, keep, metric_rows


def synthetic_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for image in range(90):
        stratum = ("t_only", "s_only", "t_and_s")[image % 3]
        weight = (1.0, 1.7, 2.3)[image % 3]
        for gt in range(3):
            margin = (3 * image + gt + 0.5) / 270.0
            log_area = rng.normal(5.2, 0.5)
            count = rng.normal(2.0, 0.4)
            log_q = rng.normal(-2.0, 0.6)
            ciou = np.clip(rng.normal(0.55, 0.12), 0.05, 0.95)
            fragile_p = 1 / (1 + np.exp(-(-0.3 - 2.4 * margin + 0.3 * log_area)))
            miss_p = 1 / (1 + np.exp(-(-1.0 - 1.2 * margin + 2.0 * (margin - 0.45) ** 2 - 0.5 * log_q)))
            rows.append(
                {
                    "image_id": f"synthetic_{image:03d}",
                    "stratum": stratum,
                    "sampling_weight": weight,
                    "margin": margin,
                    "o2o_fragile": int(rng.random() < fragile_p),
                    "fn_at_iou50": int(rng.random() < miss_p),
                    "log_area": log_area,
                    "log1p_o2m_count": count,
                    "log_q_active": log_q,
                    "active_ciou": ciou,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.self_test:
        raw = synthetic_frame(args.seed)
        input_contract = {"kind": "deterministic synthetic self-test", "path": None, "sha256": None}
        reps = min(args.bootstrap, 100)
        folds = min(args.folds, 3)
    else:
        if args.input is None or not args.input.is_file():
            raise FileNotFoundError("--input must name an existing CSV unless --self-test is used")
        raw = pd.read_csv(args.input)
        input_contract = {"kind": "existing per-GT/incremental CSV", "path": str(args.input.resolve()), "sha256": sha256(args.input)}
        reps = args.bootstrap
        folds = args.folds
    frame = validate_frame(raw, folds, args.spline_n_knots, args.spline_degree)
    args.output_dir.mkdir(parents=True)

    quintile_rows, quintile_contract = analyze_quintiles(frame, reps, args.seed + 10)
    pd.DataFrame(quintile_rows).to_csv(args.output_dir / "weighted_margin_quintiles.csv", index=False)

    model_summaries: dict[str, dict] = {}
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for offset, (name, outcome) in enumerate(OUTCOMES.items()):
        summary, predictions, rows = analyze_models(
            frame,
            name,
            outcome,
            folds,
            reps,
            args.seed + 100 * (offset + 1),
            args.spline_n_knots,
            args.spline_degree,
        )
        model_summaries[name] = summary
        prediction_parts.append(predictions)
        metric_rows.extend(rows)
    predictions = frame[["image_id", "stratum", "sampling_weight", "margin", *BASE_COVARIATES, *OUTCOMES.values()]].copy()
    for part in prediction_parts:
        new_columns = [column for column in part.columns if column not in predictions.columns]
        predictions = pd.concat([predictions, part[new_columns]], axis=1)
    predictions.to_csv(args.output_dir / "margin_shape_oof_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "margin_shape_metrics_by_fold.csv", index=False)

    summary = {
        "status": "self_test_complete" if args.self_test else "complete",
        "purpose": "test whether linear-logit margin conclusions are driven by the extreme low-margin region",
        "scope": "diagnostic shape sensitivity; no causal or best-predictor claim",
        "n_gt": len(frame),
        "n_images": int(frame["image_id"].nunique()),
        "weighted_gt_mass": float(frame["sampling_weight"].sum()),
        "base_covariates": BASE_COVARIATES,
        "quintiles": quintile_contract,
        "models": {
            "linear_margin": "base covariates plus one linear margin term",
            "restricted_cubic_margin": "base covariates plus cubic B-spline margin basis with linear-tail extrapolation",
            "spline_knots": "training-fold-only inverse-CDF weighted margin quantiles",
            "spline_n_knots": args.spline_n_knots,
            "spline_degree": args.spline_degree,
            "regularization": "near-unpenalized logistic regression, C=1e6",
        },
        "outcomes": model_summaries,
    }
    (args.output_dir / "margin_shape_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.self_test:
        checks = {
            "five_weighted_quintiles_present": len({row["margin_quintile"] for row in quintile_rows}) == 5,
            "both_outcomes_analyzed": set(model_summaries) == set(OUTCOMES),
            "all_oof_predictions_finite": bool(np.isfinite(predictions.filter(like="prob_").to_numpy(float)).all()),
            "one_fold_per_image": all(
                predictions.groupby("image_id")[column].nunique().max() == 1
                for column in predictions.columns
                if column.startswith("oof_fold_")
            ),
        }
        checks["passed"] = all(checks.values())
        if not checks["passed"]:
            raise RuntimeError(f"self-test failed: {checks}")
        (args.output_dir / "self_test.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    output_files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {
        "status": summary["status"],
        "read_only_input": True,
        "cpu_only": True,
        "input": input_contract,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "arguments": {
            "bootstrap": reps,
            "seed": args.seed,
            "folds": folds,
            "spline_n_knots": args.spline_n_knots,
            "spline_degree": args.spline_degree,
            "self_test": args.self_test,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs_sha256": {path.name: sha256(path) for path in output_files},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
