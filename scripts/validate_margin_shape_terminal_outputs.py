"""Independently validate terminal outputs from the formal margin-shape run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


OUTCOMES = {
    "fragility": "o2o_fragile",
    "coverage_based_miss": "fn_at_iou50",
}
MODELS = {
    "linear_margin": "prob_linear_{outcome}",
    "restricted_cubic_margin": "prob_spline_{outcome}",
}
METRICS = ("roc_auc", "pr_auc", "log_loss")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_set(y: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, probability, sample_weight=weight)),
        "pr_auc": float(average_precision_score(y, probability, sample_weight=weight)),
        "log_loss": float(log_loss(y, probability, sample_weight=weight, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    artifact = run_dir / "artifact"
    config = json.loads((run_dir / "CONFIG.json").read_text(encoding="utf-8"))
    summary = json.loads((artifact / "margin_shape_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(artifact / "margin_shape_oof_predictions.csv")
    per_fold = pd.read_csv(artifact / "margin_shape_metrics_by_fold.csv")
    quintiles = pd.read_csv(artifact / "weighted_margin_quintiles.csv")

    checks: dict[str, bool] = {}

    def check(name: str, value: object) -> None:
        checks[name] = bool(value)

    config_path = run_dir / "CONFIG.json"
    input_path = project_root / config["input"]
    script_path = project_root / config["script"]
    check("summary_and_manifest_complete", summary.get("status") == "complete" and manifest.get("status") == "complete")
    check("config_hash_matches_status", sha256(config_path) == json.loads((run_dir / "STATUS.json").read_text(encoding="utf-8"))["config_sha256"])
    check("input_hash_matches_config", sha256(input_path) == config["input_sha256"])
    check("script_hash_matches_config", sha256(script_path) == config["script_sha256"])
    check("manifest_input_hash_matches_config", manifest["input"]["sha256"] == config["input_sha256"])
    check("manifest_script_hash_matches_config", manifest["script_sha256"] == config["script_sha256"])
    check("manifest_arguments_match_config", all(manifest["arguments"][key] == config[key] for key in ("bootstrap_replicates", "seed", "folds", "spline_n_knots", "spline_degree")) if "bootstrap_replicates" in manifest["arguments"] else (
        manifest["arguments"]["bootstrap"] == config["bootstrap_replicates"]
        and manifest["arguments"]["seed"] == config["seed"]
        and manifest["arguments"]["folds"] == config["folds"]
        and manifest["arguments"]["spline_n_knots"] == config["spline_n_knots"]
        and manifest["arguments"]["spline_degree"] == config["spline_degree"]
    ))

    for filename, expected in manifest["outputs_sha256"].items():
        check(f"output_hash::{filename}", sha256(artifact / filename) == expected)

    check("row_count_matches_summary", len(predictions) == summary["n_gt"])
    check("image_count_matches_summary", predictions["image_id"].nunique() == summary["n_images"])
    check("weighted_mass_matches_summary", abs(predictions["sampling_weight"].sum() - summary["weighted_gt_mass"]) < 1e-9)
    probability_frame = predictions.filter(regex=r"^prob_")
    check("probabilities_finite", np.isfinite(probability_frame.to_numpy(float)).all())
    check("probabilities_bounded", ((probability_frame >= 0.0) & (probability_frame <= 1.0)).all().all())
    check("two_outcomes_present", set(quintiles["outcome"]) == set(OUTCOMES))
    check("five_quintiles_per_outcome", quintiles.groupby("outcome")["margin_quintile"].nunique().eq(5).all())
    check("quintile_bootstrap_complete", quintiles["bootstrap_replicates_ok"].eq(config["bootstrap_replicates"]).all())
    check("empty_stderr", (run_dir / "stderr.log").stat().st_size == 0)

    metric_differences: dict[str, float] = {}
    for outcome_name, outcome_column in OUTCOMES.items():
        fold_column = f"oof_fold_{outcome_name}"
        check(f"{outcome_name}::five_folds", set(predictions[fold_column]) == set(range(config["folds"])))
        check(
            f"{outcome_name}::one_fold_per_image",
            predictions.groupby("image_id")[fold_column].nunique().max() == 1,
        )
        fold_results: list[tuple[str, dict[str, float], float]] = []
        for fold, part in predictions.groupby(fold_column):
            y = part[outcome_column].to_numpy(int)
            weight = part["sampling_weight"].to_numpy(float)
            check(f"{outcome_name}::fold_{fold}::two_classes", set(y) == {0, 1})
            for model_name, template in MODELS.items():
                probability_column = template.format(outcome=outcome_name)
                recalculated = metric_set(y, part[probability_column].to_numpy(float), weight)
                recorded_row = per_fold[
                    (per_fold["outcome"] == outcome_name)
                    & (per_fold["fold"] == fold)
                    & (per_fold["model"] == model_name)
                ]
                check(f"{outcome_name}::fold_{fold}::{model_name}::one_metric_row", len(recorded_row) == 1)
                recorded = recorded_row.iloc[0]
                maximum_difference = max(abs(recalculated[metric] - float(recorded[metric])) for metric in METRICS)
                metric_differences[f"{outcome_name}::fold_{fold}::{model_name}"] = maximum_difference
                check(f"{outcome_name}::fold_{fold}::{model_name}::metrics_reproduced", maximum_difference < 1e-12)
                fold_results.append((model_name, recalculated, float(weight.sum())))

        total_weight = sum(weight for model, values, weight in fold_results if model == "linear_margin")
        for model_name in MODELS:
            aggregate = {
                metric: sum(values[metric] * weight for model, values, weight in fold_results if model == model_name)
                / total_weight
                for metric in METRICS
            }
            recorded = summary["outcomes"][outcome_name]["fold_weighted_oof_metrics"][model_name]
            maximum_difference = max(abs(aggregate[metric] - recorded[metric]) for metric in METRICS)
            metric_differences[f"{outcome_name}::aggregate::{model_name}"] = maximum_difference
            check(f"{outcome_name}::aggregate::{model_name}::metrics_reproduced", maximum_difference < 1e-12)
        outcome_summary = summary["outcomes"][outcome_name]
        check(f"{outcome_name}::bootstrap_complete", outcome_summary["bootstrap_replicates_ok"] == config["bootstrap_replicates"])
        check(f"{outcome_name}::fold_contract_count", len(outcome_summary["fold_contracts"]) == config["folds"])
        check(
            f"{outcome_name}::four_training_fold_knots",
            all(len(contract["spline_knots_from_training_weighted_quantiles"]) == config["spline_n_knots"] for contract in outcome_summary["fold_contracts"]),
        )

    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "run_id": config["run_id"],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "maximum_metric_reproduction_difference": max(metric_differences.values()),
        "checks": checks,
    }
    args.output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
