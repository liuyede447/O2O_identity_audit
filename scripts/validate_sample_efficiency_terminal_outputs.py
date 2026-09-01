"""Validate terminal outputs from the formal 1,000-repeat sample-efficiency run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    run = args.run_dir.resolve()
    artifact = run / "artifact"
    config = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    replicates = pd.read_csv(artifact / "sample_efficiency_replicates.csv")
    summary = pd.read_csv(artifact / "sample_efficiency_summary.csv")
    status = json.loads((run / "STATUS.json").read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}

    def check(name: str, value: object) -> None:
        checks[name] = bool(value)

    check("manifest_complete", manifest.get("status") == "complete")
    check("config_hash_matches_status", sha256(run / "CONFIG.json") == status["config_sha256"])
    check("analyzer_hash_matches_config", sha256(root / config["script"]) == config["script_sha256"])
    check(
        "input_hash_matches_config",
        sha256(root / config["input_per_gt"]) == config["input_per_gt_sha256"],
    )
    check(
        "helper_hash_matches_config",
        sha256(root / config["normalized_contrasts_helper"])
        == config["normalized_contrasts_helper_sha256"],
    )
    check("manifest_analyzer_hash_matches_config", manifest["source_sha256"]["analyzer"] == config["script_sha256"])
    check("manifest_input_hash_matches_config", manifest["source_sha256"]["input_per_gt"] == config["input_per_gt_sha256"])
    check(
        "manifest_helper_hash_matches_config",
        manifest["source_sha256"]["normalized_contrasts_helper"]
        == config["normalized_contrasts_helper_sha256"],
    )
    check("budgets_match_config", manifest["budgets"] == config["budgets"])
    check("repeats_match_config", manifest["repeats_per_budget"] == config["repeats"])
    check("nested_bootstrap_disabled", manifest["cluster_bootstrap_replicates_per_repeat"] == 0 == config["nested_bootstrap_replicates"])
    check("empty_stderr", (run / "stderr.log").stat().st_size == 0)
    check("row_count", len(replicates) == len(config["budgets"]) * config["repeats"])
    check("summary_row_count", len(summary) == len(config["budgets"]))
    check("budget_set", set(replicates["budget_images"]) == set(config["budgets"]) == set(summary["budget_images"]))
    check("replicate_values_finite", np.isfinite(replicates[["paired_contrast", "absolute_error_vs_full", "four_rate_spearman"]].to_numpy(float)).all())
    check("binary_indicators", set(replicates["sign_consistent"]).issubset({0, 1}) and set(replicates["branch_order_consistent"]).issubset({0, 1}))
    check("bootstrap_width_empty_when_disabled", replicates["bootstrap_ci_width"].isna().all() and summary["median_bootstrap_ci_width_pp"].isna().all())

    maximum_summary_difference = 0.0
    for budget in config["budgets"]:
        group = replicates[replicates["budget_images"] == budget].sort_values("repeat")
        recorded = summary[summary["budget_images"] == budget]
        check(f"budget_{budget}::one_summary_row", len(recorded) == 1)
        check(f"budget_{budget}::repeat_count", len(group) == config["repeats"])
        check(f"budget_{budget}::repeat_ids", group["repeat"].tolist() == list(range(config["repeats"])))
        expected = {
            "repeats": float(len(group)),
            "median_absolute_error_pp": 100.0 * float(group["absolute_error_vs_full"].median()),
            "p95_absolute_error_pp": 100.0 * float(group["absolute_error_vs_full"].quantile(0.95)),
            "sign_consistency": float(group["sign_consistent"].mean()),
            "branch_order_consistency": float(group["branch_order_consistent"].mean()),
            "median_four_rate_spearman": float(group["four_rate_spearman"].median()),
        }
        row = recorded.iloc[0]
        difference = max(abs(expected[name] - float(row[name])) for name in expected)
        maximum_summary_difference = max(maximum_summary_difference, difference)
        check(f"budget_{budget}::summary_reproduced", difference < 1e-12)
        check(
            f"budget_{budget}::absolute_error_identity",
            np.max(np.abs(group["absolute_error_vs_full"].to_numpy(float) - np.abs(group["paired_contrast"].to_numpy(float) - manifest["full_paired_contrast"]))) < 1e-12,
        )
        check(
            f"budget_{budget}::sign_indicator_identity",
            np.array_equal(
                group["sign_consistent"].to_numpy(int),
                (np.sign(group["paired_contrast"].to_numpy(float)) == np.sign(manifest["full_paired_contrast"])).astype(int),
            ),
        )

    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "run_id": config["run_id"],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "maximum_summary_reproduction_difference": maximum_summary_difference,
        "checks": checks,
    }
    args.output.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if validation["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
