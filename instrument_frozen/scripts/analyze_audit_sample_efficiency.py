"""Estimate normalized-audit accuracy as the selected-image budget decreases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_reviewer_killer_controls import normalized_contrasts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="25,50,100,150,200")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-helper-sha256")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate(counts: dict[str, int], budget: int) -> dict[str, int]:
    exact = {key: budget * value / sum(counts.values()) for key, value in counts.items()}
    allocation = {key: min(counts[key], int(np.floor(value))) for key, value in exact.items()}
    remaining = budget - sum(allocation.values())
    order = sorted(counts, key=lambda key: (exact[key] - allocation[key], counts[key]), reverse=True)
    for key in order:
        if remaining <= 0:
            break
        if allocation[key] < counts[key]:
            allocation[key] += 1; remaining -= 1
    return allocation


def adjusted_subset(frame: pd.DataFrame, selected: dict[str, list[str]], counts: dict[str, int]) -> pd.DataFrame:
    pieces = []
    for stratum, images in selected.items():
        part = frame[frame["image_id"].isin(images)].copy()
        part["sampling_weight"] = part["sampling_weight"].astype(float) * counts[stratum] / len(images)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_ci_width(frame: pd.DataFrame, reps: int, rng: np.random.Generator) -> float:
    if reps == 0:
        return float("nan")
    values = []
    by_stratum = {
        stratum: sorted(group["image_id"].unique())
        for stratum, group in frame.groupby("stratum")
    }
    for _ in range(reps):
        pieces = []
        for stratum, images in by_stratum.items():
            sampled = rng.choice(images, size=len(images), replace=True)
            multiplicity = pd.Series(sampled).value_counts()
            part = frame[frame["stratum"].eq(stratum) & frame["image_id"].isin(multiplicity.index)].copy()
            part["sampling_weight"] = part.apply(lambda row: float(row["sampling_weight"]) * int(multiplicity[row["image_id"]]), axis=1)
            pieces.append(part)
        sample = pd.concat(pieces, ignore_index=True)
        values.append(normalized_contrasts(sample)["paired_difference_in_scale_gradients"])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(hi - lo)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.repeats <= 0 or args.bootstrap < 0:
        raise ValueError("repeats must be positive and bootstrap must be non-negative")
    per_gt = args.audit_dir / "per_gt.csv"
    if not per_gt.is_file():
        raise FileNotFoundError(per_gt)
    helper = Path(__file__).resolve().with_name("run_reviewer_killer_controls.py")
    input_sha256 = sha256(per_gt)
    helper_sha256 = sha256(helper)
    if args.expected_input_sha256 and input_sha256 != args.expected_input_sha256.lower():
        raise RuntimeError(
            f"per_gt.csv SHA-256 mismatch: expected {args.expected_input_sha256}, got {input_sha256}"
        )
    if args.expected_helper_sha256 and helper_sha256 != args.expected_helper_sha256.lower():
        raise RuntimeError(
            f"normalized_contrasts helper SHA-256 mismatch: expected {args.expected_helper_sha256}, got {helper_sha256}"
        )
    frame = pd.read_csv(per_gt)
    frame = frame[frame["common_valid_margin"].eq(1)].copy()
    full = normalized_contrasts(frame)
    full_order_values = np.array([full[key] for key in ("o2o_8_16", "o2o_16_32", "o2m_8_16", "o2m_16_32")])
    images_by_stratum = {
        stratum: sorted(group["image_id"].unique())
        for stratum, group in frame.groupby("stratum")
    }
    counts = {key: len(value) for key, value in images_by_stratum.items()}
    budgets = [int(value) for value in args.budgets.split(",")]
    if max(budgets) >= sum(counts.values()):
        raise ValueError("budget must be below the full common-valid image count")
    rng = np.random.default_rng(args.seed)
    records = []
    for budget in budgets:
        allocation = allocate(counts, budget)
        for repeat in range(args.repeats):
            selected = {
                stratum: list(rng.choice(images_by_stratum[stratum], size=allocation[stratum], replace=False))
                for stratum in sorted(allocation) if allocation[stratum] > 0
            }
            sample = adjusted_subset(frame, selected, counts)
            estimate = normalized_contrasts(sample)
            order_values = np.array([estimate[key] for key in ("o2o_8_16", "o2o_16_32", "o2m_8_16", "o2m_16_32")])
            rho = float(spearmanr(full_order_values, order_values).statistic)
            ci_width = bootstrap_ci_width(sample, args.bootstrap, rng)
            records.append({
                "budget_images": budget,
                "repeat": repeat,
                "paired_contrast": estimate["paired_difference_in_scale_gradients"],
                "absolute_error_vs_full": abs(estimate["paired_difference_in_scale_gradients"] - full["paired_difference_in_scale_gradients"]),
                "sign_consistent": int(np.sign(estimate["paired_difference_in_scale_gradients"]) == np.sign(full["paired_difference_in_scale_gradients"])),
                "branch_order_consistent": int(estimate["o2o_scale_contrast"] > estimate["o2m_scale_contrast"]),
                "four_rate_spearman": rho,
                "bootstrap_ci_width": ci_width,
            })
    args.output_dir.mkdir(parents=True)
    raw = pd.DataFrame(records)
    raw.to_csv(args.output_dir / "sample_efficiency_replicates.csv", index=False)
    summary_rows = []
    for budget, group in raw.groupby("budget_images"):
        summary_rows.append({
            "budget_images": int(budget),
            "repeats": len(group),
            "median_absolute_error_pp": 100 * float(group["absolute_error_vs_full"].median()),
            "p95_absolute_error_pp": 100 * float(group["absolute_error_vs_full"].quantile(0.95)),
            "sign_consistency": float(group["sign_consistent"].mean()),
            "branch_order_consistency": float(group["branch_order_consistent"].mean()),
            "median_four_rate_spearman": float(group["four_rate_spearman"].median()),
            "median_bootstrap_ci_width_pp": 100 * float(group["bootstrap_ci_width"].median()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "sample_efficiency_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "protocol": "outcome-blind stratified image-budget subsampling of frozen normalized audit",
        "source_audit": str(args.audit_dir),
        "full_common_valid_images": int(frame["image_id"].nunique()),
        "full_paired_contrast": full["paired_difference_in_scale_gradients"],
        "budgets": budgets,
        "repeats_per_budget": args.repeats,
        "cluster_bootstrap_replicates_per_repeat": args.bootstrap,
        "nested_bootstrap_note": (
            "disabled; precision of the full audit is estimated separately by the prospective precision-design analysis"
            if args.bootstrap == 0 else
            "used descriptively for each budget replicate"
        ),
        "seed": args.seed,
        "ranking_definition": "Spearman ordering of O2O tiny/small and O2M tiny/small rates against the full audit",
        "source_sha256": {
            "analyzer": sha256(Path(__file__)),
            "input_per_gt": input_sha256,
            "normalized_contrasts_helper": helper_sha256,
        },
        "expected_sha256": {
            "input_per_gt": args.expected_input_sha256,
            "normalized_contrasts_helper": args.expected_helper_sha256,
        },
        "outputs": [
            "sample_efficiency_replicates.csv",
            "sample_efficiency_summary.csv",
            "manifest.json",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
