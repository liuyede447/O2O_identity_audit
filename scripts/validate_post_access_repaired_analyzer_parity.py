"""Compare the repaired analyzer with the frozen post-access correction.

This is read-only software QA.  It imports the repaired aggregation and
bootstrap functions but never invokes the confirmatory CLI, never creates an
outcome-access receipt, and never writes a normative verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
import analyze_confirmatory_lockbox as repaired  # noqa: E402


METRICS = {
    "H1": ("normalized_paired_gap", "greater_than_zero"),
    "H2": ("normalized_o2m_gap", "less_than_zero"),
    "H3": ("fixed_minus_normalized_o2o_gap", "greater_than_zero"),
    "H4": ("fixed_eligibility_minus_normalized", "greater_than_zero"),
    "H5": ("normalized_within_minus_fixed", "greater_than_zero"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def calculate(
    *,
    selected: Path,
    fixed_dir: Path,
    normalized_dir: Path,
    fixed_anatomy: Path,
    normalized_anatomy: Path,
    reps: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict]:
    selection = pd.read_csv(selected, usecols=["image_id", "stratum"])
    selection_roster = repaired.normalize_selection_roster(selection)
    fixed = pd.read_csv(fixed_dir / "per_gt.csv")
    normalized = pd.read_csv(normalized_dir / "per_gt.csv")
    fixed_a = pd.read_csv(fixed_anatomy / "per_direction_anatomy.csv")
    normalized_a = pd.read_csv(normalized_anatomy / "per_direction_anatomy.csv")
    branch_images = repaired.branch_image_aggregate(fixed, normalized, selection_roster)
    pathway_images = repaired.pathway_aggregate(fixed_a, normalized_a, selection_roster)

    fixed_point = repaired.branch_from_aggregate(branch_images, "fixed")
    normalized_point = repaired.branch_from_aggregate(branch_images, "norm")
    pathway_point = repaired.pathway_stats(pathway_images)
    points = {
        "normalized_paired_gap": normalized_point["paired_gap"],
        "normalized_o2m_gap": normalized_point["o2m_gap"],
        "fixed_minus_normalized_o2o_gap": (
            fixed_point["o2o_gap"] - normalized_point["o2o_gap"]
        ),
        **pathway_point,
    }

    samples = {key: [] for key in points}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        branch_sample = repaired.resample(branch_images, rng)
        pathway_sample = repaired.resample(pathway_images, rng)
        fixed_stats = repaired.branch_from_aggregate(branch_sample, "fixed")
        normalized_stats = repaired.branch_from_aggregate(branch_sample, "norm")
        path_stats = repaired.pathway_stats(pathway_sample)
        samples["normalized_paired_gap"].append(normalized_stats["paired_gap"])
        samples["normalized_o2m_gap"].append(normalized_stats["o2m_gap"])
        samples["fixed_minus_normalized_o2o_gap"].append(
            fixed_stats["o2o_gap"] - normalized_stats["o2o_gap"]
        )
        samples["fixed_eligibility_minus_normalized"].append(
            path_stats["fixed_eligibility_minus_normalized"]
        )
        samples["normalized_within_minus_fixed"].append(
            path_stats["normalized_within_minus_fixed"]
        )
    sample_arrays = {key: np.asarray(value, dtype=float) for key, value in samples.items()}
    return points, sample_arrays, {
        "selected_image_clusters": int(len(selection_roster)),
        **branch_images.attrs["support_audit"],
    }


def validate(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    receipt_sidecar = Path(str(args.receipt) + ".sha256")
    if not args.receipt.is_file() or not receipt_sidecar.is_file():
        raise FileNotFoundError("historical receipt or sidecar is missing")
    receipt_before = sha256(args.receipt)
    sidecar_before = sha256(receipt_sidecar)

    expected = json.loads(args.corrective_results.read_text(encoding="utf-8"))
    if expected.get("artifact_role") != "exploratory_not_normative_verdict":
        raise RuntimeError("reference artifact is not the frozen exploratory correction")
    if expected.get("normative_lockbox_verdict_exists") is not False:
        raise RuntimeError("reference artifact incorrectly claims a normative verdict")
    if expected.get("bootstrap_replicates") != args.reps or expected.get("seed") != args.seed:
        raise RuntimeError("requested bootstrap contract differs from the corrective artifact")

    points, samples, support = calculate(
        selected=args.selected,
        fixed_dir=args.fixed_dir,
        normalized_dir=args.normalized_dir,
        fixed_anatomy=args.fixed_anatomy,
        normalized_anatomy=args.normalized_anatomy,
        reps=args.reps,
        seed=args.seed,
    )
    expected_samples = pd.read_csv(args.corrective_bootstrap)
    if list(expected_samples.columns) != list(samples):
        raise RuntimeError("corrective bootstrap columns differ from repaired analyzer metrics")

    point_differences: dict[str, float] = {}
    interval_differences: dict[str, float] = {}
    p_differences: dict[str, float] = {}
    bootstrap_differences: dict[str, float] = {}
    for hypothesis, (metric, direction) in METRICS.items():
        reference = expected["results"][hypothesis]
        point_differences[hypothesis] = abs(points[metric] - reference["point_estimate"])
        interval_differences[hypothesis] = float(
            np.max(
                np.abs(
                    np.asarray(repaired.ci(samples[metric]))
                    - np.asarray(reference["ci_95_two_sided"])
                )
            )
        )
        p_differences[hypothesis] = abs(
            repaired.empirical_one_sided_p(samples[metric], direction)
            - reference["one_sided_empirical_p_descriptive"]
        )
        bootstrap_differences[hypothesis] = float(
            np.max(
                np.abs(
                    samples[metric]
                    - expected_samples[metric].to_numpy(dtype=float)
                )
            )
        )

    expected_support = expected["support_audit"]["branch"]
    support_expected_counts = {
        "selected_image_clusters": expected_support["selected_images"],
        "fixed_common_valid": expected_support["fixed_common_valid_gt"],
        "normalized_common_valid": expected_support["normalized_common_valid_gt"],
        "intersection": expected_support["intersection_common_valid_gt"],
        "fixed_only": expected_support["fixed_only_gt_count"],
        "normalized_only": expected_support["normalized_only_gt_count"],
    }
    tolerance = 1e-15
    maximums = {
        "point": max(point_differences.values()),
        "interval": max(interval_differences.values()),
        "empirical_p": max(p_differences.values()),
        "bootstrap": max(bootstrap_differences.values()),
    }
    if any(value > tolerance for value in maximums.values()):
        raise AssertionError(f"repaired analyzer parity exceeded tolerance: {maximums}")
    if support != support_expected_counts:
        raise AssertionError(
            f"repaired analyzer support differs: actual={support}, expected={support_expected_counts}"
        )

    receipt_after = sha256(args.receipt)
    sidecar_after = sha256(receipt_sidecar)
    if receipt_after != receipt_before or sidecar_after != sidecar_before:
        raise AssertionError("historical outcome-access receipt changed during parity validation")

    result = {
        "status": "PASS",
        "artifact_role": "post_access_software_qa_not_normative_verdict",
        "prospective_status": "historical_lockbox_v1_remains_invalidated",
        "normative_lockbox_verdict_created": False,
        "repaired_analyzer_sha256": sha256(SCRIPT_ROOT / "analyze_confirmatory_lockbox.py"),
        "validator_sha256": sha256(Path(__file__)),
        "corrective_results_sha256": sha256(args.corrective_results),
        "corrective_bootstrap_sha256": sha256(args.corrective_bootstrap),
        "bootstrap_replicates": args.reps,
        "seed": args.seed,
        "tolerance": tolerance,
        "maximum_absolute_difference": maximums,
        "support_counts": support,
        "receipt_sha256_before": receipt_before,
        "receipt_sha256_after": receipt_after,
        "receipt_sidecar_sha256_before": sidecar_before,
        "receipt_sidecar_sha256_after": sidecar_after,
        "historical_receipt_unchanged": True,
        "scientific_claim_status_changed": False,
    }
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "PARITY_RESULT.json"
    write_json(result_path, result)
    report_path = args.output_dir / "PARITY_REPORT.md"
    report_path.write_text(
        "# Repaired analyzer / Lockbox-v1 corrective parity\n\n"
        "Status: **PASS**.\n\n"
        "The repaired aggregation and bootstrap implementation reproduced all five "
        "frozen post-access exploratory point estimates, confidence intervals, "
        "empirical p-values, 5,000 bootstrap trajectories, and support counts within "
        f"an absolute tolerance of {tolerance:g}.\n\n"
        "The historical outcome-access receipt and its sidecar were byte-hash "
        "unchanged. No confirmatory CLI was invoked and no `LOCKBOX_VERDICT.json` "
        "was created. Lockbox v1 remains invalidated and the comparison is software "
        "QA only.\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in (result_path, report_path)
        ),
        encoding="ascii",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--normalized-dir", type=Path, required=True)
    parser.add_argument("--fixed-anatomy", type=Path, required=True)
    parser.add_argument("--normalized-anatomy", type=Path, required=True)
    parser.add_argument("--corrective-results", type=Path, required=True)
    parser.add_argument("--corrective-bootstrap", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    print(json.dumps(validate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
