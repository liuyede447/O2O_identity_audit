"""Assess whether dense-grid results preserve the legacy scientific structure.

This is a read-only comparison over geometry post-processing artifacts.  It
does not recompute detector assignments or alter manuscript evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.05, 0.0625, 0.10, 0.125, 0.20, 0.25)
SIZE_GROUPS = ("all", "t_8_16", "s_16_32")
O2M = ("o2m_legacy", "o2m_pre_topk", "o2m_assigned_positive")
CAUSES = ("eligibility", "topk", "conflict", "within_set", "other")
TAU_LABELS = {
    0.05: "tau_00500",
    0.0625: "tau_00625",
    0.10: "tau_01000",
    0.125: "tau_01250",
    0.20: "tau_02000",
    0.25: "tau_02500",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=ROOT / "evidence" / "artifacts" / "runs" / "20260903_boundary_full_grid_1024_postprocess_v2",
    )
    parser.add_argument(
        "--legacy-tau-root",
        type=Path,
        default=ROOT / "evidence" / "artifacts" / "runs" / "20260901_rmcbd_tau_sensitivity",
    )
    parser.add_argument(
        "--legacy-primary",
        type=Path,
        default=ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_geometry_formal_v2" / "artifact",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_dir(root: Path, primary: Path, tau: float) -> Path:
    if math.isclose(tau, 0.125, abs_tol=1e-12):
        return primary
    label = "tau_00500_v2" if math.isclose(tau, 0.05, abs_tol=1e-12) else TAU_LABELS[tau]
    return root / label


def load_artifact(directory: Path, expected_tau: float) -> dict:
    required = ("summary.json", "survival_cif.csv", "rmsr.csv", "rmsr_contrasts.csv", "margin_construct.csv")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete artifact {directory}: {missing}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or not math.isclose(float(summary["tau"]), expected_tau, abs_tol=1e-12):
        raise ValueError(f"artifact is not complete for tau={expected_tau}: {directory}")
    outputs = summary.get("output_sha256", {})
    for name in required[1:]:
        if outputs.get(name) != sha256(directory / name):
            raise ValueError(f"artifact hash mismatch: {directory / name}")
    return {
        "directory": str(directory.resolve()),
        "summary_sha256": sha256(directory / "summary.json"),
        "event_csv_sha256": summary.get("event_csv_sha256"),
        "curve_csv_sha256": summary.get("curve_csv_sha256"),
        "survival": pd.read_csv(directory / "survival_cif.csv"),
        "rmsr": pd.read_csv(directory / "rmsr.csv"),
        "contrasts": pd.read_csv(directory / "rmsr_contrasts.csv"),
        "margin": pd.read_csv(directory / "margin_construct.csv"),
    }


def unique_row(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        if isinstance(value, float):
            selected = selected.loc[selected[column].map(lambda item: math.isclose(float(item), value, abs_tol=1e-12))]
        else:
            selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def estimate(frame: pd.DataFrame, **filters) -> float:
    value = float(unique_row(frame, **filters)["estimate"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite estimate for {filters}")
    return value


def verdict(legacy_signature, dense_signature) -> str:
    return "KEEP" if legacy_signature == dense_signature else "CHANGED"


def survival_ordering(legacy: dict[float, dict], dense: dict[float, dict]) -> dict:
    rows = []
    legacy_signature, dense_signature = [], []
    for tau in TAUS:
        for size in SIZE_GROUPS:
            o2o_legacy = estimate(legacy[tau]["survival"], size_group=size, estimand="o2o", metric="survival", radius=tau)
            o2o_dense = estimate(dense[tau]["survival"], size_group=size, estimand="o2o", metric="survival", radius=tau)
            for o2m in O2M:
                old = estimate(legacy[tau]["survival"], size_group=size, estimand=o2m, metric="survival", radius=tau)
                new = estimate(dense[tau]["survival"], size_group=size, estimand=o2m, metric="survival", radius=tau)
                old_gap, new_gap = o2o_legacy - old, o2o_dense - new
                old_sign, new_sign = old_gap > 0, new_gap > 0
                legacy_signature.append(old_sign)
                dense_signature.append(new_sign)
                rows.append({
                    "tau": tau, "size_group": size, "o2m_estimand": o2m,
                    "legacy_o2o_survival": o2o_legacy, "legacy_o2m_survival": old, "legacy_gap": old_gap,
                    "dense_o2o_survival": o2o_dense, "dense_o2m_survival": new, "dense_gap": new_gap,
                    "gap_delta": new_gap - old_gap,
                })
    return {
        "verdict": verdict(legacy_signature, dense_signature),
        "criterion": "At each tau endpoint and size group, O2O survival exceeds each O2M survival.",
        "legacy_all_pass": all(legacy_signature), "dense_all_pass": all(dense_signature), "rows": rows,
    }


def tiny_small_structure(legacy: dict[float, dict], dense: dict[float, dict]) -> dict:
    rows = []
    legacy_signature, dense_signature = [], []
    for tau in TAUS:
        old_values = {
            estimand: {
                size: estimate(legacy[tau]["rmsr"], size_group=size, estimand=estimand, tau=tau)
                for size in ("t_8_16", "s_16_32")
            }
            for estimand in ("o2o", *O2M)
        }
        new_values = {
            estimand: {
                size: estimate(dense[tau]["rmsr"], size_group=size, estimand=estimand, tau=tau)
                for size in ("t_8_16", "s_16_32")
            }
            for estimand in ("o2o", *O2M)
        }
        for estimand in ("o2o", *O2M):
            old_delta = old_values[estimand]["s_16_32"] - old_values[estimand]["t_8_16"]
            new_delta = new_values[estimand]["s_16_32"] - new_values[estimand]["t_8_16"]
            legacy_signature.append(old_delta > 0)
            dense_signature.append(new_delta > 0)
            rows.append({
                "tau": tau, "component": f"small_minus_tiny_{estimand}",
                "legacy": old_delta, "dense": new_delta, "delta": new_delta - old_delta,
            })
        old_advantage = (
            old_values["o2o"]["s_16_32"] - old_values["o2m_assigned_positive"]["s_16_32"]
            - old_values["o2o"]["t_8_16"] + old_values["o2m_assigned_positive"]["t_8_16"]
        )
        new_advantage = (
            new_values["o2o"]["s_16_32"] - new_values["o2m_assigned_positive"]["s_16_32"]
            - new_values["o2o"]["t_8_16"] + new_values["o2m_assigned_positive"]["t_8_16"]
        )
        legacy_signature.append(old_advantage > 0)
        dense_signature.append(new_advantage > 0)
        rows.append({
            "tau": tau, "component": "small_minus_tiny_o2o_advantage_over_o2m_assigned",
            "legacy": old_advantage, "dense": new_advantage, "delta": new_advantage - old_advantage,
        })
    return {
        "verdict": verdict(legacy_signature, dense_signature),
        "criterion": "Preserve signs of small-minus-tiny RMCBD for every branch and of the small-minus-tiny O2O advantage.",
        "legacy_signature": legacy_signature, "dense_signature": dense_signature, "rows": rows,
    }


def cause_composition(legacy: dict[float, dict], dense: dict[float, dict]) -> dict:
    rows = []
    legacy_signature, dense_signature = [], []
    for tau in TAUS:
        for size in SIZE_GROUPS:
            old = {
                cause: estimate(legacy[tau]["survival"], size_group=size, estimand="o2o", metric=f"cif_{cause}", radius=tau)
                for cause in CAUSES
            }
            new = {
                cause: estimate(dense[tau]["survival"], size_group=size, estimand="o2o", metric=f"cif_{cause}", radius=tau)
                for cause in CAUSES
            }
            old_total, new_total = sum(old.values()), sum(new.values())
            old_dominant = max(CAUSES, key=lambda cause: (old[cause], -CAUSES.index(cause)))
            new_dominant = max(CAUSES, key=lambda cause: (new[cause], -CAUSES.index(cause)))
            old_relation = old["eligibility"] >= old["within_set"]
            new_relation = new["eligibility"] >= new["within_set"]
            legacy_signature.append((old_dominant, old_relation))
            dense_signature.append((new_dominant, new_relation))
            rows.append({
                "tau": tau, "size_group": size,
                "legacy_dominant": old_dominant, "dense_dominant": new_dominant,
                "legacy_eligibility_ge_within_set": old_relation,
                "dense_eligibility_ge_within_set": new_relation,
                "legacy_total_incidence": old_total, "dense_total_incidence": new_total,
                "legacy_shares": {cause: old[cause] / old_total if old_total else None for cause in CAUSES},
                "dense_shares": {cause: new[cause] / new_total if new_total else None for cause in CAUSES},
                "cif_delta": {cause: new[cause] - old[cause] for cause in CAUSES},
            })
    return {
        "verdict": verdict(legacy_signature, dense_signature),
        "criterion": "Preserve the dominant first-boundary cause and eligibility-versus-within-set ordering by tau and size group.",
        "rows": rows,
    }


def rmcbd_direction(legacy: dict[float, dict], dense: dict[float, dict]) -> dict:
    rows = []
    legacy_signature, dense_signature = [], []
    for tau in TAUS:
        for size in SIZE_GROUPS:
            for o2m in O2M:
                contrast = f"o2o_minus_{o2m}"
                old_row = unique_row(legacy[tau]["contrasts"], size_group=size, contrast=contrast, tau=tau)
                new_row = unique_row(dense[tau]["contrasts"], size_group=size, contrast=contrast, tau=tau)
                old, new = float(old_row["estimate"]), float(new_row["estimate"])
                old_positive, new_positive = old > 0, new > 0
                legacy_signature.append(old_positive)
                dense_signature.append(new_positive)
                rows.append({
                    "tau": tau, "size_group": size, "contrast": contrast,
                    "legacy": old, "legacy_ci": [float(old_row["ci_low"]), float(old_row["ci_high"])],
                    "dense": new, "dense_ci": [float(new_row["ci_low"]), float(new_row["ci_high"])],
                    "delta": new - old,
                })
    return {
        "verdict": verdict(legacy_signature, dense_signature),
        "criterion": "Preserve the sign of every O2O-minus-O2M RMCBD contrast across six tau values and three size groups.",
        "legacy_all_positive": all(legacy_signature), "dense_all_positive": all(dense_signature), "rows": rows,
    }


def margin_relation(legacy: dict[float, dict], dense: dict[float, dict]) -> dict:
    tau = 0.125
    analyses = (
        "margin_vs_observed_rho_R",
        "specificity_control_margin_vs_eligibility_cause_overall_radius",
        "specificity_difference_rhoR_minus_eligibility_correlation",
    )
    rows = []
    legacy_signature, dense_signature = [], []
    for size in SIZE_GROUPS:
        values = {}
        for analysis in analyses:
            old_row = unique_row(legacy[tau]["margin"], size_group=size, analysis=analysis)
            new_row = unique_row(dense[tau]["margin"], size_group=size, analysis=analysis)
            old, new = float(old_row["estimate"]), float(new_row["estimate"])
            values[analysis] = (old, new)
            rows.append({
                "size_group": size, "analysis": analysis,
                "legacy": old, "dense": new, "delta": new - old,
                "legacy_ci": [float(old_row["ci_low"]), float(old_row["ci_high"])],
                "dense_ci": [float(new_row["ci_low"]), float(new_row["ci_high"])],
                "legacy_n_rows": None if pd.isna(old_row.get("n_rows")) else int(float(old_row["n_rows"])),
                "dense_n_rows": None if pd.isna(new_row.get("n_rows")) else int(float(new_row["n_rows"])),
            })
        old_rho, new_rho = values[analyses[0]]
        old_difference, new_difference = values[analyses[2]]
        legacy_signature.append((old_rho > 0, old_difference > 0))
        dense_signature.append((new_rho > 0, new_difference > 0))
    return {
        "verdict": verdict(legacy_signature, dense_signature),
        "criterion": "Margin remains positively related to observed rho_R and more strongly than to eligibility-boundary distance.",
        "legacy_all_pass": all(all(pair) for pair in legacy_signature),
        "dense_all_pass": all(all(pair) for pair in dense_signature),
        "rows": rows,
    }


def render_markdown(payload: dict) -> str:
    sections = payload["assessments"]
    lines = [
        "# Dense-grid scientific-structure freeze assessment", "",
        f"Overall verdict: **{payload['overall_verdict']}**", "",
        "The verdict concerns qualitative scientific structure. Numerical shifts are reported below and remain subject to manuscript-level interpretation.", "",
        "| Criterion | Verdict | Numerical evidence |", "|---|---|---|",
    ]
    survival = sections["o2o_o2m_survival_ordering"]
    min_dense_gap = min(row["dense_gap"] for row in survival["rows"])
    max_gap_change = max(abs(row["gap_delta"]) for row in survival["rows"])
    branch = sections["tiny_small_branch_structure"]
    branch_changes = sum(a != b for a, b in zip(branch["legacy_signature"], branch["dense_signature"]))
    causes = sections["first_boundary_cause_composition"]
    cause_changes = sum(
        (row["legacy_dominant"], row["legacy_eligibility_ge_within_set"])
        != (row["dense_dominant"], row["dense_eligibility_ge_within_set"])
        for row in causes["rows"]
    )
    rmcbd = sections["rmcbd_direction_across_tau"]
    min_contrast = min(row["dense"] for row in rmcbd["rows"])
    margin = sections["margin_rho_R_relation"]
    dense_rho = [row["dense"] for row in margin["rows"] if row["analysis"] == "margin_vs_observed_rho_R"]
    dense_specificity = [row["dense"] for row in margin["rows"] if row["analysis"].startswith("specificity_difference")]
    lines.extend([
        f"| O2O/O2M survival ordering | {survival['verdict']} | minimum dense O2O–O2M survival gap {min_dense_gap:.6f}; maximum absolute legacy-to-dense gap change {max_gap_change:.6f} |",
        f"| Tiny/small branch structure | {branch['verdict']} | {branch_changes} sign changes among {len(branch['legacy_signature'])} prespecified comparisons |",
        f"| First-boundary cause composition | {causes['verdict']} | {cause_changes} dominant/relation changes among {len(causes['rows'])} tau-size cells |",
        f"| RMCBD direction across tau | {rmcbd['verdict']} | minimum dense O2O–O2M contrast {min_contrast:.6f} |",
        f"| Margin–rho_R relationship | {margin['verdict']} | dense rho range {min(dense_rho):.4f}–{max(dense_rho):.4f}; specificity-difference range {min(dense_specificity):.4f}–{max(dense_specificity):.4f} |",
        "", "## Interpretation", "",
    ])
    if payload["overall_verdict"] == "KEEP":
        lines.append("All five prespecified structural conclusions are retained by the full-domain 1/1024 analysis.")
    elif payload["overall_verdict"] == "CHANGED":
        lines.append("At least one prespecified structural conclusion differs from the legacy analysis; manuscript claims must be revised before freezing.")
    else:
        lines.append("The comparison failed an input or consistency requirement and cannot support a scientific freeze.")
    lines.extend(["", "## Provenance", ""])
    lines.append(f"- Dense root: `{payload['dense_root']}`")
    lines.append(f"- Legacy tau root: `{payload['legacy_tau_root']}`")
    lines.append(f"- Legacy primary tau=0.125: `{payload['legacy_primary']}`")
    lines.append(f"- Assessment script SHA-256: `{payload['script_sha256']}`")
    lines.append("")
    lines.append("Machine-readable per-tau and per-size values are in `assessment.json`.")
    return "\n".join(lines) + "\n"


def assess(args: argparse.Namespace) -> dict:
    dense, legacy = {}, {}
    sources = {"dense": {}, "legacy": {}}
    for tau in TAUS:
        dense_dir = args.dense_root / TAU_LABELS[tau]
        legacy_path = legacy_dir(args.legacy_tau_root, args.legacy_primary, tau)
        dense[tau] = load_artifact(dense_dir, tau)
        legacy[tau] = load_artifact(legacy_path, tau)
        for label, artifact in (("dense", dense[tau]), ("legacy", legacy[tau])):
            sources[label][str(tau)] = {
                key: artifact[key] for key in ("directory", "summary_sha256", "event_csv_sha256", "curve_csv_sha256")
            }
    assessments = {
        "o2o_o2m_survival_ordering": survival_ordering(legacy, dense),
        "tiny_small_branch_structure": tiny_small_structure(legacy, dense),
        "first_boundary_cause_composition": cause_composition(legacy, dense),
        "rmcbd_direction_across_tau": rmcbd_direction(legacy, dense),
        "margin_rho_R_relation": margin_relation(legacy, dense),
    }
    individual = [item["verdict"] for item in assessments.values()]
    overall = "FAIL" if "FAIL" in individual else "CHANGED" if "CHANGED" in individual else "KEEP"
    return {
        "status": "complete", "overall_verdict": overall,
        "decision_rule": "FAIL for incomplete/inconsistent inputs; CHANGED for any structural-signature change; otherwise KEEP.",
        "dense_root": str(args.dense_root.resolve()),
        "legacy_tau_root": str(args.legacy_tau_root.resolve()),
        "legacy_primary": str(args.legacy_primary.resolve()),
        "taus": list(TAUS), "sources": sources, "assessments": assessments,
        "script_sha256": sha256(Path(__file__)),
    }


def self_test() -> dict:
    assert verdict([True, False], [True, False]) == "KEEP"
    assert verdict([True], [False]) == "CHANGED"
    frame = pd.DataFrame([{"size_group": "all", "estimand": "o2o", "metric": "survival", "radius": 0.05, "estimate": 0.9}])
    assert math.isclose(estimate(frame, size_group="all", estimand="o2o", metric="survival", radius=0.05), 0.9)
    return {"status": "PASS", "tests": 3}


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.output_dir is None:
        raise ValueError("--output-dir is required")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        payload = assess(args)
    except Exception as exc:
        payload = {
            "status": "failed", "overall_verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}",
            "dense_root": str(args.dense_root.resolve()), "legacy_tau_root": str(args.legacy_tau_root.resolve()),
            "legacy_primary": str(args.legacy_primary.resolve()), "script_sha256": sha256(Path(__file__)),
        }
        raise
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "assessment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "ASSESSMENT.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "overall_verdict": payload["overall_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
