"""Freeze the six-condition equivalent-side kappa sensitivity matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "stress_contract_sensitivity_20260829"
LEGACY = ROOT / "results" / "reviewer_controls_20260829"
OUT = BASE / "frozen_matrix"


SOURCES = [
    ("AI-TOD-v2", 0.03125, BASE / "ai_s0e300_equiv_k003125"),
    ("AI-TOD-v2", 0.0625, LEGACY / "formal_s0e300_k00625"),
    ("AI-TOD-v2", 0.125, BASE / "ai_s0e300_equiv_k0125"),
    ("VisDrone", 0.03125, BASE / "vis_b4e150_equiv_k003125"),
    ("VisDrone", 0.0625, LEGACY / "vis_b4e150_equiv_k00625"),
    ("VisDrone", 0.125, BASE / "vis_b4e150_equiv_k0125"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    rows, sources = [], []
    for dataset, kappa, directory in SOURCES:
        summary_path = directory / "normalized_perturbation_summary.json"
        manifest_path = directory / "manifest.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or float(manifest["kappa"]) != kappa:
            raise RuntimeError(f"invalid source: {directory}")
        point, ci = summary["point"], summary["ci_95"]
        row = {
            "dataset": dataset,
            "kappa": kappa,
            "selected_images": int(manifest["selected_images"]),
            "common_valid_gt": int(manifest["common_valid_normalized_gt"]),
            "contributing_images": int(summary["n_images"]),
        }
        for key in ("o2o_scale_contrast", "o2m_scale_contrast", "paired_difference_in_scale_gradients"):
            row[key] = point[key]
            row[f"{key}_ci_low"] = ci[key][0]
            row[f"{key}_ci_high"] = ci[key][1]
        rows.append(row)
        sources.extend([
            {"path": str(summary_path), "sha256": sha256(summary_path)},
            {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        ])
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "kappa_sensitivity_matrix.csv", index=False)
    verdict = {
        "all_o2m_intervals_below_zero": bool((frame["o2m_scale_contrast_ci_high"] < 0).all()),
        "all_paired_intervals_above_zero": bool((frame["paired_difference_in_scale_gradients_ci_low"] > 0).all()),
        "all_o2o_intervals_include_zero": bool(((frame["o2o_scale_contrast_ci_low"] <= 0) & (frame["o2o_scale_contrast_ci_high"] >= 0)).all()),
        "interpretation": "branch divergence is stable across tested kappa; absolute O2O response is kappa- and dataset-dependent",
    }
    frozen = {
        "status": "complete",
        "protocol": "stress_contract_sensitivity_v1",
        "read_only": True,
        "rows": rows,
        "verdict": verdict,
        "sources": sources,
    }
    (OUT / "stress_contract_sensitivity.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Equivalent-side kappa sensitivity", "",
        "All values are 8--16 minus 16--32 px any-direction fragility contrasts.", "",
        frame.to_markdown(index=False), "",
        f"- O2M intervals below zero in all conditions: **{verdict['all_o2m_intervals_below_zero']}**",
        f"- Paired intervals above zero in all conditions: **{verdict['all_paired_intervals_above_zero']}**",
        f"- O2O intervals include zero in all conditions: **{verdict['all_o2o_intervals_include_zero']}**",
        "- Frozen interpretation: branch divergence is stable across the tested kappa values; the absolute O2O response is not.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
