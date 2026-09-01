"""Freeze learning-dynamics trajectories after all queue tasks complete."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "five_experiment_upgrade_20260830" / "learning_dynamics"


def main() -> None:
    status = json.loads((OUT / "queue_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError("learning-dynamics queue is not complete")
    inventory = pd.read_csv(OUT / "checkpoint_inventory.csv")
    validation_path = OUT / "formal_validation_metrics.csv"
    validation = pd.read_csv(validation_path) if validation_path.is_file() else None
    records = []
    for row in inventory.to_dict("records"):
        epoch = int(row["trajectory_epoch"])
        for stress_mode in ("fixed_1px", "normalized"):
            directory = OUT / f"epoch_{epoch:03d}_{stress_mode}"
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((directory / "normalized_perturbation_summary.json").read_text(encoding="utf-8"))
            point, ci = summary["point"], summary["ci_95"]
            record = {
                **row,
                "stress_mode": stress_mode,
                "common_valid_gt": summary["n_gt"],
                "o2o_scale_contrast_pp": 100 * point["o2o_scale_contrast"],
                "o2o_ci_low_pp": 100 * ci["o2o_scale_contrast"][0],
                "o2o_ci_high_pp": 100 * ci["o2o_scale_contrast"][1],
                "o2m_scale_contrast_pp": 100 * point["o2m_scale_contrast"],
                "o2m_ci_low_pp": 100 * ci["o2m_scale_contrast"][0],
                "o2m_ci_high_pp": 100 * ci["o2m_scale_contrast"][1],
                "paired_contrast_pp": 100 * point["paired_difference_in_scale_gradients"],
                "paired_ci_low_pp": 100 * ci["paired_difference_in_scale_gradients"][0],
                "paired_ci_high_pp": 100 * ci["paired_difference_in_scale_gradients"][1],
                "audit_elapsed_sec": manifest["audit_elapsed_sec"],
            }
            if validation is not None:
                match = validation[validation["trajectory_epoch"].eq(epoch)]
                if len(match) == 1:
                    record.update({key: float(match.iloc[0][key]) for key in ("precision", "recall", "map50", "map50_95")})
            records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "learning_dynamics_trajectory.csv", index=False)
    findings = {"status": "complete", "trajectory_points": int(inventory.shape[0]), "stress_contracts": ["fixed_1px", "normalized"], "descriptive_correlations": {}}
    for stress, group in frame.groupby("stress_mode"):
        findings["descriptive_correlations"][stress] = {
            "epoch_vs_paired_spearman": float(spearmanr(group["trajectory_epoch"], group["paired_contrast_pp"]).statistic),
            "train_cls_loss_vs_paired_spearman": float(spearmanr(group["train_cls_loss"], group["paired_contrast_pp"]).statistic),
        }
        if "map50_95" in group and group["map50_95"].notna().sum() >= 3:
            findings["descriptive_correlations"][stress]["map_vs_paired_spearman"] = float(spearmanr(group["map50_95"], group["paired_contrast_pp"]).statistic)
    (OUT / "learning_dynamics_summary.json").write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    print(frame[["trajectory_epoch", "stress_mode", "o2o_scale_contrast_pp", "o2m_scale_contrast_pp", "paired_contrast_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
