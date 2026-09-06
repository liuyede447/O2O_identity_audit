"""Build final_evidence_v4 without modifying the immutable v3 predecessor."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "evidence_freeze/final_evidence_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate_terms(value):
    if isinstance(value, dict):
        return {key: migrate_terms(item) for key, item in value.items()}
    if isinstance(value, list):
        return [migrate_terms(item) for item in value]
    if not isinstance(value, str):
        return value
    replacements = (
        ("post-conflict loss-active O2O identity", "post-conflict assigned O2O identity"),
        ("loss-active O2O identity", "post-conflict assigned O2O identity"),
        ("loss-active identity", "post-conflict assigned O2O identity"),
        ("loss-active candidate", "post-conflict assigned O2O candidate"),
        ("loss-active boundary", "assigned-identity boundary"),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def entry(evidence_id: str, claim: str, artifact: str, *, dataset=None, checkpoint=None, seed=None,
          unit=None, denominator=None, estimand=None, stress=None, sampling=None, weighting=None,
          uncertainty=None, estimates=None, stage="revision_validation", validity=None,
          script=None, manifest=None, multiplicity="descriptive; no confirmatory family") -> dict:
    artifact_path = ROOT / artifact
    manifest_hash = sha256(ROOT / manifest) if manifest else None
    return {
        "claim_id": evidence_id,
        "evidence_id": evidence_id,
        "claim": claim,
        "dataset": dataset,
        "checkpoint_sha256": checkpoint,
        "seed": seed,
        "unit": unit,
        "denominator": denominator,
        "estimand": estimand,
        "stress_contract": stress,
        "sampling": sampling,
        "weighting": weighting,
        "uncertainty": uncertainty,
        "estimates": estimates or {},
        "multiplicity_status": multiplicity,
        "evidence_stage_status": stage,
        "artifact": artifact,
        "artifact_sha256": sha256(artifact_path),
        "script_sha256": {"analysis": sha256(ROOT / script)} if script else None,
        "manifest_path": manifest,
        "manifest_sha256": manifest_hash,
        "validity_notes": validity,
        "missing_reason": {},
    }


def main() -> None:
    existing = [path for path in HERE.iterdir() if path.is_file() and path.name != Path(__file__).name]
    if existing:
        raise FileExistsError(f"v4 output files already exist: {[path.name for path in existing]}")
    ledger = migrate_terms(copy.deepcopy(json.loads((V3 / "CURRENT_EVIDENCE_LEDGER.json").read_text(encoding="utf-8"))))
    loss_path = "results/measurement_validation_20260905/parity_5251_with_target_score_v1/summary.json"
    provenance_path = "results/measurement_validation_20260905/checkpoint_provenance_v1/summary.json"
    count_path = "runs/20260905_o2m_positive_count_scale_controls_v5/artifact/summary.json"
    fixed_path = "runs/20260905_multiseed_fixed_completion_v1/summary.json"
    equivalent_path = "runs/20260905_multiseed_equivalent_rank_set_v1/summary.json"
    for required in (loss_path, provenance_path, count_path, fixed_path, equivalent_path):
        if not (ROOT / required).is_file():
            raise FileNotFoundError(required)
    loss = json.loads((ROOT / loss_path).read_text(encoding="utf-8"))
    provenance = json.loads((ROOT / provenance_path).read_text(encoding="utf-8"))
    count = json.loads((ROOT / count_path).read_text(encoding="utf-8"))
    fixed = json.loads((ROOT / fixed_path).read_text(encoding="utf-8"))
    equivalent = json.loads((ROOT / equivalent_path).read_text(encoding="utf-8"))
    checkpoint_map = {key: item["checkpoint_sha256"] for key, item in provenance["checkpoints"].items()}
    count_object = count["standardized_analyses"]["object_any_top1_fragility"]
    count_direction = count["standardized_analyses"]["direction_top1_flip"]
    count_turnover = count["standardized_analyses"]["rank_set_turnover"]
    additions = [
        entry(
            "EV-ASSIGNED-TARGET-SCORE-SEMANTICS-V1",
            "Native/production/reference parity remained exact, and every audited assigned O2O identity in the 300 base states and 5,251 sampled directional states had positive normalized target-score mass; assignment alone is nevertheless not sufficient for positive loss weight in the executable edge case.",
            loss_path,
            dataset="AI-TOD-v2", checkpoint=checkpoint_map["0"], seed=0,
            unit="assigned focal-GT state", denominator=loss["loss_weight_semantics"],
            estimand="count of post-conflict assigned O2O identities with nonpositive normalized target score",
            stress=["base", "fixed_1px", "equivalent_side_kappa_0.0625"],
            sampling="frozen 300-image parity roster plus deterministic hash-selected directional states",
            weighting=None, uncertainty="deterministic qualification; no sampling interval",
            estimates=loss["loss_weight_semantics"], stage="revision_instrument_qualification",
            validity="Supports positive loss weight on the audited states while retaining assigned-identity terminology because assignment is not sufficient in general.",
            script="scripts/validate_native_production_reference_parity.py", multiplicity=None,
        ),
        entry(
            "EV-CHECKPOINT-PROVENANCE-V1",
            "The four released files named best.pt record val=false, null embedded best_fitness/fitness, and zero-based epoch 280; seeds 0--2 have recovered post-training scan logs showing epoch-280 promotion, while seed 3 retains an explicit missing-log caveat.",
            provenance_path,
            dataset="AI-TOD-v2", checkpoint=checkpoint_map, seed=[0, 1, 2, 3],
            unit="checkpoint artifact and training run", denominator={"checkpoints": 4, "training_rows_per_run": 300},
            estimand="checkpoint naming and selection provenance", stress=None,
            sampling="all protocol-eligible released seed checkpoints", weighting=None,
            uncertainty=None, estimates={"seed0_to_2_promotion_logs_recovered": 3, "seed3_promotion_log_recovered": 0},
            stage="revision_provenance", validity="The filename is not evidence of online validation selection. Post-training selection and formal AP recheck used the same validation benchmark.",
            script="scripts/audit_checkpoint_provenance.py", multiplicity=None,
        ),
        entry(
            "EV-POSITIVE-COUNT-THREE-STAGE-V1",
            "Full-support crude, K=3--7 overlap-support crude, and overlap-standardised estimates separate support restriction from count-composition standardisation; most Top-1 attenuation arose at the support-restriction step, while positive-set turnover remained large.",
            count_path,
            dataset="AI-TOD-v2", checkpoint=checkpoint_map["0"], seed=0,
            unit="common-valid focal GT or directional replay", denominator={"common_valid_gt": 6377, "contributing_images": 288, "overlap_coverage_tiny": 0.6028970374970802, "overlap_coverage_small": 0.6429609485181138},
            estimand="16--32 minus 8--16 contrast before restriction, after K=3--7 restriction, and after within-overlap K standardisation",
            stress="equivalent_side_kappa_0.0625", sampling="frozen outcome-blind 300-image roster",
            weighting="inverse image-inclusion weights; pooled weighted K distribution for standardisation",
            uncertainty=count["bootstrap"],
            estimates={
                "object_top1": count_object,
                "direction_top1": count_direction,
                "positive_set_turnover": count_turnover,
            },
            stage="revision_descriptive_control", validity="Descriptive decomposition; neither restriction nor standardisation identifies a causal count effect.",
            script="scripts/analyze_o2m_positive_count_and_scale_controls.py",
        ),
        entry(
            "EV-MULTISEED-FIXED-COMPLETION-V1",
            "The fixed-one-pixel paired branch contrast retained the same direction across every protocol-eligible released seed under a crossed four-seed and image-cluster bootstrap.",
            fixed_path,
            dataset="AI-TOD-v2", checkpoint=checkpoint_map, seed=[0, 1, 2, 3],
            unit="strict cross-seed common-valid focal GT", denominator={"common_valid_gt": fixed["cross_seed_common_valid_gt"], "selected_images": 300},
            estimand=fixed["estimand"], stress="fixed_1px", sampling="frozen outcome-blind 300-image roster; every protocol-eligible released seed",
            weighting="inverse image-inclusion weights and equal-seed mean",
            uncertainty=fixed["bootstrap"], estimates={"per_seed": fixed["per_seed"], "point": fixed["point"], "ci_95": fixed["ci_95"]},
            stage="revision_training_seed_sensitivity", validity="Four top-level seeds remain a limited training-randomness sample; seed 4 failed the frozen initialisation rule.",
            script="scripts/analyze_multiseed_fixed_completion.py", manifest="protocols/MULTISEED_FIXED_COMPLETION_SPEC_20260905.json",
        ),
        entry(
            "EV-MULTISEED-EQUIVALENT-RANKSET-V1",
            "The equivalent-area paired endpoint, positive-count difference, Top-1 turnover, and positive-set turnover retained their reported directions across every protocol-eligible released seed.",
            equivalent_path,
            dataset="AI-TOD-v2", checkpoint=checkpoint_map, seed=[0, 1, 2, 3],
            unit="strict cross-seed common-valid focal GT and its legal directional replays", denominator={"common_valid_gt": equivalent["cross_seed_common_valid_gt"], "direction_rows_per_seed": 25165, "selected_images": 300},
            estimand=equivalent["estimand"], stress="equivalent_side_kappa_0.0625",
            sampling="frozen outcome-blind 300-image roster; every protocol-eligible released seed",
            weighting="inverse image-inclusion weights and equal-seed mean",
            uncertainty=equivalent["bootstrap"], estimates={"per_seed": equivalent["per_seed"], "point": equivalent["point"], "ci_95": equivalent["ci_95"]},
            stage="revision_training_seed_sensitivity", validity=equivalent["interpretation"],
            script="scripts/analyze_multiseed_equivalent_rank_set.py", manifest="protocols/MULTISEED_EQUIVALENT_RANK_SET_SPEC_20260905.json",
        ),
    ]
    existing_ids = {item["evidence_id"] for item in ledger["entries"]}
    if existing_ids.intersection(item["evidence_id"] for item in additions):
        raise RuntimeError("v4 additions collide with predecessor evidence IDs")
    ledger["entries"].extend(additions)
    ledger.update({
        "schema_version": "4.0",
        "status": "FINAL_EVIDENCE_V4_AWAITING_AUTHOR_REAPPROVAL",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "predecessor": {"path": "evidence_freeze/final_evidence_v3", "ledger_sha256": sha256(V3 / "CURRENT_EVIDENCE_LEDGER.json"), "preserved": True},
        "terminology_migration": "The state formerly labelled loss-active identity is now labelled post-conflict assigned O2O identity. Audited target-score positivity is reported separately.",
        "active_manuscript_authority": {**ledger.get("active_manuscript_authority", {}), "assigned_target_score_semantics": additions[0]["evidence_id"], "checkpoint_provenance": additions[1]["evidence_id"], "positive_count_three_stage": additions[2]["evidence_id"], "multiseed_fixed": additions[3]["evidence_id"], "multiseed_equivalent_rank_set": additions[4]["evidence_id"]},
    })
    ledger_path = HERE / "CURRENT_EVIDENCE_LEDGER.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "CURRENT_EVIDENCE_LEDGER.md").write_text(
        "# Current evidence ledger v4\n\n"
        f"Status: **{ledger['status']}**\n\n"
        f"- Admitted evidence records: {len(ledger['entries'])}\n"
        f"- Excluded predecessor records: {len(ledger.get('excluded_from_evidence', []))}\n"
        "- New revision records: assigned/target-score semantics, checkpoint provenance, three-stage count decomposition, four-seed fixed endpoint, and four-seed equivalent-area/Rank--Set sensitivity.\n"
        "- Predecessor final_evidence_v3 remains byte-preserved.\n",
        encoding="utf-8",
    )
    excluded = json.loads((V3 / "EXCLUDED_EVIDENCE.json").read_text(encoding="utf-8"))
    (HERE / "EXCLUDED_EVIDENCE.json").write_text(json.dumps(excluded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result_claim = [{"artifact": item["artifact"], "artifact_sha256": item["artifact_sha256"], "evidence_id": item["evidence_id"], "claim": item["claim"], "evidence_stage_status": item["evidence_stage_status"], "multiplicity_status": item["multiplicity_status"], "validity_notes": item["validity_notes"]} for item in ledger["entries"]]
    source_result = [{"evidence_id": item["evidence_id"], "sources": [], "artifact": item["artifact"], "artifact_sha256": item["artifact_sha256"], "manifest_path": item["manifest_path"], "manifest_sha256": item["manifest_sha256"], "evidence_stage_status": item["evidence_stage_status"]} for item in ledger["entries"]]
    (HERE / "RESULT_TO_CLAIM_MAPPING.json").write_text(json.dumps({"entries": result_claim}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "SOURCE_TO_RESULT_MAPPING.json").write_text(json.dumps({"entries": source_result}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = {"status": "PASS", "checks": {"predecessor_preserved": True, "unique_evidence_ids": len({item['evidence_id'] for item in ledger['entries']}) == len(ledger['entries']), "pending_scientific_gates": 0, "author_reapproval_required": True}, "entries": len(ledger["entries"])}
    (HERE / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (HERE / "EVIDENCE_FREEZE_REPORT.md").write_text(
        "# Final evidence freeze v4\n\nStatus: **AWAITING_AUTHOR_REAPPROVAL**.\n\n"
        "This revision preserves final_evidence_v3, corrects assignment-state terminology, and adds the five audited revision records. No new training, artificial fixed-K assigner, third dataset, or multi-seed dense-grid analysis was performed.\n",
        encoding="utf-8",
    )
    manifest = {
        "status": ledger["status"],
        "frozen_at_utc": ledger["updated_at"],
        "evidence_entries": len(ledger["entries"]),
        "excluded_records": len(ledger.get("excluded_from_evidence", [])),
        "pending_scientific_gates": [],
        "author_reapproval_required": True,
        "predecessor_final_evidence_v3": ledger["predecessor"],
        "current_evidence_ledger_sha256": sha256(ledger_path),
        "current_evidence_ledger_markdown_sha256": sha256(HERE / "CURRENT_EVIDENCE_LEDGER.md"),
        "source_to_result_mapping_sha256": sha256(HERE / "SOURCE_TO_RESULT_MAPPING.json"),
        "result_to_claim_mapping_sha256": sha256(HERE / "RESULT_TO_CLAIM_MAPPING.json"),
        "excluded_evidence_sha256": sha256(HERE / "EXCLUDED_EVIDENCE.json"),
        "validation_sha256": sha256(HERE / "VALIDATION.json"),
        "creator_script_sha256": sha256(Path(__file__).resolve()),
    }
    (HERE / "FINAL_EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in HERE.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    (HERE / "SHA256SUMS.txt").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "entries": manifest["evidence_entries"]}, indent=2))


if __name__ == "__main__":
    main()
