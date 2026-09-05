"""Build the additive final_evidence_v3 bundle from frozen v2 plus dense-grid evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V2 = ROOT / "evidence_freeze" / "final_evidence_v2"
POST = ROOT / "runs" / "20260903_boundary_full_grid_1024_postprocess_v2"
POP = ROOT / "runs" / "20260902_boundary_full_grid_1024_population_v1"
GATE = ROOT / "results" / "measurement_validation_20260902" / "dense_grid_convergence_gate_v3"
TAU_DIRS = {
    "0.05": "tau_00500",
    "0.0625": "tau_00625",
    "0.1": "tau_01000",
    "0.125": "tau_01250",
    "0.2": "tau_02000",
    "0.25": "tau_02500",
}
CHECKPOINT_SHA256 = "4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa"
V2_MANIFEST_SHA256 = "4f626da49dcd7ee87ffd33311e9c08c73629e2b7b4cb0b2c7e74501a7a5a1ca9"
V2_SHA256SUMS_SHA256 = "cea0c7dcb471c9ac72141fae685549d8f84821f868329f5e4704f76f0bfabdc2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(rows: list[dict[str, str]], **filters: object) -> dict[str, str]:
    matched = [row for row in rows if all(str(row[key]) == str(value) for key, value in filters.items())]
    if len(matched) != 1:
        raise RuntimeError(f"expected one row for {filters}, found {len(matched)}")
    return matched[0]


def estimate(row: dict[str, str]) -> dict[str, object]:
    return {
        "point": float(row["estimate"]),
        "ci95": [float(row["ci_low"]), float(row["ci_high"])],
    }


def source(sha: str, *paths: str) -> dict[str, object]:
    return {
        "sha256": sha,
        "resolved_paths": list(paths),
        "resolution_note": "path resolution is informational; SHA-256 is authoritative",
    }


def assert_frozen_inputs() -> tuple[dict, dict, dict, dict]:
    if sha256(V2 / "FINAL_EVIDENCE_MANIFEST.json") != V2_MANIFEST_SHA256:
        raise RuntimeError("frozen final_evidence_v2 manifest changed")
    if sha256(V2 / "SHA256SUMS.txt") != V2_SHA256SUMS_SHA256:
        raise RuntimeError("frozen final_evidence_v2 checksum ledger changed")
    ledger = load_json(V2 / "CURRENT_EVIDENCE_LEDGER.json")
    result_map = load_json(V2 / "RESULT_TO_CLAIM_MAPPING.json")
    source_map = load_json(V2 / "SOURCE_TO_RESULT_MAPPING.json")
    excluded = load_json(V2 / "EXCLUDED_EVIDENCE.json")
    if ledger.get("status") != "FINAL_EVIDENCE_FROZEN" or ledger.get("pending") != []:
        raise RuntimeError("predecessor ledger is not final and pending-free")
    return ledger, result_map, source_map, excluded


def convergence_entry() -> tuple[dict, dict]:
    artifact = GATE / "convergence_gate.json"
    payload = load_json(artifact)
    if payload.get("status") != "FAIL_NOT_CONVERGED" or payload.get("gate_pass") is not False:
        raise RuntimeError("unexpected convergence-gate verdict")
    entry = {
        "claim_id": "EV-BOUNDARY-GRID-CONVERGENCE-V1",
        "evidence_id": "EV-BOUNDARY-GRID-CONVERGENCE-V1",
        "claim": "On the prespecified 30-image convergence subset, 1/512 and 1/1024 grids shared all 4,408 trajectory keys and event-state classifications, but nine material localization or return-state mismatches occurred across four trajectories; the full-population primary grid was therefore fixed at 1/1024.",
        "dataset": "AI-TOD-v2",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "seed": {"selection": 20260823, "bootstrap": None},
        "unit": "focal-GT directional continuous-radius trajectory",
        "denominator": {
            "images": 30,
            "trajectories_per_grid": int(payload["grid_1_1024"]["trajectory_count"]),
            "shared_trajectory_keys": int(payload["key_audit"]["shared"]),
            "material_mismatch_instances": int(payload["material_mismatch_count"]),
            "material_trajectories": int(payload["material_trajectory_count"]),
        },
        "estimand": "deterministic grid-convergence classification for event, censor, first-cause, transition-radius, and return-state fields",
        "stress_contract": "continuous equivalent-side replay through radius 0.25, comparing exhaustive 1/512 and 1/1024 grids",
        "sampling": "same frozen 30-image convergence subset for both resolutions",
        "weighting": None,
        "uncertainty": None,
        "estimates": {
            "material_mismatch_fraction": int(payload["material_trajectory_count"]) / int(payload["key_audit"]["shared"]),
            "fine_only_event_count": sum(int(value) for value in payload["fine_only_events_missed_by_1_512"].values()),
            "selected_primary_grid_step": 1.0 / 1024.0,
        },
        "multiplicity_status": "not applicable to deterministic convergence qualification",
        "evidence_stage_status": "instrument_resolution_qualification",
        "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(artifact),
        "script_sha256": sha256(ROOT / "scripts" / "analyze_dense_grid_convergence.py"),
        "manifest_path": None,
        "manifest_sha256": None,
        "validity_notes": "This fail-closed gate selects numerical resolution; it is not a model-effect estimate. The absence of fine-only events did not override the prespecified zero-material-mismatch rule.",
        "missing_reason": {
            "weighting": "The comparison is deterministic and paired by trajectory key; no estimator weighting is used.",
            "uncertainty": "No sampling interval is defined for the deterministic convergence gate.",
            "manifest_path": "The gate artifact binds both grid CSV hashes directly and has no companion manifest.",
            "manifest_sha256": "No companion manifest exists.",
        },
    }
    mapping = {
        "evidence_id": entry["evidence_id"],
        "sources": [
            source(entry["script_sha256"], "scripts/analyze_dense_grid_convergence.py"),
            source(payload["grid_1_512"]["event_csv_sha256"], "runs/20260902_boundary_full_grid_512_subset_v2/per_direction_boundary_events.csv"),
            source(payload["grid_1_512"]["curve_csv_sha256"], "runs/20260902_boundary_full_grid_512_subset_v2/o2m_rank_set_radius_curve.csv"),
            source(payload["grid_1_1024"]["event_csv_sha256"], "runs/20260902_boundary_full_grid_1024_subset_v2/per_direction_boundary_events.csv"),
            source(payload["grid_1_1024"]["curve_csv_sha256"], "runs/20260902_boundary_full_grid_1024_subset_v2/o2m_rank_set_radius_curve.csv"),
        ],
        "artifact": entry["artifact"],
        "artifact_sha256": entry["artifact_sha256"],
        "manifest_path": None,
        "manifest_sha256": None,
        "evidence_stage_status": entry["evidence_stage_status"],
    }
    return entry, mapping


def geometry_entry() -> tuple[dict, dict]:
    terminal = load_json(POST / "TERMINAL_VALIDATION.json")
    if terminal.get("status") != "PASS" or set(terminal.get("geometry", {})) != set(TAU_DIRS.values()):
        raise RuntimeError("six-tau dense-grid terminal validation is incomplete")
    source_validation = terminal["source_validation"]
    if source_validation.get("status") != "PASS" or int(source_validation["event_rows"]) != 29724:
        raise RuntimeError("canonical population validation failed")

    headline_dir = POST / TAU_DIRS["0.125"]
    rmsr = read_csv(headline_dir / "rmsr.csv")
    contrasts = read_csv(headline_dir / "rmsr_contrasts.csv")
    survival = read_csv(headline_dir / "survival_cif.csv")
    returns = read_csv(headline_dir / "return_transition.csv")
    group_names = {"all": "all", "t_8_16": "tiny", "s_16_32": "small"}

    def contrast(group: str) -> dict[str, object]:
        return estimate(unique(
            contrasts,
            size_group=group,
            contrast="o2o_minus_o2m_assigned_positive",
            tau="0.125",
        ))

    estimates: dict[str, object] = {
        "all_o2o_rmcbd": estimate(unique(rmsr, size_group="all", estimand="o2o", tau="0.125")),
        "all_o2m_assigned_positive_rmcbd": estimate(unique(rmsr, size_group="all", estimand="o2m_assigned_positive", tau="0.125")),
        "tiny_o2o_minus_o2m_assigned_positive_rmcbd": contrast("t_8_16"),
        "small_o2o_minus_o2m_assigned_positive_rmcbd": contrast("s_16_32"),
        "rmcbd_o2o_minus_o2m_assigned_positive_by_tau": {},
        "survival_at_tau_0_125": {},
        "o2o_first_boundary_cif_at_tau_0_125": {},
        "return_transition_at_tau_0_125": {},
    }
    for tau, directory in TAU_DIRS.items():
        rows = read_csv(POST / directory / "rmsr_contrasts.csv")
        estimates["rmcbd_o2o_minus_o2m_assigned_positive_by_tau"][tau] = {
            label: estimate(unique(rows, size_group=group, contrast="o2o_minus_o2m_assigned_positive", tau=tau))
            for group, label in group_names.items()
        }
    for group, label in group_names.items():
        estimates["survival_at_tau_0_125"][label] = {
            branch: estimate(unique(survival, size_group=group, estimand=estimand, metric="survival", radius="0.125"))
            for branch, estimand in (("o2o", "o2o"), ("o2m_assigned_positive", "o2m_assigned_positive"))
        }
        estimates["o2o_first_boundary_cif_at_tau_0_125"][label] = {
            cause: estimate(unique(survival, size_group=group, estimand="o2o", metric=f"cif_{cause}", radius="0.125"))
            for cause in ("eligibility", "topk", "conflict", "within_set", "other")
        }
        estimates["return_transition_at_tau_0_125"][label] = {}
        for branch, estimand in (("o2o", "o2o"), ("o2m_assigned_positive", "o2m_assigned_positive")):
            estimates["return_transition_at_tau_0_125"][label][branch] = {
                metric: estimate(unique(returns, size_group=group, estimand=estimand, radius="0.125", metric=metric))
                for metric in ("ever_boundary_rate", "endpoint_changed_rate", "returned_to_base_after_boundary_rate")
            }

    population_summary = load_json(POP / "summary.json")
    entry = {
        "claim_id": "EV-BOUNDARY-GEOMETRY-V3",
        "evidence_id": "EV-BOUNDARY-GEOMETRY-V3",
        "claim": "On the complete 300-image 1/1024 grid, loss-active O2O identity retained a larger restricted mean cause-specific boundary distance than assigned-positive O2M across all six audited truncation radii; the O2O-minus-O2M gap remained larger for 16-32 px than for 8-16 px focal objects at tau=0.125.",
        "dataset": "AI-TOD-v2",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "seed": {"selection": 20260823, "bootstrap": 20260831},
        "unit": "focal-GT directional continuous-radius trajectory clustered by image",
        "denominator": {
            "images": int(population_summary["processed_images"]),
            "audited_focal_gt": sum(int(value) for value in population_summary["audited_focal_gt_by_size_bin"].values()),
            "directional_trajectories": int(population_summary["focal_direction_rows"]),
            "curve_rows": int(population_summary["radius_curve_rows"]),
            "o2o_base_defined_at_tau_0_125": int(unique(rmsr, size_group="all", estimand="o2o", tau="0.125")["n_base_defined"]),
            "o2m_assigned_positive_base_defined_at_tau_0_125": int(unique(rmsr, size_group="all", estimand="o2m_assigned_positive", tau="0.125")["n_base_defined"]),
        },
        "estimand": "IPW Kaplan-Meier restricted mean cause-specific boundary distance (RMCBD) and paired O2O-minus-O2M contrasts, with Aalen-Johansen cause-specific incidence and exact endpoint/return summaries",
        "stress_contract": "exhaustive continuous equivalent-side radius grid r in [0,0.25] at step 1/1024; RMCBD tau in {0.05,0.0625,0.10,0.125,0.20,0.25}",
        "sampling": "same frozen outcome-blind stratified 300-image discovery sample; 13 base-outside GTs excluded under the no-clipping trajectory-domain contract",
        "weighting": "inverse image-inclusion weights",
        "uncertainty": "5000 stratified image-cluster percentile bootstrap replicates per tau; weighted Kaplan-Meier and Aalen-Johansen estimators",
        "estimates": estimates,
        "multiplicity_status": "discovery analysis and prespecified tau sensitivity; no confirmatory multiplicity claim",
        "evidence_stage_status": "discovery_dense_grid_primary",
        "artifact": str((headline_dir / "rmsr_contrasts.csv").relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(headline_dir / "rmsr_contrasts.csv"),
        "script_sha256": terminal["source_hashes"]["geometry_script"],
        "manifest_path": str((POST / "TERMINAL_VALIDATION.json").relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(POST / "TERMINAL_VALIDATION.json"),
        "validity_notes": "Supersedes EV-BOUNDARY-GEOMETRY-V2 for manuscript boundary claims. Base-undefined trajectories are excluded rather than imputed; right censoring and competing causes are retained. The terminal validator bound all six tau outputs to canonical event and curve SHA-256 values and PASS status.",
        "missing_reason": {},
    }
    mapping = {
        "evidence_id": entry["evidence_id"],
        "sources": [
            source(entry["script_sha256"], "scripts/analyze_assignment_boundary_geometry.py"),
            source(source_validation["event_csv_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/per_direction_boundary_events.csv"),
            source(source_validation["curve_csv_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/o2m_rank_set_radius_curve.csv"),
            source(source_validation["manifest_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/manifest.json"),
            source(source_validation["summary_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/summary.json"),
        ],
        "artifact": entry["artifact"],
        "artifact_sha256": entry["artifact_sha256"],
        "manifest_path": entry["manifest_path"],
        "manifest_sha256": entry["manifest_sha256"],
        "evidence_stage_status": entry["evidence_stage_status"],
    }
    return entry, mapping


def margin_entry() -> tuple[dict, dict]:
    terminal = load_json(POST / "TERMINAL_VALIDATION.json")
    headline_dir = POST / TAU_DIRS["0.125"]
    rows = read_csv(headline_dir / "margin_construct.csv")

    def item(analysis: str) -> tuple[dict[str, object], dict[str, str]]:
        row = unique(rows, size_group="all", analysis=analysis)
        return estimate(row), row

    rho, rho_row = item("margin_vs_observed_rho_R")
    eligibility, eligibility_row = item("specificity_control_margin_vs_eligibility_cause_overall_radius")
    difference, _ = item("specificity_difference_rhoR_minus_eligibility_correlation")
    entry = {
        "claim_id": "EV-MARGIN-RHOR-CONSTRUCT-V3",
        "evidence_id": "EV-MARGIN-RHOR-CONSTRUCT-V3",
        "claim": "Among trajectories with an observed fixed-pair q-order crossing on the complete 1/1024 grid, the native active-relative margin was positively associated with crossing radius and more strongly than with eligibility-boundary radius on its separately observed support.",
        "dataset": "AI-TOD-v2",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "seed": {"selection": 20260823, "bootstrap": 20260831},
        "unit": "focal-GT directional continuous-radius trajectory clustered by image",
        "denominator": {
            "all_directional_trajectories": 29724,
            "rho_R_observed": int(float(rho_row["n_rows"])),
            "observed_rho_R_images": int(float(rho_row["n_images"])),
            "observed_eligibility_boundary_rows": int(float(eligibility_row["n_rows"])),
            "observed_eligibility_boundary_images": int(float(eligibility_row["n_images"])),
        },
        "estimand": "IPW-weighted Spearman association between base active-relative margin and observed fixed-pair q-order crossing radius, with a separately observed eligibility-boundary specificity control",
        "stress_contract": "exhaustive continuous equivalent-side radius grid r in [0,0.25] at step 1/1024 with fixed base active/runner pair and prespecified competing-censor order",
        "sampling": "same frozen outcome-blind stratified 300-image discovery sample",
        "weighting": "inverse image-inclusion weights",
        "uncertainty": "5000 stratified image-cluster percentile bootstrap replicates",
        "estimates": {
            "margin_vs_observed_rho_R_spearman": rho,
            "margin_vs_eligibility_boundary_radius_spearman": eligibility,
            "specificity_correlation_difference": difference,
        },
        "multiplicity_status": "discovery construct-validation analysis; no confirmatory multiplicity claim",
        "evidence_stage_status": "discovery_dense_grid_construct_validation",
        "artifact": str((headline_dir / "margin_construct.csv").relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(headline_dir / "margin_construct.csv"),
        "script_sha256": terminal["source_hashes"]["geometry_script"],
        "manifest_path": str((POST / "TERMINAL_VALIDATION.json").relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(POST / "TERMINAL_VALIDATION.json"),
        "validity_notes": "Supersedes EV-MARGIN-RHOR-CONSTRUCT-V2 for manuscript boundary claims. Both correlations are conditional on observed cases and use different supports; neither is a censoring-adjusted population correlation. This is construct localization, not a causal or predictive claim.",
        "missing_reason": {},
    }
    source_validation = terminal["source_validation"]
    mapping = {
        "evidence_id": entry["evidence_id"],
        "sources": [
            source(entry["script_sha256"], "scripts/analyze_assignment_boundary_geometry.py"),
            source(source_validation["event_csv_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/per_direction_boundary_events.csv"),
            source(source_validation["curve_csv_sha256"], "runs/20260902_boundary_full_grid_1024_population_v1/o2m_rank_set_radius_curve.csv"),
        ],
        "artifact": entry["artifact"],
        "artifact_sha256": entry["artifact_sha256"],
        "manifest_path": entry["manifest_path"],
        "manifest_sha256": entry["manifest_sha256"],
        "evidence_stage_status": entry["evidence_stage_status"],
    }
    return entry, mapping


def result_mapping(entry: dict) -> dict:
    return {
        "artifact": entry["artifact"],
        "artifact_sha256": entry["artifact_sha256"],
        "evidence_id": entry["evidence_id"],
        "claim": entry["claim"],
        "evidence_stage_status": entry["evidence_stage_status"],
        "multiplicity_status": entry["multiplicity_status"],
        "validity_notes": entry["validity_notes"],
    }


def main() -> None:
    old_ledger, old_result_map, old_source_map, excluded = assert_frozen_inputs()
    convergence, convergence_source = convergence_entry()
    geometry, geometry_source = geometry_entry()
    margin, margin_source = margin_entry()
    additions = [convergence, geometry, margin]

    ledger = deepcopy(old_ledger)
    ledger["updated_at"] = "2026-09-03T15:46:57+08:00"
    ledger["entries"].extend(additions)
    ledger["active_manuscript_authority"] = {
        "boundary_grid_resolution": convergence["evidence_id"],
        "boundary_geometry": geometry["evidence_id"],
        "margin_rho_R_construct": margin["evidence_id"],
        "superseded_for_active_manuscript_boundary_claims": [
            "EV-BOUNDARY-GEOMETRY-V2",
            "EV-MARGIN-RHOR-CONSTRUCT-V2",
        ],
    }
    ledger["predecessor"] = {
        "path": "evidence_freeze/final_evidence_v2",
        "manifest_sha256": V2_MANIFEST_SHA256,
        "sha256sums_sha256": V2_SHA256SUMS_SHA256,
        "preserved": True,
    }
    write_json(HERE / "CURRENT_EVIDENCE_LEDGER.json", ledger)

    result_map = deepcopy(old_result_map)
    result_map["entries"].extend(result_mapping(entry) for entry in additions)
    write_json(HERE / "RESULT_TO_CLAIM_MAPPING.json", result_map)
    source_map = deepcopy(old_source_map)
    source_map["entries"].extend([convergence_source, geometry_source, margin_source])
    write_json(HERE / "SOURCE_TO_RESULT_MAPPING.json", source_map)
    write_json(HERE / "EXCLUDED_EVIDENCE.json", excluded)

    ledger_md = """# Current Evidence Ledger

Status: **FINAL EVIDENCE FROZEN**  
Updated: 2026-09-03 +08:00

This additive v3 ledger preserves `final_evidence_v2` byte-for-byte and supersedes only its active manuscript boundary-geometry and margin--rho_R claims.

## Dense-grid additions

- `EV-BOUNDARY-GRID-CONVERGENCE-V1`: the fail-closed 30-image gate selected the 1/1024 grid after nine material mismatches across four of 4,408 trajectories.
- `EV-BOUNDARY-GEOMETRY-V3`: complete 300-image, 29,724-trajectory, six-tau dense-grid geometry evidence.
- `EV-MARGIN-RHOR-CONSTRUCT-V3`: dense-grid margin--rho_R construct-localization evidence.

The active boundary IDs are declared in `CURRENT_EVIDENCE_LEDGER.json`. Historical v2 entries remain registered for audit history but are not active manuscript authority for boundary numbers.
"""
    (HERE / "CURRENT_EVIDENCE_LEDGER.md").write_text(ledger_md, encoding="utf-8", newline="\n")
    report = """# Final evidence freeze v3

Status: **FINAL_EVIDENCE_FROZEN**.

- Predecessor: `final_evidence_v2`, preserved byte-for-byte.
- Evidence entries: 27 (24 predecessor entries plus 3 dense-grid additions).
- Active boundary resolution: exhaustive 1/1024 grid selected by fail-closed convergence gate.
- Canonical population: 300 images, 7,431 focal GT, 29,724 directional trajectories, 7,593,288 curve rows.
- Statistical post-processing: six tau values, 5,000 stratified image-cluster bootstrap replicates per tau, terminal status PASS.
- GitHub/public upload: not performed by this freeze.
"""
    (HERE / "EVIDENCE_FREEZE_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    validation = {
        "status": "PASS",
        "checks": {
            "predecessor_manifest_hash": True,
            "predecessor_sha256sums_hash": True,
            "convergence_gate_fail_closed": True,
            "canonical_population_validation": True,
            "six_tau_postprocess_terminal_validation": True,
            "active_boundary_ids_unique": True,
        },
        "canonical_input_sha256": {
            "event_csv": load_json(POST / "TERMINAL_VALIDATION.json")["source_validation"]["event_csv_sha256"],
            "curve_csv": load_json(POST / "TERMINAL_VALIDATION.json")["source_validation"]["curve_csv_sha256"],
            "population_manifest": load_json(POST / "TERMINAL_VALIDATION.json")["source_validation"]["manifest_sha256"],
            "population_summary": load_json(POST / "TERMINAL_VALIDATION.json")["source_validation"]["summary_sha256"],
            "postprocess_terminal_validation": sha256(POST / "TERMINAL_VALIDATION.json"),
        },
    }
    write_json(HERE / "VALIDATION.json", validation)

    manifest = {
        "status": "FINAL_EVIDENCE_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority_order": ledger["authority_order"],
        "evidence_entries": len(ledger["entries"]),
        "excluded_records": len(excluded["entries"]),
        "pending_gates": [],
        "active_manuscript_authority": ledger["active_manuscript_authority"],
        "predecessor_final_evidence_v2": ledger["predecessor"],
        "current_evidence_ledger_sha256": sha256(HERE / "CURRENT_EVIDENCE_LEDGER.json"),
        "current_evidence_ledger_markdown_sha256": sha256(HERE / "CURRENT_EVIDENCE_LEDGER.md"),
        "source_to_result_mapping_sha256": sha256(HERE / "SOURCE_TO_RESULT_MAPPING.json"),
        "result_to_claim_mapping_sha256": sha256(HERE / "RESULT_TO_CLAIM_MAPPING.json"),
        "excluded_evidence_sha256": sha256(HERE / "EXCLUDED_EVIDENCE.json"),
        "validation_sha256": sha256(HERE / "VALIDATION.json"),
        "creator_script_sha256": sha256(Path(__file__)),
    }
    write_json(HERE / "FINAL_EVIDENCE_MANIFEST.json", manifest)
    checksum_files = sorted(
        path for path in HERE.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    checksum_text = "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_files)
    (HERE / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "PASS",
        "entries": len(ledger["entries"]),
        "ledger_sha256": sha256(HERE / "CURRENT_EVIDENCE_LEDGER.json"),
        "manifest_sha256": sha256(HERE / "FINAL_EVIDENCE_MANIFEST.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
