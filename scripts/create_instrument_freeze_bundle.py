"""Create the immutable audit-instrument source snapshot and preregistration.

This utility has two deliberately separate phases.  ``stage`` copies the exact
validated sources and validation summaries into an isolated directory.  After
that directory has received a source-only Git commit, ``finalize`` records the
commit, environment, frozen analysis rules, and prospective H1--H5 predictions.
It never reads lockbox outcomes and never selects lockbox images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


# The isolated freeze repository must not acquire untracked import caches while
# finalization or seal validation imports frozen helper modules.
sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TAG = "audit_instrument_v1.0"
PREREGISTRATION_TAG = "audit_preregistration_v1.0"

CORE_SCRIPTS = (
    "create_instrument_freeze_bundle.py",
    "create_run_output_inventory.py",
    "margin_o2o_replay.py",
    "reference_o2o_assigner_numpy.py",
    "validate_audit_instrument_oracles.py",
    "validate_audit_instrument_mutations.py",
    "validate_native_production_reference_parity.py",
    "run_reviewer_killer_controls.py",
    "analyze_native_alignment_path_anatomy.py",
    "bootstrap_native_path_anatomy.py",
    "analyze_eligibility_stable_and_pathway_contrasts.py",
    "run_eligibility_lock_counterfactual.py",
    "prepare_disjoint_audit_lockbox.py",
    "analyze_confirmatory_lockbox.py",
    "analyze_o2m_rank_set_decoupling.py",
    "analyze_assignment_boundary_radius.py",
    "analyze_assignment_boundary_geometry.py",
    "validate_boundary_radius_reference_subset.py",
    "analyze_assignment_spillover.py",
    "analyze_o2o_preresolution_control.py",
    "analyze_margin_shape_sensitivity.py",
    "analyze_lockbox_precision_design.py",
    "analyze_audit_sample_efficiency.py",
    "validate_current_evidence_ledger.py",
    "validate_sealed_lockbox_bundle.py",
    "validate_frozen_extraction_contract.py",
)

CORE_PROTOCOLS = (
    "BOUNDARY_DISCOVERY_CONTRACT.json",
    "BOUNDARY_DISCOVERY_CONTRACT.md",
    "LOCKBOX_SEALED_EXTRACTION_SPEC.md",
    "LOCKBOX_EXECUTION_CHECKLIST.md",
    "LOCKBOX_COMMAND_RUNBOOK_TEMPLATE.md",
    "SPILLOVER_GATED_PILOT_SPEC.md",
    "NUMERICAL_REPEATABILITY_CONTRACT.md",
)

GOVERNANCE_FILES = (
    "PROJECT_STATE.json",
    "NEXT_ACTION.md",
    "RUN_REGISTRY.csv",
)

LEDGER_FILES = (
    "CURRENT_EVIDENCE_LEDGER.json",
    "CURRENT_EVIDENCE_LEDGER.md",
    "SCHEMA.md",
)

VALIDATION_FILES = {
    "oracle_validation.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "oracle_suite_v1" / "oracle_validation.json",
    "mutation_validation.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "mutation_suite_v1" / "mutation_validation.json",
    "real_parity_summary.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "parity_full_v1" / "summary.json",
    "real_parity_deterministic_5k_summary.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "parity_deterministic_5k_v1" / "summary.json",
    "eligibility_lock_summary.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "eligibility_lock_v1" / "summary.json",
    "eligibility_stable_summary.json": ROOT / "evidence" / "artifacts" / "results" / "five_experiment_upgrade_20260830" / "eligibility_stable_sensitivity_v2" / "summary.json",
    "corrected_anatomy_bootstrap.json": ROOT / "evidence" / "artifacts" / "results" / "five_experiment_upgrade_20260830" / "anatomy_bootstrap_summary_v2.json",
    "rank_set_fixed_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T161235_rank_set_fixed_v1" / "artifact" / "rank_set_summary.json",
    "rank_set_normalized_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T162551_rank_set_normalized_v1" / "artifact" / "rank_set_summary.json",
    "preresolution_fixed_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T163900_preresolution_fixed_v1" / "artifact" / "summary.json",
    "preresolution_normalized_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T164000_preresolution_normalized_v1" / "artifact" / "summary.json",
    "boundary_numeric_repro_summary.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "boundary_illegal_shift_repro_v4" / "summary.json",
    "boundary_reference_smoke_summary.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "boundary_reference_smoke_v3" / "summary.json",
    "boundary_discovery_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T174200_boundary_discovery_v2" / "artifact" / "summary.json",
    "boundary_discovery_manifest.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831T174200_boundary_discovery_v2" / "artifact" / "manifest.json",
    "boundary_reference_formal_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_reference_formal_v2" / "artifact" / "summary.json",
    "boundary_reference_formal_manifest.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_reference_formal_v2" / "artifact" / "manifest.json",
    "boundary_reference_output_inventory.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_reference_formal_v2" / "artifact" / "OUTPUT_INVENTORY.json",
    "boundary_geometry_summary.json": ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_geometry_formal_v2" / "artifact" / "summary.json",
    "freeze_builder_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "freeze_builder_selftest.json",
    "lockbox_selector_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "selector_selftest.json",
    "sealed_bundle_validator_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "sealed_bundle_validator_selftest.json",
    "confirmatory_analyzer_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "confirmatory_analyzer_selftest.json",
    "lockbox_chain_selftest_manifest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "manifest.json",
    "run_output_inventory_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "run_output_inventory_selftest.json",
    "boundary_geometry_analyzer_selftest.json": ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "lockbox_chain_selftests_pre_freeze_current" / "boundary_geometry_analyzer_selftest.json",
}

EXPECTED_VALIDATION_STATUSES = {
    "oracle_validation.json": "PASS",
    "mutation_validation.json": "PASS",
    "real_parity_summary.json": "PASS",
    "real_parity_deterministic_5k_summary.json": "PASS",
    "eligibility_lock_summary.json": "complete",
    "eligibility_stable_summary.json": "complete",
    "corrected_anatomy_bootstrap.json": "complete",
    "rank_set_fixed_summary.json": "complete",
    "rank_set_normalized_summary.json": "complete",
    "preresolution_fixed_summary.json": "complete",
    "preresolution_normalized_summary.json": "complete",
    "boundary_numeric_repro_summary.json": "smoke_complete",
    "boundary_reference_smoke_summary.json": "PASS",
    "boundary_discovery_summary.json": "complete",
    "boundary_discovery_manifest.json": "complete",
    "boundary_reference_formal_summary.json": "PASS",
    "boundary_reference_formal_manifest.json": "PASS",
    "boundary_reference_output_inventory.json": "complete",
    "boundary_geometry_summary.json": "complete",
    "freeze_builder_selftest.json": "PASS",
    "lockbox_selector_selftest.json": "PASS",
    "sealed_bundle_validator_selftest.json": "PASS",
    "confirmatory_analyzer_selftest.json": "PASS",
    "lockbox_chain_selftest_manifest.json": "PASS",
    "run_output_inventory_selftest.json": "PASS",
    "boundary_geometry_analyzer_selftest.json": "PASS",
}

DISCOVERY_SELECTION = ROOT / "evidence" / "artifacts" / "results" / "five_experiment_upgrade_20260830" / "anatomy_fixed_1px_v2" / "selected_images.csv"
DATA_YAML = ROOT / "configs" / "aitod_v2_local_g.yaml"
CHECKPOINT = Path(r"E:\two_paper\beifen\archive_backup\artifacts\analysis_runs\detect\experiments\baseline_13zone\yolo26s_aitodv2_baseline_13zone_e300_s0\weights\best.pt")
LOCAL_TAL = ROOT / "ultralytics_local" / "ultralytics" / "utils" / "tal.py"
EXTERNAL_TAL = Path(r"D:\anaconda3\Lib\site-packages\ultralytics\utils\tal.py")


def frozen_discovery_exclusion(output: Path) -> Path:
    """Return the staged discovery-exclusion manifest inside the isolated repository."""
    return output / "frozen_inputs" / "discovery_selected_images_exclusion.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def inventory(directory: Path, excluded: set[str] | None = None) -> list[dict[str, object]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def validate_qualification_inputs() -> None:
    for target_name, source in VALIDATION_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = read_json(source, target_name)
        expected_status = EXPECTED_VALIDATION_STATUSES[target_name]
        if payload.get("status") != expected_status:
            raise RuntimeError(
                f"{target_name} status must be {expected_status!r}, got {payload.get('status')!r}"
            )
        if target_name in {
            "real_parity_summary.json",
            "real_parity_deterministic_5k_summary.json",
            "boundary_reference_smoke_summary.json",
            "boundary_reference_formal_summary.json",
        } and int(payload.get("mismatches", -1)) != 0:
            raise RuntimeError(f"{target_name} must report zero mismatches")

    oracle = read_json(VALIDATION_FILES["oracle_validation.json"], "oracle validation")
    oracle_cases = oracle.get("oracle_cases")
    properties = oracle.get("properties")
    if not isinstance(oracle_cases, list) or len(oracle_cases) != 6 or not all(
        isinstance(case, dict) and case.get("pass") is True for case in oracle_cases
    ):
        raise RuntimeError("oracle qualification must contain exactly 6/6 passing cases")
    if not isinstance(properties, dict) or len(properties) != 7 or not all(
        value is True for value in properties.values()
    ):
        raise RuntimeError("oracle qualification must contain exactly 7/7 passing properties")

    mutation = read_json(VALIDATION_FILES["mutation_validation.json"], "mutation validation")
    if mutation.get("killed") != 6 or mutation.get("total") != 6:
        raise RuntimeError("mutation qualification must report 6/6 killed mutations")
    parity = read_json(
        VALIDATION_FILES["real_parity_deterministic_5k_summary.json"],
        "deterministic parity summary",
    )
    if (
        parity.get("base_states") != 300
        or parity.get("sampled_direction_states") != 5251
        or parity.get("mismatches") != 0
    ):
        raise RuntimeError("deterministic parity qualification must report 300 base, 5,251 direction, zero mismatch")

    selftest_manifest = read_json(
        VALIDATION_FILES["lockbox_chain_selftest_manifest.json"],
        "lockbox chain self-test manifest",
    )
    report_hashes = selftest_manifest.get("report_sha256")
    script_hashes = selftest_manifest.get("script_sha256")
    if not isinstance(report_hashes, dict) or not isinstance(script_hashes, dict):
        raise RuntimeError("lockbox chain self-test manifest lacks report/script hashes")
    selftest_root = VALIDATION_FILES["lockbox_chain_selftest_manifest.json"].parent
    for name, expected in report_hashes.items():
        path = selftest_root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"lockbox self-test report hash mismatch: {name}")
    for relative, expected in script_hashes.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"lockbox self-test script hash mismatch: {relative}")

    boundary_run = ROOT / "evidence" / "artifacts" / "runs" / "20260831T174200_boundary_discovery_v2"
    boundary_summary = read_json(
        boundary_run / "artifact" / "summary.json", "boundary discovery summary"
    )
    boundary_manifest = read_json(
        boundary_run / "artifact" / "manifest.json", "boundary discovery manifest"
    )
    boundary_config = read_json(boundary_run / "CONFIG.json", "boundary discovery config")
    if (
        boundary_summary.get("selected_images") != 300
        or boundary_summary.get("focal_direction_rows") != 29724
        or sum(boundary_summary.get("audited_focal_gt_by_size_bin", {}).values()) != 7431
    ):
        raise RuntimeError("formal boundary discovery counts differ from the qualified contract")
    if (
        boundary_manifest.get("script_sha256") != boundary_config.get("script_sha256")
        or boundary_manifest.get("checkpoint_sha256") != boundary_config.get("checkpoint_sha256")
        or boundary_manifest.get("selected_images_source_sha256") != boundary_config.get("selected_images_sha256")
        or boundary_manifest.get("rmax") != 0.25
        or boundary_manifest.get("coarse_step") != 0.015625
        or boundary_manifest.get("fine_step") != 0.001953125
        or boundary_manifest.get("directions") != ["left", "right", "up", "down"]
    ):
        raise RuntimeError("formal boundary discovery manifest differs from the qualified contract")

    reference_run = ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_reference_formal_v2"
    reference_summary = read_json(
        reference_run / "artifact" / "summary.json", "formal boundary reference summary"
    )
    reference_manifest = read_json(
        reference_run / "artifact" / "manifest.json", "formal boundary reference manifest"
    )
    reference_config = read_json(reference_run / "CONFIG.json", "formal boundary reference config")
    reference_trajectories = int(reference_summary.get("reference_trajectories", -1))
    tolerance = float(reference_summary.get("allowed_radius_and_bracket_tolerance", -1.0))
    if (
        reference_summary.get("selected_images") != 30
        or reference_summary.get("forward_calls") != 30
        or reference_summary.get("one_forward_per_image") is not True
        or reference_trajectories <= 0
        or reference_summary.get("production_trajectories_in_subset") != reference_trajectories
        or reference_summary.get("mismatches") != 0
        or abs(tolerance - 0.001953126) > 1e-12
        or float(reference_summary.get("maximum_event_radius_absolute_difference", float("inf"))) > tolerance
    ):
        raise RuntimeError("formal independent boundary reference did not satisfy its exact qualification contract")
    reference_sources = reference_manifest.get("source_sha256")
    if not isinstance(reference_sources, dict):
        raise RuntimeError("formal boundary reference source hashes are missing")
    production_artifact = boundary_run / "artifact"
    if (
        reference_manifest.get("checkpoint_sha256") != reference_config.get("checkpoint_sha256")
        or reference_manifest.get("subset", {}).get("max_images_after_hash_ranking") != 30
        or reference_manifest.get("scan", {}).get("rmax") != 0.25
        or reference_manifest.get("scan", {}).get("coarse_step") != 0.015625
        or reference_manifest.get("scan", {}).get("fine_step") != 0.001953125
        or reference_sources.get("validator") != reference_config.get("script_sha256")
        or reference_sources.get("reference_assigner") != sha256(ROOT / "scripts" / "reference_o2o_assigner_numpy.py")
        or reference_sources.get("selected_images") != reference_config.get("selected_images_sha256")
        or reference_sources.get("production_manifest") != sha256(production_artifact / "manifest.json")
        or reference_sources.get("production_events") != sha256(production_artifact / "per_direction_boundary_events.csv")
    ):
        raise RuntimeError("formal independent boundary reference provenance differs from the frozen inputs")
    from create_run_output_inventory import validate as validate_output_inventory

    reference_inventory = read_json(
        reference_run / "artifact" / "OUTPUT_INVENTORY.json",
        "formal boundary reference output inventory",
    )
    inventory_validation = validate_output_inventory(reference_run / "artifact")
    if (
        reference_inventory.get("run_id") != "20260831_boundary_reference_formal_v2"
        or reference_inventory.get("generator_sha256") != sha256(ROOT / "scripts" / "create_run_output_inventory.py")
        or inventory_validation.get("status") != "PASS"
    ):
        raise RuntimeError("formal boundary reference output inventory is incomplete or stale")

    geometry_run = ROOT / "evidence" / "artifacts" / "runs" / "20260831_boundary_geometry_formal_v2"
    geometry_summary = read_json(
        geometry_run / "artifact" / "summary.json", "boundary geometry summary"
    )
    geometry_config = read_json(geometry_run / "CONFIG.json", "boundary geometry config")
    expected_geometry_outputs = {
        "survival_cif.csv",
        "rmsr.csv",
        "rmsr_contrasts.csv",
        "o2m_endpoint_rank_set.csv",
        "return_transition.csv",
        "rho_R_status.csv",
        "margin_construct.csv",
        "analysis_results.json",
    }
    output_hashes = geometry_summary.get("output_sha256")
    if (
        geometry_summary.get("bootstrap_reps") != 5000
        or geometry_summary.get("seed") != 20260831
        or geometry_summary.get("tau") != 0.125
        or geometry_summary.get("endpoint_radii") != [0.03125, 0.0625, 0.125]
        or geometry_summary.get("ci_level") != 0.95
        or geometry_summary.get("script_sha256") != geometry_config.get("script_sha256")
        or geometry_summary.get("event_csv_sha256") != sha256(production_artifact / "per_direction_boundary_events.csv")
        or geometry_summary.get("curve_csv_sha256") != sha256(production_artifact / "o2m_rank_set_radius_curve.csv")
        or set(geometry_summary.get("outputs", [])) != expected_geometry_outputs
        or not isinstance(output_hashes, dict)
        or set(output_hashes) != expected_geometry_outputs
    ):
        raise RuntimeError("boundary geometry summary differs from the qualified contract")
    for name, expected in output_hashes.items():
        path = geometry_run / "artifact" / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"boundary geometry output hash mismatch: {name}")


def validate_staged_source_inventory(output: Path, *, allow_finalization_files: bool = False) -> None:
    inventory_path = output / "SOURCE_INVENTORY.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    declared = read_json(inventory_path, "SOURCE_INVENTORY.json").get("files")
    if not isinstance(declared, list):
        raise RuntimeError("SOURCE_INVENTORY.json files must be a list")
    actual = inventory(output, {"SOURCE_INVENTORY.json"})
    if allow_finalization_files:
        actual_by_path = {row["path"]: row for row in actual}
        if any(actual_by_path.get(row.get("path")) != row for row in declared if isinstance(row, dict)):
            raise RuntimeError("one or more staged source files changed after the source-only commit")
    elif declared != actual:
        raise RuntimeError("staged source inventory no longer matches the staged files")


def validate_frozen_script_identity(
    output: Path,
    current_script: Path,
    *,
    use_tool_manifest: bool,
) -> str:
    relative = f"scripts/{current_script.name}"
    expected_path = output.resolve() / relative
    if current_script.resolve() != expected_path.resolve():
        raise RuntimeError(f"entrypoint must execute the frozen script: {relative}")
    if use_tool_manifest:
        manifest = read_json(output / "TOOL_FREEZE_MANIFEST.json", "TOOL_FREEZE_MANIFEST.json")
        rows = manifest.get("frozen_files")
    else:
        rows = read_json(output / "SOURCE_INVENTORY.json", "SOURCE_INVENTORY.json").get("files")
    if not isinstance(rows, list):
        raise RuntimeError("frozen script inventory is missing")
    hashes = {
        row.get("path"): row.get("sha256")
        for row in rows
        if isinstance(row, dict)
    }
    actual = sha256(current_script)
    if hashes.get(relative) != actual:
        raise RuntimeError(f"entrypoint hash differs from the frozen inventory: {relative}")
    return actual


def git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_source_commit_and_tag(
    output: Path,
    source_commit: str,
    *,
    require_clean: bool = True,
    require_head_match: bool = True,
) -> None:
    if git_text(output, "config", "--bool", "core.autocrlf").lower() != "false":
        raise RuntimeError("isolated freeze repository must set core.autocrlf=false")
    head = git_text(output, "rev-parse", "HEAD").lower()
    if require_head_match and head != source_commit:
        raise RuntimeError(f"source commit {source_commit} differs from isolated repository HEAD {head}")
    if not require_head_match:
        ancestry = subprocess.run(
            ["git", "-C", str(output), "merge-base", "--is-ancestor", source_commit, head],
            capture_output=True,
        )
        if ancestry.returncode != 0:
            raise RuntimeError("frozen source commit is not an ancestor of the current freeze HEAD")
    if require_clean and git_text(output, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("isolated freeze repository must be clean before finalize")
    tag_commit = git_text(output, "rev-list", "-n", "1", SOURCE_TAG).lower()
    if tag_commit != source_commit:
        raise RuntimeError(f"{SOURCE_TAG} does not resolve to the frozen source commit")


def validate_preregistration_seal(output: Path) -> dict[str, object]:
    validate_finalized_bundle(output)
    head = git_text(output, "rev-parse", "HEAD").lower()
    tag_commit = git_text(output, "rev-list", "-n", "1", PREREGISTRATION_TAG).lower()
    if tag_commit != head:
        raise RuntimeError(f"{PREREGISTRATION_TAG} does not resolve to the current freeze HEAD")
    if git_text(output, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("freeze repository must be clean after preregistration sealing")
    manifest = read_json(output / "TOOL_FREEZE_MANIFEST.json", "TOOL_FREEZE_MANIFEST.json")
    source_commit = str(manifest.get("instrument_source_commit", "")).lower()
    parent = git_text(output, "rev-parse", f"{head}^").lower()
    if parent != source_commit:
        raise RuntimeError("preregistration commit must be the direct child of the frozen source commit")
    finalization_files = {
        "CONFIRMATORY_AUDIT_PLAN.json",
        "CONFIRMATORY_AUDIT_PLAN.json.sha256",
        "CONFIRMATORY_AUDIT_PLAN.md",
        "TOOL_FREEZE_MANIFEST.json",
        "TOOL_FREEZE_MANIFEST.json.sha256",
    }
    changed = {
        line.strip().replace("\\", "/")
        for line in git_text(output, "diff", "--name-only", source_commit, head).splitlines()
        if line.strip()
    }
    if changed != finalization_files:
        raise RuntimeError(
            f"preregistration commit must change exactly the five finalization files; got {sorted(changed)}"
        )
    for name in sorted(finalization_files):
        current = output / name
        committed = subprocess.run(
            ["git", "-C", str(output), "show", f"{head}:{name}"],
            check=True,
            capture_output=True,
        ).stdout
        if not current.is_file() or hashlib.sha256(committed).hexdigest() != sha256(current):
            raise RuntimeError(f"committed preregistration artifact differs from the worktree: {name}")
    return {
        "status": "PASS",
        "preregistration_commit": head,
        "preregistration_tag": PREREGISTRATION_TAG,
    }


def seal_preregistration(output: Path) -> dict[str, object]:
    validate_finalized_bundle(output)
    expected_paths = {
        "CONFIRMATORY_AUDIT_PLAN.json",
        "CONFIRMATORY_AUDIT_PLAN.json.sha256",
        "CONFIRMATORY_AUDIT_PLAN.md",
        "TOOL_FREEZE_MANIFEST.json",
        "TOOL_FREEZE_MANIFEST.json.sha256",
    }
    status_lines = git_text(output, "status", "--porcelain", "--untracked-files=all").splitlines()
    changed_paths = {line[3:].strip().replace("\\", "/") for line in status_lines if len(line) >= 4}
    if changed_paths != expected_paths:
        raise RuntimeError(
            f"unexpected files before preregistration commit: expected {sorted(expected_paths)}, got {sorted(changed_paths)}"
        )
    existing_tag = subprocess.run(
        ["git", "-C", str(output), "rev-parse", "-q", "--verify", f"refs/tags/{PREREGISTRATION_TAG}"],
        capture_output=True,
        text=True,
    )
    if existing_tag.returncode == 0:
        raise RuntimeError(f"{PREREGISTRATION_TAG} already exists")
    subprocess.run(["git", "-C", str(output), "add", *sorted(expected_paths)], check=True)
    subprocess.run(
        ["git", "-C", str(output), "commit", "-m", "Freeze prospective audit preregistration v1.0"],
        check=True,
        capture_output=True,
    )
    commit = git_text(output, "rev-parse", "HEAD").lower()
    subprocess.run(["git", "-C", str(output), "tag", PREREGISTRATION_TAG, commit], check=True)
    return validate_preregistration_seal(output)


def validate_finalized_bundle(output: Path) -> dict[str, object]:
    manifest_path = output / "TOOL_FREEZE_MANIFEST.json"
    sidecar_path = output / "TOOL_FREEZE_MANIFEST.json.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("freeze manifest or sidecar is missing")
    actual_manifest_hash = sha256(manifest_path)
    tokens = sidecar_path.read_text(encoding="ascii").strip().split()
    if not tokens or tokens[0].lower() != actual_manifest_hash:
        raise RuntimeError("freeze manifest SHA-256 sidecar is malformed or mismatched")
    manifest = read_json(manifest_path, "TOOL_FREEZE_MANIFEST.json")
    if manifest.get("status") != "FROZEN_BEFORE_LOCKBOX_SELECTION":
        raise RuntimeError("freeze manifest status is not FROZEN_BEFORE_LOCKBOX_SELECTION")
    source_commit = manifest.get("instrument_source_commit")
    if not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise RuntimeError("freeze manifest instrument_source_commit is malformed")
    validate_source_commit_and_tag(
        output,
        source_commit.lower(),
        require_clean=False,
        require_head_match=False,
    )
    declared = manifest.get("frozen_files")
    if not isinstance(declared, list):
        raise RuntimeError("freeze manifest frozen_files must be a list")
    actual = inventory(
        output,
        {"TOOL_FREEZE_MANIFEST.json", "TOOL_FREEZE_MANIFEST.json.sha256"},
    )
    if declared != actual:
        raise RuntimeError("freeze manifest frozen_files inventory is stale or tampered")
    validate_staged_source_inventory(output, allow_finalization_files=True)
    return {
        "status": "PASS",
        "instrument_manifest_sha256": actual_manifest_hash,
        "frozen_files": len(actual),
    }


def run_self_test() -> dict[str, object]:
    checks: dict[str, bool] = {}
    checks["bytecode_writes_disabled"] = sys.dont_write_bytecode is True
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "freeze"
        output.mkdir()
        source = output / "source.txt"
        source.write_text("frozen source\n", encoding="utf-8")
        frozen_discovery_exclusion(output).parent.mkdir(parents=True)
        frozen_discovery_exclusion(output).write_text(
            "image_id\nsynthetic.png\n", encoding="utf-8", newline="\n"
        )
        checks["frozen_discovery_exclusion_resolves_inside_bundle"] = (
            frozen_discovery_exclusion(output).resolve().is_relative_to(output.resolve())
            and sha256(frozen_discovery_exclusion(output))
            == hashlib.sha256(b"image_id\nsynthetic.png\n").hexdigest()
        )
        write_json(
            output / "SOURCE_INVENTORY.json",
            {"files": inventory(output, {"SOURCE_INVENTORY.json"})},
        )
        validate_staged_source_inventory(output)
        checks["staged_inventory_accepts_exact_files"] = True

        subprocess.run(["git", "init", str(output)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(output), "config", "core.autocrlf", "false"], check=True)
        subprocess.run(["git", "-C", str(output), "config", "user.email", "selftest@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(output), "config", "user.name", "freeze-selftest"], check=True)
        subprocess.run(["git", "-C", str(output), "add", "."], check=True)
        subprocess.run(["git", "-C", str(output), "commit", "-m", "frozen source"], check=True, capture_output=True)
        commit = git_text(output, "rev-parse", "HEAD").lower()
        subprocess.run(["git", "-C", str(output), "tag", SOURCE_TAG, commit], check=True)
        validate_source_commit_and_tag(output, commit)
        checks["commit_and_tag_accept_exact_head"] = True
        try:
            validate_source_commit_and_tag(output, "0" * 40)
        except RuntimeError:
            checks["wrong_commit_rejected"] = True
        else:
            raise AssertionError("wrong source commit was accepted")

        preregistration = output / "CONFIRMATORY_AUDIT_PLAN.json"
        write_json(preregistration, {"status": "FROZEN_BEFORE_OUTCOMES"})
        (output / "CONFIRMATORY_AUDIT_PLAN.json.sha256").write_text(
            f"{sha256(preregistration)}  {preregistration.name}\n",
            encoding="ascii",
            newline="\n",
        )
        (output / "CONFIRMATORY_AUDIT_PLAN.md").write_text(
            "# Synthetic preregistration\n", encoding="utf-8", newline="\n"
        )
        manifest_path = output / "TOOL_FREEZE_MANIFEST.json"
        sidecar_path = output / "TOOL_FREEZE_MANIFEST.json.sha256"
        manifest = {
            "status": "FROZEN_BEFORE_LOCKBOX_SELECTION",
            "instrument_source_commit": commit,
            "frozen_files": inventory(
                output,
                {"TOOL_FREEZE_MANIFEST.json", "TOOL_FREEZE_MANIFEST.json.sha256"},
            ),
        }
        write_json(manifest_path, manifest)
        sidecar_path.write_text(
            f"{sha256(manifest_path)}  {manifest_path.name}\n", encoding="ascii", newline="\n"
        )
        validate_finalized_bundle(output)
        checks["finalized_inventory_accepts_exact_bundle"] = True
        seal_result = seal_preregistration(output)
        assert seal_result["preregistration_tag"] == PREREGISTRATION_TAG
        checks["preregistration_commit_and_tag_created"] = True
        validate_preregistration_seal(output)
        checks["preregistration_seal_revalidates"] = True

        source.write_text("tampered\n", encoding="utf-8")
        try:
            validate_preregistration_seal(output)
        except RuntimeError:
            checks["frozen_source_tamper_rejected"] = True
        else:
            raise AssertionError("tampered frozen source was accepted")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "tests": len(checks),
        "checks": checks,
    }


def stage(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    from validate_current_evidence_ledger import validate as validate_ledger
    from prepare_disjoint_audit_lockbox import (
        build_eligible_population_contract,
        canonical_eligible_population_bytes,
    )

    ledger_path = ROOT / "evidence_ledger" / "CURRENT_EVIDENCE_LEDGER.json"
    ledger_result = validate_ledger(ledger_path, ROOT)
    if ledger_result["status"] != "PASS":
        raise RuntimeError(f"current evidence ledger is not freeze-ready: {ledger_result['errors']}")
    ledger = read_json(ledger_path, "CURRENT_EVIDENCE_LEDGER.json")
    pending = set(ledger.get("pending", []))
    unresolved_core = pending.intersection(
        {"boundary_radius_and_geometry", "independent_boundary_reference"}
    )
    if unresolved_core:
        raise RuntimeError(f"core evidence ledger gates remain pending: {sorted(unresolved_core)}")
    validate_qualification_inputs()
    (output / "scripts").mkdir(parents=True)
    (output / "protocols").mkdir()
    (output / "configs").mkdir()
    (output / "validation_evidence").mkdir()
    (output / "frozen_inputs").mkdir()
    (output / "governance").mkdir()
    (output / "evidence_ledger").mkdir()

    for name in CORE_SCRIPTS:
        source = ROOT / "scripts" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / "scripts" / name)

    for name in CORE_PROTOCOLS:
        source = ROOT / "protocols" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / "protocols" / name)

    for name in GOVERNANCE_FILES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / "governance" / name)

    for name in LEDGER_FILES:
        source = ROOT / "evidence_ledger" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / "evidence_ledger" / name)

    local_package = ROOT / "ultralytics_local" / "ultralytics"
    shutil.copytree(
        local_package,
        output / "ultralytics_local" / "ultralytics",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(DATA_YAML, output / "configs" / DATA_YAML.name)
    shutil.copy2(DISCOVERY_SELECTION, output / "frozen_inputs" / "discovery_selected_images_exclusion.csv")
    eligible_rows, eligible_contract = build_eligible_population_contract(DATA_YAML, imgsz=800)
    (output / "frozen_inputs" / "eligible_population_roster.json").write_bytes(
        canonical_eligible_population_bytes(eligible_rows)
    )
    write_json(
        output / "frozen_inputs" / "eligible_population_roster_contract.json",
        eligible_contract,
    )
    for target_name, source in VALIDATION_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output / "validation_evidence" / target_name)

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout
    (output / "requirements_frozen.txt").write_text(pip_freeze, encoding="utf-8", newline="\n")
    (output / "README.md").write_text(
        "# Audit instrument freeze v1.0\n\n"
        "This isolated snapshot contains the exact source and validation evidence frozen "
        "before the prospective audit-outcome lockbox was selected or executed.\n\n"
        "The manuscript and publication figures are outside this snapshot and remained frozen "
        "during the experimental window.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_json(output / "SOURCE_INVENTORY.json", {"files": inventory(output, {"SOURCE_INVENTORY.json"})})


def environment_payload() -> dict[str, object]:
    local_package_root = ROOT / "ultralytics_local"
    if str(local_package_root) not in sys.path:
        sys.path.insert(0, str(local_package_root))
    import torch
    import ultralytics

    return {
        "python": sys.version.replace("\n", " "),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def finalize(output: Path, source_commit: str, selection_seed: int) -> None:
    if not (output / ".git").is_dir():
        raise RuntimeError("freeze directory is not an initialized Git repository")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit.lower()):
        raise ValueError("source_commit must be a full 40-character Git hash")
    source_commit = source_commit.lower()
    validate_staged_source_inventory(output)
    validate_source_commit_and_tag(output, source_commit)
    validate_frozen_script_identity(
        output,
        Path(__file__),
        use_tool_manifest=False,
    )
    eligible_population_contract = read_json(
        output / "frozen_inputs" / "eligible_population_roster_contract.json",
        "eligible population roster contract",
    )
    environment = environment_payload()
    if environment["cuda_available"] is not True or environment["torch_default_dtype"] != "torch.float32":
        raise RuntimeError("prospective extraction requires CUDA availability and torch.float32 default dtype")

    plan = {
        "status": "FROZEN_BEFORE_OUTCOMES",
        "protocol": "prospective_disjoint_audit_outcome_lockbox_H1_H5_v2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "instrument_source_commit": source_commit,
        "instrument_source_tag": "audit_instrument_v1.0",
        "population": "AI-TOD-v2 validation images eligible for the pre-existing t_only/s_only/t_and_s audit strata",
        "sample": {
            "name": "disjoint prospective audit-outcome lockbox",
            "size_images": 300,
            "selection_seed": selection_seed,
            "selection_is_outcome_blind": True,
            "disjoint_from_discovery_manifest": True,
            "allocation": {"t_only": 160, "s_only": 2, "t_and_s": 138},
        },
        "frozen_checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": sha256(CHECKPOINT),
        },
        "stress_contracts": {
            "fixed": "one-pixel centre shift in each of four legal cardinal directions",
            "normalized": "equivalent-side centre shift with kappa=0.0625 in each legal cardinal direction",
        },
        "stress_contract_parameters": {
            "imgsz": 800,
            "detector_contract": "yolo26",
            "fixed": {
                "branch_stress_mode": "fixed_1px",
                "anatomy_stress": "fixed_1px",
                "pixel_shift": 1.0
            },
            "normalized": {
                "branch_stress_mode": "normalized",
                "anatomy_stress": "equivalent_side",
                "normalization": "equivalent_side",
                "kappa": 0.0625
            }
        },
        "statistics": {
            "unit": "ground-truth object for branch fragility; directional identity exchange for pathway composition",
            "branch_estimand_domain": "fixed and normalized rows with common_valid_margin == 1 on exactly equal GT support; selected images with zero common-valid contribution remain zero-valued bootstrap clusters",
            "pathway_estimand_domain": "all selected image clusters; pathway numerators and denominators use directional o2o_flip == 1 rows, while images with zero flips remain zero-valued bootstrap clusters",
            "weighting": "inverse inclusion weights from the frozen stratified image selection",
            "uncertainty": "5000 image-cluster bootstrap replicates within sampling stratum",
            "bootstrap_replicates": 5000,
            "confidence_level": 0.95,
            "confirmatory_seed": 20260831,
            "decision_rule": "one-sided empirical bootstrap p values at alpha=0.05; all outcomes reported regardless of verdict",
        },
        "runtime_contract": {
            "device_argument": "0",
            "device_type": "cuda",
            "device_index": 0,
            "workers": 0,
            "imgsz": 800,
            "torch_default_dtype": "torch.float32",
            "environment": environment,
        },
        "confirmatory_families": {
            "tier_a": {
                "procedure": "hierarchical_gatekeeping",
                "bootstrap_test": "one_sided_empirical_bootstrap_p",
                "alpha": 0.05,
                "order": ["H1", "H2", "H3"],
                "hypotheses": {
                    "H1": {
                        "id": "H1",
                        "metric": "normalized_paired_gap",
                        "direction": "greater_than_zero",
                        "estimand": "normalized O2O scale gap minus normalized O2M scale gap",
                    },
                    "H2": {
                        "id": "H2",
                        "metric": "normalized_o2m_gap",
                        "direction": "less_than_zero",
                        "estimand": "normalized 8-16 px minus 16-32 px O2M fragility gap",
                    },
                    "H3": {
                        "id": "H3",
                        "metric": "fixed_minus_normalized_o2o_gap",
                        "direction": "greater_than_zero",
                        "estimand": "fixed O2O scale gap minus normalized O2O scale gap on the same lockbox images",
                    },
                },
            },
            "tier_b": {
                "procedure": "holm",
                "bootstrap_test": "one_sided_empirical_bootstrap_p",
                "alpha": 0.05,
                "order": ["H4", "H5"],
                "hypotheses": {
                    "H4": {
                        "id": "H4",
                        "metric": "fixed_eligibility_minus_normalized",
                        "direction": "greater_than_zero",
                        "estimand": "fixed minus normalized eligibility first-divergence fraction",
                    },
                    "H5": {
                        "id": "H5",
                        "metric": "normalized_within_minus_fixed",
                        "direction": "greater_than_zero",
                        "estimand": "normalized minus fixed within-set geometry first-divergence fraction",
                    },
                },
            },
        },
        "one_time_rule": "The selected 300-image lockbox is executed once. No replacement sample, threshold change, or primary re-analysis is permitted after outcomes are opened.",
    }
    prereg = output / "CONFIRMATORY_AUDIT_PLAN.json"
    write_json(prereg, plan)
    prereg_hash = sha256(prereg)
    (output / "CONFIRMATORY_AUDIT_PLAN.json.sha256").write_text(prereg_hash + "  CONFIRMATORY_AUDIT_PLAN.json\n", encoding="ascii", newline="\n")

    markdown = "# Prospective confirmatory audit plan\n\n"
    markdown += f"Status: **FROZEN_BEFORE_OUTCOMES**  \nInstrument source commit: `{source_commit}`  \nPreregistration SHA-256: `{prereg_hash}`\n\n"
    markdown += "The JSON file is normative. It freezes a single outcome-blind, disjoint 300-image audit sample, the fixed and normalized replay contracts, IPW plus 5,000 stratified image-cluster bootstrap replicates, H1--H3 hierarchical gatekeeping, and Holm correction for H4--H5. All primary verdicts will be reported regardless of direction. No secondary confirmatory family is generated automatically.\n"
    (output / "CONFIRMATORY_AUDIT_PLAN.md").write_text(markdown, encoding="utf-8", newline="\n")

    frozen_oracle = read_json(output / "validation_evidence" / "oracle_validation.json", "frozen oracle validation")
    frozen_mutation = read_json(output / "validation_evidence" / "mutation_validation.json", "frozen mutation validation")
    frozen_parity = read_json(
        output / "validation_evidence" / "real_parity_deterministic_5k_summary.json",
        "frozen deterministic parity summary",
    )
    frozen_reference = read_json(
        output / "validation_evidence" / "boundary_reference_formal_summary.json",
        "frozen boundary reference summary",
    )
    frozen_geometry = read_json(
        output / "validation_evidence" / "boundary_geometry_summary.json",
        "frozen boundary geometry summary",
    )

    manifest = {
        "status": "FROZEN_BEFORE_LOCKBOX_SELECTION",
        "protocol": "audit_instrument_v1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "instrument_source_commit": source_commit,
        "instrument_source_tag": "audit_instrument_v1.0",
        "environment": environment,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256(CHECKPOINT)},
        "data_yaml": {"path": f"configs/{DATA_YAML.name}", "sha256": sha256(DATA_YAML)},
        "eligible_population_roster": eligible_population_contract,
        "assignment_code_provenance": {
            "extractor_import_contract": "frozen scripts prepend frozen ultralytics_local before importing assignment code",
            "frozen_local_tal": {"path": str(LOCAL_TAL), "sha256": sha256(LOCAL_TAL)},
            "external_environment_tal": {"path": str(EXTERNAL_TAL), "sha256": sha256(EXTERNAL_TAL)},
            "roles_are_distinct": True,
        },
        "discovery_exclusion_manifest_sha256": sha256(frozen_discovery_exclusion(output)),
        "selection_seed": selection_seed,
        "preregistration_sha256": prereg_hash,
        "qualification": {
            "oracle_cases": {
                "passed": sum(case.get("pass") is True for case in frozen_oracle["oracle_cases"]),
                "total": len(frozen_oracle["oracle_cases"]),
            },
            "property_invariants": {
                "passed": sum(value is True for value in frozen_oracle["properties"].values()),
                "total": len(frozen_oracle["properties"]),
            },
            "mutations": {
                "killed": frozen_mutation["killed"],
                "total": frozen_mutation["total"],
            },
            "real_three_way_stage_parity": {
                "base_states": frozen_parity["base_states"],
                "direction_states": frozen_parity["sampled_direction_states"],
                "mismatches": frozen_parity["mismatches"],
            },
            "formal_boundary_reference": {
                "selected_images": frozen_reference["selected_images"],
                "trajectories": frozen_reference["reference_trajectories"],
                "mismatches": frozen_reference["mismatches"],
                "maximum_radius_difference": frozen_reference["maximum_event_radius_absolute_difference"],
            },
            "boundary_geometry": {
                "bootstrap_replicates": frozen_geometry["bootstrap_reps"],
                "tau": frozen_geometry["tau"],
                "endpoint_radii": frozen_geometry["endpoint_radii"],
            },
        },
        "frozen_files": inventory(
            output,
            {"TOOL_FREEZE_MANIFEST.json", "TOOL_FREEZE_MANIFEST.json.sha256"},
        ),
    }
    freeze_manifest = output / "TOOL_FREEZE_MANIFEST.json"
    write_json(freeze_manifest, manifest)
    (output / "TOOL_FREEZE_MANIFEST.json.sha256").write_text(
        sha256(freeze_manifest) + "  TOOL_FREEZE_MANIFEST.json\n", encoding="ascii", newline="\n"
    )
    validate_finalized_bundle(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.add_argument("--source-commit", required=True)
    finalize_parser.add_argument("--selection-seed", type=int, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal-preregistration")
    seal_parser.add_argument("--output", type=Path, required=True)
    validate_seal_parser = subparsers.add_parser("validate-preregistration-seal")
    validate_seal_parser.add_argument("--output", type=Path, required=True)
    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.output)
    elif args.command == "finalize":
        finalize(args.output, args.source_commit.lower(), args.selection_seed)
    elif args.command == "validate":
        print(json.dumps(validate_finalized_bundle(args.output), indent=2, sort_keys=True))
    elif args.command == "seal-preregistration":
        print(json.dumps(seal_preregistration(args.output), indent=2, sort_keys=True))
    elif args.command == "validate-preregistration-seal":
        print(json.dumps(validate_preregistration_seal(args.output), indent=2, sort_keys=True))
    else:
        result = run_self_test()
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
