"""Run a fully synthetic, file-backed rehearsal of the lockbox analyzer CLI.

The rehearsal builds isolated Git-sealed instrument and selection repositories,
four sealed raw-extraction directories, and a readiness artifact.  It then runs
the repaired analyzer through its real command-line entry point.  No project
dataset, checkpoint, lockbox-v1 output, or outcome-access receipt is read or
modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


sys.dont_write_bytecode = True

SCRIPT_ROOT = Path(__file__).resolve().parent
REPS = 99
SEED = 7
FROZEN_AT = "2026-01-01T00:00:00+00:00"

sys.path.insert(0, str(SCRIPT_ROOT))
import create_instrument_freeze_bundle as freeze_builder  # noqa: E402
import prepare_disjoint_audit_lockbox as selector  # noqa: E402
import validate_sealed_lockbox_bundle as sealed_validator  # noqa: E402


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


def write_with_sidecar(path: Path, payload: object) -> str:
    write_json(path, payload)
    digest = sha256(path)
    Path(str(path) + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*arguments: str, cwd: Path | None = None) -> str:
    command = ["git"]
    if cwd is not None:
        command.extend(["-C", str(cwd)])
    command.extend(arguments)
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def selected_rows() -> list[dict]:
    rows = []
    for index in range(300):
        if index == 299:
            stratum = "empty_stratum"
        else:
            stratum = ("t_only", "s_only", "t_and_s")[index % 3]
        rows.append(
            {
                "dataset_index": index,
                "image_id": f"image_{index:03d}",
                "stratum": stratum,
            }
        )
    return rows


def allocation(rows: list[dict]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        result[row["stratum"]] = result.get(row["stratum"], 0) + 1
    return result


def branch_rows(contract: str) -> list[dict]:
    rows: list[dict] = []
    missing_index = 0 if contract == "fixed" else 1
    for index in range(299):
        if index == missing_index:
            continue
        image_id = f"image_{index:03d}"
        stratum = ("t_only", "s_only", "t_and_s")[index % 3]
        all_zero = index == 2
        for gt_id, size_bin in ((0, "t_8_16"), (1, "s_16_32")):
            if all_zero:
                o2o_fragile = 0
                o2m_fragile = 0
            elif contract == "fixed":
                o2o_fragile = int(size_bin == "t_8_16")
                o2m_fragile = 0
            else:
                o2o_fragile = int(size_bin == "t_8_16" and index % 2 == 0)
                o2m_fragile = int(size_bin == "s_16_32")
            rows.append(
                {
                    "stratum": stratum,
                    "image_id": image_id,
                    "gt_id": gt_id,
                    "common_valid_margin": 1,
                    "size_bin": size_bin,
                    "sampling_weight": 1.0,
                    "o2o_fragile": o2o_fragile,
                    "o2m_fragile": o2m_fragile,
                }
            )
    extra_index = 2 if contract == "fixed" else 3
    rows.append(
        {
            "stratum": ("t_only", "s_only", "t_and_s")[extra_index % 3],
            "image_id": f"image_{extra_index:03d}",
            "gt_id": 90 if contract == "fixed" else 91,
            "common_valid_margin": 1,
            "size_bin": "t_8_16",
            "sampling_weight": 1.0,
            "o2o_fragile": 0,
            "o2m_fragile": 0,
        }
    )
    return rows


def anatomy_rows(contract: str) -> list[dict]:
    rows = []
    missing_index = 0 if contract == "fixed" else 1
    for index in range(299):
        if index == missing_index:
            continue
        rows.append(
            {
                "stratum": ("t_only", "s_only", "t_and_s")[index % 3],
                "image_id": f"image_{index:03d}",
                "sampling_weight": 1.0,
                "o2o_flip": 0 if index == 2 else 1,
                "first_divergence": (
                    "eligibility_boundary"
                    if contract == "fixed"
                    else "within_set_geometry_rank_reversal"
                ),
            }
        )
    return rows


def copy_frozen_scripts(freeze: Path) -> None:
    scripts = freeze / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "analyze_confirmatory_lockbox.py",
        "validate_sealed_lockbox_bundle.py",
        "create_instrument_freeze_bundle.py",
        "prepare_disjoint_audit_lockbox.py",
    ):
        shutil.copy2(SCRIPT_ROOT / name, scripts / name)
    (scripts / "run_reviewer_killer_controls.py").write_text(
        "def image_stratum(*args, **kwargs):\n    return 't_only'\n",
        encoding="utf-8",
        newline="\n",
    )
    (scripts / "analyze_native_alignment_path_anatomy.py").write_text(
        "# synthetic extractor identity\n", encoding="utf-8", newline="\n"
    )


def build_scenario(base: Path, *, reverse: bool, duplicate: bool) -> dict[str, Path]:
    freeze = base / "freeze"
    frozen_inputs = freeze / "frozen_inputs"
    frozen_inputs.mkdir(parents=True)
    copy_frozen_scripts(freeze)

    exclusion = frozen_inputs / "discovery_selected_images_exclusion.csv"
    write_csv(
        exclusion,
        [{"dataset_index": 999, "image_id": "discovery_only", "stratum": "t_only"}],
        ["dataset_index", "image_id", "stratum"],
    )
    exclusion_hash = sha256(exclusion)

    source_inventory = freeze / "SOURCE_INVENTORY.json"
    write_json(
        source_inventory,
        {"files": freeze_builder.inventory(freeze, {source_inventory.name})},
    )
    git("init", str(freeze))
    git("config", "core.autocrlf", "false", cwd=freeze)
    git("config", "user.email", "synthetic-e2e@example.invalid", cwd=freeze)
    git("config", "user.name", "synthetic lockbox e2e", cwd=freeze)
    git("add", ".", cwd=freeze)
    git("commit", "-m", "freeze synthetic analyzer source", cwd=freeze)
    source_commit = git("rev-parse", "HEAD", cwd=freeze).lower()
    git("tag", freeze_builder.SOURCE_TAG, source_commit, cwd=freeze)

    rows = selected_rows()
    frozen_allocation = allocation(rows)
    runtime_contract = {
        "device_argument": "cpu",
        "workers": 0,
        "imgsz": 800,
        "torch_default_dtype": "torch.float32",
        "environment": {"synthetic": True},
    }
    hypotheses = {
        key: {
            "metric": value["metric"],
            "direction": value["direction"],
            "estimand": f"synthetic {key}",
        }
        for key, value in {
            "H1": {"metric": "normalized_paired_gap", "direction": "greater_than_zero"},
            "H2": {"metric": "normalized_o2m_gap", "direction": "less_than_zero"},
            "H3": {"metric": "fixed_minus_normalized_o2o_gap", "direction": "greater_than_zero"},
            "H4": {"metric": "fixed_eligibility_minus_normalized", "direction": "greater_than_zero"},
            "H5": {"metric": "normalized_within_minus_fixed", "direction": "greater_than_zero"},
        }.items()
    }
    preregistration = freeze / "CONFIRMATORY_AUDIT_PLAN.json"
    preregistration_payload = {
        "status": "FROZEN_BEFORE_OUTCOMES",
        "frozen_at_utc": FROZEN_AT,
        "instrument_source_commit": source_commit,
        "instrument_source_tag": freeze_builder.SOURCE_TAG,
        "sample": {
            "size_images": len(rows),
            "selection_seed": SEED,
            "allocation": frozen_allocation,
        },
        "runtime_contract": runtime_contract,
        "statistics": {"bootstrap_replicates": REPS, "confirmatory_seed": SEED},
        "confirmatory_families": {
            "tier_a": {
                "procedure": "hierarchical_gatekeeping",
                "bootstrap_test": "one_sided_empirical_bootstrap_p",
                "alpha": 0.05,
                "order": ["H1", "H2", "H3"],
                "hypotheses": {key: hypotheses[key] for key in ("H1", "H2", "H3")},
            },
            "tier_b": {
                "procedure": "holm",
                "bootstrap_test": "one_sided_empirical_bootstrap_p",
                "alpha": 0.05,
                "order": ["H4", "H5"],
                "hypotheses": {key: hypotheses[key] for key in ("H4", "H5")},
            },
        },
        "secondary_confirmatory_family": {
            "procedure": "holm",
            "bootstrap_test": "one_sided_empirical_bootstrap_p",
            "alpha": 0.05,
            "hypotheses": [],
        },
    }
    preregistration_hash = write_with_sidecar(preregistration, preregistration_payload)
    (freeze / "CONFIRMATORY_AUDIT_PLAN.md").write_text(
        "# Synthetic CLI rehearsal only\n", encoding="utf-8", newline="\n"
    )

    checkpoint_hash = "2" * 64
    data_hash = "3" * 64
    roster_contract = {
        "protocol": "eligible_population_roster_v1",
        "sha256": "6" * 64,
        "count": len(rows) + 1,
        "strata_counts": {**frozen_allocation, "discovery_only": 1},
    }
    instrument = freeze / "TOOL_FREEZE_MANIFEST.json"
    instrument_sidecar = Path(str(instrument) + ".sha256")
    instrument_payload = {
        "status": "FROZEN_BEFORE_LOCKBOX_SELECTION",
        "instrument_source_commit": source_commit,
        "instrument_source_tag": freeze_builder.SOURCE_TAG,
        "preregistration_sha256": preregistration_hash,
        "checkpoint": {"sha256": checkpoint_hash},
        "data_yaml": {"sha256": data_hash},
        "eligible_population_roster": roster_contract,
        "discovery_exclusion_manifest_sha256": exclusion_hash,
        "frozen_files": freeze_builder.inventory(
            freeze, {instrument.name, instrument_sidecar.name}
        ),
    }
    instrument_hash = write_with_sidecar(instrument, instrument_payload)
    preregistration_seal = freeze_builder.seal_preregistration(freeze)

    selection = base / "selection"
    selection.mkdir()
    selected = selection / "selected_images.csv"
    write_csv(selected, rows, ["dataset_index", "image_id", "stratum"])
    selected_hash = sha256(selected)
    selection_manifest = selection / "selection_manifest.json"
    selection_manifest_payload = {
        "status": "FROZEN_SELECTION_BEFORE_OUTCOMES",
        "protocol": "disjoint_audit_outcome_lockbox_selection_v1",
        "outcome_blind": True,
        "outcome_accessed": False,
        "seed": SEED,
        "requested_images": len(rows),
        "allocation": frozen_allocation,
        "disjoint_overlap": 0,
        "instrument_source_commit": source_commit,
        "instrument_source_tag": freeze_builder.SOURCE_TAG,
        "preregistration_sha256": preregistration_hash,
        "instrument_manifest_sha256": instrument_hash,
        "preregistration_commit": preregistration_seal["preregistration_commit"],
        "preregistration_tag": preregistration_seal["preregistration_tag"],
        "source_sha256": {
            "dataset_config": data_hash,
            "exclusion_manifest": exclusion_hash,
            "selection_script": sha256(freeze / "scripts" / "prepare_disjoint_audit_lockbox.py"),
        },
        "eligible_population_roster": roster_contract,
        "selected_images_sha256": selected_hash,
    }
    selection_manifest_hash = write_with_sidecar(
        selection_manifest, selection_manifest_payload
    )
    selector.seal_selection_repository(selection)

    branch_fields = [
        "stratum",
        "image_id",
        "gt_id",
        "common_valid_margin",
        "size_bin",
        "sampling_weight",
        "o2o_fragile",
        "o2m_fragile",
    ]
    anatomy_fields = [
        "stratum",
        "image_id",
        "sampling_weight",
        "o2o_flip",
        "first_divergence",
    ]
    directories: dict[str, Path] = {}
    for role, outputs in sealed_validator.EXPECTED_OUTPUTS.items():
        directory = base / role
        directory.mkdir()
        directories[role] = directory
        shutil.copy2(selected, directory / "selected_images.csv")
        if role.endswith("branch"):
            contract = "fixed" if role.startswith("fixed") else "normalized"
            per_gt = branch_rows(contract)
            if duplicate and role == "fixed_branch":
                per_gt.append(dict(per_gt[0]))
            if reverse:
                per_gt.reverse()
            write_csv(directory / "per_gt.csv", per_gt, branch_fields)
            write_csv(
                directory / "per_direction.csv",
                [{"image_id": "synthetic_raw_not_summarized"}],
                ["image_id"],
            )
        else:
            contract = "fixed" if role.startswith("fixed") else "normalized"
            per_direction = anatomy_rows(contract)
            if reverse:
                per_direction.reverse()
            write_csv(
                directory / "per_direction_anatomy.csv",
                per_direction,
                anatomy_fields,
            )
            write_csv(
                directory / "pairwise_q_gap.csv",
                [{"image_id": "synthetic_raw_not_summarized"}],
                ["image_id"],
            )
        extractor = (
            freeze / "scripts" / "run_reviewer_killer_controls.py"
            if role.endswith("branch")
            else freeze / "scripts" / "analyze_native_alignment_path_anatomy.py"
        )
        manifest = {
            "status": "sealed_extraction_complete",
            "sealed_outcomes_not_summarized": True,
            "outcome_access_permitted": False,
            "selected_images": len(rows),
            "outputs": list(outputs),
            "artifact_sha256": {
                name: sha256(directory / name) for name in outputs
            },
            "frozen_contract": {
                "role": role,
                "instrument_manifest_sha256": instrument_hash,
                "preregistration_sha256": preregistration_hash,
                "selection_manifest_sha256": selection_manifest_hash,
                "selected_images_sha256": selected_hash,
                "checkpoint_sha256": checkpoint_hash,
                "data_yaml_sha256": data_hash,
                "instrument_source_commit": source_commit,
                "instrument_source_tag": freeze_builder.SOURCE_TAG,
                "extractor_sha256": sha256(extractor),
                "runtime_contract": runtime_contract,
            },
        }
        if role.endswith("branch"):
            manifest["stress_mode"] = "fixed_1px" if role.startswith("fixed") else "normalized"
            if role.startswith("normalized"):
                manifest["normalization"] = "equivalent_side"
        else:
            manifest["stress"] = "fixed_1px" if role.startswith("fixed") else "equivalent_side"
        write_json(directory / "manifest.json", manifest)

    ready = base / "SEALED_BUNDLE_READY.json"
    sealed_validator.validate_and_write(
        fixed_branch=directories["fixed_branch"],
        normalized_branch=directories["normalized_branch"],
        fixed_anatomy=directories["fixed_anatomy"],
        normalized_anatomy=directories["normalized_anatomy"],
        selected_manifest_path=selection_manifest,
        preregistration_path=preregistration,
        instrument_manifest_path=instrument,
        output=ready,
    )
    return {
        **directories,
        "freeze": freeze,
        "preregistration": preregistration,
        "selection_manifest": selection_manifest,
        "instrument": instrument,
        "ready": ready,
        "output": base / "formal_output",
    }


def run_cli(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    analyzer = paths["freeze"] / "scripts" / "analyze_confirmatory_lockbox.py"
    command = [
        sys.executable,
        str(analyzer),
        "--fixed-dir",
        str(paths["fixed_branch"]),
        "--normalized-dir",
        str(paths["normalized_branch"]),
        "--fixed-anatomy",
        str(paths["fixed_anatomy"]),
        "--normalized-anatomy",
        str(paths["normalized_anatomy"]),
        "--preregistration",
        str(paths["preregistration"]),
        "--selected-manifest",
        str(paths["selection_manifest"]),
        "--instrument-manifest",
        str(paths["instrument"]),
        "--sealed-ready",
        str(paths["ready"]),
        "--output-dir",
        str(paths["output"]),
        "--reps",
        str(REPS),
        "--seed",
        str(SEED),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command,
        cwd=paths["freeze"] / "scripts",
        env=environment,
        capture_output=True,
        text=True,
    )


def scientific_payload(verdict: dict) -> dict:
    return {
        "tier_a": verdict["tier_a"],
        "tier_b": verdict["tier_b"],
        "branch_support_audit": verdict["branch_support_audit"],
        "selected_image_clusters": verdict["selected_image_clusters"],
        "bootstrap_replicates": verdict["bootstrap_replicates"],
        "seed": verdict["seed"],
    }


def validate_output_bundle(paths: dict[str, Path]) -> None:
    output = paths["output"]
    required = {
        "LOCKBOX_VERDICT.json",
        "LOCKBOX_REPORT.md",
        "confirmatory_results.json",
        "SHA256SUMS.txt",
    }
    actual = {path.name for path in output.iterdir() if path.is_file()}
    if actual != required:
        raise AssertionError(f"unexpected analyzer output bundle: {sorted(actual)}")
    for line in (output / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split(maxsplit=1)
        if sha256(output / name) != expected:
            raise AssertionError(f"analyzer output hash mismatch: {name}")


def run_rehearsal(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with tempfile.TemporaryDirectory(prefix="lockbox_cli_e2e_") as temporary:
        base = Path(temporary)
        normal_paths = build_scenario(base / "normal", reverse=False, duplicate=False)
        reverse_paths = build_scenario(base / "reverse", reverse=True, duplicate=False)
        duplicate_paths = build_scenario(base / "duplicate", reverse=False, duplicate=True)

        normal_run = run_cli(normal_paths)
        reverse_run = run_cli(reverse_paths)
        duplicate_run = run_cli(duplicate_paths)
        if normal_run.returncode != 0:
            raise RuntimeError(f"normal CLI rehearsal failed:\n{normal_run.stderr}")
        if reverse_run.returncode != 0:
            raise RuntimeError(f"reverse-order CLI rehearsal failed:\n{reverse_run.stderr}")
        if duplicate_run.returncode == 0 or "duplicate" not in duplicate_run.stderr.lower():
            raise RuntimeError("duplicate-key CLI scenario did not fail closed as expected")
        validate_output_bundle(normal_paths)
        validate_output_bundle(reverse_paths)
        for paths in (normal_paths, reverse_paths, duplicate_paths):
            receipt = paths["ready"].parent / "OUTCOME_ACCESS_RECEIPT.json"
            if not receipt.is_file() or not Path(str(receipt) + ".sha256").is_file():
                raise AssertionError("case-local outcome-access receipt was not created")
        if duplicate_paths["output"].exists():
            raise AssertionError("duplicate-key scenario wrote an output bundle after failure")

        normal_verdict_path = normal_paths["output"] / "LOCKBOX_VERDICT.json"
        reverse_verdict_path = reverse_paths["output"] / "LOCKBOX_VERDICT.json"
        normal_verdict = json.loads(normal_verdict_path.read_text(encoding="utf-8"))
        reverse_verdict = json.loads(reverse_verdict_path.read_text(encoding="utf-8"))
        if scientific_payload(normal_verdict) != scientific_payload(reverse_verdict):
            raise AssertionError("scientific output changed after raw merge-order permutation")
        if not normal_verdict["all_primary_formally_confirmed"]:
            raise AssertionError("synthetic expected verdict was not reproduced")
        support = normal_verdict["branch_support_audit"]
        expected_support = {
            "fixed_common_valid": 597,
            "normalized_common_valid": 597,
            "intersection": 594,
            "fixed_only": 3,
            "normalized_only": 3,
        }
        if support != expected_support:
            raise AssertionError(f"unexpected support audit: {support}")
        if normal_verdict["selected_image_clusters"] != 300:
            raise AssertionError("selected zero-row clusters were not retained")

        result = {
            "status": "PASS",
            "artifact_role": "synthetic_post_failure_software_qa_not_scientific_evidence",
            "analyzer_sha256": sha256(SCRIPT_ROOT / "analyze_confirmatory_lockbox.py"),
            "rehearsal_script_sha256": sha256(Path(__file__)),
            "selected_clusters": 300,
            "bootstrap_replicates": REPS,
            "expected_verdict_reproduced": True,
            "raw_merge_order_invariant": True,
            "duplicate_key_failed_closed": True,
            "zero_row_selected_cluster_retained": True,
            "empty_stratum_retained": True,
            "all_zero_outcome_cluster_retained": True,
            "unequal_raw_rosters_handled": True,
            "support_audit": support,
            "normal_verdict_sha256": sha256(normal_verdict_path),
            "reverse_verdict_sha256": sha256(reverse_verdict_path),
            "scientific_payload_sha256": hashlib.sha256(
                (json.dumps(scientific_payload(normal_verdict), sort_keys=True) + "\n").encode("utf-8")
            ).hexdigest(),
            "synthetic_scientific_payload": scientific_payload(normal_verdict),
            "historical_lockbox_v1_touched": False,
            "normative_or_scientific_status_changed": False,
        }
    output_dir.mkdir(parents=True)
    result_path = output_dir / "E2E_RESULT.json"
    write_json(result_path, result)
    report_path = output_dir / "E2E_REPORT.md"
    report_path.write_text(
        "# Synthetic lockbox analyzer CLI rehearsal\n\n"
        "Status: **PASS**.\n\n"
        "This post-failure software-QA rehearsal used only generated fixtures. "
        "It exercised the real analyzer CLI from sealed manifests and raw tables "
        "through zero-fill, common-support intersection, clustered bootstrap, "
        "Tier-A gatekeeping, Tier-B Holm correction, and `LOCKBOX_VERDICT.json`.\n\n"
        "It retained 300 selected clusters, including zero-row and empty-stratum "
        "clusters; handled fixed-only and normalized-only GT keys; reproduced the "
        "expected synthetic verdict; produced identical scientific outputs after "
        "raw-row reversal; and rejected a sealed duplicate-key scenario.\n\n"
        "No real outcome, checkpoint, project lockbox receipt, or scientific claim "
        "was accessed or changed. This does not validate or restore Lockbox v1.\n",
        encoding="utf-8",
        newline="\n",
    )
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n" for path in (result_path, report_path)
        ),
        encoding="ascii",
        newline="\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_rehearsal(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
