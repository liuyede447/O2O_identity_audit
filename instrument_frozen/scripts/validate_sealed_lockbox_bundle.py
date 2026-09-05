"""Validate four sealed lockbox extractions without opening outcome rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


EXPECTED_OUTPUTS = {
    "fixed_branch": ("selected_images.csv", "per_direction.csv", "per_gt.csv"),
    "normalized_branch": ("selected_images.csv", "per_direction.csv", "per_gt.csv"),
    "fixed_anatomy": ("selected_images.csv", "per_direction_anatomy.csv", "pairwise_q_gap.csv"),
    "normalized_anatomy": ("selected_images.csv", "per_direction_anatomy.csv", "pairwise_q_gap.csv"),
}
PROHIBITED_NAMES = {
    "normalized_perturbation_summary.json",
    "incremental_value_summary.json",
    "anatomy_summary.json",
    "confirmatory_results.json",
    "lockbox_verdict.json",
    "lockbox_report.md",
}
FROZEN_TAG = "audit_instrument_v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-branch", type=Path)
    parser.add_argument("--normalized-branch", type=Path)
    parser.add_argument("--fixed-anatomy", type=Path)
    parser.add_argument("--normalized-anatomy", type=Path)
    parser.add_argument("--selected-manifest", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--instrument-manifest", type=Path)
    parser.add_argument("--secondary-results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        required = (
            "fixed_branch",
            "normalized_branch",
            "fixed_anatomy",
            "normalized_anatomy",
            "selected_manifest",
            "preregistration",
            "instrument_manifest",
            "output",
        )
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error("the following arguments are required unless --self-test is used: " + ", ".join(missing))
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def read_verified_json(path: Path, label: str) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    actual = sha256(path)
    if not tokens or not is_sha256(tokens[0]) or tokens[0].lower() != actual:
        raise RuntimeError(f"{label} SHA-256 sidecar is malformed or mismatched")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload, actual


def require_full_commit(value: object, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    ):
        raise RuntimeError(f"{label} must be a full 40-character Git commit")
    return value.lower()


def validate_frozen_chain(
    selected_manifest_path: Path,
    preregistration_path: Path,
    instrument_manifest_path: Path,
) -> tuple[dict, str, str, str, str, str]:
    preregistration, preregistration_hash = read_verified_json(preregistration_path, "preregistration")
    instrument, instrument_hash = read_verified_json(instrument_manifest_path, "instrument manifest")
    selected_manifest, selected_manifest_hash = read_verified_json(selected_manifest_path, "selection manifest")

    if preregistration.get("status") != "FROZEN_BEFORE_OUTCOMES":
        raise RuntimeError("preregistration status must be FROZEN_BEFORE_OUTCOMES")
    if instrument.get("status") != "FROZEN_BEFORE_LOCKBOX_SELECTION":
        raise RuntimeError("instrument manifest status must be FROZEN_BEFORE_LOCKBOX_SELECTION")
    if selected_manifest.get("status") != "FROZEN_SELECTION_BEFORE_OUTCOMES":
        raise RuntimeError("selection manifest status must be FROZEN_SELECTION_BEFORE_OUTCOMES")
    if selected_manifest.get("outcome_blind") is not True or selected_manifest.get("outcome_accessed") is not False:
        raise RuntimeError("selection manifest must remain outcome-blind and unopened")
    if selected_manifest.get("protocol") != "disjoint_audit_outcome_lockbox_selection_v1":
        raise RuntimeError("selection manifest protocol is incorrect")
    if selected_manifest.get("disjoint_overlap") != 0:
        raise RuntimeError("selection manifest must report zero discovery overlap")

    sample = preregistration.get("sample")
    if not isinstance(sample, dict):
        raise RuntimeError("preregistration sample definition is missing")
    if selected_manifest.get("seed") != sample.get("selection_seed"):
        raise RuntimeError("selection seed differs from the preregistration")
    if selected_manifest.get("requested_images") != sample.get("size_images"):
        raise RuntimeError("selection size differs from the preregistration")
    if selected_manifest.get("allocation") != sample.get("allocation"):
        raise RuntimeError("selection allocation differs from the preregistration")

    commits = {
        require_full_commit(preregistration.get("instrument_source_commit"), "preregistration commit"),
        require_full_commit(instrument.get("instrument_source_commit"), "instrument manifest commit"),
        require_full_commit(selected_manifest.get("instrument_source_commit"), "selection manifest commit"),
    }
    if len(commits) != 1:
        raise RuntimeError("preregistration, instrument, and selection commits differ")
    tags = {
        preregistration.get("instrument_source_tag"),
        instrument.get("instrument_source_tag"),
        selected_manifest.get("instrument_source_tag"),
    }
    if tags != {FROZEN_TAG}:
        raise RuntimeError(f"preregistration, instrument, and selection tags must all be {FROZEN_TAG}")
    if instrument.get("preregistration_sha256") != preregistration_hash:
        raise RuntimeError("instrument manifest does not bind the supplied preregistration")
    if selected_manifest.get("preregistration_sha256") != preregistration_hash:
        raise RuntimeError("selection manifest does not bind the supplied preregistration")
    if selected_manifest.get("instrument_manifest_sha256") != instrument_hash:
        raise RuntimeError("selection manifest does not bind the supplied instrument manifest")
    from create_instrument_freeze_bundle import validate_preregistration_seal

    preregistration_seal = validate_preregistration_seal(
        instrument_manifest_path.resolve().parent
    )
    if selected_manifest.get("preregistration_commit") != preregistration_seal["preregistration_commit"]:
        raise RuntimeError("selection manifest preregistration commit differs from the sealed freeze repository")
    if selected_manifest.get("preregistration_tag") != preregistration_seal["preregistration_tag"]:
        raise RuntimeError("selection manifest preregistration tag differs from the sealed freeze repository")

    from prepare_disjoint_audit_lockbox import validate_selection_repository

    selection_seal = validate_selection_repository(selected_manifest_path.resolve().parent)
    source_hashes = selected_manifest.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("selection source hashes are missing")
    data_yaml = instrument.get("data_yaml")
    if not isinstance(data_yaml, dict) or source_hashes.get("dataset_config") != data_yaml.get("sha256"):
        raise RuntimeError("selection dataset hash differs from the frozen instrument")
    if source_hashes.get("exclusion_manifest") != instrument.get("discovery_exclusion_manifest_sha256"):
        raise RuntimeError("selection exclusion hash differs from the frozen instrument")
    frozen_files = instrument.get("frozen_files")
    if not isinstance(frozen_files, list):
        raise RuntimeError("instrument frozen_files inventory is missing")
    frozen_by_path = {
        row.get("path"): row.get("sha256")
        for row in frozen_files
        if isinstance(row, dict)
    }
    if source_hashes.get("selection_script") != frozen_by_path.get("scripts/prepare_disjoint_audit_lockbox.py"):
        raise RuntimeError("selection script differs from the frozen-files inventory")
    if selected_manifest.get("eligible_population_roster") != instrument.get("eligible_population_roster"):
        raise RuntimeError("selection eligible-population contract differs from the frozen instrument")

    selected_images_hash = selected_manifest.get("selected_images_sha256")
    if not is_sha256(selected_images_hash):
        raise RuntimeError("selection manifest selected_images_sha256 is missing or malformed")
    selected_images_path = selected_manifest_path.parent / "selected_images.csv"
    if not selected_images_path.is_file() or sha256(selected_images_path) != selected_images_hash:
        raise RuntimeError("selection manifest selected_images.csv is missing or hash-mismatched")
    selected_ids = read_image_ids(selected_images_path)
    exclusion_path = instrument_manifest_path.resolve().parent / "frozen_inputs" / "discovery_selected_images_exclusion.csv"
    if not exclusion_path.is_file() or sha256(exclusion_path) != instrument.get("discovery_exclusion_manifest_sha256"):
        raise RuntimeError("frozen discovery exclusion manifest is missing or hash-mismatched")
    if selected_ids.intersection(read_image_ids(exclusion_path)):
        raise RuntimeError("sealed selection overlaps the discovery exclusion manifest")
    return (
        selected_manifest,
        selected_manifest_hash,
        preregistration_hash,
        instrument_hash,
        next(iter(commits)),
        selected_images_hash,
    )


def read_image_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise RuntimeError(f"{path} does not contain an image_id column")
        identifiers = []
        for row in reader:
            identifier = str(row.get("image_id", "")).strip()
            if not identifier:
                raise RuntimeError(f"{path} contains an empty image_id")
            identifiers.append(identifier)
    if not identifiers:
        raise RuntimeError(f"{path} contains no selected images")
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"{path} contains duplicate image_id values")
    return set(identifiers)


def validate_role(role: str, manifest: dict) -> None:
    if role == "fixed_branch" and manifest.get("stress_mode") != "fixed_1px":
        raise RuntimeError("fixed branch manifest does not declare stress_mode=fixed_1px")
    if role == "normalized_branch" and (
        manifest.get("stress_mode") != "normalized" or manifest.get("normalization") != "equivalent_side"
    ):
        raise RuntimeError("normalized branch manifest does not declare normalized equivalent-side stress")
    if role == "fixed_anatomy" and manifest.get("stress") != "fixed_1px":
        raise RuntimeError("fixed anatomy manifest does not declare stress=fixed_1px")
    if role == "normalized_anatomy" and manifest.get("stress") != "equivalent_side":
        raise RuntimeError("normalized anatomy manifest does not declare stress=equivalent_side")


def validate_extraction(
    role: str,
    directory: Path,
    selected_images_hash: str,
    selected_ids: set[str],
    expected_binding: dict[str, object],
) -> str:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    prohibited = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name.lower() in PROHIBITED_NAMES
    )
    if prohibited:
        raise RuntimeError(f"{role} contains prohibited summary or verdict artifact: {prohibited[0]}")

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{role} manifest must contain a JSON object")
    if manifest.get("status") != "sealed_extraction_complete":
        raise RuntimeError(f"{role} status must be sealed_extraction_complete")
    if manifest.get("outcome_access_permitted") is not False:
        raise RuntimeError(f"{role} outcome_access_permitted must be false")
    if manifest.get("sealed_outcomes_not_summarized") is not True:
        raise RuntimeError(f"{role} must declare sealed_outcomes_not_summarized=true")
    validate_role(role, manifest)
    binding = manifest.get("frozen_contract")
    if not isinstance(binding, dict):
        raise RuntimeError(f"{role} frozen_contract binding is missing")
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise RuntimeError(f"{role} frozen_contract {key} differs from the frozen chain")

    expected_outputs = EXPECTED_OUTPUTS[role]
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_outputs) or set(outputs) != set(expected_outputs):
        raise RuntimeError(f"{role} outputs must be exactly {list(expected_outputs)}")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(expected_outputs):
        raise RuntimeError(f"{role} artifact_sha256 must cover exactly the sealed raw outputs")
    for name in expected_outputs:
        path = directory / name
        expected_hash = artifact_hashes.get(name)
        if not path.is_file() or not is_sha256(expected_hash) or sha256(path) != expected_hash.lower():
            raise RuntimeError(f"{role} raw output is missing or hash-mismatched: {name}")

    selected_path = directory / "selected_images.csv"
    if artifact_hashes["selected_images.csv"].lower() != selected_images_hash:
        raise RuntimeError(f"{role} selected_images.csv hash differs from the frozen selection")
    ids = read_image_ids(selected_path)
    if ids != selected_ids:
        raise RuntimeError(f"{role} image_id set differs from the frozen selection")
    if manifest.get("selected_images") != len(selected_ids):
        raise RuntimeError(f"{role} selected_images count differs from the frozen selection")
    return sha256(manifest_path)


def build_ready_payload(
    *,
    fixed_branch: Path,
    normalized_branch: Path,
    fixed_anatomy: Path,
    normalized_anatomy: Path,
    selected_manifest_path: Path,
    preregistration_path: Path,
    instrument_manifest_path: Path,
    secondary_results: Path | None = None,
) -> dict:
    from create_instrument_freeze_bundle import validate_finalized_bundle

    validate_finalized_bundle(instrument_manifest_path.resolve().parent)
    (
        selected_manifest,
        selected_manifest_hash,
        preregistration_hash,
        instrument_hash,
        source_commit,
        selected_images_hash,
    ) = validate_frozen_chain(selected_manifest_path, preregistration_path, instrument_manifest_path)
    selected_ids = read_image_ids(selected_manifest_path.parent / "selected_images.csv")
    declared_count = selected_manifest.get("requested_images")
    if declared_count != len(selected_ids):
        raise RuntimeError("selection manifest requested_images differs from selected image IDs")

    directories = {
        "fixed_branch": fixed_branch,
        "normalized_branch": normalized_branch,
        "fixed_anatomy": fixed_anatomy,
        "normalized_anatomy": normalized_anatomy,
    }
    instrument, _ = read_verified_json(instrument_manifest_path, "instrument manifest")
    preregistration, _ = read_verified_json(preregistration_path, "preregistration")
    runtime_contract = preregistration.get("runtime_contract")
    if not isinstance(runtime_contract, dict):
        raise RuntimeError("preregistration runtime_contract is missing")
    checkpoint = instrument.get("checkpoint")
    data_yaml = instrument.get("data_yaml")
    frozen_files = instrument.get("frozen_files")
    if not isinstance(checkpoint, dict) or not is_sha256(checkpoint.get("sha256")):
        raise RuntimeError("instrument checkpoint SHA-256 is missing")
    if not isinstance(data_yaml, dict) or not is_sha256(data_yaml.get("sha256")):
        raise RuntimeError("instrument data YAML SHA-256 is missing")
    if not isinstance(frozen_files, list):
        raise RuntimeError("instrument frozen_files inventory is missing")
    source_hashes = {
        row.get("path"): row.get("sha256")
        for row in frozen_files
        if isinstance(row, dict)
    }
    role_scripts = {
        "fixed_branch": "scripts/run_reviewer_killer_controls.py",
        "normalized_branch": "scripts/run_reviewer_killer_controls.py",
        "fixed_anatomy": "scripts/analyze_native_alignment_path_anatomy.py",
        "normalized_anatomy": "scripts/analyze_native_alignment_path_anatomy.py",
    }
    expected_bindings = {}
    for role, script_path in role_scripts.items():
        script_hash = source_hashes.get(script_path)
        if not is_sha256(script_hash):
            raise RuntimeError(f"instrument frozen_files is missing {script_path}")
        expected_bindings[role] = {
            "role": role,
            "instrument_manifest_sha256": instrument_hash,
            "preregistration_sha256": preregistration_hash,
            "selection_manifest_sha256": selected_manifest_hash,
            "selected_images_sha256": selected_images_hash,
            "checkpoint_sha256": checkpoint["sha256"],
            "data_yaml_sha256": data_yaml["sha256"],
            "instrument_source_commit": source_commit,
            "instrument_source_tag": FROZEN_TAG,
            "extractor_sha256": script_hash,
            "runtime_contract": runtime_contract,
        }
    manifest_hashes = {
        role: validate_extraction(
            role,
            directory,
            selected_images_hash,
            selected_ids,
            expected_bindings[role],
        )
        for role, directory in directories.items()
    }
    from prepare_disjoint_audit_lockbox import validate_selection_repository

    selection_seal = validate_selection_repository(selected_manifest_path.resolve().parent)
    secondary_family = preregistration.get("secondary_confirmatory_family")
    secondary_hypotheses = (
        secondary_family.get("hypotheses", [])
        if isinstance(secondary_family, dict)
        else []
    )
    if secondary_hypotheses and (secondary_results is None or not secondary_results.is_file()):
        raise RuntimeError("preregistered secondary hypotheses require sealed secondary results")
    if not secondary_hypotheses and secondary_results is not None:
        raise RuntimeError("secondary results were supplied without preregistered secondary hypotheses")
    secondary_results_hash = sha256(secondary_results) if secondary_results is not None else None
    return {
        "status": "SEALED_BUNDLE_READY",
        "outcome_access_permitted": False,
        "selected_image_count": len(selected_ids),
        "selected_images_sha256": selected_images_hash,
        "selection_manifest_sha256": selected_manifest_hash,
        "preregistration_sha256": preregistration_hash,
        "instrument_manifest_sha256": instrument_hash,
        "instrument_source_commit": source_commit,
        "instrument_source_tag": FROZEN_TAG,
        "preregistration_commit": selected_manifest["preregistration_commit"],
        "preregistration_tag": selected_manifest["preregistration_tag"],
        "selection_commit": selection_seal["selection_commit"],
        "selection_tag": selection_seal["selection_tag"],
        "secondary_results_sha256": secondary_results_hash,
        "extraction_manifest_sha256": manifest_hashes,
    }


def validate_and_write(
    *,
    fixed_branch: Path,
    normalized_branch: Path,
    fixed_anatomy: Path,
    normalized_anatomy: Path,
    selected_manifest_path: Path,
    preregistration_path: Path,
    instrument_manifest_path: Path,
    output: Path,
    secondary_results: Path | None = None,
) -> tuple[int, str]:
    if output.exists():
        raise FileExistsError(output)
    if output.name != "SEALED_BUNDLE_READY.json":
        raise ValueError("--output filename must be SEALED_BUNDLE_READY.json")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    payload = build_ready_payload(
        fixed_branch=fixed_branch,
        normalized_branch=normalized_branch,
        fixed_anatomy=fixed_anatomy,
        normalized_anatomy=normalized_anatomy,
        selected_manifest_path=selected_manifest_path,
        preregistration_path=preregistration_path,
        instrument_manifest_path=instrument_manifest_path,
        secondary_results=secondary_results,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    ready_hash = sha256(output)
    sidecar_path(output).write_text(
        f"{ready_hash}  {output.name}\n", encoding="ascii", newline="\n"
    )
    return int(payload["selected_image_count"]), ready_hash


def validate_existing_ready_bundle(
    *,
    fixed_branch: Path,
    normalized_branch: Path,
    fixed_anatomy: Path,
    normalized_anatomy: Path,
    selected_manifest_path: Path,
    preregistration_path: Path,
    instrument_manifest_path: Path,
    ready_path: Path,
    secondary_results: Path | None = None,
) -> tuple[dict, str]:
    """Revalidate the full sealed chain without interpreting outcome rows."""
    ready, ready_hash = read_verified_json(ready_path, "sealed ready bundle")
    if ready.get("status") != "SEALED_BUNDLE_READY":
        raise RuntimeError("sealed ready bundle status must be SEALED_BUNDLE_READY")
    if ready.get("outcome_access_permitted") is not False:
        raise RuntimeError("sealed ready bundle must keep outcome_access_permitted=false")
    current = build_ready_payload(
        fixed_branch=fixed_branch,
        normalized_branch=normalized_branch,
        fixed_anatomy=fixed_anatomy,
        normalized_anatomy=normalized_anatomy,
        selected_manifest_path=selected_manifest_path,
        preregistration_path=preregistration_path,
        instrument_manifest_path=instrument_manifest_path,
        secondary_results=secondary_results,
    )
    if ready != current:
        raise RuntimeError("sealed ready bundle no longer matches the frozen chain or raw artifacts")
    return ready, ready_hash


def write_with_sidecar(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    digest = sha256(path)
    sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    return digest


def write_selected(path: Path, identifiers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset_index", "image_id", "stratum"])
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            writer.writerow({"dataset_index": index, "image_id": identifier, "stratum": "t_only"})


def run_self_test() -> dict[str, object]:
    identifiers = ["image_a", "image_b", "image_c"]
    checkpoint_hash = "2" * 64
    data_hash = "3" * 64
    runtime_contract = {
        "device_argument": "0",
        "workers": 0,
        "imgsz": 800,
        "torch_default_dtype": "torch.float32",
        "environment": {"cuda_available": True},
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        base = Path(temporary_directory)
        freeze = base / "freeze"
        scripts = freeze / "scripts"
        scripts.mkdir(parents=True)
        branch_script = scripts / "run_reviewer_killer_controls.py"
        anatomy_script = scripts / "analyze_native_alignment_path_anatomy.py"
        selector_script = scripts / "prepare_disjoint_audit_lockbox.py"
        branch_script.write_text("# frozen branch extractor\n", encoding="utf-8")
        anatomy_script.write_text("# frozen anatomy extractor\n", encoding="utf-8")
        selector_script.write_text("# frozen selector\n", encoding="utf-8")
        branch_script_hash = sha256(branch_script)
        anatomy_script_hash = sha256(anatomy_script)
        selector_script_hash = sha256(selector_script)
        frozen_inputs = freeze / "frozen_inputs"
        frozen_inputs.mkdir()
        exclusion_path = frozen_inputs / "discovery_selected_images_exclusion.csv"
        write_selected(exclusion_path, ["discovery_image"])
        exclusion_hash = sha256(exclusion_path)
        eligible_population_roster = {
            "protocol": "eligible_population_roster_v1",
            "sha256": "6" * 64,
            "count": 4,
            "strata_counts": {"t_only": 4},
        }

        from create_instrument_freeze_bundle import inventory as freeze_inventory

        (freeze / "SOURCE_INVENTORY.json").write_text(
            json.dumps(
                {"files": freeze_inventory(freeze, {"SOURCE_INVENTORY.json"})},
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(["git", "init", str(freeze)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(freeze), "config", "core.autocrlf", "false"], check=True)
        subprocess.run(["git", "-C", str(freeze), "config", "user.email", "selftest@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(freeze), "config", "user.name", "sealed-selftest"], check=True)
        subprocess.run(["git", "-C", str(freeze), "add", "."], check=True)
        subprocess.run(["git", "-C", str(freeze), "commit", "-m", "frozen source"], check=True, capture_output=True)
        commit = subprocess.run(
            ["git", "-C", str(freeze), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        subprocess.run(["git", "-C", str(freeze), "tag", FROZEN_TAG, commit], check=True)

        preregistration_path = freeze / "CONFIRMATORY_AUDIT_PLAN.json"
        preregistration_hash = write_with_sidecar(
            preregistration_path,
            {
                "status": "FROZEN_BEFORE_OUTCOMES",
                "instrument_source_commit": commit,
                "instrument_source_tag": FROZEN_TAG,
                "sample": {
                    "size_images": len(identifiers),
                    "selection_seed": 20260831,
                    "allocation": {"t_only": len(identifiers)},
                },
                "runtime_contract": runtime_contract,
            },
        )
        (freeze / "CONFIRMATORY_AUDIT_PLAN.md").write_text(
            "# Synthetic preregistration\n", encoding="utf-8", newline="\n"
        )
        instrument_path = freeze / "TOOL_FREEZE_MANIFEST.json"
        instrument_sidecar = sidecar_path(instrument_path)
        instrument_hash = write_with_sidecar(
            instrument_path,
            {
                "status": "FROZEN_BEFORE_LOCKBOX_SELECTION",
                "instrument_source_commit": commit,
                "instrument_source_tag": FROZEN_TAG,
                "preregistration_sha256": preregistration_hash,
                "checkpoint": {"sha256": checkpoint_hash},
                "data_yaml": {"sha256": data_hash},
                "eligible_population_roster": eligible_population_roster,
                "discovery_exclusion_manifest_sha256": exclusion_hash,
                "frozen_files": freeze_inventory(
                    freeze,
                    {instrument_path.name, instrument_sidecar.name},
                ),
            },
        )
        from create_instrument_freeze_bundle import seal_preregistration

        preregistration_seal = seal_preregistration(freeze)
        selection_directory = base / "selection"
        selection_directory.mkdir()
        selected_path = selection_directory / "selected_images.csv"
        write_selected(selected_path, identifiers)
        selected_hash = sha256(selected_path)
        selection_manifest_path = selection_directory / "selection_manifest.json"
        selection_manifest_hash = write_with_sidecar(
            selection_manifest_path,
            {
                "status": "FROZEN_SELECTION_BEFORE_OUTCOMES",
                "protocol": "disjoint_audit_outcome_lockbox_selection_v1",
                "outcome_blind": True,
                "outcome_accessed": False,
                "seed": 20260831,
                "requested_images": len(identifiers),
                "allocation": {"t_only": len(identifiers)},
                "disjoint_overlap": 0,
                "instrument_source_commit": commit,
                "instrument_source_tag": FROZEN_TAG,
                "preregistration_sha256": preregistration_hash,
                "instrument_manifest_sha256": instrument_hash,
                "preregistration_commit": preregistration_seal["preregistration_commit"],
                "preregistration_tag": preregistration_seal["preregistration_tag"],
                "source_sha256": {
                    "dataset_config": data_hash,
                    "exclusion_manifest": exclusion_hash,
                    "selection_script": selector_script_hash,
                },
                "eligible_population_roster": eligible_population_roster,
                "selected_images_sha256": selected_hash,
            },
        )
        from prepare_disjoint_audit_lockbox import seal_selection_repository

        selection_seal = seal_selection_repository(selection_directory)
        assert selection_seal["selection_tag"] == "audit_lockbox_selection_v1.0"

        directories = {}
        for role, outputs in EXPECTED_OUTPUTS.items():
            directory = base / role
            directory.mkdir()
            directories[role] = directory
            write_selected(directory / "selected_images.csv", identifiers)
            for name in outputs[1:]:
                (directory / name).write_text("sealed raw bytes\n", encoding="utf-8")
            manifest = {
                "status": "sealed_extraction_complete",
                "sealed_outcomes_not_summarized": True,
                "outcome_access_permitted": False,
                "selected_images": len(identifiers),
                "outputs": list(outputs),
                "artifact_sha256": {name: sha256(directory / name) for name in outputs},
                "frozen_contract": {
                    "role": role,
                    "instrument_manifest_sha256": instrument_hash,
                    "preregistration_sha256": preregistration_hash,
                    "selection_manifest_sha256": selection_manifest_hash,
                    "selected_images_sha256": selected_hash,
                    "checkpoint_sha256": checkpoint_hash,
                    "data_yaml_sha256": data_hash,
                    "instrument_source_commit": commit,
                    "instrument_source_tag": FROZEN_TAG,
                    "extractor_sha256": branch_script_hash if role.endswith("branch") else anatomy_script_hash,
                    "runtime_contract": runtime_contract,
                },
            }
            if role.endswith("branch"):
                manifest["stress_mode"] = "fixed_1px" if role.startswith("fixed") else "normalized"
                if role.startswith("normalized"):
                    manifest["normalization"] = "equivalent_side"
            else:
                manifest["stress"] = "fixed_1px" if role.startswith("fixed") else "equivalent_side"
            (directory / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
            )

        output = base / "SEALED_BUNDLE_READY.json"
        count, ready_hash = validate_and_write(
            fixed_branch=directories["fixed_branch"],
            normalized_branch=directories["normalized_branch"],
            fixed_anatomy=directories["fixed_anatomy"],
            normalized_anatomy=directories["normalized_anatomy"],
            selected_manifest_path=selection_manifest_path,
            preregistration_path=preregistration_path,
            instrument_manifest_path=instrument_path,
            output=output,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["status"] == "SEALED_BUNDLE_READY"
        assert set(payload["extraction_manifest_sha256"]) == set(EXPECTED_OUTPUTS)
        assert sidecar_path(output).is_file()
        revalidated, revalidated_hash = validate_existing_ready_bundle(
            fixed_branch=directories["fixed_branch"],
            normalized_branch=directories["normalized_branch"],
            fixed_anatomy=directories["fixed_anatomy"],
            normalized_anatomy=directories["normalized_anatomy"],
            selected_manifest_path=selection_manifest_path,
            preregistration_path=preregistration_path,
            instrument_manifest_path=instrument_path,
            ready_path=output,
        )
        assert revalidated == payload
        assert revalidated_hash == ready_hash

        try:
            validate_and_write(
                fixed_branch=directories["fixed_branch"],
                normalized_branch=directories["normalized_branch"],
                fixed_anatomy=directories["fixed_anatomy"],
                normalized_anatomy=directories["normalized_anatomy"],
                selected_manifest_path=selection_manifest_path,
                preregistration_path=preregistration_path,
                instrument_manifest_path=instrument_path,
                output=output,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("an existing readiness artifact was overwritten")

        prohibited = directories["fixed_branch"] / "normalized_perturbation_summary.json"
        prohibited.write_text("{}\n", encoding="utf-8")
        try:
            build_ready_payload(
                fixed_branch=directories["fixed_branch"],
                normalized_branch=directories["normalized_branch"],
                fixed_anatomy=directories["fixed_anatomy"],
                normalized_anatomy=directories["normalized_anatomy"],
                selected_manifest_path=selection_manifest_path,
                preregistration_path=preregistration_path,
                instrument_manifest_path=instrument_path,
            )
        except RuntimeError as error:
            assert "prohibited" in str(error)
        else:
            raise AssertionError("a prohibited outcome summary was accepted")
        prohibited.unlink()

        tampered = directories["normalized_anatomy"] / "pairwise_q_gap.csv"
        tampered.write_text("tampered\n", encoding="utf-8")
        try:
            build_ready_payload(
                fixed_branch=directories["fixed_branch"],
                normalized_branch=directories["normalized_branch"],
                fixed_anatomy=directories["fixed_anatomy"],
                normalized_anatomy=directories["normalized_anatomy"],
                selected_manifest_path=selection_manifest_path,
                preregistration_path=preregistration_path,
                instrument_manifest_path=instrument_path,
            )
        except RuntimeError as error:
            assert "hash-mismatched" in str(error)
        else:
            raise AssertionError("a hash-mismatched raw output was accepted")
        return {
            "status": "PASS",
            "tests": 6,
            "ready_count": count,
            "ready_hash": ready_hash,
        }


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_self_test()
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        print(json.dumps(result, separators=(",", ":")))
    else:
        from create_instrument_freeze_bundle import validate_frozen_script_identity

        validate_frozen_script_identity(
            args.instrument_manifest.resolve().parent,
            Path(__file__),
            use_tool_manifest=True,
        )
        count, ready_hash = validate_and_write(
            fixed_branch=args.fixed_branch,
            normalized_branch=args.normalized_branch,
            fixed_anatomy=args.fixed_anatomy,
            normalized_anatomy=args.normalized_anatomy,
            selected_manifest_path=args.selected_manifest,
            preregistration_path=args.preregistration,
            instrument_manifest_path=args.instrument_manifest,
            output=args.output,
            secondary_results=args.secondary_results,
        )
        print(json.dumps({"ready": True, "count": count, "hash": ready_hash}, separators=(",", ":")))


if __name__ == "__main__":
    main()
