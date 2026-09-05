"""Fail-closed validation for each sealed lockbox extraction role."""

from __future__ import annotations

from pathlib import Path

from create_instrument_freeze_bundle import (
    environment_payload,
    sha256,
    validate_finalized_bundle,
)
from validate_sealed_lockbox_bundle import read_verified_json, validate_frozen_chain


ROLES = {"fixed_branch", "normalized_branch", "fixed_anatomy", "normalized_anatomy"}


def validate_extraction_contract(
    *,
    role: str,
    current_script: Path,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    data: Path,
    selected_images: Path,
    selected_manifest: Path,
    preregistration: Path,
    instrument_manifest: Path,
    imgsz: int,
    device: str,
    workers: int,
    detector_contract: str,
    stress_mode: str,
    normalization: str | None,
    kappa: float | None,
) -> dict[str, object]:
    if role not in ROLES:
        raise RuntimeError(f"unsupported sealed extraction role: {role}")

    freeze_root = instrument_manifest.resolve().parent
    freeze_validation = validate_finalized_bundle(freeze_root)
    instrument, instrument_hash = read_verified_json(instrument_manifest, "instrument manifest")
    prereg_plan, prereg_hash = read_verified_json(preregistration, "preregistration")
    (
        selection,
        selection_hash,
        chained_prereg_hash,
        chained_instrument_hash,
        source_commit,
        selected_images_hash,
    ) = validate_frozen_chain(selected_manifest, preregistration, instrument_manifest)
    if prereg_hash != chained_prereg_hash or instrument_hash != chained_instrument_hash:
        raise RuntimeError("frozen-chain hashes changed during extraction validation")

    canonical_selected = selected_manifest.resolve().parent / "selected_images.csv"
    if selected_images.resolve() != canonical_selected.resolve():
        raise RuntimeError("--selected-images must be the CSV bound by the frozen selection manifest")
    if sha256(selected_images) != selected_images_hash:
        raise RuntimeError("selected image CSV differs from the frozen selection")

    frozen_checkpoint = instrument.get("checkpoint")
    if not isinstance(frozen_checkpoint, dict):
        raise RuntimeError("instrument checkpoint contract is missing")
    frozen_checkpoint_hash = str(frozen_checkpoint.get("sha256", "")).lower()
    if expected_checkpoint_sha256.lower() != frozen_checkpoint_hash:
        raise RuntimeError("caller checkpoint hash differs from the frozen instrument")
    if not checkpoint.is_file() or sha256(checkpoint) != frozen_checkpoint_hash:
        raise RuntimeError("checkpoint bytes differ from the frozen instrument")

    frozen_data = instrument.get("data_yaml")
    if not isinstance(frozen_data, dict):
        raise RuntimeError("instrument data_yaml contract is missing")
    frozen_data_hash = str(frozen_data.get("sha256", "")).lower()
    if not data.is_file() or sha256(data) != frozen_data_hash:
        raise RuntimeError("data YAML differs from the frozen instrument")

    parameters = prereg_plan.get("stress_contract_parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("preregistration stress_contract_parameters are missing")
    if int(parameters.get("imgsz", -1)) != int(imgsz):
        raise RuntimeError("image size differs from the frozen stress contract")
    if parameters.get("detector_contract") != detector_contract:
        raise RuntimeError("detector contract differs from the preregistration")
    runtime = prereg_plan.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise RuntimeError("preregistration runtime_contract is missing")
    if str(device) != str(runtime.get("device_argument")) or int(workers) != int(runtime.get("workers", -1)):
        raise RuntimeError("device/workers differ from the frozen runtime contract")
    if int(runtime.get("imgsz", -1)) != int(imgsz):
        raise RuntimeError("runtime image size differs from the frozen contract")
    actual_environment = environment_payload()
    if actual_environment != runtime.get("environment") or actual_environment != instrument.get("environment"):
        raise RuntimeError("runtime software/backend environment differs from the frozen instrument")
    if actual_environment.get("cuda_available") is not True or actual_environment.get("torch_default_dtype") != "torch.float32":
        raise RuntimeError("sealed extraction requires CUDA and torch.float32")
    fixed = parameters.get("fixed", {})
    normalized = parameters.get("normalized", {})
    if role == "fixed_branch" and stress_mode != fixed.get("branch_stress_mode"):
        raise RuntimeError("fixed branch stress mode differs from the preregistration")
    if role == "fixed_anatomy" and stress_mode != fixed.get("anatomy_stress"):
        raise RuntimeError("fixed anatomy stress differs from the preregistration")
    if role == "normalized_branch":
        if stress_mode != normalized.get("branch_stress_mode"):
            raise RuntimeError("normalized branch stress mode differs from the preregistration")
        if normalization != normalized.get("normalization"):
            raise RuntimeError("normalized branch normalization differs from the preregistration")
        if float(kappa) != float(normalized.get("kappa")):
            raise RuntimeError("normalized branch kappa differs from the preregistration")
    if role == "normalized_anatomy":
        if stress_mode != normalized.get("anatomy_stress"):
            raise RuntimeError("normalized anatomy stress differs from the preregistration")
        if float(kappa) != float(normalized.get("kappa")):
            raise RuntimeError("normalized anatomy kappa differs from the preregistration")

    relative_script = f"scripts/{current_script.name}"
    frozen_files = instrument.get("frozen_files")
    if not isinstance(frozen_files, list):
        raise RuntimeError("instrument frozen_files inventory is missing")
    inventory = {
        row.get("path"): row.get("sha256")
        for row in frozen_files
        if isinstance(row, dict)
    }
    expected_script = freeze_root / relative_script
    if current_script.resolve() != expected_script.resolve():
        raise RuntimeError("sealed extraction must execute the script inside the frozen instrument")
    if inventory.get(relative_script) != sha256(current_script):
        raise RuntimeError("executing extractor differs from the frozen-files inventory")

    return {
        "role": role,
        "instrument_manifest_sha256": instrument_hash,
        "preregistration_sha256": prereg_hash,
        "selection_manifest_sha256": selection_hash,
        "selected_images_sha256": selected_images_hash,
        "checkpoint_sha256": frozen_checkpoint_hash,
        "data_yaml_sha256": frozen_data_hash,
        "instrument_source_commit": source_commit,
        "instrument_source_tag": selection.get("instrument_source_tag"),
        "extractor_sha256": sha256(current_script),
        "runtime_contract": runtime,
        "freeze_validation": freeze_validation,
    }
