#!/usr/bin/env python3
"""Audit and atomically merge parallel boundary-population image shards.

The canonical scanner output and three worker outputs remain immutable inputs.
An executed merge first copies every selected shard into a complete staging
directory, revalidates it, then atomically swaps the canonical ``image_shards``
directory while retaining the original directory as a timestamped backup.
Root CSVs and summaries are intentionally rebuilt later by the canonical
scanner's ``--resume`` path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARD_FILES = (
    "per_direction_boundary_events.csv",
    "o2m_rank_set_radius_curve.csv",
    "summary.json",
)
CONTRACT_FIELDS = (
    "checkpoint_sha256",
    "fine_step",
    "replay_engine",
    "replay_batch_size",
)


class AuditError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--canonical-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--expected-workers", type=int, default=3)
    parser.add_argument("--expected-images", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-final", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--python", default=r"D:\anaconda3\python.exe")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing {label or 'JSON'}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise AuditError(f"invalid {label or 'JSON'}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{label or 'JSON'} must contain an object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_path(value: str | Path, plan_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = ((plan_root / path), (PROJECT_ROOT / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def require_hash(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise AuditError(f"{label} is not a SHA-256 digest: {value!r}")
    return text


def read_roster(path: Path, expected_count: int | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing selected-images CSV: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"dataset_index", "image_id"}.issubset(fields):
            raise AuditError(f"selected-images CSV lacks dataset_index/image_id: {path}")
        rows = list(reader)
    if expected_count is not None and len(rows) != expected_count:
        raise AuditError(f"selected-images count mismatch in {path}: expected {expected_count}, got {len(rows)}")
    ids = [row["image_id"] for row in rows]
    indices = [int(row["dataset_index"]) for row in rows]
    if any(not image_id for image_id in ids):
        raise AuditError(f"empty image_id in {path}")
    if len(ids) != len(set(ids)):
        raise AuditError(f"duplicate image_id in {path}")
    if len(indices) != len(set(indices)):
        raise AuditError(f"duplicate dataset_index in {path}")
    return rows


def roster_signature(rows: Iterable[dict[str, str]]) -> list[tuple[str, int]]:
    return [(str(row["image_id"]), int(row["dataset_index"])) for row in rows]


def validate_worker_root_roster(
    actual: list[tuple[str, int]], assigned: list[tuple[str, int]], worker_id: str
) -> None:
    require_equal(set(actual), set(assigned), f"{worker_id} root selected-images identity set")
    expected_scanner_order = sorted(assigned, key=lambda item: item[0])
    require_equal(actual, expected_scanner_order, f"{worker_id} root selected-images scanner order")


def require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> tuple[str, ...]:
    fields = tuple(fieldnames or ())
    missing = sorted(required - set(fields))
    if missing:
        raise AuditError(f"missing columns in {path}: {missing}")
    return fields


def validate_shard(shard_dir: Path, expected_image_id: str, expected_dataset_index: int) -> dict[str, Any]:
    if not shard_dir.is_dir():
        raise AuditError(f"missing shard directory: {shard_dir}")
    partials = list(shard_dir.glob("*.partial"))
    if partials:
        raise AuditError(f"partial files inside completed shard: {shard_dir}")
    marker = read_json(shard_dir / "complete.json", "complete marker")
    summary = read_json(shard_dir / "summary.json", "shard summary")
    require_equal(marker.get("status"), "complete", f"{shard_dir} marker status")
    require_equal(marker.get("image_id"), expected_image_id, f"{shard_dir} marker image_id")
    require_equal(int(marker.get("dataset_index", -1)), expected_dataset_index, f"{shard_dir} marker dataset_index")
    marker_hashes = marker.get("sha256")
    if not isinstance(marker_hashes, dict) or set(marker_hashes) != set(SHARD_FILES):
        raise AuditError(f"{shard_dir} marker must hash exactly {list(SHARD_FILES)}")
    for filename in SHARD_FILES:
        target = shard_dir / filename
        expected_hash = require_hash(marker_hashes.get(filename), f"{shard_dir}/{filename} hash")
        if not target.is_file() or sha256(target) != expected_hash:
            raise AuditError(f"completed shard hash mismatch: {target}")
    marker_without_hash = {key: value for key, value in marker.items() if key != "sha256"}
    if marker_without_hash != summary:
        raise AuditError(f"marker/summary payload mismatch: {shard_dir}")
    require_equal(summary.get("status"), "complete", f"{shard_dir} summary status")
    require_equal(summary.get("image_id"), expected_image_id, f"{shard_dir} summary image_id")
    require_equal(int(summary.get("dataset_index", -1)), expected_dataset_index, f"{shard_dir} summary dataset_index")

    event_path = shard_dir / "per_direction_boundary_events.csv"
    event_keys: set[tuple[str, str, str]] = set()
    with event_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        event_header = require_columns(reader.fieldnames, {"image_id", "gt_id", "direction"}, event_path)
        event_rows = 0
        for row in reader:
            require_equal(row["image_id"], expected_image_id, f"{event_path} row image_id")
            key = (row["image_id"], row["gt_id"], row["direction"])
            if key in event_keys:
                raise AuditError(f"duplicate trajectory key in {event_path}: {key}")
            event_keys.add(key)
            event_rows += 1
    require_equal(event_rows, int(marker.get("trajectories", -1)), f"{event_path} row count")

    curve_path = shard_dir / "o2m_rank_set_radius_curve.csv"
    curve_rows = 0
    curve_trajectory_keys: set[tuple[str, str, str]] = set()
    finished_keys: set[tuple[str, str, str]] = set()
    current_key: tuple[str, str, str] | None = None
    current_radii: set[str] = set()
    with curve_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        curve_header = require_columns(reader.fieldnames, {"image_id", "gt_id", "direction", "radius"}, curve_path)
        for row in reader:
            require_equal(row["image_id"], expected_image_id, f"{curve_path} row image_id")
            key = (row["image_id"], row["gt_id"], row["direction"])
            if key != current_key:
                if current_key is not None:
                    finished_keys.add(current_key)
                if key in finished_keys:
                    raise AuditError(f"non-contiguous curve trajectory in {curve_path}: {key}")
                current_key = key
                current_radii = set()
            if row["radius"] in current_radii:
                raise AuditError(f"duplicate curve radius in {curve_path}: {key} radius={row['radius']}")
            current_radii.add(row["radius"])
            curve_trajectory_keys.add(key)
            curve_rows += 1
    if curve_trajectory_keys != event_keys:
        raise AuditError(f"curve/event trajectory-key set mismatch: {shard_dir}")
    require_equal(curve_rows, int(marker.get("curve_rows", -1)), f"{curve_path} row count")
    return {
        "image_id": expected_image_id,
        "dataset_index": expected_dataset_index,
        "trajectories": event_rows,
        "curve_rows": curve_rows,
        "event_header": event_header,
        "curve_header": curve_header,
        "marker_sha256": sha256(shard_dir / "complete.json"),
    }


def enumerate_shards(
    root: Path,
    allowed_superseded_partials: set[str] | None = None,
    observed_superseded_partials: set[str] | None = None,
) -> set[str]:
    shards_root = root / "image_shards"
    if not shards_root.is_dir():
        raise AuditError(f"missing image_shards directory: {shards_root}")
    names: set[str] = set()
    for entry in shards_root.iterdir():
        if entry.name.endswith(".partial"):
            image_id = entry.name[: -len(".partial")]
            if allowed_superseded_partials is not None and image_id in allowed_superseded_partials:
                if observed_superseded_partials is not None:
                    observed_superseded_partials.add(image_id)
                continue
            raise AuditError(f"incomplete partial shard present without an assigned complete replacement: {entry}")
        if not entry.is_dir():
            raise AuditError(f"unexpected non-directory in image_shards: {entry}")
        names.add(entry.name)
    return names


def plan_path_and_hash(plan: dict[str, Any], key: str, plan_root: Path) -> tuple[Path, str]:
    aliases = {"full_selection": "full_selected_images"}
    actual_key = key if key in plan else aliases.get(key, key)
    value = plan.get(actual_key)
    if isinstance(value, dict):
        path_value = value.get("path") or value.get("file")
        hash_value = value.get("sha256")
    else:
        path_value = value
        hash_value = plan.get(f"{actual_key}_sha256")
    if not path_value:
        raise AuditError(f"parallel plan is missing {key} path")
    return resolve_path(path_value, plan_root), require_hash(hash_value, f"parallel plan {key} hash")


def worker_entries(plan: dict[str, Any], plan_root: Path, expected_workers: int) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    entries = plan.get("workers")
    if not isinstance(entries, list) or len(entries) != expected_workers:
        raise AuditError(f"parallel plan must contain exactly {expected_workers} workers")
    result = []
    ids: set[str] = set()
    for raw in entries:
        entry = {"worker_id": raw} if isinstance(raw, str) else raw
        if not isinstance(entry, dict):
            raise AuditError("invalid worker entry in parallel plan")
        worker_id = str(entry.get("worker_id") or "")
        if not worker_id or worker_id in ids:
            raise AuditError(f"missing or duplicate worker_id: {worker_id!r}")
        ids.add(worker_id)
        contract_ref = entry.get("contract") or entry.get("contract_path") or f"contracts/{worker_id}_contract.json"
        contract_path = resolve_path(contract_ref, plan_root)
        expected_hash = entry.get("contract_sha256")
        if not expected_hash:
            hashes = plan.get("worker_contract_sha256", {})
            expected_hash = hashes.get(worker_id) if isinstance(hashes, dict) else None
        if sha256(contract_path) != require_hash(expected_hash, f"{worker_id} contract hash"):
            raise AuditError(f"worker contract hash mismatch: {contract_path}")
        contract = read_json(contract_path, f"{worker_id} contract")
        require_equal(str(contract.get("worker_id")), worker_id, f"{worker_id} contract worker_id")
        result.append((entry, contract_path, contract))
    return result


def contract_value(contract: dict[str, Any], field: str) -> Any:
    if field == "scanner_sha256":
        return contract.get(field) or contract.get("script_sha256")
    return contract.get(field)


def validate_root_terminal(root: Path, expected_images: int, role: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_json(root / "manifest.json", f"{role} manifest")
    progress = read_json(root / "progress.json", f"{role} progress")
    summary = read_json(root / "summary.json", f"{role} summary")
    require_equal(manifest.get("status"), "complete", f"{role} manifest status")
    require_equal(progress.get("status"), "complete", f"{role} progress status")
    require_equal(summary.get("status"), "complete", f"{role} summary status")
    require_equal(int(progress.get("processed_images", -1)), expected_images, f"{role} progress processed_images")
    require_equal(int(progress.get("total_images", -1)), expected_images, f"{role} progress total_images")
    require_equal(int(summary.get("selected_images", -1)), expected_images, f"{role} summary selected_images")
    require_equal(int(summary.get("processed_images", -1)), expected_images, f"{role} summary processed_images")
    return manifest, progress, summary


def audit_merge(
    plan_root: Path,
    canonical_override: Path | None,
    expected_workers: int,
    expected_images: int,
) -> dict[str, Any]:
    plan_root = plan_root.resolve()
    plan_path = plan_root / "contracts" / "parallel_plan.json"
    plan = read_json(plan_path, "parallel plan")
    protocol = str(plan.get("protocol") or "")
    if not protocol:
        raise AuditError("parallel plan protocol is missing")
    checks = plan.get("checks")
    if not isinstance(checks, dict) or not all(checks.get(name) is True for name in ("complete_assignment", "mutually_exclusive", "no_overlap_with_canonical")):
        raise AuditError("parallel plan partition checks are missing or false")

    canonical_value = plan.get("canonical_output")
    if isinstance(canonical_value, dict):
        canonical_value = canonical_value.get("path") or canonical_value.get("directory")
    canonical_dir = canonical_override.resolve() if canonical_override else resolve_path(canonical_value, plan_root)
    canonical_manifest = read_json(canonical_dir / "manifest.json", "canonical manifest")
    expected_manifest_hash = plan.get("canonical_manifest_sha256") or plan.get("canonical_manifest_sha")
    if expected_manifest_hash and sha256(canonical_dir / "manifest.json") != require_hash(expected_manifest_hash, "canonical manifest hash"):
        raise AuditError("canonical manifest differs from the frozen parallel-plan snapshot")

    full_selection, full_selection_hash = plan_path_and_hash(plan, "full_selection", plan_root)
    if sha256(full_selection) != full_selection_hash:
        raise AuditError("full selected-images source hash mismatch")
    full_rows = read_roster(full_selection, expected_images)
    full_signature = roster_signature(full_rows)
    full_map = dict(full_signature)
    canonical_rows = read_roster(canonical_dir / "selected_images.csv", expected_images)
    require_equal(roster_signature(canonical_rows), full_signature, "canonical selected-images order/content")
    require_equal(canonical_manifest.get("selected_images_source_sha256"), full_selection_hash, "canonical selection-source hash")

    canonical_contract = {
        "checkpoint_sha256": canonical_manifest.get("checkpoint_sha256"),
        "scanner_sha256": canonical_manifest.get("script_sha256"),
        "fine_step": canonical_manifest.get("fine_step"),
        "replay_engine": canonical_manifest.get("replay_engine"),
        "replay_batch_size": canonical_manifest.get("replay_batch_size"),
    }
    plan_contract = plan.get("run_contract") or plan.get("canonical_contract")
    if isinstance(plan_contract, dict):
        for field in (*CONTRACT_FIELDS, "scanner_sha256"):
            require_equal(contract_value(plan_contract, field), canonical_contract[field], f"parallel-plan {field}")

    worker_info = []
    source_roots: list[tuple[str, Path, set[str]]] = []
    worker_assigned_union: set[str] = set()
    for entry, contract_path, contract in worker_entries(plan, plan_root, expected_workers):
        worker_id = str(contract["worker_id"])
        require_equal(contract.get("protocol"), canonical_manifest.get("protocol"), f"{worker_id} contract protocol")
        for field in (*CONTRACT_FIELDS, "scanner_sha256"):
            require_equal(contract_value(contract, field), canonical_contract[field], f"{worker_id} contract {field}")
        if contract.get("full_selected_images_sha256") is not None:
            require_equal(contract.get("full_selected_images_sha256"), full_selection_hash, f"{worker_id} full selection hash")
        selection_path = resolve_path(contract.get("selected_images"), plan_root)
        selection_hash = require_hash(contract.get("selected_images_sha256"), f"{worker_id} selected hash")
        if sha256(selection_path) != selection_hash:
            raise AuditError(f"{worker_id} selected-images hash mismatch")
        assigned_rows = read_roster(selection_path)
        assigned_signature = roster_signature(assigned_rows)
        assigned_ids = [image_id for image_id, _ in assigned_signature]
        declared_ids = contract.get("assigned_images")
        declared_count = contract.get("assigned_count")
        if declared_count is None and isinstance(declared_ids, int):
            declared_count = declared_ids
        require_equal(len(assigned_rows), int(declared_count if declared_count is not None else -1), f"{worker_id} assigned_count")
        if isinstance(declared_ids, list):
            require_equal(set(str(item) for item in declared_ids), set(assigned_ids), f"{worker_id} assigned_images")
        declared = contract.get("assigned")
        if declared is not None:
            if declared and isinstance(declared[0], dict):
                declared_signature = [(str(item["image_id"]), int(item["dataset_index"])) for item in declared]
                require_equal(set(declared_signature), set(assigned_signature), f"{worker_id} assigned identities")
            else:
                require_equal(set(str(item) for item in declared), set(assigned_ids), f"{worker_id} assigned IDs")
        for image_id, dataset_index in assigned_signature:
            if image_id not in full_map or full_map[image_id] != dataset_index:
                raise AuditError(f"{worker_id} has image/dataset identity outside full roster: {image_id}/{dataset_index}")
        output_value = contract.get("output_dir") or entry.get("output_dir")
        worker_root = resolve_path(output_value, plan_root)
        manifest, progress, summary = validate_root_terminal(worker_root, len(assigned_rows), worker_id)
        require_equal(manifest.get("protocol"), canonical_manifest.get("protocol"), f"{worker_id} manifest protocol")
        require_equal(manifest.get("selected_images_source_sha256"), selection_hash, f"{worker_id} manifest selection hash")
        worker_root_rows = read_roster(worker_root / "selected_images.csv", len(assigned_rows))
        validate_worker_root_roster(roster_signature(worker_root_rows), assigned_signature, worker_id)
        for field in CONTRACT_FIELDS:
            require_equal(manifest.get(field), canonical_contract[field], f"{worker_id} manifest {field}")
        require_equal(manifest.get("script_sha256"), canonical_contract["scanner_sha256"], f"{worker_id} manifest scanner hash")
        status_value = contract.get("status_path") or entry.get("status_path")
        if status_value:
            status_payload = read_json(resolve_path(status_value, plan_root), f"{worker_id} status")
            require_equal(status_payload.get("status"), "complete", f"{worker_id} external status")
        shard_ids = enumerate_shards(worker_root)
        require_equal(shard_ids, set(assigned_ids), f"{worker_id} shard roster")
        overlap = worker_assigned_union & set(assigned_ids)
        if overlap:
            raise AuditError(f"worker assignments overlap: {sorted(overlap)}")
        worker_assigned_union.update(assigned_ids)
        source_roots.append((worker_id, worker_root, shard_ids))
        worker_info.append(
            {
                "worker_id": worker_id,
                "contract": str(contract_path),
                "output_dir": str(worker_root),
                "assigned_images": len(assigned_ids),
                "selection_sha256": selection_hash,
                "manifest_sha256": sha256(worker_root / "manifest.json"),
            }
        )

    superseded_canonical_partials: set[str] = set()
    canonical_ids = enumerate_shards(
        canonical_dir,
        allowed_superseded_partials=worker_assigned_union,
        observed_superseded_partials=superseded_canonical_partials,
    )
    partial_completed_overlap = superseded_canonical_partials & canonical_ids
    if partial_completed_overlap:
        raise AuditError(
            f"canonical partial conflicts with a canonical completed shard: {sorted(partial_completed_overlap)}"
        )
    canonical_declared = plan.get("completed_image_ids") or plan.get("completed_ids") or plan.get("canonical_completed_images")
    if canonical_declared is not None:
        require_equal(set(str(item) for item in canonical_declared), canonical_ids, "canonical completed-image snapshot")
    remaining_declared = plan.get("remaining_image_ids") or plan.get("remaining_ids") or plan.get("remaining_images")
    if remaining_declared is not None:
        require_equal(set(str(item) for item in remaining_declared), worker_assigned_union, "parallel remaining-image assignment")
    overlap = canonical_ids & worker_assigned_union
    if overlap:
        raise AuditError(f"canonical/worker duplicate shard assignment: {sorted(overlap)}")
    coverage = canonical_ids | worker_assigned_union
    expected_ids = set(full_map)
    if coverage != expected_ids:
        raise AuditError(f"full-roster coverage failure; missing={sorted(expected_ids - coverage)}, extra={sorted(coverage - expected_ids)}")
    source_roots.insert(0, ("canonical", canonical_dir, canonical_ids))

    source_by_image: dict[str, tuple[str, Path]] = {}
    shard_audit = []
    event_header: tuple[str, ...] | None = None
    curve_header: tuple[str, ...] | None = None
    total_trajectories = 0
    total_curve_rows = 0
    for role, root, ids in source_roots:
        for image_id in ids:
            if image_id in source_by_image:
                raise AuditError(f"duplicate image shard across sources: {image_id}")
            source_by_image[image_id] = (role, root / "image_shards" / image_id)
    for image_id, dataset_index in full_signature:
        role, shard_dir = source_by_image[image_id]
        result = validate_shard(shard_dir, image_id, dataset_index)
        if event_header is None:
            event_header = result["event_header"]
            curve_header = result["curve_header"]
        else:
            require_equal(result["event_header"], event_header, f"{image_id} event CSV header")
            require_equal(result["curve_header"], curve_header, f"{image_id} curve CSV header")
        total_trajectories += int(result["trajectories"])
        total_curve_rows += int(result["curve_rows"])
        shard_audit.append({**result, "source": role, "source_dir": str(shard_dir)})

    return {
        "protocol": "boundary_population_worker_merge_audit_v1",
        "status": "PASS_READY_TO_MERGE",
        "plan_root": str(plan_root),
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "parallel_protocol": protocol,
        "canonical_dir": str(canonical_dir),
        "canonical_existing_shards": len(canonical_ids),
        "superseded_canonical_partial": sorted(superseded_canonical_partials),
        "worker_count": len(worker_info),
        "workers": worker_info,
        "full_selection": str(full_selection),
        "full_selection_sha256": full_selection_hash,
        "selected_images": len(full_rows),
        "coverage_exactly_once": True,
        "total_trajectories": total_trajectories,
        "total_curve_rows": total_curve_rows,
        "contract": canonical_contract,
        "selected_order": [image_id for image_id, _ in full_signature],
        "source_by_image": {image_id: {"role": role, "shard_dir": str(path)} for image_id, (role, path) in source_by_image.items()},
        "shards": shard_audit,
    }


def powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def resume_command(audit: dict[str, Any], python: str, imgsz: int, device: str, workers: int) -> str:
    canonical = Path(audit["canonical_dir"])
    manifest = read_json(canonical / "manifest.json", "canonical manifest")
    selected_source = Path(str(manifest["selected_images_source"]))
    if not selected_source.is_absolute():
        selected_source = PROJECT_ROOT / selected_source
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "analyze_assignment_boundary_radius.py"),
        "--checkpoint", str(manifest["checkpoint"]),
        "--expected-sha256", str(manifest["checkpoint_sha256"]),
        "--data", str(manifest["data"]),
        "--selected-images", str(selected_source.resolve()),
        "--output-dir", str(canonical),
        "--rmax", str(manifest["rmax"]),
        "--coarse-step", str(manifest["coarse_step"]),
        "--fine-step", str(manifest["fine_step"]),
        "--search-mode", str(manifest["search_mode"]),
        "--replay-engine", str(manifest["replay_engine"]),
        "--replay-batch-size", str(manifest["replay_batch_size"]),
        "--imgsz", str(imgsz),
        "--device", str(device),
        "--workers", str(workers),
        "--resume",
    ]
    if manifest.get("incremental_assert_native"):
        args.append("--incremental-assert-native")
    max_focal = int(manifest.get("max_focal_gt_per_image", 0))
    if max_focal:
        args.extend(("--max-focal-gt-per-image", str(max_focal)))
    return "& " + " ".join(powershell_quote(value) for value in args)


def execute_merge(audit: dict[str, Any], python: str, imgsz: int, device: str, workers: int) -> dict[str, Any]:
    canonical = Path(audit["canonical_dir"])
    original = canonical / "image_shards"
    if not original.is_dir():
        raise AuditError(f"canonical image_shards disappeared before merge: {original}")
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    stage = canonical / f"image_shards.merge-stage-{token}"
    backup = canonical / f"image_shards.premerge-{token}"
    if stage.exists() or backup.exists():
        raise AuditError("merge stage/backup path collision")
    stage.mkdir()
    source_by_image = audit["source_by_image"]
    try:
        for image_id in audit["selected_order"]:
            source = Path(source_by_image[image_id]["shard_dir"])
            shutil.copytree(source, stage / image_id, copy_function=shutil.copy2)
        # A complete second validation occurs before the canonical directory is touched.
        expected_index = {item["image_id"]: int(item["dataset_index"]) for item in audit["shards"]}
        stage_ids = {path.name for path in stage.iterdir() if path.is_dir()}
        require_equal(stage_ids, set(audit["selected_order"]), "staged image coverage")
        trajectories = 0
        curve_rows = 0
        for image_id in audit["selected_order"]:
            result = validate_shard(stage / image_id, image_id, expected_index[image_id])
            trajectories += int(result["trajectories"])
            curve_rows += int(result["curve_rows"])
        require_equal(trajectories, int(audit["total_trajectories"]), "staged trajectories")
        require_equal(curve_rows, int(audit["total_curve_rows"]), "staged curve rows")
    except Exception:
        # Leave a failed stage in place for forensic inspection; no source was changed.
        raise
    os.replace(original, backup)
    try:
        os.replace(stage, original)
    except Exception:
        os.replace(backup, original)
        raise
    return {
        "status": "MERGED_AWAITING_CANONICAL_RESUME",
        "canonical_image_shards": str(original),
        "preserved_premerge_backup": str(backup),
        "superseded_canonical_partial": audit.get("superseded_canonical_partial", []),
        "worker_sources_preserved": [worker["output_dir"] for worker in audit["workers"]],
        "selected_images": audit["selected_images"],
        "total_trajectories": audit["total_trajectories"],
        "total_curve_rows": audit["total_curve_rows"],
        "resume_command": resume_command(audit, python, imgsz, device, workers),
    }


def verify_final(canonical: Path, expected_images: int) -> dict[str, Any]:
    canonical = canonical.resolve()
    manifest, progress, summary = validate_root_terminal(canonical, expected_images, "canonical final")
    rows = read_roster(canonical / "selected_images.csv", expected_images)
    order = [row["image_id"] for row in rows]
    position = {image_id: index for index, image_id in enumerate(order)}
    index_by_id = {row["image_id"]: int(row["dataset_index"]) for row in rows}
    shard_ids = enumerate_shards(canonical)
    require_equal(shard_ids, set(order), "final canonical shard coverage")
    markers = {}
    expected_event_header = None
    expected_curve_header = None
    for image_id in order:
        result = validate_shard(canonical / "image_shards" / image_id, image_id, index_by_id[image_id])
        markers[image_id] = result
        expected_event_header = expected_event_header or result["event_header"]
        expected_curve_header = expected_curve_header or result["curve_header"]
        require_equal(result["event_header"], expected_event_header, f"{image_id} event header")
        require_equal(result["curve_header"], expected_curve_header, f"{image_id} curve header")

    event_path = canonical / "per_direction_boundary_events.csv"
    event_keys: set[tuple[str, str, str]] = set()
    event_blocks: list[str] = []
    last_position = -1
    current_image = None
    with event_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_equal(tuple(reader.fieldnames or ()), expected_event_header, "final event header")
        event_rows = 0
        for row in reader:
            image_id = row["image_id"]
            if image_id not in position or position[image_id] < last_position:
                raise AuditError(f"final event CSV is not in selected-image order at {image_id}")
            if image_id != current_image:
                if image_id in event_blocks:
                    raise AuditError(f"non-contiguous final event image block: {image_id}")
                event_blocks.append(image_id)
                current_image = image_id
                last_position = position[image_id]
            key = (image_id, row["gt_id"], row["direction"])
            if key in event_keys:
                raise AuditError(f"duplicate final trajectory key: {key}")
            event_keys.add(key)
            event_rows += 1
    expected_event_rows = sum(int(item["trajectories"]) for item in markers.values())
    require_equal(event_rows, expected_event_rows, "final event-row total")

    curve_path = canonical / "o2m_rank_set_radius_curve.csv"
    curve_blocks: list[str] = []
    last_position = -1
    current_image = None
    current_key = None
    current_radii: set[str] = set()
    finished_keys: set[tuple[str, str, str]] = set()
    with curve_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_equal(tuple(reader.fieldnames or ()), expected_curve_header, "final curve header")
        curve_rows = 0
        for row in reader:
            image_id = row["image_id"]
            if image_id not in position or position[image_id] < last_position:
                raise AuditError(f"final curve CSV is not in selected-image order at {image_id}")
            if image_id != current_image:
                if image_id in curve_blocks:
                    raise AuditError(f"non-contiguous final curve image block: {image_id}")
                curve_blocks.append(image_id)
                current_image = image_id
                last_position = position[image_id]
            key = (image_id, row["gt_id"], row["direction"])
            if key not in event_keys:
                raise AuditError(f"final curve trajectory lacks event row: {key}")
            if key != current_key:
                if current_key is not None:
                    finished_keys.add(current_key)
                if key in finished_keys:
                    raise AuditError(f"non-contiguous final curve trajectory: {key}")
                current_key = key
                current_radii = set()
            if row["radius"] in current_radii:
                raise AuditError(f"duplicate final curve radius: {key} radius={row['radius']}")
            current_radii.add(row["radius"])
            curve_rows += 1
    expected_curve_rows = sum(int(item["curve_rows"]) for item in markers.values())
    require_equal(curve_rows, expected_curve_rows, "final curve-row total")
    require_equal(int(progress.get("trajectories", -1)), event_rows, "final progress trajectories")
    require_equal(int(progress.get("curve_rows", -1)), curve_rows, "final progress curve_rows")
    require_equal(int(summary.get("focal_direction_rows", -1)), event_rows, "final summary focal_direction_rows")
    require_equal(int(summary.get("radius_curve_rows", -1)), curve_rows, "final summary radius_curve_rows")
    return {
        "protocol": "boundary_population_final_root_validation_v1",
        "status": "PASS",
        "canonical_dir": str(canonical),
        "selected_images": len(order),
        "selected_order_valid": True,
        "unique_trajectory_keys": len(event_keys),
        "event_rows": event_rows,
        "curve_rows": curve_rows,
        "manifest_sha256": sha256(canonical / "manifest.json"),
        "event_csv_sha256": sha256(event_path),
        "curve_csv_sha256": sha256(curve_path),
    }


def concat_shards_for_selftest(canonical: Path, order: list[str]) -> None:
    for filename in ("per_direction_boundary_events.csv", "o2m_rank_set_radius_curve.csv"):
        destination = canonical / filename
        with destination.open("wb") as output:
            wrote_header = False
            for image_id in order:
                with (canonical / "image_shards" / image_id / filename).open("rb") as source:
                    header = source.readline()
                    if not wrote_header:
                        output.write(header)
                        wrote_header = True
                    output.writelines(source)


def make_selftest_fixture(root: Path) -> tuple[Path, Path]:
    plan_root = root / "plan"
    contracts = plan_root / "contracts"
    canonical = root / "canonical"
    contracts.mkdir(parents=True)
    (canonical / "image_shards").mkdir(parents=True)
    rows = [
        {"dataset_index": str(index), "image_id": f"image_{index:02d}", "stratum": "t_only"}
        for index in range(4)
    ]

    def write_csv(path: Path, selected: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dataset_index", "image_id", "stratum"))
            writer.writeheader()
            writer.writerows(selected)

    def write_shard(path: Path, row: dict[str, str]) -> None:
        path.mkdir(parents=True)
        event = path / "per_direction_boundary_events.csv"
        with event.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_id", "gt_id", "direction"))
            writer.writeheader()
            writer.writerow({"image_id": row["image_id"], "gt_id": "0", "direction": "left"})
        curve = path / "o2m_rank_set_radius_curve.csv"
        with curve.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_id", "gt_id", "direction", "radius"))
            writer.writeheader()
            writer.writerow({"image_id": row["image_id"], "gt_id": "0", "direction": "left", "radius": "0.0"})
            writer.writerow({"image_id": row["image_id"], "gt_id": "0", "direction": "left", "radius": "0.0009765625"})
        summary = {
            "status": "complete", "image_id": row["image_id"], "dataset_index": int(row["dataset_index"]),
            "total_gt": 1, "density": "low_lt32", "focal_gt": 1, "focal_gt_by_size_bin": {"t_8_16": 1},
            "excluded_base_box_outside_image_by_size_bin": {}, "audited_focal_gt_by_size_bin": {"t_8_16": 1},
            "trajectories": 1, "curve_rows": 2, "incremental_native_validation_scenarios": 0, "elapsed_sec": 1.0,
        }
        write_json(path / "summary.json", summary)
        marker = {**summary, "sha256": {filename: sha256(path / filename) for filename in SHARD_FILES}}
        write_json(path / "complete.json", marker)

    full_selection = plan_root / "full_selected_images.csv"
    write_csv(full_selection, rows)
    write_csv(canonical / "selected_images.csv", rows)
    scanner_hash = "a" * 64
    checkpoint_hash = "b" * 64
    canonical_manifest = {
        "status": "running", "protocol": "continuous_assignment_boundary_radius_full_grid_v2",
        "checkpoint": "checkpoint.pt", "checkpoint_sha256": checkpoint_hash, "data": "data.yaml",
        "selected_images_source": str(full_selection), "selected_images_source_sha256": sha256(full_selection),
        "audited_selected_images": 4, "rmax": 0.25, "coarse_step": 0.015625, "fine_step": 0.0009765625,
        "search_mode": "full-grid", "replay_batch_size": 64, "replay_engine": "focal-row",
        "incremental_assert_native": False, "max_focal_gt_per_image": 0, "script_sha256": scanner_hash, "smoke": False,
    }
    write_json(canonical / "manifest.json", canonical_manifest)
    write_shard(canonical / "image_shards" / rows[0]["image_id"], rows[0])
    # Model an interrupted canonical image that is reassigned to worker 1.
    interrupted = canonical / "image_shards" / f"{rows[1]['image_id']}.partial"
    interrupted.mkdir()
    (interrupted / "incomplete.txt").write_text("preserve in premerge backup\n", encoding="utf-8")
    workers = []
    for number, row in enumerate(rows[1:], start=1):
        worker_id = f"worker_{number:02d}"
        worker_root = plan_root / worker_id
        selection = contracts / f"{worker_id}_selected_images.csv"
        write_csv(selection, [row])
        write_csv(worker_root / "selected_images.csv", [row])
        (worker_root / "image_shards").mkdir(parents=True)
        write_shard(worker_root / "image_shards" / row["image_id"], row)
        worker_manifest = {**canonical_manifest, "status": "complete", "selected_images_source": str(selection), "selected_images_source_sha256": sha256(selection), "audited_selected_images": 1}
        write_json(worker_root / "manifest.json", worker_manifest)
        write_json(worker_root / "progress.json", {"status": "complete", "processed_images": 1, "total_images": 1, "trajectories": 1, "curve_rows": 2})
        write_json(worker_root / "summary.json", {"status": "complete", "selected_images": 1, "processed_images": 1, "focal_direction_rows": 1, "radius_curve_rows": 2})
        status_path = plan_root / f"{worker_id}_status.json"
        write_json(status_path, {"status": "complete"})
        contract = {
            "protocol": "continuous_assignment_boundary_radius_full_grid_v2", "worker_id": worker_id,
            "selected_images": str(selection), "selected_images_sha256": sha256(selection),
            "full_selected_images_sha256": sha256(full_selection), "output_dir": str(worker_root),
            "status_path": str(status_path), "assigned_images": 1,
            "assigned": [{"image_id": row["image_id"], "dataset_index": int(row["dataset_index"])}],
            "checkpoint_sha256": checkpoint_hash, "scanner_sha256": scanner_hash,
            "fine_step": 0.0009765625, "replay_engine": "focal-row", "replay_batch_size": 64,
        }
        contract_path = contracts / f"{worker_id}_contract.json"
        write_json(contract_path, contract)
        workers.append({"worker_id": worker_id, "contract": str(contract_path), "contract_sha256": sha256(contract_path)})
    plan = {
        "protocol": "boundary_population_parallel_plan_v1", "status": "complete",
        "canonical_output": str(canonical), "canonical_manifest_sha256": sha256(canonical / "manifest.json"),
        "full_selection": str(full_selection), "full_selection_sha256": sha256(full_selection),
        "completed_image_ids": [rows[0]["image_id"]], "remaining_image_ids": [row["image_id"] for row in rows[1:]],
        "run_contract": {"checkpoint_sha256": checkpoint_hash, "scanner_sha256": scanner_hash, "fine_step": 0.0009765625, "replay_engine": "focal-row", "replay_batch_size": 64},
        "workers": workers, "checks": {"complete_assignment": True, "mutually_exclusive": True, "no_overlap_with_canonical": True},
    }
    write_json(contracts / "parallel_plan.json", plan)
    return plan_root, canonical


def expect_rejection(root: Path, mutation: str) -> bool:
    plan_root, canonical = make_selftest_fixture(root)
    plan_path = plan_root / "contracts" / "parallel_plan.json"
    plan = read_json(plan_path)
    worker_entry = plan["workers"][0]
    contract_path = Path(worker_entry["contract"])
    contract = read_json(contract_path)
    worker_root = Path(contract["output_dir"])
    if mutation == "duplicate":
        source = worker_root / "image_shards" / "image_01"
        shutil.copytree(source, canonical / "image_shards" / "image_01")
    elif mutation == "missing":
        target = worker_root / "image_shards" / "image_01"
        if root not in target.parents:
            raise AssertionError("unsafe self-test removal target")
        shutil.rmtree(target)
    elif mutation == "hash":
        with (worker_root / "image_shards" / "image_01" / "per_direction_boundary_events.csv").open("a", encoding="utf-8") as handle:
            handle.write("tamper\n")
    elif mutation == "contract":
        contract["fine_step"] = 0.001953125
        write_json(contract_path, contract)
        worker_entry["contract_sha256"] = sha256(contract_path)
        write_json(plan_path, plan)
    elif mutation == "incomplete":
        manifest_path = worker_root / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
    elif mutation == "worker_partial":
        (worker_root / "image_shards" / "unrelated.partial").mkdir()
    elif mutation == "orphan_partial":
        (canonical / "image_shards" / "orphan.partial").mkdir()
    else:
        raise AssertionError(mutation)
    try:
        audit_merge(plan_root, canonical, 3, 4)
    except AuditError:
        return True
    return False


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="boundary-merge-selftest-") as temporary:
        root = Path(temporary).resolve()
        mixed_assigned = [("a_image", 1), ("P_image", 2), ("b_image", 3)]
        mixed_scanner_order = sorted(mixed_assigned, key=lambda item: item[0])
        validate_worker_root_roster(mixed_scanner_order, mixed_assigned, "mixed_case")
        wrong_order_rejected = False
        try:
            validate_worker_root_roster(mixed_assigned, mixed_assigned, "mixed_case")
        except AuditError:
            wrong_order_rejected = True
        plan_root, canonical = make_selftest_fixture(root / "success")
        audit = audit_merge(plan_root, canonical, 3, 4)
        merge = execute_merge(audit, sys.executable, 800, "cpu", 0)
        backup = Path(merge["preserved_premerge_backup"])
        if not backup.is_dir():
            raise AssertionError("premerge backup was not preserved")
        if not (backup / "image_01.partial" / "incomplete.txt").is_file():
            raise AssertionError("superseded canonical partial was not preserved in backup")
        if not all(Path(path).is_dir() for path in merge["worker_sources_preserved"]):
            raise AssertionError("worker source was not preserved")
        concat_shards_for_selftest(canonical, audit["selected_order"])
        manifest = read_json(canonical / "manifest.json")
        manifest["status"] = "complete"
        write_json(canonical / "manifest.json", manifest)
        write_json(canonical / "progress.json", {"status": "complete", "processed_images": 4, "total_images": 4, "trajectories": 4, "curve_rows": 8})
        write_json(canonical / "summary.json", {"status": "complete", "selected_images": 4, "processed_images": 4, "focal_direction_rows": 4, "radius_curve_rows": 8})
        final = verify_final(canonical, 4)
        checks = {
            "success_audit": audit["status"] == "PASS_READY_TO_MERGE",
            "superseded_partial_recorded": audit["superseded_canonical_partial"] == ["image_01"],
            "atomic_copy_merge": merge["status"] == "MERGED_AWAITING_CANONICAL_RESUME",
            "final_root_validation": final["status"] == "PASS",
            "duplicate_rejected": expect_rejection(root / "duplicate", "duplicate"),
            "missing_rejected": expect_rejection(root / "missing", "missing"),
            "hash_mismatch_rejected": expect_rejection(root / "hash", "hash"),
            "contract_mismatch_rejected": expect_rejection(root / "contract", "contract"),
            "incomplete_rejected": expect_rejection(root / "incomplete", "incomplete"),
            "worker_partial_rejected": expect_rejection(root / "worker_partial", "worker_partial"),
            "orphan_partial_rejected": expect_rejection(root / "orphan_partial", "orphan_partial"),
            "mixed_case_scanner_order_accepted": mixed_scanner_order[0][0] == "P_image",
            "non_scanner_worker_order_rejected": wrong_order_rejected,
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        return {"status": "PASS", "checks": checks}


def main() -> int:
    args = parse_args()
    self_test = run_self_test() if args.self_test else None
    if args.plan_root is None:
        if self_test is None:
            raise AuditError("--plan-root is required unless only --self-test is requested")
        print(json.dumps(self_test, sort_keys=True))
        return 0
    plan_root = args.plan_root.resolve()
    report_dir = (args.report_dir or (plan_root / "merge_validation")).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.execute and args.verify_final:
            raise AuditError("--execute and --verify-final are separate phases")
        if args.verify_final:
            plan = read_json(plan_root / "contracts" / "parallel_plan.json", "parallel plan")
            canonical_value = plan.get("canonical_output")
            if isinstance(canonical_value, dict):
                canonical_value = canonical_value.get("path") or canonical_value.get("directory")
            canonical = args.canonical_dir.resolve() if args.canonical_dir else resolve_path(canonical_value, plan_root)
            final = verify_final(canonical, args.expected_images)
            if self_test is not None:
                final["self_test"] = self_test
            write_json(report_dir / "final_validation.json", final)
            print(json.dumps({"status": final["status"], "report_dir": str(report_dir)}, sort_keys=True))
            return 0
        audit = audit_merge(plan_root, args.canonical_dir, args.expected_workers, args.expected_images)
        if self_test is not None:
            audit["self_test"] = self_test
        write_json(report_dir / "premerge_audit.json", audit)
        if args.execute:
            merge = execute_merge(audit, args.python, args.imgsz, args.device, args.workers)
            write_json(report_dir / "merge_receipt.json", merge)
            (report_dir / "canonical_resume_command.ps1").write_text(
                "Set-Location " + powershell_quote(PROJECT_ROOT) + "\n" + merge["resume_command"] + "\n",
                encoding="utf-8",
            )
            print(json.dumps({"status": merge["status"], "report_dir": str(report_dir)}, sort_keys=True))
            return 0
        print(json.dumps({"status": audit["status"], "report_dir": str(report_dir)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        write_json(report_dir / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
