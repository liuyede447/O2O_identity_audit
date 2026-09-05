"""Prepare one outcome-blind disjoint stratified audit sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_reviewer_killer_controls import image_stratum


SELECTION_TAG = "audit_lockbox_selection_v1.0"
SELECTION_FILES = {
    "selected_images.csv",
    "selection_manifest.json",
    "selection_manifest.json.sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--exclude-selected", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-images", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--instrument-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        required = (
            "data",
            "exclude_selected",
            "output_dir",
            "seed",
            "preregistration",
            "instrument_manifest",
        )
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error("the following arguments are required unless --self-test is used: " + ", ".join(missing))
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_text(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_selection_repository(directory: Path) -> dict[str, str]:
    if not (directory / ".git").is_dir():
        raise RuntimeError("selection directory is not a Git repository")
    head = git_text(directory, "rev-parse", "HEAD").lower()
    tag_commit = git_text(directory, "rev-list", "-n", "1", SELECTION_TAG).lower()
    if tag_commit != head:
        raise RuntimeError(f"{SELECTION_TAG} does not resolve to the selection HEAD")
    if git_text(directory, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("selection repository must be clean")
    for name in sorted(SELECTION_FILES):
        path = directory / name
        committed = subprocess.run(
            ["git", "-C", str(directory), "show", f"{head}:{name}"],
            check=True,
            capture_output=True,
        ).stdout
        if not path.is_file() or hashlib.sha256(committed).hexdigest() != sha256(path):
            raise RuntimeError(f"committed selection artifact differs from the worktree: {name}")
    return {"selection_commit": head, "selection_tag": SELECTION_TAG}


def seal_selection_repository(directory: Path) -> dict[str, str]:
    if (directory / ".git").exists():
        raise RuntimeError("selection repository already exists")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != SELECTION_FILES:
        raise RuntimeError(
            f"selection bundle must contain exactly {sorted(SELECTION_FILES)}, got {sorted(actual)}"
        )
    subprocess.run(["git", "init", str(directory)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(directory), "config", "core.autocrlf", "false"], check=True)
    subprocess.run(["git", "-C", str(directory), "config", "user.email", "selection-freeze@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(directory), "config", "user.name", "O2O audit selection freeze"], check=True)
    subprocess.run(["git", "-C", str(directory), "add", *sorted(SELECTION_FILES)], check=True)
    subprocess.run(
        ["git", "-C", str(directory), "commit", "-m", "Freeze disjoint audit-outcome selection v1"],
        check=True,
        capture_output=True,
    )
    head = git_text(directory, "rev-parse", "HEAD").lower()
    subprocess.run(["git", "-C", str(directory), "tag", SELECTION_TAG, head], check=True)
    return validate_selection_repository(directory)


def sidecar_path(path: Path) -> Path:
    return Path(str(path) + ".sha256")


def read_verified_json(path: Path, label: str) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if not fields or len(fields[0]) != 64 or any(character not in "0123456789abcdef" for character in fields[0].lower()):
        raise RuntimeError(f"{label} SHA-256 sidecar is malformed")
    expected = fields[0].lower()
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 sidecar mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload, actual


def require_frozen_at(payload: dict, label: str) -> str:
    value = payload.get("frozen_at_utc")
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} frozen_at_utc is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} frozen_at_utc must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"{label} frozen_at_utc must include a UTC offset")
    return value


def validate_frozen_inputs(
    preregistration: Path,
    instrument_manifest: Path,
    *,
    sample_images: int,
    seed: int,
) -> tuple[dict, str, dict, str, dict[str, int]]:
    prereg, prereg_hash = read_verified_json(preregistration, "preregistration")
    if prereg.get("status") != "FROZEN_BEFORE_OUTCOMES":
        raise RuntimeError("preregistration status must be FROZEN_BEFORE_OUTCOMES")
    require_frozen_at(prereg, "preregistration")
    sample = prereg.get("sample")
    if not isinstance(sample, dict):
        raise RuntimeError("preregistration sample definition is missing")
    if sample.get("size_images") != sample_images:
        raise RuntimeError("preregistration sample size does not match --sample-images")
    if sample.get("selection_seed") != seed:
        raise RuntimeError("preregistration selection seed does not match --seed")
    allocation = sample.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise RuntimeError("preregistration allocation is missing")
    if any(not isinstance(key, str) or not isinstance(value, int) or value < 0 for key, value in allocation.items()):
        raise RuntimeError("preregistration allocation must contain non-negative integer counts")
    if sum(allocation.values()) != sample_images:
        raise RuntimeError("preregistration allocation does not sum to --sample-images")

    instrument, instrument_hash = read_verified_json(instrument_manifest, "instrument manifest")
    if instrument.get("status") != "FROZEN_BEFORE_LOCKBOX_SELECTION":
        raise RuntimeError("instrument manifest status must be FROZEN_BEFORE_LOCKBOX_SELECTION")
    require_frozen_at(instrument, "instrument manifest")
    source_commit = instrument.get("instrument_source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit.lower()
    ):
        raise RuntimeError("instrument_source_commit must be a full 40-character Git hash")
    if instrument.get("instrument_source_tag") != "audit_instrument_v1.0":
        raise RuntimeError("instrument_source_tag must be audit_instrument_v1.0")
    if prereg.get("instrument_source_commit") != source_commit:
        raise RuntimeError("preregistration and instrument source commits differ")
    if instrument.get("preregistration_sha256") != prereg_hash:
        raise RuntimeError("instrument manifest does not bind the supplied preregistration hash")
    if instrument.get("selection_seed") != seed:
        raise RuntimeError("instrument manifest selection seed does not match --seed")
    return prereg, prereg_hash, instrument, instrument_hash, dict(allocation)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def allocation_from_exclusion(rows: list[dict], requested: int) -> dict[str, int]:
    counts = Counter(str(row["stratum"]) for row in rows)
    if sum(counts.values()) != requested:
        raise ValueError("exclusion manifest does not match requested sample size")
    return dict(counts)


def validate_data_yaml_binding(data_path: Path, instrument: dict) -> str:
    data_yaml = instrument.get("data_yaml")
    if not isinstance(data_yaml, dict):
        raise RuntimeError("instrument manifest data_yaml contract is missing")
    expected = data_yaml.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected.lower()
    ):
        raise RuntimeError("instrument manifest data_yaml SHA-256 is malformed")
    actual = sha256(data_path)
    if actual != expected.lower():
        raise RuntimeError("--data SHA-256 differs from the frozen instrument manifest")
    return actual


def canonical_eligible_population_bytes(rows: list[dict]) -> bytes:
    normalized = []
    for row in rows:
        image_id = row.get("image_id")
        stratum = row.get("stratum")
        if not isinstance(image_id, str) or not image_id or not isinstance(stratum, str) or not stratum:
            raise RuntimeError("eligible population rows require non-empty image_id and stratum strings")
        normalized.append({"image_id": image_id, "stratum": stratum})
    normalized.sort(key=lambda row: (row["image_id"], row["stratum"]))
    image_ids = [row["image_id"] for row in normalized]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError("eligible population contains duplicate image_id values")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def eligible_population_contract(rows: list[dict]) -> dict:
    canonical_bytes = canonical_eligible_population_bytes(rows)
    normalized = json.loads(canonical_bytes.decode("utf-8"))
    strata_counts = Counter(row["stratum"] for row in normalized)
    return {
        "protocol": "eligible_population_roster_v1",
        "sha256": sha256_bytes(canonical_bytes),
        "count": len(normalized),
        "strata_counts": dict(sorted(strata_counts.items())),
    }


def validate_eligible_population_binding(instrument: dict, actual: dict) -> None:
    expected = instrument.get("eligible_population_roster")
    if not isinstance(expected, dict):
        raise RuntimeError("instrument manifest eligible population roster contract is missing")
    required = ("protocol", "sha256", "count", "strata_counts")
    missing = [field for field in required if field not in expected]
    if missing:
        raise RuntimeError("instrument manifest eligible population roster contract is incomplete")
    if expected.get("protocol") != "eligible_population_roster_v1":
        raise RuntimeError("instrument manifest eligible population roster protocol is unsupported")
    if any(expected.get(field) != actual.get(field) for field in required):
        raise RuntimeError("eligible population roster differs from the frozen instrument manifest")


def _load_selection_dataset(data_path: Path, imgsz: int):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_path))
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": imgsz,
            "batch": 1,
            "workers": 0,
            "rect": False,
            "cache": False,
            "fraction": 1.0,
        }
    )
    return build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)


def _eligible_population_rows(dataset, imgsz: int) -> tuple[list[dict], dict[str, int]]:
    rows = []
    dataset_indices = {}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, imgsz)
        if stratum is None:
            continue
        image_id = Path(label["im_file"]).stem
        if image_id in dataset_indices:
            raise RuntimeError("eligible population contains duplicate image_id values")
        dataset_indices[image_id] = index
        rows.append({"image_id": image_id, "stratum": stratum})
    rows.sort(key=lambda row: (row["image_id"], row["stratum"]))
    return rows, dataset_indices


def build_eligible_population_contract(data_path: Path, imgsz: int = 800) -> tuple[list[dict], dict]:
    dataset = _load_selection_dataset(data_path, imgsz)
    rows, _ = _eligible_population_rows(dataset, imgsz)
    return rows, eligible_population_contract(rows)


def run_self_test() -> dict[str, object]:
    frozen_at = "2026-08-31T08:00:00+00:00"
    commit = "1" * 40
    allocation = {"t_only": 160, "s_only": 2, "t_and_s": 138}
    prereg = {
        "status": "FROZEN_BEFORE_OUTCOMES",
        "frozen_at_utc": frozen_at,
        "instrument_source_commit": commit,
        "sample": {"size_images": 300, "selection_seed": 20260831, "allocation": allocation},
    }

    def write_with_sidecar(path: Path, payload: dict) -> str:
        path.write_text(json.dumps(payload), encoding="utf-8")
        digest = sha256(path)
        sidecar_path(path).write_text(f"{digest}  {path.name}\n", encoding="ascii")
        return digest

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        data_path = base / "data.yaml"
        data_path.write_text("path: frozen-dataset\nval: images/val\n", encoding="utf-8")
        roster_rows = [
            {"image_id": "image-001", "stratum": "t_only"},
            {"image_id": "image-002", "stratum": "s_only"},
            {"image_id": "image-003", "stratum": "t_and_s"},
        ]
        roster_contract = eligible_population_contract(roster_rows)
        prereg_path = base / "CONFIRMATORY_AUDIT_PLAN.json"
        prereg_hash = write_with_sidecar(prereg_path, prereg)
        instrument = {
            "status": "FROZEN_BEFORE_LOCKBOX_SELECTION",
            "frozen_at_utc": frozen_at,
            "instrument_source_commit": commit,
            "instrument_source_tag": "audit_instrument_v1.0",
            "preregistration_sha256": prereg_hash,
            "selection_seed": 20260831,
            "data_yaml": {"path": str(data_path), "sha256": sha256(data_path)},
            "eligible_population_roster": roster_contract,
        }
        instrument_path = base / "TOOL_FREEZE_MANIFEST.json"
        write_with_sidecar(instrument_path, instrument)
        validate_frozen_inputs(
            prereg_path,
            instrument_path,
            sample_images=300,
            seed=20260831,
        )
        validate_data_yaml_binding(data_path, instrument)
        validate_eligible_population_binding(instrument, roster_contract)

        sidecar_path(prereg_path).write_text(f"{'0' * 64}  {prereg_path.name}\n", encoding="ascii")
        try:
            validate_frozen_inputs(prereg_path, instrument_path, sample_images=300, seed=20260831)
        except RuntimeError as exc:
            assert "sidecar mismatch" in str(exc)
        else:
            raise AssertionError("bad preregistration hash was accepted")

        write_with_sidecar(prereg_path, {**prereg, "status": "DRAFT"})
        try:
            validate_frozen_inputs(prereg_path, instrument_path, sample_images=300, seed=20260831)
        except RuntimeError as exc:
            assert "status" in str(exc)
        else:
            raise AssertionError("bad preregistration status was accepted")

        prereg_hash = write_with_sidecar(prereg_path, prereg)
        instrument["preregistration_sha256"] = prereg_hash
        write_with_sidecar(instrument_path, {**instrument, "status": "DRAFT"})
        try:
            validate_frozen_inputs(prereg_path, instrument_path, sample_images=300, seed=20260831)
        except RuntimeError as exc:
            assert "status" in str(exc)
        else:
            raise AssertionError("bad instrument status was accepted")

        write_with_sidecar(instrument_path, instrument)
        try:
            validate_frozen_inputs(prereg_path, instrument_path, sample_images=299, seed=20260831)
        except RuntimeError as exc:
            assert "sample size" in str(exc)
        else:
            raise AssertionError("mismatched CLI sample size was accepted")

        try:
            validate_frozen_inputs(prereg_path, instrument_path, sample_images=300, seed=7)
        except RuntimeError as exc:
            assert "selection seed" in str(exc)
        else:
            raise AssertionError("mismatched CLI seed was accepted")

        original_data = data_path.read_bytes()
        data_path.write_bytes(original_data + b"# tampered\n")
        try:
            validate_data_yaml_binding(data_path, instrument)
        except RuntimeError as exc:
            assert "--data SHA-256" in str(exc)
        else:
            raise AssertionError("wrong data YAML hash was accepted")
        data_path.write_bytes(original_data)

        tampered_rows = [dict(row) for row in roster_rows]
        tampered_rows[0]["image_id"] = "image-tampered"
        try:
            validate_eligible_population_binding(instrument, eligible_population_contract(tampered_rows))
        except RuntimeError as exc:
            assert "eligible population roster differs" in str(exc)
        else:
            raise AssertionError("tampered eligible population roster was accepted")

        selection_bundle = base / "selection_bundle"
        selection_bundle.mkdir()
        (selection_bundle / "selected_images.csv").write_text(
            "image_id,stratum\nimage-001,t_only\n", encoding="utf-8", newline="\n"
        )
        selection_manifest = selection_bundle / "selection_manifest.json"
        selection_manifest.write_text("{}\n", encoding="utf-8", newline="\n")
        sidecar_path(selection_manifest).write_text(
            f"{sha256(selection_manifest)}  {selection_manifest.name}\n",
            encoding="ascii",
            newline="\n",
        )
        seal = seal_selection_repository(selection_bundle)
        assert seal["selection_tag"] == SELECTION_TAG
        validate_selection_repository(selection_bundle)
        (selection_bundle / "selected_images.csv").write_text(
            "image_id,stratum\ntampered,t_only\n", encoding="utf-8", newline="\n"
        )
        try:
            validate_selection_repository(selection_bundle)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered sealed selection was accepted")
    return {"status": "PASS", "tests": 9}


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
        return
    if args.sample_images < 1:
        raise ValueError("--sample-images must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    # These cryptographic and protocol gates deliberately run before either the
    # dataset or the discovery-selection exclusion manifest is opened.
    _, prereg_hash, instrument, instrument_hash, frozen_allocation = validate_frozen_inputs(
        args.preregistration,
        args.instrument_manifest,
        sample_images=args.sample_images,
        seed=args.seed,
    )
    from create_instrument_freeze_bundle import (
        validate_frozen_script_identity,
        validate_preregistration_seal,
    )

    preregistration_seal = validate_preregistration_seal(
        args.instrument_manifest.resolve().parent
    )
    validate_frozen_script_identity(
        args.instrument_manifest.resolve().parent,
        Path(__file__),
        use_tool_manifest=True,
    )
    if not args.data.is_file() or not args.exclude_selected.is_file():
        raise FileNotFoundError("data or exclusion manifest")
    data_yaml_hash = validate_data_yaml_binding(args.data, instrument)
    if instrument.get("discovery_exclusion_manifest_sha256") != sha256(args.exclude_selected):
        raise RuntimeError("exclusion manifest hash differs from the frozen instrument manifest")
    with args.exclude_selected.open(encoding="utf-8", newline="") as stream:
        exclusion_rows = list(csv.DictReader(stream))
    allocation = allocation_from_exclusion(exclusion_rows, args.sample_images)
    if allocation != frozen_allocation:
        raise RuntimeError("exclusion-manifest allocation differs from the preregistration")
    excluded_ids = {str(row["image_id"]) for row in exclusion_rows}

    dataset = _load_selection_dataset(args.data, args.imgsz)
    eligible_rows, dataset_indices = _eligible_population_rows(dataset, args.imgsz)
    actual_roster_contract = eligible_population_contract(eligible_rows)
    validate_eligible_population_binding(instrument, actual_roster_contract)
    remaining = {key: [] for key in allocation}
    full_counts = Counter()
    for row in eligible_rows:
        image_id = row["image_id"]
        stratum = row["stratum"]
        full_counts[stratum] += 1
        if image_id not in excluded_ids and stratum in remaining:
            remaining[stratum].append((dataset_indices[image_id], image_id))
    for stratum, required in allocation.items():
        if len(remaining[stratum]) < required:
            raise RuntimeError(f"insufficient disjoint images in {stratum}")

    rng = random.Random(args.seed)
    selected_rows = []
    for stratum in sorted(allocation):
        chosen = rng.sample(remaining[stratum], allocation[stratum])
        probability = allocation[stratum] / len(remaining[stratum])
        for index, image_id in chosen:
            selected_rows.append({
                "dataset_index": index, "image_id": image_id, "stratum": stratum,
                "inclusion_probability": probability, "sampling_weight": 1.0 / probability,
            })
    selected_rows.sort(key=lambda row: row["image_id"])
    selected_ids = {row["image_id"] for row in selected_rows}
    if selected_ids & excluded_ids or len(selected_ids) != args.sample_images:
        raise RuntimeError("disjointness or uniqueness assertion failed")

    args.output_dir.mkdir(parents=True)
    selected_path = args.output_dir / "selected_images.csv"
    write_rows(selected_path, selected_rows)
    manifest = {
        "status": "FROZEN_SELECTION_BEFORE_OUTCOMES",
        "protocol": "disjoint_audit_outcome_lockbox_selection_v1",
        "outcome_blind": True,
        "outcome_accessed": False,
        "seed": args.seed,
        "requested_images": args.sample_images,
        "allocation": allocation,
        "full_population_counts": dict(full_counts),
        "remaining_population_counts": {key: len(value) for key, value in remaining.items()},
        "excluded_images": len(excluded_ids),
        "disjoint_overlap": 0,
        "preregistration_sha256": prereg_hash,
        "instrument_manifest_sha256": instrument_hash,
        "instrument_source_commit": instrument["instrument_source_commit"],
        "instrument_source_tag": instrument["instrument_source_tag"],
        "preregistration_commit": preregistration_seal["preregistration_commit"],
        "preregistration_tag": preregistration_seal["preregistration_tag"],
        "source_sha256": {
            "dataset_config": data_yaml_hash,
            "exclusion_manifest": sha256(args.exclude_selected),
            "selection_script": sha256(Path(__file__)),
        },
        "eligible_population_roster": actual_roster_contract,
        "selected_images_sha256": sha256(selected_path),
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = sha256(manifest_path)
    sidecar_path(manifest_path).write_text(f"{manifest_hash}  {manifest_path.name}\n", encoding="ascii")
    selection_seal = seal_selection_repository(args.output_dir)
    print(json.dumps({
        "selected_image_count": len(selected_rows),
        "allocation": allocation,
        "selected_images_sha256": manifest["selected_images_sha256"],
        "selection_manifest_sha256": manifest_hash,
        "preregistration_sha256": prereg_hash,
        "instrument_manifest_sha256": instrument_hash,
        "selection_commit": selection_seal["selection_commit"],
        "selection_tag": selection_seal["selection_tag"],
    }, indent=2))


if __name__ == "__main__": main()
