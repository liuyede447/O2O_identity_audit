"""Validate the current evidence ledger before manuscript population."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = {
    "claim_id",
    "evidence_id",
    "claim",
    "dataset",
    "checkpoint_sha256",
    "seed",
    "unit",
    "denominator",
    "estimand",
    "stress_contract",
    "sampling",
    "weighting",
    "uncertainty",
    "estimates",
    "multiplicity_status",
    "evidence_stage_status",
    "artifact",
    "artifact_sha256",
    "script_sha256",
    "manifest_path",
    "manifest_sha256",
    "validity_notes",
    "missing_reason",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hash_value(value, field: str, errors: list[str], evidence_id: str) -> None:
    if value is None:
        return
    values = value.values() if isinstance(value, dict) else [value]
    for item in values:
        if not isinstance(item, str) or not HEX64.fullmatch(item.lower()):
            errors.append(f"{evidence_id}: {field} contains a non-SHA256 value")


def validate(ledger_path: Path, root: Path) -> dict:
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ledger entries must be a list")
    errors: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index}: not an object")
            continue
        evidence_id = str(entry.get("evidence_id", f"entry_{index}"))
        missing = REQUIRED.difference(entry)
        if missing:
            errors.append(f"{evidence_id}: missing fields {sorted(missing)}")
        if evidence_id in seen:
            errors.append(f"{evidence_id}: duplicate evidence_id")
        seen.add(evidence_id)
        if entry.get("claim_id") != evidence_id:
            errors.append(f"{evidence_id}: claim_id must equal evidence_id")
        artifact_text = entry.get("artifact")
        artifact_hash = entry.get("artifact_sha256")
        if not isinstance(artifact_text, str) or not artifact_text:
            errors.append(f"{evidence_id}: missing artifact path")
        else:
            artifact = root / artifact_text
            if not artifact.is_file():
                errors.append(f"{evidence_id}: artifact does not exist: {artifact_text}")
            elif not isinstance(artifact_hash, str) or not HEX64.fullmatch(artifact_hash.lower()):
                errors.append(f"{evidence_id}: invalid artifact_sha256")
            elif sha256(artifact) != artifact_hash.lower():
                errors.append(f"{evidence_id}: artifact SHA-256 mismatch")
            lowered = artifact_text.lower()
            if any(token in lowered for token in ("smoke", "incomplete")):
                errors.append(f"{evidence_id}: smoke/incomplete artifact cannot be evidence")
        manifest_text = entry.get("manifest_path")
        manifest_hash = entry.get("manifest_sha256")
        if isinstance(manifest_text, dict):
            if not isinstance(manifest_hash, dict) or set(manifest_text) != set(manifest_hash):
                errors.append(f"{evidence_id}: manifest path/hash mappings must have identical keys")
            else:
                for key, relative in manifest_text.items():
                    manifest = root / str(relative)
                    expected = manifest_hash[key]
                    if not manifest.is_file():
                        errors.append(f"{evidence_id}: manifest does not exist for {key}: {relative}")
                    elif not isinstance(expected, str) or not HEX64.fullmatch(expected.lower()):
                        errors.append(f"{evidence_id}: invalid manifest_sha256 for {key}")
                    elif sha256(manifest) != expected.lower():
                        errors.append(f"{evidence_id}: manifest SHA-256 mismatch for {key}")
        elif manifest_text is not None:
            manifest = root / str(manifest_text)
            if not manifest.is_file():
                errors.append(f"{evidence_id}: manifest does not exist: {manifest_text}")
            elif not isinstance(manifest_hash, str) or not HEX64.fullmatch(manifest_hash.lower()):
                errors.append(f"{evidence_id}: invalid manifest_sha256")
            elif sha256(manifest) != manifest_hash.lower():
                errors.append(f"{evidence_id}: manifest SHA-256 mismatch")
        elif manifest_hash is not None:
            errors.append(f"{evidence_id}: manifest hash supplied without a manifest path")
        validate_hash_value(entry.get("script_sha256"), "script_sha256", errors, evidence_id)
    return {
        "status": "PASS" if not errors else "FAIL",
        "ledger": str(ledger_path),
        "entries": len(entries),
        "errors": errors,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "result.json"
        artifact.write_text("{}\n", encoding="utf-8")
        entry = {key: None for key in REQUIRED}
        entry.update({
            "claim_id": "E1",
            "evidence_id": "E1",
            "claim": "synthetic",
            "artifact": "result.json",
            "artifact_sha256": sha256(artifact),
            "evidence_stage_status": "self_test",
            "validity_notes": "synthetic",
            "missing_reason": {},
        })
        ledger = root / "ledger.json"
        ledger.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
        result = validate(ledger, root)
        assert result["status"] == "PASS", result
        entry["artifact_sha256"] = "0" * 64
        ledger.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
        assert validate(ledger, root)["status"] == "FAIL"
    print(json.dumps({"self_test": "PASS"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.ledger is None:
        raise ValueError("--ledger is required outside self-test")
    result = validate(args.ledger, args.root)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
