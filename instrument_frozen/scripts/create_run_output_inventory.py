"""Create or validate a recursive SHA-256 inventory for a completed run artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile


INVENTORY_NAME = "OUTPUT_INVENTORY.json"
SIDECAR_NAME = "OUTPUT_INVENTORY.json.sha256"
SUMS_NAME = "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(directory: Path) -> list[dict[str, object]]:
    excluded = {INVENTORY_NAME, SIDECAR_NAME, SUMS_NAME}
    rows = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        rows.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def validate(directory: Path) -> dict[str, object]:
    inventory_path = directory / INVENTORY_NAME
    sidecar_path = directory / SIDECAR_NAME
    sums_path = directory / SUMS_NAME
    if not inventory_path.is_file() or not sidecar_path.is_file() or not sums_path.is_file():
        raise FileNotFoundError("output inventory, sidecar, or SHA256SUMS is missing")
    actual_inventory_hash = sha256(inventory_path)
    tokens = sidecar_path.read_text(encoding="ascii").strip().split()
    if not tokens or tokens[0].lower() != actual_inventory_hash:
        raise RuntimeError("output inventory sidecar is malformed or mismatched")
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise RuntimeError("output inventory status is not complete")
    actual_rows = collect(directory)
    if payload.get("files") != actual_rows:
        raise RuntimeError("output inventory no longer matches the artifact directory")
    if payload.get("file_count") != len(actual_rows):
        raise RuntimeError("output inventory file_count is inconsistent")
    if payload.get("total_bytes") != sum(int(row["bytes"]) for row in actual_rows):
        raise RuntimeError("output inventory total_bytes is inconsistent")
    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in actual_rows
    ) + f"{actual_inventory_hash}  {INVENTORY_NAME}\n"
    if sums_path.read_text(encoding="ascii") != expected_sums:
        raise RuntimeError("SHA256SUMS.txt differs from the recursive output inventory")
    return {
        "status": "PASS",
        "inventory_sha256": actual_inventory_hash,
        "file_count": len(actual_rows),
        "total_bytes": payload["total_bytes"],
    }


def create(directory: Path, run_id: str) -> dict[str, object]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    targets = [directory / name for name in (INVENTORY_NAME, SIDECAR_NAME, SUMS_NAME)]
    if any(path.exists() for path in targets):
        raise FileExistsError("output inventory artifacts already exist")
    rows = collect(directory)
    if not rows:
        raise RuntimeError("cannot inventory an empty artifact directory")
    payload = {
        "status": "complete",
        "protocol": "recursive_run_output_sha256_inventory_v1",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
        "generator_sha256": sha256(Path(__file__)),
    }
    inventory_path, sidecar_path, sums_path = targets
    inventory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    inventory_hash = sha256(inventory_path)
    sidecar_path.write_text(
        f"{inventory_hash}  {INVENTORY_NAME}\n", encoding="ascii", newline="\n"
    )
    sums_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
        + f"{inventory_hash}  {INVENTORY_NAME}\n",
        encoding="ascii",
        newline="\n",
    )
    return validate(directory)


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory) / "artifact"
        (directory / "nested").mkdir(parents=True)
        (directory / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
        (directory / "nested" / "b.bin").write_bytes(b"\x00\x01\x02")
        created = create(directory, "synthetic")
        assert created["file_count"] == 2
        try:
            create(directory, "synthetic")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing inventory was overwritten")
        (directory / "a.txt").write_text("tampered\n", encoding="utf-8", newline="\n")
        try:
            validate(directory)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered output passed inventory validation")
    return {"status": "PASS", "tests": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--directory", type=Path, required=True)
    create_parser.add_argument("--run-id", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--directory", type=Path, required=True)
    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create(args.directory, args.run_id)
    elif args.command == "validate":
        result = validate(args.directory)
    else:
        result = self_test()
        if args.report:
            if args.report.exists():
                raise FileExistsError(args.report)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
