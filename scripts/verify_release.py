"""Verify integrity and public-release boundaries for this repository."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8-sig")))
    for row in rows:
        path = ROOT / row["relative_path"]
        if not path.is_file():
            errors.append(f"missing: {row['relative_path']}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"size: {row['relative_path']}")
        if sha256(path) != row["sha256"]:
            errors.append(f"sha256: {row['relative_path']}")

    unexpected_weights = sorted(ROOT.rglob("*.pt")) + sorted(ROOT.rglob("*.pth"))
    if unexpected_weights:
        errors.extend(f"model binary included: {p.relative_to(ROOT)}" for p in unexpected_weights)

    for relative in ("data/primary/aitodv2/selected_images.csv", "data/primary/visdrone/selected_images.csv"):
        path = ROOT / relative
        with path.open(newline="", encoding="utf-8-sig") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        if count != 300:
            errors.append(f"expected 300 selected images in {relative}, found {count}")

    for path in ROOT.rglob("*.json"):
        with path.open(encoding="utf-8") as stream:
            json.load(stream)
    for path in ROOT.rglob("*.csv"):
        if path == MANIFEST:
            continue
        with path.open(newline="", encoding="utf-8-sig") as stream:
            list(csv.reader(stream))

    if errors:
        raise SystemExit("FAIL\n" + "\n".join(errors))
    print(f"PASS: {len(rows)} files verified; primary samples are 300 + 300 images; no model binaries included.")


if __name__ == "__main__":
    main()
