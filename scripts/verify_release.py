#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

rows = list(csv.DictReader((ROOT / "FILE_MANIFEST.csv").open(encoding="utf-8")))
failures = []
for row in rows:
    path = ROOT / row["path"]
    if not path.is_file():
        failures.append(f"missing: {row['path']}")
    elif path.stat().st_size != int(row["bytes"]):
        failures.append(f"size: {row['path']}")
    elif digest(path) != row["sha256"]:
        failures.append(f"sha256: {row['path']}")

listed = {row["path"] for row in rows}
actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
unlisted = sorted(actual - listed - {"FILE_MANIFEST.csv", "SHA256SUMS.txt"})
failures.extend(f"unlisted: {path}" for path in unlisted)

forbidden = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT).as_posix().lower()
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}:
        forbidden.append(rel)
    if rel.endswith("o2m_rank_set_radius_curve.csv"):
        forbidden.append(rel)
    if any(part in {"images", "labels", "raw_predictions", "prediction_dumps"} for part in rel.split("/")):
        forbidden.append(rel)

if failures or forbidden:
    for item in failures: print("FAIL", item)
    for item in forbidden: print("FORBIDDEN", item)
    raise SystemExit(1)
print(f"PASS: {len(rows)} files verified; repository exclusion policy passed")
