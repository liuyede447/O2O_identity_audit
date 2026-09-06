"""Reassemble the complete dense-grid gzip asset from GitHub release parts."""
from pathlib import Path
import argparse
import hashlib

parser = argparse.ArgumentParser()
parser.add_argument("parts_dir", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
parts = [args.parts_dir / f"boundary_full_grid_1024_o2m_rank_set_radius_curve.csv.gz.part0{i}" for i in range(1, 5)]
out = args.output
h = hashlib.sha256()
with out.open("wb") as dst:
    for part in parts:
        with part.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(block)
                h.update(block)
print(out)
print(h.hexdigest())
