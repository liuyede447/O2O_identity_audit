"""Verify the derived inputs and reproduce the supplement's added calculations."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
EXPECTED = HERE / "expected"

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def compare(actual, expected, where="result"):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{where}: keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{where}: lengths differ")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, f"{where}[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, atol=2e-10, rtol=1e-9, equal_nan=True):
            raise ValueError(f"{where}: {actual} vs {expected}")
    elif actual != expected:
        raise ValueError(f"{where}: {actual!r} vs {expected!r}")

def csv_match(path, reference):
    a, b = (pd.read_csv(f) for f in (path, reference))
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-8, atol=1e-9)

def extract_inputs():
    manifest = read_json(HERE / "inputs_sha256.json")
    with tarfile.open(HERE / "derived_inputs.tar.gz") as archive:
        if len(archive.getmembers()) != len(manifest) or set(archive.getnames()) != set(manifest):
            raise ValueError("Archive members differ from the manifest")
        if any(not m.isfile() for m in archive.getmembers()):
            raise ValueError("Expected regular files in the derived archive")
        archive.extractall(HERE / "inputs", filter="data")
    for name, digest in manifest.items():
        if hashlib.sha256((HERE / "inputs" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input hash mismatch for {name}")

def run(file, *args):
    env = {**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"}
    p = subprocess.run([sys.executable, "-B", str(HERE / file), *map(str, args)],
                       cwd=HERE, env=env, check=True, capture_output=True)
    (OUT / (Path(file).stem + ".log")).write_bytes(p.stdout)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-draws", type=int, default=5000,
                        help="Use 100 for a quick point-estimate verification instead of reproducing the full intervals.")
    parser.add_argument("--only", choices=("controls", "margin", "two-track"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    extract_inputs()
    if args.only in (None, "controls"):
        run("half_pixel_final.py")
        compare(read_json(OUT / "half_pixel_final.json"), read_json(EXPECTED / "half_pixel_final.json"), "S55")
        print("S55 half-pixel results and intervals match", flush=True)
        run("analyze_controls.py")
        csv_match(OUT / "seed0_controls_summary.csv", EXPECTED / "seed0_controls_summary.csv")
        run("first_divergence.py")
        expected_fd = read_json(EXPECTED / "exp3_first_divergence_summary.json")
        expected_fd.pop("parity_gate_25_images")
        compare(read_json(OUT / "first_divergence.json"), expected_fd, "S54")
        run("augmentation.py")
        print("S53 controls and S54 first-divergence shares match", flush=True)
        run("miss_comparison.py")
        print("Discovery miss-definition checks match", flush=True)
    if args.only in (None, "margin"):
        run("analyze.py", HERE / "inputs" / "margin" / "formal", OUT / "margin", HERE / "images.csv", args.margin_draws)
        for filename in ("per_seed_points.csv", "seed_mean_points.csv", "counts.csv"):
            csv_match(OUT / "margin" / filename, EXPECTED / "margin" / filename)
        if args.margin_draws == 5000:
            for filename in ("decisions.json", "intervals.json"):
                compare(read_json(OUT / "margin" / filename), read_json(EXPECTED / "margin" / filename), filename)
        print(f"S51 margin replay matches ({args.margin_draws} resamples)", flush=True)
    if args.only in (None, "two-track"):
        run("posthoc_two_track.py", args.margin_draws)
        for filename in ("per_seed_points.csv", "seed_mean_points.csv"):
            csv_match(OUT / "two_track" / filename, EXPECTED / "two_track" / filename)
        if args.margin_draws == 5000:
            compare(read_json(OUT / "two_track" / "summary.json"), read_json(EXPECTED / "two_track" / "summary.json"), "S52")
        print(f"S52 two-track screen matches ({args.margin_draws} resamples)", flush=True)

if __name__ == "__main__":
    main()
