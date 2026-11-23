"""Rerun Supplementary Tables S43 and S44 on the archived derived inputs."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "source_snapshots"
PAIR = SNAPSHOTS / "R1_2_checkpoint_last"
EARLY = SNAPSHOTS / "R1_2_eligibility_lock_seeds"

def write_with_paths(src: Path, dst: Path, rules: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")
    for pattern, replacement in rules.items():
        text, count = re.subn(pattern, lambda _: replacement, text, flags=re.M)
        if count != 1:
            raise ValueError(f"Could not bind {pattern!r} in {src.name}")
    dst.write_text(text, encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="checkpoint_recheck_") as tmp:
        work = Path(tmp)
        inputs = work / "inputs"
        with tarfile.open(ROOT / "checkpoint_lock_inputs.tar.gz") as archive:
            archive.extractall(inputs, filter="data")
        early = work / EARLY.name
        pair = work / PAIR.name
        early.mkdir()
        pair.mkdir()
        for name in ("base_gate_fixed_only.csv", "lock_by_seed.csv"):
            shutil.copy2(EARLY / name, early / name)
        for src in (inputs / "base_gate_all.csv", PAIR / "support_flag_fixed_only.csv"):
            shutil.copy2(src, pair / src.name)
        corr = early / "correction" / "scripts"
        corr.mkdir(parents=True)
        shutil.copy2(PAIR / "compute_corrected_lock_estimates.py", corr)
        write_with_paths(EARLY / "aggregate_lock_seeds.py", early / "aggregate_lock_seeds.py", {
            r"^CORR = .*$": 'CORR = Path(__file__).parent / "correction"',
            r"^SEEDS = \{\n[\s\S]+?^\}": (
                'IN = HERE.parent / "inputs"\n'
                'SEEDS = {seed: (IN / f"seed{seed}" / "epoch280.csv", '
                'IN / f"seed{seed}" / "epoch280_gate.csv") for seed in range(4)}'
            ),
        })
        write_with_paths(PAIR / "aggregate_paired.py", pair / "aggregate_paired.py", {
            r'^    return LOCK_LAST / f"seed\{seed\}" / "per_gt.csv"$': (
                '    return HERE.parent / "inputs" / f"seed{seed}" / "final_epoch.csv"'
            ),
        })
        write_with_paths(PAIR / "aggregate_table2_support.py", pair / "aggregate_table2_support.py", {
            r"^FLAGS = \{\n[\s\S]+?^\}": (
                'FLAGS = {seed: HERE.parent / "inputs" / f"seed{seed}" / "competition_flag.csv" '
                'for seed in range(4)}'
            ),
        })
        for name in ("aggregate_paired.py", "aggregate_table2_support.py"):
            subprocess.run([sys.executable, "-B", str(pair / name)], check=True)
        for name in ("paired_by_seed.csv", "summary.json", "table2_support_by_seed.csv", "table2_support_summary.json"):
            shutil.copy2(pair / name, args.output_dir)
    print(f"S43 and S44 results written to {args.output_dir}")

if __name__ == "__main__":
    main()
