"""Rerun the revision analyses, including bootstrap intervals, from the archived derived inputs.

The scripts in source_snapshots/ are the unmodified originals. They name their
inputs by absolute Windows paths (G:\\revision_20261120\\..., E:\\two_paper\\beifen\\...).
This program extracts revision_inputs.tar.gz, which mirrors exactly the files those
scripts read under the same relative layout (G/..., E/...), copies the scripts to a
temporary directory, rewrites each absolute path literal to point into the mirror,
runs the scripts, and compares every output with the archived copy.

Usage (from the bundle root, requirements installed):
    python -B rerun_revision_analyses.py --output-dir rerun_check
    python -B rerun_revision_analyses.py --output-dir rerun_check --only table5 table4

No GPU, detector checkpoint or benchmark dataset is needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "source_snapshots"
GEO_REF = SNAP / "R1_3_multidirection" / "geometry_reference"
DIRECTION_SETS = ("cardinal", "diagonal", "eight", "min8", "cardinal_legal", "diagonal_legal", "min8_legal")
GEO_TABLES = ("rmsr.csv", "rmsr_contrasts.csv", "survival_cif.csv", "o2m_endpoint_rank_set.csv")

# name -> (manuscript items, script relative to source_snapshots, extra args, outputs to compare)
ANALYSES: dict[str, tuple[str, str, list[str], list[str]]] = {
    "table5": ("Table 5 (eligibility lock, seeds 0-3)",
               "R1_2_eligibility_lock_seeds/aggregate_lock_seeds.py", [],
               ["R1_2_eligibility_lock_seeds/lock_by_seed.csv", "R1_2_eligibility_lock_seeds/summary.json"]),
    "table8": ("Table 8 and S42 (fixed-K replay)",
               "R1_4_fixed_k/aggregate_fixed_k.py", [],
               ["R1_4_fixed_k/fixed_k_by_seed.csv", "R1_4_fixed_k/summary.json"]),
    "table8_strata": ("Fixed-K stratified by positive count",
                      "R1_4_fixed_k/fixed_k_stratified.py", [],
                      ["R1_4_fixed_k/stratified_by_seed.csv", "R1_4_fixed_k/stratified_summary.json"]),
    "r2_1": ("Like-for-like O2O/O2M measure (R2.1)",
             "R2_1_comparable_measure/comparable_measure.py", [],
             ["R2_1_comparable_measure/comparable_measure_long.csv", "R2_1_comparable_measure/summary.json"]),
    "table4": ("Table 4 (RFLA and ATSS rule replay)",
               "R2_2_independent_assigners/summarise_base_nonempty.py", [],
               ["R2_2_independent_assigners/summary_base_nonempty.json"]),
    "s40_per_direction": ("Table S40 (RFLA and ATSS per-direction sensitivity)",
                          "R2_2_independent_assigners/per_direction_sensitivity.py", [],
                          ["R2_2_independent_assigners/per_direction_sensitivity.json"]),
    "s47": ("Table S47 and Section 17 (short-side strata)",
            "R3_eligibility_rule_strata/analyze_rule_strata.py", [],
            ["R3_eligibility_rule_strata/rule_strata_results.json"]),
    "s47_sensitivity": ("Section 17 tolerance sensitivity",
                        "R3_eligibility_rule_strata/sensitivity_rule_strata.py", [],
                        ["R3_eligibility_rule_strata/rule_strata_sensitivity.json"]),
    "s47_seeds": ("Table S47 discovery seeds 0-3",
                  "R3_eligibility_rule_strata/seeds_boundary8.py", [],
                  ["R3_eligibility_rule_strata/seeds_boundary8.json"]),
    "table7_sets": ("Table 7 inputs (direction sets)",
                    "R1_3_multidirection/build_direction_sets.py", [], []),
}
for _set in DIRECTION_SETS:
    ANALYSES[f"table7_{_set}"] = (
        f"Table 7 and Supplementary multidirection tables ({_set})",
        "analyze_assignment_boundary_geometry.py", ["@GEOMETRY", _set], [])

PATH_LITERAL = re.compile(r'r(["\'])([A-Za-z]):\\([^"\']*)\1')


def rewrite_paths(text: str, mirror: Path) -> str:
    def repl(m: re.Match) -> str:
        drive, rest = m.group(2).upper(), m.group(3).replace("\\", "/")
        return 'r"' + (mirror / drive).as_posix() + "/" + rest + '"'
    return PATH_LITERAL.sub(repl, text)


def close(a, b, tol=1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol * max(1.0, abs(b))
    return a == b


def looks_like_path(v) -> bool:
    return isinstance(v, str) and bool(re.search(r"^[A-Za-z]:[\\/]|^/|[\\/].*\.(csv|json|py|pt|yaml)$", v))


def compare_json(a, b, where="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in set(a) | set(b):
            if looks_like_path(k):
                continue  # path-keyed hash maps identify the local copy
            if k not in a or k not in b:
                out.append(f"{where}/{k}: key missing")
                continue
            out += compare_json(a[k], b[k], f"{where}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare_json(x, y, f"{where}[{i}]")]
    if looks_like_path(a) and looks_like_path(b):
        return []
    return [] if close(a, b) else [f"{where}: {a!r} vs {b!r}"]


def compare_csv(a: Path, b: Path) -> list[str]:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return [f"shape/columns differ: {x.shape} vs {y.shape}"]
    out = []
    for c in x.columns:
        if pd.api.types.is_bool_dtype(x[c]) or pd.api.types.is_bool_dtype(y[c]):
            if not x[c].astype(str).equals(y[c].astype(str)):
                out.append(f"column {c}: values differ")
        elif pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            d = (x[c] - y[c]).abs()
            bad = ~((d <= 1e-9 * y[c].abs().clip(lower=1)) | (x[c].isna() & y[c].isna()))
            if bad.any():
                out.append(f"column {c}: {int(bad.sum())} rows differ (max {d[bad].max():.3g})")
        elif not x[c].astype(str).equals(y[c].astype(str)):
            xs, ys = x[c].astype(str).tolist(), y[c].astype(str).tolist()
            if all(looks_like_path(v) for v in xs if v != "nan"):
                continue
            parsed = [(_numbers(a), _numbers(b)) for a, b in zip(xs, ys)]
            if all(p is not None and q is not None and len(p) == len(q) and all(map(close, p, q))
                   for p, q in parsed):
                continue  # bracketed intervals such as "[0.12, 0.19]" agree numerically
            out.append(f"column {c}: text differs")
    return out


def _numbers(text: str):
    """Parse a bracketed list of numbers written by the analysis scripts, else None."""
    t = text.strip()
    if not (t.startswith("[") and t.endswith("]")):
        return None
    try:
        return [float(v) for v in t[1:-1].split(",") if v.strip()]
    except ValueError:
        return None


def compare(new: Path, ref: Path) -> list[str]:
    if not new.exists():
        return ["output not produced"]
    if ref.suffix == ".json":
        return compare_json(json.loads(new.read_text(encoding="utf-8")), json.loads(ref.read_text(encoding="utf-8")))
    return compare_csv(new, ref)


def write_size_only_labels(mirror: Path) -> None:
    """Recreate the label lines analyze_rule_strata.py reads, with width and height only.

    For four fixed-only GTs without a size-scaled row, analyze_rule_strata.py reads the box size from
    tokens 4 and 5 of the YOLO label line, and it checks that mapping on 2,000 other GTs.
    val_label_sizes.csv holds just those two tokens, copied from the converted validation labels.
    Class and position tokens are written as '-', so the files carry no other annotation data.
    """
    sizes = pd.read_csv(SNAP / "R3_eligibility_rule_strata" / "val_label_sizes.csv", dtype=str)
    labels = mirror / "G" / "revision_20261120" / "heldout_replication" / "AI-TOD-v2-officialval-YOLO" / "val" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for image_id, rows in sizes.groupby("image_id", sort=False):
        lines = [""] * (rows["label_line"].astype(int).max() + 1)
        for r in rows.itertuples():
            lines[int(r.label_line)] = f"- - - {r.width_norm} {r.height_norm}"
        (labels / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--only", nargs="+", choices=sorted(ANALYSES), help="run a subset")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    selected = args.only or list(ANALYSES)
    if any(n.startswith("table7_") and n != "table7_sets" for n in selected) and "table7_sets" not in selected:
        selected = ["table7_sets"] + selected
    args.output_dir.mkdir(parents=True, exist_ok=False)

    work = Path(tempfile.mkdtemp(prefix="revision_rerun_"))
    try:
        mirror = work / "mirror"
        print("extracting revision_inputs.tar.gz ...", flush=True)
        with tarfile.open(ROOT / "revision_inputs.tar.gz") as archive:
            try:
                archive.extractall(mirror, filter="data")
            except TypeError:  # Python < 3.12
                archive.extractall(mirror)
        snap = work / "source_snapshots"
        shutil.copytree(SNAP, snap, ignore=shutil.ignore_patterns("__pycache__"))
        for py in list(snap.rglob("*.py")) + list(mirror.rglob("*.py")):
            text = py.read_text(encoding="utf-8")
            new = rewrite_paths(text, mirror)
            if new != text:
                py.write_text(new, encoding="utf-8")

        if {"s47", "s47_sensitivity"} & set(selected):
            write_size_only_labels(mirror)
        env = dict(os.environ)
        if importlib.util.find_spec("torch") is None:
            # assigner_audit.py imports torch at module level for its replay code. The analyses
            # here call only its numpy/pandas summarise(); this placeholder satisfies the import
            # and raises on any attribute access, so real torch use cannot pass silently.
            stub = work / "stubs" / "torch"
            stub.mkdir(parents=True)
            (stub / "__init__.py").write_text(
                "def __getattr__(name):\n"
                "    raise RuntimeError(f'torch.{name} was used, but this rerun has no torch installed')\n",
                encoding="utf-8")
            env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), str(stub.parent)) if p)
        report, failed = [], 0
        for name in selected:
            label, script, extra, outputs = ANALYSES[name]
            cmd = [sys.executable, "-B", str(snap / script)]
            if extra[:1] == ["@GEOMETRY"]:
                ds = extra[1]
                sets = mirror / "G" / "revision_20261120" / "boundary_sets" / ds
                out = work / "geometry" / ds
                cmd += ["--event-csv", str(sets / "events.csv"), "--curve-csv", str(sets / "curves.csv"),
                        "--output-dir", str(out), "--tau", "0.125"]
                outputs = [(out / t, GEO_REF / ds / t) for t in GEO_TABLES]
            else:
                outputs = [(snap / o, SNAP / o) for o in outputs]
            for new, _ in outputs:
                new.unlink(missing_ok=True)  # archived copies must not stand in for a missing output
            t0 = time.time()
            print(f"[{name}] {label} ...", flush=True)
            proc = subprocess.run(cmd, cwd=(snap / script).parent, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=env)
            log = args.output_dir / f"{name}.log"
            log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
            problems = [] if proc.returncode == 0 else [f"exit code {proc.returncode}; see {log.name}"]
            for new, ref in outputs:
                diffs = compare(new, ref) if not problems else []
                if new.exists():
                    dest = args.output_dir / name / new.name
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copy2(new, dest)
                problems += [f"{ref.name}: {d}" for d in diffs[:5]]
            status = "PASS" if not problems else "FAIL"
            failed += status == "FAIL"
            report.append({"analysis": name, "manuscript": label, "status": status,
                           "seconds": round(time.time() - t0, 1),
                           "compared": [ref.relative_to(ROOT).as_posix() for _, ref in outputs],
                           "problems": problems})
            print(f"[{name}] {status} ({report[-1]['seconds']} s)" + "".join(f"\n    {p}" for p in problems), flush=True)
        (args.output_dir / "rerun_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{len(report) - failed}/{len(report)} analyses reproduced the archived outputs; "
              f"report in {args.output_dir / 'rerun_report.json'}")
        return 1 if failed else 0
    finally:
        if args.keep_workdir:
            print("work directory kept at", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
