"""Run and validate the serial CPU post-processing chain for a boundary grid.

The parent process never loads the full curve CSV into a dataframe. Each tau
analysis runs in a fresh child process so its pandas/numpy allocations are
released before the next tau starts. Optional convergence checks keep only one
compact per-trajectory signature in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from create_run_output_inventory import create as create_inventory
from create_run_output_inventory import sha256 as inventory_sha256
from create_run_output_inventory import validate as validate_inventory


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SCRIPT = ROOT / "scripts" / "analyze_assignment_boundary_geometry.py"
INVENTORY_SCRIPT = ROOT / "scripts" / "create_run_output_inventory.py"
EVENT_NAME = "per_direction_boundary_events.csv"
CURVE_NAME = "o2m_rank_set_radius_curve.csv"
REQUIRED_GEOMETRY_OUTPUTS = (
    "survival_cif.csv",
    "rmsr.csv",
    "rmsr_contrasts.csv",
    "o2m_endpoint_rank_set.csv",
    "return_transition.csv",
    "rho_R_status.csv",
    "margin_construct.csv",
    "analysis_results.json",
)
DEFAULT_TAUS = (0.05, 0.0625, 0.10, 0.125, 0.20, 0.25)
ENDPOINTS_BY_TAU = {
    0.05: (0.03125,),
    0.0625: (0.03125, 0.0625),
    0.10: (0.03125, 0.0625, 0.10),
    0.125: (0.03125, 0.0625, 0.125),
    0.20: (0.03125, 0.0625, 0.125, 0.20),
    0.25: (0.03125, 0.0625, 0.125, 0.25),
}
KEY_COLUMNS = ("image_id", "gt_id", "direction")
IDENTITY_COLUMNS = {
    "o2o": "o2o_active",
    "o2m_legacy": "o2m_legacy_rank",
    "o2m_pre_topk": "o2m_pre_topk_rank",
    "o2m_assigned_positive": "o2m_assigned_positive_rank",
}
EVENT_ESTIMANDS = (*IDENTITY_COLUMNS, "rho_R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Completed radius artifact directory or its run root")
    parser.add_argument("--output-root", type=Path, help="New post-processing output directory")
    parser.add_argument("--tau-values", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--geometry-script", type=Path, default=GEOMETRY_SCRIPT)
    parser.add_argument("--canonical-primary-dir", type=Path)
    parser.add_argument("--canonical-tau-root", type=Path)
    parser.add_argument("--convergence-reference-dir", type=Path)
    parser.add_argument("--require-convergence-exact", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-report", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def resolve_artifact_dir(path: Path) -> Path:
    path = path.resolve()
    candidates = (path, path / "artifact")
    for candidate in candidates:
        if (candidate / EVENT_NAME).is_file() and (candidate / CURVE_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"could not find {EVENT_NAME} and {CURVE_NAME} under {path}")


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[column] for column in KEY_COLUMNS)  # type: ignore[return-value]


def require_columns(fieldnames: list[str] | None, required: set[str], label: str) -> None:
    missing = sorted(required.difference(fieldnames or []))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_radius_input(artifact: Path) -> dict[str, object]:
    event_path = artifact / EVENT_NAME
    curve_path = artifact / CURVE_NAME
    summary_path = artifact / "summary.json"
    manifest_path = artifact / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("radius summary.json or manifest.json is missing")
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if summary.get("status") not in {"complete", "smoke_complete"}:
        raise RuntimeError("radius summary status is not terminal complete")
    if manifest.get("status") not in {"complete", "smoke_complete"}:
        raise RuntimeError("radius manifest status is not terminal complete")

    event_keys: set[tuple[str, str, str]] = set()
    with event_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, set(KEY_COLUMNS), "event CSV")
        for row in reader:
            key = row_key(row)
            if key in event_keys:
                raise RuntimeError(f"duplicate event trajectory key: {key}")
            event_keys.add(key)

    curve_rows = 0
    curve_keys: set[tuple[str, str, str]] = set()
    completed_keys: set[tuple[str, str, str]] = set()
    previous_key: tuple[str, str, str] | None = None
    previous_radius = -math.inf
    with curve_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, {*KEY_COLUMNS, "radius"}, "curve CSV")
        for row in reader:
            curve_rows += 1
            key = row_key(row)
            radius = float(row["radius"])
            if key != previous_key:
                if previous_key is not None:
                    completed_keys.add(previous_key)
                if key in completed_keys:
                    raise RuntimeError(f"curve trajectory is not contiguous: {key}")
                if not math.isclose(radius, 0.0, abs_tol=1e-12):
                    raise RuntimeError(f"curve trajectory does not start at radius zero: {key}")
                curve_keys.add(key)
                previous_key = key
                previous_radius = radius
            else:
                if radius <= previous_radius + 1e-12:
                    raise RuntimeError(f"curve radii are not strictly increasing: {key}")
                previous_radius = radius
    if curve_keys != event_keys:
        raise RuntimeError(
            f"event/curve trajectory keys differ: events={len(event_keys)}, curves={len(curve_keys)}"
        )
    if int(summary.get("focal_direction_rows", -1)) != len(event_keys):
        raise RuntimeError("summary focal_direction_rows does not match event CSV")
    if int(summary.get("radius_curve_rows", -1)) != curve_rows:
        raise RuntimeError("summary radius_curve_rows does not match curve CSV")
    return {
        "status": "PASS",
        "artifact_dir": str(artifact),
        "event_rows": len(event_keys),
        "curve_rows": curve_rows,
        "event_csv_sha256": sha256(event_path),
        "curve_csv_sha256": sha256(curve_path),
        "summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
    }


def tau_key(value: float) -> float:
    for known in ENDPOINTS_BY_TAU:
        if math.isclose(value, known, abs_tol=1e-12):
            return known
    raise ValueError(f"no prespecified endpoint schedule for tau={value}")


def tau_label(value: float) -> str:
    return f"tau_{round(value * 10000):05d}"


def geometry_command(args: argparse.Namespace, artifact: Path, output: Path, tau: float) -> list[str]:
    endpoints = ENDPOINTS_BY_TAU[tau_key(tau)]
    return [
        str(args.python),
        str(args.geometry_script),
        "--event-csv",
        str(artifact / EVENT_NAME),
        "--curve-csv",
        str(artifact / CURVE_NAME),
        "--output-dir",
        str(output),
        "--tau",
        f"{tau:.12g}",
        "--endpoint-radii",
        *(f"{value:.12g}" for value in endpoints),
        "--bootstrap-reps",
        str(args.bootstrap_reps),
        "--seed",
        str(args.seed),
        "--ci-level",
        f"{args.ci_level:.12g}",
    ]


def validate_geometry_output(output: Path, source: dict[str, object], args: argparse.Namespace, tau: float) -> dict[str, object]:
    summary_path = output / "summary.json"
    stderr_path = output / "stderr.log"
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"geometry status is not complete: {output}")
    if not math.isclose(float(summary.get("tau")), tau, abs_tol=1e-12):
        raise RuntimeError(f"geometry tau mismatch: {output}")
    if int(summary.get("bootstrap_reps", -1)) != args.bootstrap_reps:
        raise RuntimeError(f"geometry bootstrap count mismatch: {output}")
    if int(summary.get("seed", -1)) != args.seed:
        raise RuntimeError(f"geometry seed mismatch: {output}")
    if summary.get("event_csv_sha256") != source["event_csv_sha256"]:
        raise RuntimeError(f"geometry event source hash mismatch: {output}")
    if summary.get("curve_csv_sha256") != source["curve_csv_sha256"]:
        raise RuntimeError(f"geometry curve source hash mismatch: {output}")
    if stderr_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"geometry stderr is not empty: {output}")
    if tuple(summary.get("outputs", [])) != REQUIRED_GEOMETRY_OUTPUTS:
        raise RuntimeError(f"geometry output contract mismatch: {output}")
    for name in REQUIRED_GEOMETRY_OUTPUTS:
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if summary["output_sha256"].get(name) != sha256(path):
            raise RuntimeError(f"geometry output hash mismatch: {path}")
    return {
        "status": "PASS",
        "tau": tau,
        "summary_sha256": sha256(summary_path),
        "output_sha256": summary["output_sha256"],
    }


def run_geometry(args: argparse.Namespace, artifact: Path, output: Path, tau: float, source: dict[str, object]) -> dict[str, object]:
    command = geometry_command(args, artifact, output, tau)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
    (output / "stdout.log").write_text(completed.stdout, encoding="utf-8", newline="\n")
    (output / "stderr.log").write_text(completed.stderr, encoding="utf-8", newline="\n")
    (output / "COMMAND.json").write_text(
        json.dumps({"cwd": str(ROOT), "argv": command}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"geometry analyzer failed for tau={tau}; see {output / 'stderr.log'}")
    validation = validate_geometry_output(output, source, args, tau)
    inventory = create_inventory(output, tau_label(tau))
    if validate_inventory(output) != inventory:
        raise RuntimeError(f"geometry inventory validation changed unexpectedly: {output}")
    return {
        **validation,
        "elapsed_sec": time.monotonic() - started,
        "inventory_sha256": inventory["inventory_sha256"],
    }


def compare_csv_values(actual: Path, expected: Path) -> dict[str, object]:
    mismatches: list[dict[str, object]] = []
    rows = 0
    with actual.open("r", newline="", encoding="utf-8-sig") as left, expected.open(
        "r", newline="", encoding="utf-8-sig"
    ) as right:
        left_reader = csv.reader(left)
        right_reader = csv.reader(right)
        while True:
            left_row = next(left_reader, None)
            right_row = next(right_reader, None)
            if left_row is None and right_row is None:
                break
            rows += 1
            if left_row != right_row and len(mismatches) < 20:
                mismatches.append({"row": rows, "actual": left_row, "expected": right_row})
    return {"status": "PASS" if not mismatches else "FAIL", "rows_including_header": rows, "mismatches": mismatches}


def compare_json_values(actual: Path, expected: Path) -> dict[str, object]:
    same = json.loads(actual.read_text(encoding="utf-8")) == json.loads(expected.read_text(encoding="utf-8"))
    return {"status": "PASS" if same else "FAIL"}


def compare_geometry_to_canonical(actual: Path, canonical: Path) -> dict[str, object]:
    if not canonical.is_dir():
        raise FileNotFoundError(canonical)
    files: dict[str, object] = {}
    for name in REQUIRED_GEOMETRY_OUTPUTS:
        left, right = actual / name, canonical / name
        if not right.is_file():
            raise FileNotFoundError(right)
        if name.endswith(".csv"):
            result = compare_csv_values(left, right)
        else:
            result = compare_json_values(left, right)
        result["actual_sha256"] = sha256(left)
        result["canonical_sha256"] = sha256(right)
        files[name] = result
    passed = all(result["status"] == "PASS" for result in files.values())
    return {"status": "PASS" if passed else "FAIL", "canonical_dir": str(canonical), "files": files}


def canonical_for_tau(args: argparse.Namespace, tau: float) -> Path | None:
    if math.isclose(tau, 0.125, abs_tol=1e-12) and args.canonical_primary_dir:
        return args.canonical_primary_dir.resolve()
    if not args.canonical_tau_root:
        return None
    root = args.canonical_tau_root.resolve()
    names = [tau_label(tau)]
    if math.isclose(tau, 0.05, abs_tol=1e-12):
        names.insert(0, "tau_00500_v2")
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"canonical tau directory not found for tau={tau} under {root}")


def event_signature(row: dict[str, str]) -> dict[str, tuple[str, ...]]:
    result = {}
    for estimand in EVENT_ESTIMANDS:
        fields = [
            row.get(f"{estimand}_event_observed", ""),
            row.get(f"{estimand}_radius", ""),
            row.get(f"{estimand}_right_censored", ""),
            row.get(f"{estimand}_competing_censored", ""),
            row.get(f"{estimand}_censor_radius", ""),
            row.get(f"{estimand}_censor_reason", ""),
        ]
        if estimand == "o2o":
            fields.append(row.get("o2o_first_divergence", ""))
        result[estimand] = tuple(fields)
    return result


def load_event_signatures(path: Path) -> dict[tuple[str, str, str], dict[str, tuple[str, ...]]]:
    result = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, set(KEY_COLUMNS), "convergence event CSV")
        for row in reader:
            key = row_key(row)
            if key in result:
                raise RuntimeError(f"duplicate convergence event key: {key}")
            result[key] = event_signature(row)
    return result


def normalize_identity(value: str) -> str | None:
    return None if value == "" else value


def curve_return_signatures(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    signatures: dict[tuple[str, str, str], dict[str, str]] = {}
    current_key: tuple[str, str, str] | None = None
    base: dict[str, str | None] = {}
    changed: dict[str, bool] = {}
    returned: dict[str, bool] = {}

    def finish() -> None:
        if current_key is None:
            return
        signatures[current_key] = {
            name: "changed_then_returned" if returned[name] else "changed_no_return" if changed[name] else "never_changed"
            for name in changed
        }

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {*KEY_COLUMNS, "radius", *IDENTITY_COLUMNS.values(), "o2m_positive_set_exact_change"}
        require_columns(reader.fieldnames, required, "convergence curve CSV")
        for row in reader:
            key = row_key(row)
            if key != current_key:
                finish()
                current_key = key
                base = {name: normalize_identity(row[column]) for name, column in IDENTITY_COLUMNS.items()}
                base["o2m_positive_set"] = "0"
                changed = {name: False for name in base}
                returned = {name: False for name in base}
            for name, column in IDENTITY_COLUMNS.items():
                is_changed = normalize_identity(row[column]) != base[name]
                if is_changed:
                    changed[name] = True
                elif changed[name]:
                    returned[name] = True
            set_changed = row["o2m_positive_set_exact_change"] == "1"
            if set_changed:
                changed["o2m_positive_set"] = True
            elif changed["o2m_positive_set"]:
                returned["o2m_positive_set"] = True
    finish()
    return signatures


def compare_convergence(artifact: Path, reference: Path) -> dict[str, object]:
    reference = resolve_artifact_dir(reference)
    left_events = load_event_signatures(reference / EVENT_NAME)
    right_events = load_event_signatures(artifact / EVENT_NAME)
    left_returns = curve_return_signatures(reference / CURVE_NAME)
    right_returns = curve_return_signatures(artifact / CURVE_NAME)
    all_keys = set(left_events) | set(right_events)
    estimand_mismatches = Counter()
    examples: list[dict[str, object]] = []
    for key in sorted(all_keys):
        if key not in left_events or key not in right_events:
            estimand_mismatches["trajectory_key"] += 1
            continue
        for estimand in EVENT_ESTIMANDS:
            if left_events[key][estimand] != right_events[key][estimand]:
                estimand_mismatches[f"event:{estimand}"] += 1
                if len(examples) < 50:
                    examples.append(
                        {
                            "key": key,
                            "estimand": estimand,
                            "reference": left_events[key][estimand],
                            "candidate": right_events[key][estimand],
                        }
                    )
        for estimand in (*IDENTITY_COLUMNS, "o2m_positive_set"):
            if left_returns.get(key, {}).get(estimand) != right_returns.get(key, {}).get(estimand):
                estimand_mismatches[f"return:{estimand}"] += 1
    exact = not estimand_mismatches
    return {
        "status": "PASS" if exact else "DIFFERENT",
        "reference_dir": str(reference),
        "candidate_dir": str(artifact),
        "reference_trajectories": len(left_events),
        "candidate_trajectories": len(right_events),
        "mismatch_counts": dict(sorted(estimand_mismatches.items())),
        "examples": examples,
        "fields": {
            "events": list(EVENT_ESTIMANDS),
            "return_states": [*IDENTITY_COLUMNS, "o2m_positive_set"],
        },
    }


def plan(args: argparse.Namespace, artifact: Path) -> dict[str, object]:
    taus = [tau_key(value) for value in args.tau_values]
    if len(set(taus)) != len(taus):
        raise ValueError("tau-values contains duplicates")
    return {
        "protocol": "dense_grid_serial_postprocess_v1",
        "input_dir": str(artifact),
        "output_root": str(args.output_root.resolve()),
        "python": str(args.python.resolve()),
        "geometry_script": str(args.geometry_script.resolve()),
        "tau_jobs": [
            {
                "tau": tau,
                "label": tau_label(tau),
                "endpoint_radii": list(ENDPOINTS_BY_TAU[tau]),
            }
            for tau in taus
        ],
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "ci_level": args.ci_level,
        "execution": "strictly serial subprocesses; at most one geometry dataframe pair resident",
    }


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        left = root / "left"
        right = root / "right"
        left.mkdir()
        right.mkdir()
        event_fields = [*KEY_COLUMNS, "o2o_event_observed", "o2o_radius", "o2o_right_censored", "o2o_censor_radius", "o2o_first_divergence"]
        curve_fields = [*KEY_COLUMNS, "radius", *IDENTITY_COLUMNS.values(), "o2m_positive_set_exact_change"]
        event_row = {name: "" for name in event_fields}
        event_row.update({"image_id": "a", "gt_id": "0", "direction": "left", "o2o_event_observed": "1", "o2o_radius": "0.1", "o2o_right_censored": "0", "o2o_first_divergence": "eligibility_boundary"})
        curve_rows = []
        for radius, active in (("0.0", "1"), ("0.1", "2"), ("0.2", "1")):
            row = {name: "" for name in curve_fields}
            row.update({"image_id": "a", "gt_id": "0", "direction": "left", "radius": radius, "o2o_active": active, "o2m_legacy_rank": "1", "o2m_pre_topk_rank": "1", "o2m_assigned_positive_rank": "1", "o2m_positive_set_exact_change": "0"})
            curve_rows.append(row)
        for directory in (left, right):
            with (directory / EVENT_NAME).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=event_fields)
                writer.writeheader()
                writer.writerow(event_row)
            with (directory / CURVE_NAME).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=curve_fields)
                writer.writeheader()
                writer.writerows(curve_rows)
        assert compare_convergence(right, left)["status"] == "PASS"
        assert curve_return_signatures(left / CURVE_NAME)[("a", "0", "left")]["o2o"] == "changed_then_returned"
        csv_comparison = compare_csv_values(left / EVENT_NAME, right / EVENT_NAME)
        assert csv_comparison["status"] == "PASS"
        event_row["o2o_radius"] = "0.2"
        with (right / EVENT_NAME).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=event_fields)
            writer.writeheader()
            writer.writerow(event_row)
        assert compare_convergence(right, left)["status"] == "DIFFERENT"
    return {"status": "PASS", "tests": 4}


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = self_test()
        if args.self_test_report:
            if args.self_test_report.exists():
                raise FileExistsError(args.self_test_report)
            args.self_test_report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.self_test_report, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.input_dir is None or args.output_root is None:
        raise ValueError("--input-dir and --output-root are required")
    if args.bootstrap_reps < 0 or not (0 < args.ci_level < 1):
        raise ValueError("bootstrap-reps must be nonnegative and ci-level must be in (0,1)")
    if not args.geometry_script.is_file() or not args.python.is_file():
        raise FileNotFoundError("python or geometry script is missing")
    artifact = resolve_artifact_dir(args.input_dir)
    pipeline_plan = plan(args, artifact)
    if args.plan_only:
        print(json.dumps(pipeline_plan, indent=2, sort_keys=True))
        return
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    write_json(output_root / "PLAN.json", pipeline_plan)
    started = time.monotonic()

    source_validation = validate_radius_input(artifact)
    write_json(output_root / "SOURCE_VALIDATION.json", source_validation)
    source_hashes = {
        "pipeline_script": sha256(Path(__file__)),
        "geometry_script": sha256(args.geometry_script),
        "inventory_script": sha256(INVENTORY_SCRIPT),
        "event_csv": source_validation["event_csv_sha256"],
        "curve_csv": source_validation["curve_csv_sha256"],
        "radius_summary": source_validation["summary_sha256"],
        "radius_manifest": source_validation["manifest_sha256"],
    }
    write_json(output_root / "SOURCE_HASHES.json", source_hashes)

    convergence = None
    if args.convergence_reference_dir:
        convergence = compare_convergence(artifact, args.convergence_reference_dir)
        write_json(output_root / "CONVERGENCE_COMPARISON.json", convergence)
        if args.require_convergence_exact and convergence["status"] != "PASS":
            raise RuntimeError("convergence comparison was not exact")

    geometry_results = {}
    canonical_results = {}
    for tau in [tau_key(value) for value in args.tau_values]:
        label = tau_label(tau)
        output = output_root / label
        geometry_results[label] = run_geometry(args, artifact, output, tau, source_validation)
        canonical = canonical_for_tau(args, tau)
        if canonical is not None:
            comparison = compare_geometry_to_canonical(output, canonical)
            canonical_results[label] = comparison
            if comparison["status"] != "PASS":
                write_json(output_root / "CANONICAL_COMPARISON.json", canonical_results)
                raise RuntimeError(f"generated geometry differs from canonical values at tau={tau}")
    if canonical_results:
        write_json(output_root / "CANONICAL_COMPARISON.json", canonical_results)

    terminal = {
        "status": "PASS",
        "protocol": "dense_grid_serial_postprocess_v1",
        "source_validation": source_validation,
        "source_hashes": source_hashes,
        "geometry": geometry_results,
        "canonical_comparison": canonical_results,
        "convergence": convergence,
        "elapsed_sec": time.monotonic() - started,
        "memory_contract": "tau jobs executed serially in fresh child processes; parent used streaming CSV validation",
    }
    write_json(output_root / "TERMINAL_VALIDATION.json", terminal)
    inventory = create_inventory(output_root, output_root.name)
    if validate_inventory(output_root) != inventory:
        raise RuntimeError("root output inventory validation changed unexpectedly")
    result = {
        **terminal,
        "output_inventory_sha256": inventory_sha256(output_root / "OUTPUT_INVENTORY.json"),
        "output_file_count": inventory["file_count"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
