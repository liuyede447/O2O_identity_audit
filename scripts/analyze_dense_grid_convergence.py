#!/usr/bin/env python3
"""Compare 1/512 and 1/1024 dense-grid boundary artifacts.

The comparison is deliberately trajectory-level and conservative.  A normal
one-fine-step localization improvement is reported as quantization-only, not as
an exact mismatch.  A fine-only event, an earlier event by more than one fine
step, a first-event-type change, a return-state change, or a nesting violation
is material and fails the convergence gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator


EVENT_FILE = "per_direction_boundary_events.csv"
CURVE_FILE = "o2m_rank_set_radius_curve.csv"
KEY_COLUMNS = ("image_id", "gt_id", "direction")
ESTIMANDS = (
    "o2o",
    "o2m_legacy",
    "o2m_pre_topk",
    "o2m_assigned_positive",
    "rho_R",
)
IDENTITY_COLUMNS = {
    "o2o": "o2o_active",
    "o2m_legacy": "o2m_legacy_rank",
    "o2m_pre_topk": "o2m_pre_topk_rank",
    "o2m_assigned_positive": "o2m_assigned_positive_rank",
}
RETURN_STATES = (
    *IDENTITY_COLUMNS,
    "o2m_positive_set",
    "rho_R_order",
    "rho_R_comparability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-512", type=Path, required=True)
    parser.add_argument("--grid-1024", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--fail-on-material", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(fieldnames: list[str] | None, required: Iterable[str], label: str) -> None:
    available = set(fieldnames or [])
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[name] for name in KEY_COLUMNS)  # type: ignore[return-value]


def key_text(key: tuple[str, str, str]) -> str:
    return "/".join(key)


def as_flag(value: str) -> bool:
    return value.strip() == "1"


def as_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def event_state(row: dict[str, str], estimand: str) -> str:
    observed = as_flag(row.get(f"{estimand}_event_observed", ""))
    competing = as_flag(row.get(f"{estimand}_competing_censored", ""))
    right = as_flag(row.get(f"{estimand}_right_censored", ""))
    if sum((observed, competing, right)) > 1:
        return "invalid_multiple_states"
    if observed:
        return "observed"
    if competing:
        return "competing_censored"
    if right:
        return "right_censored"
    return "other_censored"


def event_distance(row: dict[str, str], estimand: str) -> float | None:
    if event_state(row, estimand) == "observed":
        return as_float(row.get(f"{estimand}_radius", ""))
    return as_float(row.get(f"{estimand}_censor_radius", ""))


def event_type(row: dict[str, str], estimand: str) -> str:
    if event_state(row, estimand) == "observed":
        if estimand == "o2o":
            return row.get("o2o_first_divergence", "").strip() or "observed_unspecified"
        return "identity_change"
    return row.get(f"{estimand}_censor_reason", "").strip() or event_state(row, estimand)


def load_events(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[tuple[str, str, str]]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    order: list[tuple[str, str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(KEY_COLUMNS)
        for estimand in ESTIMANDS:
            required.update(
                {
                    f"{estimand}_event_observed",
                    f"{estimand}_right_censored",
                    f"{estimand}_censor_radius",
                    f"{estimand}_censor_reason",
                }
            )
        required.update({"o2o_first_divergence", "rho_R_competing_censored"})
        require_columns(reader.fieldnames, required, str(path))
        for row in reader:
            key = row_key(row)
            if key in rows:
                raise ValueError(f"duplicate trajectory key in {path}: {key_text(key)}")
            rows[key] = row
            order.append(key)
    return rows, order


def normalize_identity(value: str) -> str | None:
    value = value.strip()
    return value or None


def transition_label(changed: dict[str, bool], returned: dict[str, bool], name: str) -> str:
    if returned[name]:
        return "changed_then_returned"
    if changed[name]:
        return "changed_no_return"
    return "never_changed"


def curve_return_signatures(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Stream the curve and retain only seven small labels per trajectory."""
    signatures: dict[tuple[str, str, str], dict[str, str]] = {}
    current_key: tuple[str, str, str] | None = None
    base: dict[str, object] = {}
    changed: dict[str, bool] = {}
    returned: dict[str, bool] = {}

    def finish() -> None:
        if current_key is None:
            return
        if current_key in signatures:
            raise ValueError(f"non-contiguous duplicate trajectory in {path}: {key_text(current_key)}")
        signatures[current_key] = {
            name: transition_label(changed, returned, name) for name in RETURN_STATES
        }

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            *KEY_COLUMNS,
            "radius",
            *IDENTITY_COLUMNS.values(),
            "o2m_positive_set_exact_change",
            "rho_R_pair_q_gap",
            "rho_R_pair_eligibility_comparable",
            "rho_R_pair_pre_topk_comparable",
            "rho_R_pair_conflict_comparable",
            "rho_R_pair_comparability_reason",
        }
        require_columns(reader.fieldnames, required, str(path))
        for row in reader:
            key = row_key(row)
            if key != current_key:
                finish()
                current_key = key
                base = {name: normalize_identity(row[column]) for name, column in IDENTITY_COLUMNS.items()}
                base["o2m_positive_set"] = False
                base["rho_R_order"] = rho_order_state(row)
                base["rho_R_comparability"] = rho_comparability_state(row)
                changed = {name: False for name in RETURN_STATES}
                returned = {name: False for name in RETURN_STATES}

            current: dict[str, object] = {
                name: normalize_identity(row[column]) for name, column in IDENTITY_COLUMNS.items()
            }
            current["o2m_positive_set"] = as_flag(row["o2m_positive_set_exact_change"])
            current["rho_R_order"] = rho_order_state(row)
            current["rho_R_comparability"] = rho_comparability_state(row)
            for name in RETURN_STATES:
                value = current[name]
                if name == "rho_R_order" and value == "not_comparable":
                    continue
                differs = value != base[name]
                if differs:
                    changed[name] = True
                elif changed[name]:
                    returned[name] = True
    finish()
    return signatures


def rho_comparability_state(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["rho_R_pair_eligibility_comparable"].strip(),
        row["rho_R_pair_pre_topk_comparable"].strip(),
        row["rho_R_pair_conflict_comparable"].strip(),
        row["rho_R_pair_comparability_reason"].strip(),
    )


def rho_order_state(row: dict[str, str]) -> str:
    comparable = rho_comparability_state(row)[:3] == ("1", "1", "1")
    if not comparable:
        return "not_comparable"
    gap = as_float(row["rho_R_pair_q_gap"])
    if gap is None:
        return "undefined"
    if gap > 0:
        return "positive"
    if gap < 0:
        return "negative"
    return "tie"


def compare_distance(
    coarse: float | None,
    fine: float | None,
    fine_step: float,
    transition_localized: bool,
) -> tuple[str, bool, float | None]:
    if coarse is None and fine is None:
        return ("same_undefined_distance", False, None)
    if coarse is None or fine is None:
        return ("one_missing_distance", True, None)
    delta = coarse - fine
    tolerance = max(1e-12, fine_step * 1e-7)
    if not transition_localized:
        material = abs(delta) > tolerance
        return ("same_censor_distance" if not material else "censor_distance_changed", material, delta)
    if delta < -tolerance:
        return ("fine_event_later_nesting_violation", True, delta)
    if delta <= fine_step + tolerance:
        if abs(delta) <= tolerance:
            return ("same_sample", False, delta)
        return ("fine_half_step_localization", False, delta)
    return ("coarse_missed_earlier_transition", True, delta)


def add_example(examples: list[dict[str, object]], maximum: int, payload: dict[str, object]) -> None:
    signature = (payload.get("category"), payload.get("estimand", "all"))
    represented = sum(
        (item.get("category"), item.get("estimand", "all")) == signature for item in examples
    )
    if len(examples) < maximum and represented < 3:
        examples.append(payload)


def decision_row(
    category: str,
    estimand: str,
    count: int,
    denominator: int,
    material: bool,
    rule: str,
) -> dict[str, object]:
    return {
        "category": category,
        "estimand": estimand,
        "count": count,
        "denominator": denominator,
        "proportion": count / denominator if denominator else 0.0,
        "material": material,
        "threshold": "0" if material else "reported_only",
        "decision": "FAIL" if material and count else "PASS",
        "rule": rule,
    }


def compare_artifacts(
    coarse_dir: Path,
    fine_dir: Path,
    max_examples: int,
    allow_incomplete: bool,
) -> dict[str, object]:
    coarse_dir = coarse_dir.resolve()
    fine_dir = fine_dir.resolve()
    manifests = {"1/512": read_json(coarse_dir / "manifest.json"), "1/1024": read_json(fine_dir / "manifest.json")}
    if not allow_incomplete:
        for label, manifest in manifests.items():
            if manifest.get("status") != "complete":
                raise RuntimeError(f"{label} artifact is not complete: {manifest.get('status')}")
    coarse_step = float(manifests["1/512"]["fine_step"])
    fine_step = float(manifests["1/1024"]["fine_step"])
    same_grid_self_check = coarse_dir == fine_dir or math.isclose(coarse_step, fine_step, abs_tol=1e-15)
    if not same_grid_self_check and not math.isclose(coarse_step, 2.0 * fine_step, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"expected a 2:1 grid ratio, got coarse={coarse_step}, fine={fine_step}")

    coarse_events, coarse_order = load_events(coarse_dir / EVENT_FILE)
    fine_events, fine_order = load_events(fine_dir / EVENT_FILE)
    coarse_keys = set(coarse_events)
    fine_keys = set(fine_events)
    shared_keys = coarse_keys & fine_keys
    examples: list[dict[str, object]] = []
    counters: Counter[tuple[str, str]] = Counter()
    distance_classes: Counter[tuple[str, str]] = Counter()
    added_events: Counter[str] = Counter()
    material_keys: set[tuple[str, str, str]] = set()

    for key in sorted(coarse_keys - fine_keys):
        counters[("key_only_1_512", "all")] += 1
        material_keys.add(key)
        add_example(examples, max_examples, {"category": "key_only_1_512", "key": key_text(key)})
    for key in sorted(fine_keys - coarse_keys):
        counters[("key_only_1_1024", "all")] += 1
        material_keys.add(key)
        add_example(examples, max_examples, {"category": "key_only_1_1024", "key": key_text(key)})

    for key in sorted(shared_keys):
        coarse = coarse_events[key]
        fine = fine_events[key]
        for estimand in ESTIMANDS:
            coarse_state = event_state(coarse, estimand)
            fine_state = event_state(fine, estimand)
            if coarse_state != fine_state:
                counters[("event_state_mismatch", estimand)] += 1
                material_keys.add(key)
                if coarse_state != "observed" and fine_state == "observed":
                    added_events[estimand] += 1
                    category = "fine_only_event"
                elif coarse_state == "observed" and fine_state != "observed":
                    category = "coarse_only_event_nesting_violation"
                else:
                    category = "censor_state_changed"
                counters[(category, estimand)] += 1
                add_example(
                    examples,
                    max_examples,
                    {
                        "category": category,
                        "key": key_text(key),
                        "estimand": estimand,
                        "grid_1_512": {"state": coarse_state, "type": event_type(coarse, estimand), "distance": event_distance(coarse, estimand)},
                        "grid_1_1024": {"state": fine_state, "type": event_type(fine, estimand), "distance": event_distance(fine, estimand)},
                    },
                )
                continue

            coarse_type = event_type(coarse, estimand)
            fine_type = event_type(fine, estimand)
            if coarse_type != fine_type:
                counters[("first_event_type_mismatch", estimand)] += 1
                material_keys.add(key)
                add_example(
                    examples,
                    max_examples,
                    {
                        "category": "first_event_type_mismatch",
                        "key": key_text(key),
                        "estimand": estimand,
                        "grid_1_512": coarse_type,
                        "grid_1_1024": fine_type,
                    },
                )

            distance_class, material, delta = compare_distance(
                event_distance(coarse, estimand),
                event_distance(fine, estimand),
                fine_step,
                coarse_state in {"observed", "competing_censored"},
            )
            distance_classes[(distance_class, estimand)] += 1
            if material:
                counters[("material_distance_mismatch", estimand)] += 1
                material_keys.add(key)
                add_example(
                    examples,
                    max_examples,
                    {
                        "category": distance_class,
                        "key": key_text(key),
                        "estimand": estimand,
                        "grid_1_512_distance": event_distance(coarse, estimand),
                        "grid_1_1024_distance": event_distance(fine, estimand),
                        "coarse_minus_fine": delta,
                        "fine_step": fine_step,
                    },
                )

    coarse_returns = curve_return_signatures(coarse_dir / CURVE_FILE)
    fine_returns = curve_return_signatures(fine_dir / CURVE_FILE)
    return_keys = set(coarse_returns) | set(fine_returns)
    for key in sorted(return_keys):
        if key not in coarse_returns or key not in fine_returns:
            counters[("return_key_mismatch", "all")] += 1
            material_keys.add(key)
            continue
        for estimand in RETURN_STATES:
            left = coarse_returns[key][estimand]
            right = fine_returns[key][estimand]
            if left != right:
                counters[("return_state_mismatch", estimand)] += 1
                material_keys.add(key)
                add_example(
                    examples,
                    max_examples,
                    {
                        "category": "return_state_mismatch",
                        "key": key_text(key),
                        "estimand": estimand,
                        "grid_1_512": left,
                        "grid_1_1024": right,
                    },
                )

    denominator = len(shared_keys)
    key_denominator = len(coarse_keys | fine_keys)
    return_denominator = len(return_keys)
    decision_rows: list[dict[str, object]] = []
    material_rules = {
        "key_only_1_512": "trajectory key sets must be identical",
        "key_only_1_1024": "trajectory key sets must be identical",
        "event_state_mismatch": "observed/right-censored/competing-censored state must be stable",
        "fine_only_event": "any event missed by 1/512 is material",
        "coarse_only_event_nesting_violation": "a finer nested grid cannot lose a coarse-grid event",
        "censor_state_changed": "censor class must be stable",
        "first_event_type_mismatch": "first-event mechanism/type must be stable",
        "material_distance_mismatch": "observed or competing-censor transition distances may differ only by 0 or one 1/1024 step; terminal censor distances must agree",
        "return_key_mismatch": "curve trajectory key sets must be identical",
        "return_state_mismatch": "never-changed/changed-no-return/changed-then-returned state must be stable",
    }
    for (category, estimand), count in sorted(counters.items()):
        if category.startswith("key_only_"):
            row_denominator = key_denominator
        elif category == "return_key_mismatch":
            row_denominator = return_denominator
        else:
            row_denominator = denominator
        decision_rows.append(decision_row(category, estimand, count, row_denominator, True, material_rules[category]))
    for (category, estimand), count in sorted(distance_classes.items()):
        material = category in {
            "one_missing_distance",
            "censor_distance_changed",
            "fine_event_later_nesting_violation",
            "coarse_missed_earlier_transition",
        }
        decision_rows.append(
            decision_row(
                f"distance:{category}",
                estimand,
                count,
                denominator,
                material,
                "0 or one fine-grid step earlier is quantization-only; larger/late shifts are material",
            )
        )

    primary_material_categories = {
        "key_only_1_512",
        "key_only_1_1024",
        "event_state_mismatch",
        "first_event_type_mismatch",
        "material_distance_mismatch",
        "return_key_mismatch",
        "return_state_mismatch",
    }
    material_count = sum(
        count for (category, _), count in counters.items() if category in primary_material_categories
    )
    status = "PASS_CONVERGED" if material_count == 0 else "FAIL_NOT_CONVERGED"
    order_equal = coarse_order == fine_order
    return {
        "protocol": "dense_grid_convergence_gate_v1",
        "status": status,
        "gate_pass": status == "PASS_CONVERGED",
        "grid_1_512": {
            "directory": str(coarse_dir),
            "fine_step": coarse_step,
            "trajectory_count": len(coarse_events),
            "event_csv_sha256": sha256(coarse_dir / EVENT_FILE),
            "curve_csv_sha256": sha256(coarse_dir / CURVE_FILE),
        },
        "grid_1_1024": {
            "directory": str(fine_dir),
            "fine_step": fine_step,
            "trajectory_count": len(fine_events),
            "event_csv_sha256": sha256(fine_dir / EVENT_FILE),
            "curve_csv_sha256": sha256(fine_dir / CURVE_FILE),
        },
        "key_audit": {
            "shared": len(shared_keys),
            "only_1_512": len(coarse_keys - fine_keys),
            "only_1_1024": len(fine_keys - coarse_keys),
            "row_order_equal": order_equal,
        },
        "fine_only_events_missed_by_1_512": dict(sorted(added_events.items())),
        "material_mismatch_count": material_count,
        "material_trajectory_count": len(material_keys),
        "material_trajectory_examples": [key_text(key) for key in sorted(material_keys)[:max_examples]],
        "mismatch_counts": {f"{category}:{estimand}": count for (category, estimand), count in sorted(counters.items())},
        "distance_class_counts": {f"{category}:{estimand}": count for (category, estimand), count in sorted(distance_classes.items())},
        "decision_table": decision_rows,
        "examples": examples,
        "materiality_rule": {
            "pass": "zero material mismatches",
            "quantization_only": "when both grids share the same observed or competing-censor transition type, the 1/1024 radius may equal the 1/512 radius or precede it by at most one 1/1024 step",
            "material": [
                "any trajectory-key mismatch",
                "any fine-only event missed by 1/512",
                "any event/censor/competing-censor state change",
                "any first-event-type change",
                "any observed or competing-censor transition shift larger than one 1/1024 step or any later fine-grid transition",
                "any terminal right-censor-distance change",
                "any return-transition-state change",
            ],
            "consequence": "FAIL_NOT_CONVERGED requires the full run to use 1/1024; this script never starts that run",
        },
        "return_states_compared": list(RETURN_STATES),
        "memory_model": "event rows plus seven compact return labels per trajectory; curve CSVs are streamed once and never loaded as dataframes",
    }


def write_decision_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("category", "estimand", "count", "denominator", "proportion", "material", "threshold", "decision", "rule")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_report(result: dict[str, object]) -> str:
    key_audit = result["key_audit"]
    lines = [
        "# Dense-grid convergence Gate 2",
        "",
        f"**Decision: {result['status']}**",
        "",
        f"Compared {result['grid_1_512']['trajectory_count']} 1/512 trajectories with "
        f"{result['grid_1_1024']['trajectory_count']} 1/1024 trajectories.",
        f"Shared keys: {key_audit['shared']}; only 1/512: {key_audit['only_1_512']}; "
        f"only 1/1024: {key_audit['only_1_1024']}.",
        f"Primary material mismatch instances: {result['material_mismatch_count']} across "
        f"{result['material_trajectory_count']} unique trajectories.",
        "",
        "## Materiality rule",
        "",
        "The gate passes only with zero material mismatches. A common observed or competing-censor transition localized "
        "at the same radius or exactly one 1/1024 step earlier is a normal half-step quantization refinement and is not an exact mismatch. Fine-only events, "
        "larger earlier shifts, later fine-grid events, state/type changes, key changes, and return-state changes are material.",
        "",
        "## Fine-only events missed by 1/512",
        "",
    ]
    missed = result["fine_only_events_missed_by_1_512"]
    if missed:
        lines.extend([f"- {name}: {count}" for name, count in missed.items()])
    else:
        lines.append("None.")
    lines.extend(["", "## Decision table", "", "| Category | Estimand | Count | Proportion | Material | Decision |", "|---|---:|---:|---:|---:|---:|"])
    rows = result["decision_table"]
    if rows:
        for row in rows:
            lines.append(
                f"| {row['category']} | {row['estimand']} | {row['count']} | {float(row['proportion']):.6f} | "
                f"{row['material']} | {row['decision']} |"
            )
    else:
        lines.append("| no mismatches | all | 0 | 0.000000 | False | PASS |")
    lines.extend(["", "## Representative examples", ""])
    examples = result["examples"]
    if examples:
        for example in examples:
            lines.append(f"- `{json.dumps(example, ensure_ascii=False, sort_keys=True)}`")
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Execution boundary",
            "",
            "This is a read-only comparison. It does not start or authorize a 300-image run.",
            "",
        ]
    )
    return "\n".join(lines)


def run_self_test() -> dict[str, object]:
    fine_step = 1.0 / 1024.0
    checks = {
        "same_sample_is_nonmaterial": compare_distance(0.125, 0.125, fine_step, True)[:2] == ("same_sample", False),
        "half_step_is_nonmaterial": compare_distance(0.125, 0.125 - fine_step, fine_step, True)[:2] == ("fine_half_step_localization", False),
        "larger_earlier_shift_is_material": compare_distance(0.125, 0.125 - 2 * fine_step, fine_step, True)[:2] == ("coarse_missed_earlier_transition", True),
        "later_fine_event_is_material": compare_distance(0.125, 0.125 + fine_step, fine_step, True)[:2] == ("fine_event_later_nesting_violation", True),
        "changed_censor_distance_is_material": compare_distance(0.25, 0.249, fine_step, False)[:2] == ("censor_distance_changed", True),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        raise AssertionError(checks)
    return {"status": status, "checks": checks}


def main() -> int:
    args = parse_args()
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test = run_self_test() if args.self_test else None
    result = compare_artifacts(args.grid_512, args.grid_1024, args.max_examples, args.allow_incomplete)
    if self_test is not None:
        result["self_test"] = self_test
    write_json(args.output_dir / "convergence_gate.json", result)
    write_decision_csv(args.output_dir / "decision_table.csv", result["decision_table"])
    (args.output_dir / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 2 if args.fail_on_material and not result["gate_pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
