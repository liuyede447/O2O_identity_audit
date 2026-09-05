"""Evaluate frozen confirmatory families on a revalidated sealed bundle.

Outcomes are opened only after validating the instrument, preregistration,
selection, extraction manifests, raw-artifact hashes, sealed-ready sidecar,
family definitions, and outcome-access timestamp ordering.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import validate_sealed_lockbox_bundle as sealed_validator_module
from validate_sealed_lockbox_bundle import validate_existing_ready_bundle


SIZES = ("t_8_16", "s_16_32")
PATHWAYS = (
    "eligibility_boundary",
    "topk_membership_transition",
    "conflict_reassignment",
    "within_set_geometry_rank_reversal",
)
ALPHA = 0.05
BOOTSTRAP_TEST = "one_sided_empirical_bootstrap_p"
PRIMARY = {
    "H1": {"metric": "normalized_paired_gap", "direction": "greater_than_zero"},
    "H2": {"metric": "normalized_o2m_gap", "direction": "less_than_zero"},
    "H3": {"metric": "fixed_minus_normalized_o2o_gap", "direction": "greater_than_zero"},
    "H4": {"metric": "fixed_eligibility_minus_normalized", "direction": "greater_than_zero"},
    "H5": {"metric": "normalized_within_minus_fixed", "direction": "greater_than_zero"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path)
    parser.add_argument("--normalized-dir", type=Path)
    parser.add_argument("--fixed-anatomy", type=Path)
    parser.add_argument("--normalized-anatomy", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--selected-manifest", type=Path)
    parser.add_argument("--instrument-manifest", type=Path)
    parser.add_argument("--sealed-ready", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--secondary-results", type=Path)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.self_test:
        required = (
            "fixed_dir",
            "normalized_dir",
            "fixed_anatomy",
            "normalized_anatomy",
            "preregistration",
            "selected_manifest",
            "instrument_manifest",
            "sealed_ready",
            "output_dir",
        )
        missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error("the following arguments are required unless --self-test is used: " + ", ".join(missing))
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError(f"{label} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def create_outcome_access_receipt(
    *,
    sealed_ready: Path,
    output_dir: Path,
    ready_hash: str,
    ready_payload: dict,
    preregistration_hash: str,
    reps: int,
    seed: int,
) -> tuple[Path, dict, str, datetime]:
    receipt_path = sealed_ready.resolve().parent / "OUTCOME_ACCESS_RECEIPT.json"
    receipt_sidecar = Path(str(receipt_path) + ".sha256")
    timestamp = datetime.now(timezone.utc)
    payload = {
        "status": "OUTCOME_ACCESS_OPENED",
        "outcome_access_is_irreversible": True,
        "outcome_access_timestamp_utc": timestamp.isoformat(),
        "sealed_bundle_ready_sha256": ready_hash,
        "preregistration_sha256": preregistration_hash,
        "selection_manifest_sha256": ready_payload["selection_manifest_sha256"],
        "instrument_manifest_sha256": ready_payload["instrument_manifest_sha256"],
        "extraction_manifest_sha256": ready_payload["extraction_manifest_sha256"],
        "instrument_source_commit": ready_payload["instrument_source_commit"],
        "preregistration_commit": ready_payload["preregistration_commit"],
        "selection_commit": ready_payload["selection_commit"],
        "bootstrap_replicates": reps,
        "confirmatory_seed": seed,
        "intended_output_dir": str(output_dir.resolve()),
        "analyzer_sha256": sha256(Path(__file__)),
    }
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(receipt_path, flags)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive receipt, even if partial, intentionally remains as an
        # irreversible marker that outcome access may have begun.
        raise
    receipt_hash = sha256(receipt_path)
    with receipt_sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{receipt_hash}  {receipt_path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_path, payload, receipt_hash, timestamp


def validate_family(
    family: object,
    *,
    name: str,
    expected_procedure: str,
    expected_order: list[str],
) -> None:
    if not isinstance(family, dict):
        raise RuntimeError(f"{name} family definition is missing")
    if family.get("procedure") != expected_procedure:
        raise RuntimeError(f"{name} procedure must be {expected_procedure!r}")
    if family.get("bootstrap_test") != BOOTSTRAP_TEST:
        raise RuntimeError(f"{name} bootstrap_test must be {BOOTSTRAP_TEST!r}")
    if not np.isclose(float(family.get("alpha", np.nan)), ALPHA, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{name} alpha must be {ALPHA}")
    if family.get("order") != expected_order:
        raise RuntimeError(f"{name} order must be {expected_order!r}")
    hypotheses = family.get("hypotheses")
    if not isinstance(hypotheses, dict) or set(hypotheses) != set(expected_order):
        raise RuntimeError(f"{name} hypotheses must be exactly {expected_order!r}")
    for hypothesis_id in expected_order:
        definition = hypotheses[hypothesis_id]
        if not isinstance(definition, dict):
            raise RuntimeError(f"{hypothesis_id} definition must be an object")
        for field in ("metric", "direction"):
            if definition.get(field) != PRIMARY[hypothesis_id][field]:
                raise RuntimeError(f"{hypothesis_id} {field} must be {PRIMARY[hypothesis_id][field]!r}")
        if not isinstance(definition.get("estimand"), str) or not definition["estimand"].strip():
            raise RuntimeError(f"{hypothesis_id} must declare a non-empty estimand")


def validate_secondary_family(plan: dict) -> list[dict]:
    family = plan.get("secondary_confirmatory_family")
    if family is None:
        return []
    if not isinstance(family, dict):
        raise RuntimeError("secondary_confirmatory_family must be an object")
    if family.get("procedure") != "holm":
        raise RuntimeError("secondary family procedure must be 'holm'")
    if family.get("bootstrap_test") != BOOTSTRAP_TEST:
        raise RuntimeError(f"secondary bootstrap_test must be {BOOTSTRAP_TEST!r}")
    if not np.isclose(float(family.get("alpha", np.nan)), ALPHA, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"secondary family alpha must be {ALPHA}")
    hypotheses = family.get("hypotheses", [])
    if not isinstance(hypotheses, list) or len(hypotheses) > 2:
        raise RuntimeError("secondary family must contain an explicit list of at most two hypotheses")
    identifiers: set[str] = set()
    sample_keys: set[str] = set()
    for definition in hypotheses:
        if not isinstance(definition, dict):
            raise RuntimeError("each secondary hypothesis must be an object")
        identifier = definition.get("id")
        sample_key = definition.get("sample_key")
        direction = definition.get("direction")
        estimand = definition.get("estimand")
        if not isinstance(identifier, str) or not identifier or identifier in PRIMARY or identifier in identifiers:
            raise RuntimeError("secondary hypothesis IDs must be unique, non-empty, and distinct from H1--H5")
        if not isinstance(sample_key, str) or not sample_key or sample_key in sample_keys:
            raise RuntimeError("secondary sample_key values must be unique non-empty strings")
        if direction not in ("greater_than_zero", "less_than_zero"):
            raise RuntimeError(f"unsupported direction for secondary hypothesis {identifier}")
        if not isinstance(estimand, str) or not estimand.strip():
            raise RuntimeError(f"secondary hypothesis {identifier} must declare a non-empty estimand")
        identifiers.add(identifier)
        sample_keys.add(sample_key)
    return hypotheses


def validate_preregistration(
    path: Path,
    *,
    reps: int,
    seed: int,
    access_timestamp: datetime,
) -> tuple[dict, str, datetime, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prereg_hash = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("preregistration SHA-256 sidecar is missing")
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if not tokens or len(tokens[0]) != 64 or tokens[0].lower() != prereg_hash:
        raise RuntimeError("preregistration SHA-256 sidecar is malformed or mismatched")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_BEFORE_OUTCOMES":
        raise RuntimeError("preregistration is not frozen before outcomes")
    frozen_timestamp = parse_utc_timestamp(plan.get("frozen_at_utc"), "frozen_at_utc")
    if frozen_timestamp >= access_timestamp:
        raise RuntimeError("outcome access must occur strictly after preregistration freeze")
    statistics = plan.get("statistics")
    if not isinstance(statistics, dict):
        raise RuntimeError("statistics definition is missing")
    if statistics.get("bootstrap_replicates") != reps:
        raise RuntimeError("--reps differs from the frozen bootstrap_replicates")
    if statistics.get("confirmatory_seed") != seed:
        raise RuntimeError("--seed differs from the frozen confirmatory_seed")
    families = plan.get("confirmatory_families")
    if not isinstance(families, dict) or set(families) != {"tier_a", "tier_b"}:
        raise RuntimeError("confirmatory_families must contain exactly tier_a and tier_b")
    validate_family(
        families["tier_a"],
        name="tier_a",
        expected_procedure="hierarchical_gatekeeping",
        expected_order=["H1", "H2", "H3"],
    )
    validate_family(
        families["tier_b"],
        name="tier_b",
        expected_procedure="holm",
        expected_order=["H4", "H5"],
    )
    secondary = validate_secondary_family(plan)
    return plan, prereg_hash, frozen_timestamp, secondary


def ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def branch_stats(vector: np.ndarray) -> dict:
    o2o_gap = vector[0] - vector[1]
    o2m_gap = vector[2] - vector[3]
    return {"o2o_gap": float(o2o_gap), "o2m_gap": float(o2m_gap), "paired_gap": float(o2o_gap - o2m_gap)}


def normalize_selection_roster(selection: pd.DataFrame) -> pd.DataFrame:
    required = {"stratum", "image_id"}
    if not required.issubset(selection.columns):
        raise RuntimeError("sealed selection roster must contain image_id and stratum")
    roster = selection[["stratum", "image_id"]].drop_duplicates().sort_values(
        ["stratum", "image_id"]
    ).reset_index(drop=True)
    if roster.empty or roster["image_id"].duplicated().any():
        raise RuntimeError("sealed selection roster is empty or assigns an image to multiple strata")
    return roster


def branch_image_aggregate(
    fixed: pd.DataFrame,
    normalized: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["image_id", "gt_id"]
    selection_roster = normalize_selection_roster(selection)
    fixed_roster = fixed[["stratum", "image_id"]].drop_duplicates().sort_values(
        ["stratum", "image_id"]
    ).reset_index(drop=True)
    normalized_roster = normalized[["stratum", "image_id"]].drop_duplicates().sort_values(
        ["stratum", "image_id"]
    ).reset_index(drop=True)
    if not fixed_roster.equals(selection_roster) or not normalized_roster.equals(selection_roster):
        raise RuntimeError("fixed/normalized branch raw image rosters differ from the sealed selection")
    fixed = fixed[fixed["common_valid_margin"].eq(1)].copy()
    normalized = normalized[normalized["common_valid_margin"].eq(1)].copy()
    fixed_keys = set(map(tuple, fixed[keys].to_numpy()))
    normalized_keys = set(map(tuple, normalized[keys].to_numpy()))
    if fixed_keys != normalized_keys:
        raise RuntimeError(
            "fixed/normalized common-valid GT support differs; silent inner-merge dropping is forbidden"
        )
    merged = fixed.merge(normalized, on=keys, suffixes=("_fixed", "_norm"), validate="one_to_one")
    if not merged["size_bin_fixed"].equals(merged["size_bin_norm"]):
        raise RuntimeError("fixed/normalized size bins differ")
    if not merged["stratum_fixed"].equals(merged["stratum_norm"]):
        raise RuntimeError("fixed/normalized strata differ")
    if not np.allclose(merged["sampling_weight_fixed"], merged["sampling_weight_norm"]):
        raise RuntimeError("fixed/normalized sampling weights differ")
    merged["size_bin"] = merged["size_bin_fixed"]
    merged["stratum"] = merged["stratum_fixed"]
    merged["sampling_weight"] = merged["sampling_weight_fixed"].astype(float)
    columns = []
    for contract, suffix in (("fixed", "fixed"), ("norm", "norm")):
        for branch in ("o2o", "o2m"):
            for size in SIZES:
                den = f"{contract}_{branch}_{size}_den"
                num = f"{contract}_{branch}_{size}_num"
                mask = merged["size_bin"].eq(size)
                merged[den] = merged["sampling_weight"] * mask
                merged[num] = merged["sampling_weight"] * merged[f"{branch}_fragile_{suffix}"] * mask
                columns.extend([num, den])
    aggregate = merged.groupby(["stratum", "image_id"], as_index=False)[columns].sum()
    result = selection_roster.merge(
        aggregate,
        on=["stratum", "image_id"],
        how="left",
        validate="one_to_one",
    )
    result[columns] = result[columns].fillna(0.0)
    return result


def branch_from_aggregate(frame: pd.DataFrame, contract: str) -> dict:
    vector = []
    for branch in ("o2o", "o2m"):
        for size in SIZES:
            denominator = frame[f"{contract}_{branch}_{size}_den"].sum()
            if denominator <= 0:
                raise RuntimeError(f"zero denominator for {contract}/{branch}/{size}")
            vector.append(frame[f"{contract}_{branch}_{size}_num"].sum() / denominator)
    return branch_stats(np.asarray(vector))


def pathway_aggregate(
    fixed: pd.DataFrame,
    normalized: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    selection_roster = normalize_selection_roster(selection)
    rosters = []
    for label, frame in (("fixed", fixed), ("normalized", normalized)):
        roster = frame[["stratum", "image_id"]].drop_duplicates().sort_values(
            ["stratum", "image_id"]
        ).reset_index(drop=True)
        if roster["image_id"].duplicated().any():
            raise RuntimeError(f"{label} anatomy assigns one image_id to multiple strata")
        rosters.append(roster)
    if not rosters[0].equals(selection_roster) or not rosters[1].equals(selection_roster):
        raise RuntimeError("fixed/normalized anatomy raw image rosters differ from the sealed selection")

    result = selection_roster.copy()
    for contract, frame in (("fixed", fixed), ("norm", normalized)):
        subset = frame[frame["o2o_flip"].eq(1)].copy()
        subset[f"{contract}_den"] = subset["sampling_weight"].astype(float)
        for pathway in PATHWAYS:
            subset[f"{contract}_{pathway}"] = subset["sampling_weight"] * subset["first_divergence"].eq(pathway)
        cols = [f"{contract}_den"] + [f"{contract}_{pathway}" for pathway in PATHWAYS]
        aggregate = subset.groupby(["stratum", "image_id"], as_index=False)[cols].sum()
        result = result.merge(
            aggregate,
            on=["stratum", "image_id"],
            how="left",
            validate="one_to_one",
        )
        result[cols] = result[cols].fillna(0.0)
    return result


def pathway_stats(frame: pd.DataFrame) -> dict:
    fixed_denominator = frame["fixed_den"].sum()
    norm_denominator = frame["norm_den"].sum()
    if fixed_denominator <= 0 or norm_denominator <= 0:
        raise RuntimeError("pathway fractions require at least one O2O flip in each contract")
    return {
        "fixed_eligibility_minus_normalized": float(
            frame["fixed_eligibility_boundary"].sum() / fixed_denominator
            - frame["norm_eligibility_boundary"].sum() / norm_denominator
        ),
        "normalized_within_minus_fixed": float(
            frame["norm_within_set_geometry_rank_reversal"].sum() / norm_denominator
            - frame["fixed_within_set_geometry_rank_reversal"].sum() / fixed_denominator
        ),
    }


def resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("stratum"):
        indices = rng.integers(0, len(group), size=len(group))
        pieces.append(group.iloc[indices])
    return pd.concat(pieces, ignore_index=True)


def empirical_one_sided_p(values: np.ndarray, direction: str) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("bootstrap samples must be a finite, non-empty one-dimensional array")
    if direction == "greater_than_zero":
        opposite = np.count_nonzero(values <= 0.0)
    elif direction == "less_than_zero":
        opposite = np.count_nonzero(values >= 0.0)
    else:
        raise RuntimeError(f"unsupported direction: {direction}")
    return float((opposite + 1) / (len(values) + 1))


def one_sided_bound(values: np.ndarray, direction: str, alpha: float = ALPHA) -> dict:
    if direction == "greater_than_zero":
        return {"kind": "lower", "level": 1.0 - alpha, "value": float(np.quantile(values, alpha))}
    return {"kind": "upper", "level": 1.0 - alpha, "value": float(np.quantile(values, 1.0 - alpha))}


def holm_adjust(raw_p_values: dict[str, float], alpha: float = ALPHA) -> tuple[dict[str, float], dict[str, bool]]:
    ordered = sorted(raw_p_values, key=lambda identifier: (raw_p_values[identifier], identifier))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, identifier in enumerate(ordered):
        running = max(running, (count - index) * raw_p_values[identifier])
        adjusted[identifier] = min(1.0, running)
    rejected: dict[str, bool] = {}
    gate_open = True
    for index, identifier in enumerate(ordered):
        passes = gate_open and raw_p_values[identifier] <= alpha / (count - index)
        rejected[identifier] = passes
        if not passes:
            gate_open = False
    return adjusted, rejected


def tier_a_results(points: dict[str, float], samples: dict[str, np.ndarray]) -> dict:
    output = {}
    gate_open = True
    for identifier in ("H1", "H2", "H3"):
        definition = PRIMARY[identifier]
        metric = definition["metric"]
        raw_p = empirical_one_sided_p(samples[metric], definition["direction"])
        raw_pass = raw_p <= ALPHA
        if gate_open:
            formal_status = "formally_tested_pass" if raw_pass else "formally_tested_fail"
            formal_pass: bool | None = raw_pass
            if not raw_pass:
                gate_open = False
        else:
            formal_status = "descriptive_not_formally_tested"
            formal_pass = None
        output[identifier] = {
            "metric": metric,
            "direction": definition["direction"],
            "point_estimate": points[metric],
            "ci_95_two_sided": ci(samples[metric]),
            "one_sided_bound": one_sided_bound(samples[metric], definition["direction"]),
            "raw_empirical_p": raw_p,
            "raw_directional_pass": raw_pass,
            "formal_status": formal_status,
            "formal_pass": formal_pass,
        }
    return output


def holm_family_results(
    definitions: list[dict],
    points: dict[str, float],
    samples: dict[str, np.ndarray],
) -> dict:
    raw = {
        definition["id"]: empirical_one_sided_p(samples[definition["sample_key"]], definition["direction"])
        for definition in definitions
    }
    adjusted, rejected = holm_adjust(raw)
    return {
        definition["id"]: {
            "metric": definition["sample_key"],
            "direction": definition["direction"],
            "point_estimate": points[definition["sample_key"]],
            "ci_95_two_sided": ci(samples[definition["sample_key"]]),
            "one_sided_bound": one_sided_bound(samples[definition["sample_key"]], definition["direction"]),
            "raw_empirical_p": raw[definition["id"]],
            "holm_adjusted_p": adjusted[definition["id"]],
            "raw_directional_pass": raw[definition["id"]] <= ALPHA,
            "formal_status": "formally_tested_pass" if rejected[definition["id"]] else "formally_tested_fail",
            "formal_pass": rejected[definition["id"]],
        }
        for definition in definitions
    }


def load_secondary_results(path: Path, definitions: list[dict], reps: int) -> tuple[dict, dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points_source = payload.get("points")
    samples_source = payload.get("bootstrap_samples")
    if not isinstance(points_source, dict) or not isinstance(samples_source, dict):
        raise RuntimeError("secondary results must contain points and bootstrap_samples objects")
    expected = {definition["sample_key"] for definition in definitions}
    if set(points_source) != expected or set(samples_source) != expected:
        raise RuntimeError("secondary result keys must exactly match preregistered sample_key values")
    points = {key: float(points_source[key]) for key in expected}
    samples = {key: np.asarray(samples_source[key], dtype=float) for key in expected}
    if not all(np.isfinite(value) for value in points.values()):
        raise RuntimeError("secondary point estimates must be finite")
    if any(values.ndim != 1 or len(values) != reps or not np.isfinite(values).all() for values in samples.values()):
        raise RuntimeError("each secondary bootstrap sample must contain exactly the frozen number of finite replicates")
    return points, samples, sha256(path)


def evaluate(
    points: dict[str, float],
    samples: dict[str, np.ndarray],
    secondary_definitions: list[dict],
    secondary_points: dict[str, float] | None = None,
    secondary_samples: dict[str, np.ndarray] | None = None,
) -> tuple[dict, dict, dict]:
    tier_a = tier_a_results(points, samples)
    tier_b_definitions = [
        {"id": identifier, "sample_key": PRIMARY[identifier]["metric"], "direction": PRIMARY[identifier]["direction"]}
        for identifier in ("H4", "H5")
    ]
    tier_b = holm_family_results(tier_b_definitions, points, samples)
    if secondary_definitions:
        if secondary_points is None or secondary_samples is None:
            raise RuntimeError("preregistered secondary hypotheses require --secondary-results")
        secondary = holm_family_results(secondary_definitions, secondary_points, secondary_samples)
    else:
        secondary = {}
    return tier_a, tier_b, secondary


def format_number(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.8g}"


def decision_label(result: dict) -> str:
    if result["formal_status"] == "descriptive_not_formally_tested":
        return "DESCRIPTIVE"
    return "PASS" if result["formal_pass"] else "FAIL"


def family_table(results: dict) -> str:
    lines = [
        "| ID | Formal/gate status | Point | 95% CI | One-sided empirical p | Holm-adjusted p | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for identifier, result in results.items():
        interval = result["ci_95_two_sided"]
        lines.append(
            "| {identifier} | `{status}` | {point} | [{lower}, {upper}] | {raw_p} | {adjusted_p} | **{decision}** |".format(
                identifier=identifier,
                status=result["formal_status"],
                point=format_number(result["point_estimate"]),
                lower=format_number(interval[0]),
                upper=format_number(interval[1]),
                raw_p=format_number(result["raw_empirical_p"]),
                adjusted_p=format_number(result.get("holm_adjusted_p")),
                decision=decision_label(result),
            )
        )
    return "\n".join(lines)


def render_report(payload: dict) -> str:
    overall = (
        "ALL_PRIMARY_FORMALLY_CONFIRMED"
        if payload["all_primary_formally_confirmed"]
        else "ONE_OR_MORE_PRIMARY_NOT_FORMALLY_CONFIRMED"
    )
    lines = [
        "# Lockbox confirmatory verdict",
        "",
        f"- Normative verdict: `LOCKBOX_VERDICT.json`",
        f"- Overall status: **{overall}**",
        f"- Preregistration SHA-256: `{payload['preregistration_sha256']}`",
        f"- Preregistration frozen at: `{payload['preregistration_frozen_at_utc']}`",
        f"- Outcome-access receipt SHA-256: `{payload.get('outcome_access_receipt_sha256')}`",
        f"- Outcome access at: `{payload['outcome_access_timestamp_utc']}`",
        f"- Bootstrap replicates: {payload['bootstrap_replicates']}",
        "",
        "## Tier A: hierarchical gatekeeping (H1 -> H2 -> H3)",
        "",
        family_table(payload["tier_a"]),
        "",
        "Once a Tier A hypothesis fails its formal test, all later hypotheses remain reported but are descriptive and not formally tested.",
        "",
        "## Tier B: Holm family",
        "",
        family_table(payload["tier_b"]),
        "",
        "## Secondary confirmatory family",
        "",
    ]
    if payload["secondary_family"]:
        lines.append(family_table(payload["secondary_family"]))
    else:
        lines.append("No secondary confirmatory hypotheses were preregistered.")
    lines.extend(
        [
            "",
            "## One-time rule",
            "",
            "Every preregistered verdict is retained. A FAIL or descriptive result does not trigger resampling, threshold changes, estimator changes, replacement hypotheses, or a replacement lockbox sample.",
            "",
            "`confirmatory_results.json` is a compatibility copy. `LOCKBOX_VERDICT.json` is the only normative verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def write_output_bundle(output_dir: Path, payload: dict) -> dict[str, str]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    normative_payload = {
        **payload,
        "artifact_role": "normative_verdict",
        "normative_verdict_file": "LOCKBOX_VERDICT.json",
    }
    verdict = output_dir / "LOCKBOX_VERDICT.json"
    report = output_dir / "LOCKBOX_REPORT.md"
    compatibility = output_dir / "confirmatory_results.json"
    verdict.write_text(json.dumps(normative_payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    report.write_text(render_report(normative_payload), encoding="utf-8", newline="\n")
    compatibility_payload = {
        **normative_payload,
        "artifact_role": "compatibility_copy",
        "compatibility_notice": "LOCKBOX_VERDICT.json is the normative verdict.",
    }
    compatibility.write_text(
        json.dumps(compatibility_payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in (verdict, report, compatibility)
    }
    sums = output_dir / "SHA256SUMS.txt"
    sums.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in artifact_hashes.items()),
        encoding="ascii",
        newline="\n",
    )
    return {**artifact_hashes, sums.name: sha256(sums)}


def run_self_test() -> dict[str, object]:
    points = {definition["metric"]: 0.2 for definition in PRIMARY.values()}
    points["normalized_o2m_gap"] = 0.1
    samples = {
        "normalized_paired_gap": np.asarray([0.1] * 99),
        "normalized_o2m_gap": np.asarray([-0.1] * 40 + [0.1] * 59),
        "fixed_minus_normalized_o2o_gap": np.asarray([0.1] * 99),
        "fixed_eligibility_minus_normalized": np.asarray([0.1] * 99),
        "normalized_within_minus_fixed": np.asarray([0.1] * 97 + [-0.1] * 2),
    }
    tier_a, tier_b, secondary = evaluate(points, samples, [])
    assert tier_a["H1"]["formal_status"] == "formally_tested_pass"
    assert tier_a["H2"]["formal_status"] == "formally_tested_fail"
    assert tier_a["H3"]["formal_status"] == "descriptive_not_formally_tested"
    assert tier_a["H3"]["raw_directional_pass"] is True
    assert all(result["formal_pass"] for result in tier_b.values())
    assert secondary == {}
    secondary_definitions = [
        {
            "id": "S1",
            "sample_key": "boundary_metric",
            "direction": "greater_than_zero",
            "estimand": "synthetic boundary estimand",
        }
    ]
    _, _, secondary = evaluate(
        points,
        samples,
        secondary_definitions,
        {"boundary_metric": 0.1},
        {"boundary_metric": np.asarray([0.1] * 99)},
    )
    assert secondary["S1"]["formal_status"] == "formally_tested_pass"

    fixed_pathway_rows = pd.DataFrame(
        [
            {"stratum": "t_only", "image_id": "flip_image", "sampling_weight": 2.0, "o2o_flip": 1, "first_divergence": "eligibility_boundary"},
            {"stratum": "t_only", "image_id": "zero_flip_image", "sampling_weight": 2.0, "o2o_flip": 0, "first_divergence": "right_censored"},
        ]
    )
    normalized_pathway_rows = pd.DataFrame(
        [
            {"stratum": "t_only", "image_id": "flip_image", "sampling_weight": 2.0, "o2o_flip": 1, "first_divergence": "within_set_geometry_rank_reversal"},
            {"stratum": "t_only", "image_id": "zero_flip_image", "sampling_weight": 2.0, "o2o_flip": 0, "first_divergence": "right_censored"},
        ]
    )
    synthetic_selection = pd.DataFrame(
        [
            {"stratum": "t_only", "image_id": "flip_image"},
            {"stratum": "t_only", "image_id": "zero_flip_image"},
        ]
    )
    pathway_clusters = pathway_aggregate(
        fixed_pathway_rows,
        normalized_pathway_rows,
        synthetic_selection,
    )
    assert len(pathway_clusters) == 2
    assert bool((pathway_clusters[["fixed_den", "norm_den"]].sum(axis=1) == 0).any())

    fixed_branch_rows = pd.DataFrame(
        [
            {"stratum": "t_only", "image_id": "valid_image", "gt_id": 0, "common_valid_margin": 1, "size_bin": "t_8_16", "sampling_weight": 2.0, "o2o_fragile": 1, "o2m_fragile": 0},
            {"stratum": "t_only", "image_id": "zero_support_image", "gt_id": 0, "common_valid_margin": 0, "size_bin": "t_8_16", "sampling_weight": 2.0, "o2o_fragile": 0, "o2m_fragile": 0},
        ]
    )
    normalized_branch_rows = fixed_branch_rows.copy()
    branch_selection = pd.DataFrame(
        [
            {"stratum": "t_only", "image_id": "valid_image"},
            {"stratum": "t_only", "image_id": "zero_support_image"},
        ]
    )
    branch_clusters = branch_image_aggregate(
        fixed_branch_rows,
        normalized_branch_rows,
        branch_selection,
    )
    assert len(branch_clusters) == 2
    denominator_columns = [column for column in branch_clusters if column.endswith("_den")]
    assert bool((branch_clusters[denominator_columns].sum(axis=1) == 0).any())
    mismatched_support = normalized_branch_rows.copy()
    mismatched_support.loc[mismatched_support["image_id"].eq("valid_image"), "common_valid_margin"] = 0
    try:
        branch_image_aggregate(fixed_branch_rows, mismatched_support, branch_selection)
    except RuntimeError as exc:
        assert "common-valid GT support differs" in str(exc)
    else:
        raise AssertionError("mismatched common-valid branch support was silently dropped")

    frozen = datetime(2026, 8, 31, 1, 0, tzinfo=timezone.utc)
    plan = {
        "status": "FROZEN_BEFORE_OUTCOMES",
        "frozen_at_utc": frozen.isoformat(),
        "statistics": {"bootstrap_replicates": 99, "confirmatory_seed": 7},
        "confirmatory_families": {
            "tier_a": {
                "procedure": "hierarchical_gatekeeping",
                "bootstrap_test": BOOTSTRAP_TEST,
                "alpha": ALPHA,
                "order": ["H1", "H2", "H3"],
                "hypotheses": {
                    identifier: {**PRIMARY[identifier], "estimand": f"synthetic {identifier}"}
                    for identifier in ("H1", "H2", "H3")
                },
            },
            "tier_b": {
                "procedure": "holm",
                "bootstrap_test": BOOTSTRAP_TEST,
                "alpha": ALPHA,
                "order": ["H4", "H5"],
                "hypotheses": {
                    identifier: {**PRIMARY[identifier], "estimand": f"synthetic {identifier}"}
                    for identifier in ("H4", "H5")
                },
            },
        },
        "secondary_confirmatory_family": {
            "procedure": "holm",
            "bootstrap_test": BOOTSTRAP_TEST,
            "alpha": ALPHA,
            "hypotheses": [],
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        preregistration = Path(directory) / "plan.json"
        preregistration.write_text(json.dumps(plan), encoding="utf-8")
        digest = sha256(preregistration)
        preregistration.with_suffix(".json.sha256").write_text(f"{digest}  plan.json\n", encoding="ascii")
        validated = validate_preregistration(
            preregistration,
            reps=99,
            seed=7,
            access_timestamp=datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
        )
        assert validated[1] == digest

        try:
            validate_preregistration(
                preregistration,
                reps=99,
                seed=7,
                access_timestamp=frozen,
            )
        except RuntimeError as exc:
            assert "strictly after" in str(exc)
        else:
            raise AssertionError("invalid outcome-access timestamp was accepted")

        bundle_payload = {
            "status": "complete",
            "protocol": "synthetic_self_test",
            "tier_a": tier_a,
            "tier_b": tier_b,
            "secondary_family": secondary,
            "all_primary_formally_confirmed": False,
            "bootstrap_replicates": 99,
            "seed": 7,
            "preregistration_sha256": digest,
            "preregistration_frozen_at_utc": frozen.isoformat(),
            "outcome_access_timestamp_utc": datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc).isoformat(),
        }
        bundle = Path(directory) / "bundle"
        output_hashes = write_output_bundle(bundle, bundle_payload)
        assert set(output_hashes) == {
            "LOCKBOX_VERDICT.json",
            "LOCKBOX_REPORT.md",
            "confirmatory_results.json",
            "SHA256SUMS.txt",
        }
        normative = json.loads((bundle / "LOCKBOX_VERDICT.json").read_text(encoding="utf-8"))
        compatibility = json.loads((bundle / "confirmatory_results.json").read_text(encoding="utf-8"))
        report = (bundle / "LOCKBOX_REPORT.md").read_text(encoding="utf-8")
        assert normative["artifact_role"] == "normative_verdict"
        assert compatibility["artifact_role"] == "compatibility_copy"
        assert compatibility["normative_verdict_file"] == "LOCKBOX_VERDICT.json"
        assert "descriptive_not_formally_tested" in report and "Holm-adjusted p" in report
        assert "does not trigger resampling" in report
        for line in (bundle / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
            expected_hash, name = line.split(maxsplit=1)
            assert sha256(bundle / name) == expected_hash
        try:
            write_output_bundle(bundle, bundle_payload)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing output directory was overwritten")

        preregistration.write_text(json.dumps({**plan, "status": "CHANGED"}), encoding="utf-8")
        try:
            validate_preregistration(
                preregistration,
                reps=99,
                seed=7,
                access_timestamp=datetime(2026, 8, 31, 2, 0, tzinfo=timezone.utc),
            )
        except RuntimeError as exc:
            assert "mismatched" in str(exc)
        else:
            raise AssertionError("mismatched preregistration hash was accepted")

        ready_path = Path(directory) / "SEALED_BUNDLE_READY.json"
        ready_path.write_text("{}\n", encoding="utf-8", newline="\n")
        ready_payload = {
            "selection_manifest_sha256": "1" * 64,
            "instrument_manifest_sha256": "2" * 64,
            "extraction_manifest_sha256": {"fixed_branch": "3" * 64},
            "instrument_source_commit": "4" * 40,
            "preregistration_commit": "5" * 40,
            "selection_commit": "6" * 40,
        }
        receipt_path, _, receipt_hash, _ = create_outcome_access_receipt(
            sealed_ready=ready_path,
            output_dir=Path(directory) / "formal_output",
            ready_hash="7" * 64,
            ready_payload=ready_payload,
            preregistration_hash="8" * 64,
            reps=99,
            seed=7,
        )
        assert receipt_path.is_file() and sha256(receipt_path) == receipt_hash
        try:
            create_outcome_access_receipt(
                sealed_ready=ready_path,
                output_dir=Path(directory) / "different_output",
                ready_hash="7" * 64,
                ready_payload=ready_payload,
                preregistration_hash="8" * 64,
                reps=99,
                seed=7,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("a second outcome-access receipt was accepted")
    return {"status": "PASS", "tests": 26}


def main() -> None:
    args = parse_args()
    if args.self_test:
        result = run_self_test()
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
        return
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    from create_instrument_freeze_bundle import validate_frozen_script_identity

    freeze_root = args.instrument_manifest.resolve().parent
    validate_frozen_script_identity(
        freeze_root,
        Path(__file__),
        use_tool_manifest=True,
    )
    validate_frozen_script_identity(
        freeze_root,
        Path(sealed_validator_module.__file__),
        use_tool_manifest=True,
    )

    ready_payload, ready_hash = validate_existing_ready_bundle(
        fixed_branch=args.fixed_dir,
        normalized_branch=args.normalized_dir,
        fixed_anatomy=args.fixed_anatomy,
        normalized_anatomy=args.normalized_anatomy,
        selected_manifest_path=args.selected_manifest,
        preregistration_path=args.preregistration,
        instrument_manifest_path=args.instrument_manifest,
        ready_path=args.sealed_ready,
        secondary_results=args.secondary_results,
    )
    selection_roster = pd.read_csv(
        args.selected_manifest.resolve().parent / "selected_images.csv",
        usecols=["image_id", "stratum"],
    )
    normalize_selection_roster(selection_roster)

    # Complete every check that does not interpret outcomes before creating the
    # irreversible receipt. A crash after receipt creation remains a consumed,
    # incomplete lockbox rather than authorizing a second prospective run.
    preflight_timestamp = datetime.now(timezone.utc)
    _, prereg_hash, frozen_timestamp, secondary_definitions = validate_preregistration(
        args.preregistration,
        reps=args.reps,
        seed=args.seed,
        access_timestamp=preflight_timestamp,
    )
    if secondary_definitions and (args.secondary_results is None or not args.secondary_results.is_file()):
        raise RuntimeError("preregistered secondary hypotheses require an existing --secondary-results file")
    if not secondary_definitions and args.secondary_results is not None:
        raise RuntimeError("--secondary-results was supplied but no secondary hypotheses were preregistered")

    receipt_path, _, receipt_hash, access_timestamp = create_outcome_access_receipt(
        sealed_ready=args.sealed_ready,
        output_dir=args.output_dir,
        ready_hash=ready_hash,
        ready_payload=ready_payload,
        preregistration_hash=prereg_hash,
        reps=args.reps,
        seed=args.seed,
    )
    access_timestamp_source = "atomic_irreversible_receipt_before_first_outcome_read"
    _, prereg_hash, frozen_timestamp, secondary_definitions = validate_preregistration(
        args.preregistration,
        reps=args.reps,
        seed=args.seed,
        access_timestamp=access_timestamp,
    )

    # Outcome files are intentionally opened only after all frozen-chain gates pass.
    fixed = pd.read_csv(args.fixed_dir / "per_gt.csv")
    normalized = pd.read_csv(args.normalized_dir / "per_gt.csv")
    fixed_a = pd.read_csv(args.fixed_anatomy / "per_direction_anatomy.csv")
    norm_a = pd.read_csv(args.normalized_anatomy / "per_direction_anatomy.csv")
    branch_images = branch_image_aggregate(fixed, normalized, selection_roster)
    pathway_images = pathway_aggregate(fixed_a, norm_a, selection_roster)
    branch_point_fixed = branch_from_aggregate(branch_images, "fixed")
    branch_point_norm = branch_from_aggregate(branch_images, "norm")
    pathway_point = pathway_stats(pathway_images)
    points = {
        "normalized_paired_gap": branch_point_norm["paired_gap"],
        "normalized_o2m_gap": branch_point_norm["o2m_gap"],
        "fixed_minus_normalized_o2o_gap": branch_point_fixed["o2o_gap"] - branch_point_norm["o2o_gap"],
        **pathway_point,
    }

    rng = np.random.default_rng(args.seed)
    samples = {key: [] for key in points}
    for _ in range(args.reps):
        branch_sample = resample(branch_images, rng)
        pathway_sample = resample(pathway_images, rng)
        fixed_stats = branch_from_aggregate(branch_sample, "fixed")
        norm_stats = branch_from_aggregate(branch_sample, "norm")
        path_stats = pathway_stats(pathway_sample)
        samples["normalized_paired_gap"].append(norm_stats["paired_gap"])
        samples["normalized_o2m_gap"].append(norm_stats["o2m_gap"])
        samples["fixed_minus_normalized_o2o_gap"].append(fixed_stats["o2o_gap"] - norm_stats["o2o_gap"])
        samples["fixed_eligibility_minus_normalized"].append(path_stats["fixed_eligibility_minus_normalized"])
        samples["normalized_within_minus_fixed"].append(path_stats["normalized_within_minus_fixed"])
    sample_arrays = {key: np.asarray(values, dtype=float) for key, values in samples.items()}

    secondary_points = None
    secondary_samples = None
    secondary_sha256 = None
    if secondary_definitions:
        if args.secondary_results is None:
            raise RuntimeError("preregistered secondary hypotheses require --secondary-results")
        secondary_points, secondary_samples, secondary_sha256 = load_secondary_results(
            args.secondary_results, secondary_definitions, args.reps
        )
    elif args.secondary_results is not None:
        raise RuntimeError("--secondary-results was supplied but no secondary hypotheses were preregistered")

    tier_a, tier_b, secondary = evaluate(
        points,
        sample_arrays,
        secondary_definitions,
        secondary_points,
        secondary_samples,
    )
    payload = {
        "status": "complete",
        "protocol": "prospective_lockbox_gatekeeping_v2",
        "decision_contract": {
            "alpha": ALPHA,
            "bootstrap_test": BOOTSTRAP_TEST,
            "tier_a": "H1 -> H2 -> H3 hierarchical gatekeeping",
            "tier_b": "Holm familywise correction across H4 and H5",
            "secondary": "Holm familywise correction across at most two explicitly preregistered hypotheses",
        },
        "tier_a": tier_a,
        "tier_b": tier_b,
        "secondary_family": secondary,
        "all_tier_a_formally_confirmed": all(result["formal_pass"] is True for result in tier_a.values()),
        "all_tier_b_formally_confirmed": all(result["formal_pass"] is True for result in tier_b.values()),
        "all_primary_formally_confirmed": all(
            result["formal_pass"] is True for result in [*tier_a.values(), *tier_b.values()]
        ),
        "bootstrap_replicates": args.reps,
        "seed": args.seed,
        "preregistration_sha256": prereg_hash,
        "preregistration_frozen_at_utc": frozen_timestamp.isoformat(),
        "sealed_bundle_ready_sha256": ready_hash,
        "selection_manifest_sha256": ready_payload["selection_manifest_sha256"],
        "instrument_manifest_sha256": ready_payload["instrument_manifest_sha256"],
        "extraction_manifest_sha256": ready_payload["extraction_manifest_sha256"],
        "selected_images_sha256": ready_payload["selected_images_sha256"],
        "instrument_source_commit": ready_payload["instrument_source_commit"],
        "instrument_source_tag": ready_payload["instrument_source_tag"],
        "preregistration_commit": ready_payload["preregistration_commit"],
        "preregistration_tag": ready_payload["preregistration_tag"],
        "selection_commit": ready_payload["selection_commit"],
        "selection_tag": ready_payload["selection_tag"],
        "outcome_access_receipt": str(receipt_path),
        "outcome_access_receipt_sha256": receipt_hash,
        "outcome_access_timestamp_utc": access_timestamp.isoformat(),
        "outcome_access_timestamp_source": access_timestamp_source,
        "secondary_results_sha256": secondary_sha256,
    }
    output_hashes = write_output_bundle(args.output_dir, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "all_primary_formally_confirmed": payload["all_primary_formally_confirmed"],
                "normative_verdict": str(args.output_dir / "LOCKBOX_VERDICT.json"),
                "file_sha256": output_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
