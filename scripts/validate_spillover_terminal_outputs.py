"""Validate a completed assignment-spillover artifact without loading the model.

The validator streams the large pair file, reconciles it with focal and
per-image sufficient statistics, and deterministically replays the stratified
image-cluster bootstrap from frozen sufficient statistics.  A PASS report is
written only after every check succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("left", "right", "up", "down")
PRIMARY_RELATIONSHIPS = ("shared_native_claim_yes", "shared_native_claim_no")
SECONDARY_RELATIONSHIPS = ("shared_conflict", "local_nonsharing", "distant_nonsharing")
FOCAL_GROUPS = ("all", "focal_flip", "focal_stable")
FAMILIES = ("pair_spillover", "focal_any_spillover")
PRIMARY_DIFFERENCE_KEY = "pair_spillover|shared_native_claim_yes_minus_no|all"

SELECTED_FIELDS = [
    "dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"
]
FOCAL_FIELDS = [
    "image_id", "stratum", "sampling_weight", "focal_gt_id", "direction", "stress", "shift_pixels",
    "focal_size_bin", "focal_width", "focal_height", "focal_area", "focal_active_base", "focal_active_shift",
    "focal_flip", "nonfocal_neighbor_count", "nonfocal_spillover_count", "any_nonfocal_spillover",
    "shared_native_claim_yes_neighbor_count", "shared_native_claim_yes_spillover_count",
    "shared_native_claim_yes_any_spillover", "shared_native_claim_no_neighbor_count",
    "shared_native_claim_no_spillover_count", "shared_native_claim_no_any_spillover",
    "shared_conflict_neighbor_count", "shared_conflict_spillover_count", "shared_conflict_any_spillover",
    "local_nonsharing_neighbor_count", "local_nonsharing_spillover_count", "local_nonsharing_any_spillover",
    "distant_nonsharing_neighbor_count", "distant_nonsharing_spillover_count", "distant_nonsharing_any_spillover",
]
NONFOCAL_FIELDS = [
    "image_id", "stratum", "sampling_weight", "focal_gt_id", "neighbor_gt_id", "direction", "stress",
    "shift_pixels", "focal_size_bin", "focal_width", "focal_height", "focal_area", "neighbor_width",
    "neighbor_height", "neighbor_area", "local_distance_threshold", "shared_native_claim",
    "primary_relationship", "secondary_relationship", "relationship", "centre_distance_px_base",
    "normalized_centre_distance_base", "centre_distance_px_shift", "normalized_centre_distance_shift",
    "shared_claim_base", "shared_claim_shift", "shared_candidate_count_base", "shared_candidate_count_shift",
    "shared_candidate_count_union", "focal_won_shared_count_base", "neighbor_won_shared_count_base",
    "focal_won_shared_count_shift", "neighbor_won_shared_count_shift", "pair_ownership_transfer_count",
    "focal_active_base", "focal_active_shift", "focal_flip", "neighbor_active_base", "neighbor_active_shift",
    "neighbor_flip",
]
SUFFICIENT_FIELDS = [
    "image_id", "stratum", "sampling_weight", "metric_family", "estimand_role", "relationship",
    "focal_group", "raw_numerator", "raw_denominator",
]


def metric_keys() -> list[str]:
    return [
        f"{family}|{relationship}|{group}"
        for family in FAMILIES
        for relationship in ("overall", *PRIMARY_RELATIONSHIPS, *SECONDARY_RELATIONSHIPS)
        for group in FOCAL_GROUPS
    ]


METRIC_KEYS = metric_keys()
BOOTSTRAP_FIELDS = ["replicate", *METRIC_KEYS, PRIMARY_DIFFERENCE_KEY]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(a: float, b: float, atol: float, rtol: float = 1e-10) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_selected(path: Path) -> tuple[list[dict], dict[str, dict]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == SELECTED_FIELDS, "selected_images.csv schema mismatch")
        rows = list(reader)
    by_image: dict[str, dict] = {}
    indices = set()
    for row in rows:
        image_id = row["image_id"]
        index = int(row["dataset_index"])
        require(image_id not in by_image and index not in indices, "duplicate selected image/index")
        probability = float(row["inclusion_probability"])
        weight = float(row["sampling_weight"])
        require(0 < probability <= 1 and weight > 0, "invalid selected probability/weight")
        require(close(weight, 1.0 / probability, 1e-10), "sampling_weight is not reciprocal probability")
        row = dict(row)
        row["dataset_index"] = index
        row["inclusion_probability"] = probability
        row["sampling_weight"] = weight
        by_image[image_id] = row
        indices.add(index)
    return rows, by_image


def validate_source_selected(config: dict, selected_by_image: dict[str, dict]) -> None:
    source_path = resolve(ROOT, config["selected_images"])
    with source_path.open(newline="", encoding="utf-8-sig") as stream:
        source = {row["image_id"]: row for row in csv.DictReader(stream)}
    require(set(source) == set(selected_by_image), "selected output/source image roster mismatch")
    for image_id, observed in selected_by_image.items():
        expected = source[image_id]
        require(int(expected["dataset_index"]) == observed["dataset_index"], "dataset index drift")
        require(expected["stratum"] == observed["stratum"], "selection stratum drift")
        require(close(float(expected["inclusion_probability"]), observed["inclusion_probability"], 1e-12), "probability drift")
        require(close(float(expected["sampling_weight"]), observed["sampling_weight"], 1e-12), "weight drift")


def validate_focal(path: Path, selected: dict[str, dict], config: dict, atol: float) -> dict:
    records: dict[tuple[str, int, str], dict] = {}
    counts = Counter()
    spillovers = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FOCAL_FIELDS, "focal_events.csv schema mismatch")
        for row in reader:
            image_id = row["image_id"]
            require(image_id in selected, "focal image not in selected roster")
            require(row["stratum"] == selected[image_id]["stratum"], "focal stratum mismatch")
            require(close(float(row["sampling_weight"]), selected[image_id]["sampling_weight"], atol), "focal weight mismatch")
            require(row["direction"] in DIRECTIONS and row["stress"] == config["stress_mode"], "focal stress/direction mismatch")
            area = float(row["focal_area"])
            require(area > 0 and row["focal_size_bin"] in {"t_8_16", "s_16_32"}, "invalid focal area/bin")
            expected_delta = float(config["kappa"]) * math.sqrt(area)
            require(close(float(row["shift_pixels"]), expected_delta, atol), "focal shift contract mismatch")
            key = (image_id, int(row["focal_gt_id"]), row["direction"])
            require(key not in records, "duplicate focal key")
            focal_flip = int(row["focal_flip"])
            require(focal_flip in {0, 1}, "invalid focal flip")
            record = {
                "stratum": row["stratum"],
                "weight": float(row["sampling_weight"]),
                "focal_flip": focal_flip,
                "neighbor_count": int(row["nonfocal_neighbor_count"]),
                "spillover_count": int(row["nonfocal_spillover_count"]),
                "primary_counts": {}, "primary_spillovers": {},
                "secondary_counts": {}, "secondary_spillovers": {},
            }
            require(int(row["any_nonfocal_spillover"]) == int(record["spillover_count"] > 0), "focal any-spillover mismatch")
            for relationship in PRIMARY_RELATIONSHIPS:
                n = int(row[f"{relationship}_neighbor_count"])
                y = int(row[f"{relationship}_spillover_count"])
                require(0 <= y <= n, "invalid focal primary counts")
                require(int(row[f"{relationship}_any_spillover"]) == int(y > 0), "primary any mismatch")
                record["primary_counts"][relationship] = n
                record["primary_spillovers"][relationship] = y
            for relationship in SECONDARY_RELATIONSHIPS:
                n = int(row[f"{relationship}_neighbor_count"])
                y = int(row[f"{relationship}_spillover_count"])
                require(0 <= y <= n, "invalid focal secondary counts")
                require(int(row[f"{relationship}_any_spillover"]) == int(y > 0), "secondary any mismatch")
                record["secondary_counts"][relationship] = n
                record["secondary_spillovers"][relationship] = y
            require(sum(record["primary_counts"].values()) == record["neighbor_count"], "focal primary partition mismatch")
            require(sum(record["secondary_counts"].values()) == record["neighbor_count"], "focal secondary partition mismatch")
            require(sum(record["primary_spillovers"].values()) == record["spillover_count"], "focal primary spillover mismatch")
            require(sum(record["secondary_spillovers"].values()) == record["spillover_count"], "focal secondary spillover mismatch")
            records[key] = record
            counts[row["focal_size_bin"]] += 1
            spillovers["focal_flip"] += focal_flip
    return {"records": records, "size_counts": dict(counts), "focal_flips": spillovers["focal_flip"]}


def validate_nonfocal(path: Path, selected: dict[str, dict], focal_records: dict, config: dict, atol: float) -> dict:
    per_focal: dict[tuple[str, int, str], dict] = defaultdict(
        lambda: {
            "neighbors": set(), "count": 0, "spillovers": 0,
            "primary_counts": Counter(), "primary_spillovers": Counter(),
            "secondary_counts": Counter(), "secondary_spillovers": Counter(),
        }
    )
    primary_counts = Counter()
    secondary_counts = Counter()
    primary_spillovers = Counter()
    secondary_spillovers = Counter()
    focal_group_counts = Counter()
    total = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == NONFOCAL_FIELDS, "nonfocal_events.csv schema mismatch")
        for row in reader:
            total += 1
            image_id = row["image_id"]
            require(image_id in selected, "nonfocal image not selected")
            require(row["stratum"] == selected[image_id]["stratum"], "nonfocal stratum mismatch")
            require(close(float(row["sampling_weight"]), selected[image_id]["sampling_weight"], atol), "nonfocal weight mismatch")
            require(row["direction"] in DIRECTIONS and row["stress"] == config["stress_mode"], "nonfocal stress/direction mismatch")
            focal_gt = int(row["focal_gt_id"])
            neighbor_gt = int(row["neighbor_gt_id"])
            require(focal_gt != neighbor_gt, "focal equals neighbor")
            key = (image_id, focal_gt, row["direction"])
            require(key in focal_records, "nonfocal row has no focal parent")
            parent = focal_records[key]
            require(neighbor_gt not in per_focal[key]["neighbors"], "duplicate neighbor within focal event")
            per_focal[key]["neighbors"].add(neighbor_gt)
            shared = int(row["shared_native_claim"])
            require(shared in {0, 1}, "invalid shared claim")
            primary = row["primary_relationship"]
            secondary = row["secondary_relationship"]
            require(primary == ("shared_native_claim_yes" if shared else "shared_native_claim_no"), "primary relation mismatch")
            require(secondary in SECONDARY_RELATIONSHIPS and row["relationship"] == secondary, "secondary relation mismatch")
            require((secondary == "shared_conflict") == bool(shared), "shared/secondary inconsistency")
            distance = float(row["normalized_centre_distance_base"])
            require(math.isfinite(distance) and distance >= 0, "invalid normalized distance")
            if not shared:
                threshold = float(row["local_distance_threshold"])
                expected = "local_nonsharing" if distance < threshold else "distant_nonsharing"
                require(secondary == expected, "local/distant relation mismatch")
            neighbor_flip = int(row["neighbor_flip"])
            focal_flip = int(row["focal_flip"])
            require(neighbor_flip in {0, 1} and focal_flip == parent["focal_flip"], "flip mismatch")
            bucket = per_focal[key]
            bucket["count"] += 1
            bucket["spillovers"] += neighbor_flip
            bucket["primary_counts"][primary] += 1
            bucket["primary_spillovers"][primary] += neighbor_flip
            bucket["secondary_counts"][secondary] += 1
            bucket["secondary_spillovers"][secondary] += neighbor_flip
            primary_counts[primary] += 1
            secondary_counts[secondary] += 1
            primary_spillovers[primary] += neighbor_flip
            secondary_spillovers[secondary] += neighbor_flip
            focal_group_counts["focal_flip" if focal_flip else "focal_stable"] += 1
    expected_pair_support = {key for key, record in focal_records.items() if record["neighbor_count"] > 0}
    require(set(per_focal) == expected_pair_support, "focal/nonfocal parent support mismatch")
    for key, expected in focal_records.items():
        observed = per_focal.get(key, {
            "count": 0, "spillovers": 0,
            "primary_counts": Counter(), "primary_spillovers": Counter(),
            "secondary_counts": Counter(), "secondary_spillovers": Counter(),
        })
        require(observed["count"] == expected["neighbor_count"], "focal neighbor total mismatch")
        require(observed["spillovers"] == expected["spillover_count"], "focal spillover total mismatch")
        require(dict(observed["primary_counts"]) == {k: v for k, v in expected["primary_counts"].items() if v}, "focal primary pair mismatch")
        require({k: v for k, v in observed["primary_spillovers"].items() if v} == {k: v for k, v in expected["primary_spillovers"].items() if v}, "focal primary spillover mismatch")
        require(dict(observed["secondary_counts"]) == {k: v for k, v in expected["secondary_counts"].items() if v}, "focal secondary pair mismatch")
        require({k: v for k, v in observed["secondary_spillovers"].items() if v} == {k: v for k, v in expected["secondary_spillovers"].items() if v}, "focal secondary spillover mismatch")
    return {
        "rows": total,
        "primary_counts": dict(primary_counts),
        "secondary_counts": dict(secondary_counts),
        "primary_spillovers": dict(primary_spillovers),
        "secondary_spillovers": dict(secondary_spillovers),
        "focal_group_pair_counts": dict(focal_group_counts),
    }


def validate_sufficient(path: Path, selected: dict[str, dict], atol: float) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    image_ids = sorted(selected)
    image_index = {image_id: i for i, image_id in enumerate(image_ids)}
    metric_index = {key: i for i, key in enumerate(METRIC_KEYS)}
    numerators = np.zeros((len(image_ids), len(METRIC_KEYS)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    seen = set()
    strata = [selected[image_id]["stratum"] for image_id in image_ids]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == SUFFICIENT_FIELDS, "sufficient-statistics schema mismatch")
        rows = 0
        for row in reader:
            rows += 1
            image_id = row["image_id"]
            require(image_id in selected and row["stratum"] == selected[image_id]["stratum"], "sufficient roster mismatch")
            require(close(float(row["sampling_weight"]), selected[image_id]["sampling_weight"], atol), "sufficient weight mismatch")
            key = f"{row['metric_family']}|{row['relationship']}|{row['focal_group']}"
            require(key in metric_index, "unknown sufficient metric key")
            compound = (image_id, key)
            require(compound not in seen, "duplicate sufficient cell")
            seen.add(compound)
            numerator = int(row["raw_numerator"])
            denominator = int(row["raw_denominator"])
            require(0 <= numerator <= denominator, "invalid sufficient numerator/denominator")
            numerators[image_index[image_id], metric_index[key]] = numerator
            denominators[image_index[image_id], metric_index[key]] = denominator
    require(rows == len(image_ids) * len(METRIC_KEYS), "unexpected sufficient-statistics row count")
    weights = np.array([selected[image_id]["sampling_weight"] for image_id in image_ids], dtype=np.float64)
    observed_num = (numerators * weights[:, None]).sum(axis=0)
    observed_den = (denominators * weights[:, None]).sum(axis=0)
    observed_rate = np.divide(observed_num, observed_den, out=np.full_like(observed_num, np.nan), where=observed_den > 0)
    return numerators, denominators, image_ids, strata, {
        "rows": rows, "weights": weights, "weighted_num": observed_num,
        "weighted_den": observed_den, "rate": observed_rate,
        "raw_num": numerators.sum(axis=0), "raw_den": denominators.sum(axis=0),
    }


def summary_cell(summary: dict, key: str) -> dict:
    family, relationship, group = key.split("|")
    return summary["estimates"][family][relationship][group]


def validate_summary_from_sufficient(summary: dict, observed: dict, atol: float) -> float:
    max_abs = 0.0
    for j, key in enumerate(METRIC_KEYS):
        cell = summary_cell(summary, key)
        require(int(cell["raw_numerator"]) == int(observed["raw_num"][j]), f"summary raw numerator mismatch: {key}")
        require(int(cell["raw_denominator"]) == int(observed["raw_den"][j]), f"summary raw denominator mismatch: {key}")
        require(close(cell["weighted_numerator"], observed["weighted_num"][j], atol), f"weighted numerator mismatch: {key}")
        require(close(cell["weighted_denominator"], observed["weighted_den"][j], atol), f"weighted denominator mismatch: {key}")
        expected_rate = observed["rate"][j]
        if np.isnan(expected_rate):
            require(cell["rate"] is None, f"expected null rate: {key}")
        else:
            diff = abs(float(cell["rate"]) - float(expected_rate))
            max_abs = max(max_abs, diff)
            require(diff <= atol, f"summary rate mismatch: {key}")
    return max_abs


def replay_bootstrap(
    bootstrap_path: Path,
    numerators: np.ndarray,
    denominators: np.ndarray,
    image_ids: list[str],
    strata: list[str],
    weights: np.ndarray,
    seed: int,
    expected_replicates: int,
    atol: float,
) -> tuple[dict[str, np.ndarray], float]:
    by_stratum: dict[str, np.ndarray] = {}
    strata_array = np.array(strata, dtype=object)
    for stratum in sorted(set(strata)):
        by_stratum[stratum] = np.where(strata_array == stratum)[0]
    distributions = {key: np.full(expected_replicates, np.nan, dtype=np.float64) for key in (*METRIC_KEYS, PRIMARY_DIFFERENCE_KEY)}
    rng = np.random.default_rng(seed)
    max_abs = 0.0
    with bootstrap_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == BOOTSTRAP_FIELDS, "bootstrap schema mismatch")
        for replicate, row in enumerate(reader):
            require(replicate < expected_replicates, "too many bootstrap rows")
            require(int(row["replicate"]) == replicate, "bootstrap replicate sequence mismatch")
            multiplicity = np.zeros(len(image_ids), dtype=np.float64)
            for stratum in sorted(by_stratum):
                indices = by_stratum[stratum]
                sampled = rng.choice(indices, size=len(indices), replace=True)
                multiplicity += np.bincount(sampled, minlength=len(image_ids))
            factor = weights * multiplicity
            weighted_num = (numerators * factor[:, None]).sum(axis=0)
            weighted_den = (denominators * factor[:, None]).sum(axis=0)
            expected = np.divide(weighted_num, weighted_den, out=np.full_like(weighted_num, np.nan), where=weighted_den > 0)
            for j, key in enumerate(METRIC_KEYS):
                value = float(row[key]) if row[key] != "" else math.nan
                distributions[key][replicate] = value
                if np.isnan(expected[j]):
                    require(np.isnan(value), f"bootstrap expected null: {replicate} {key}")
                else:
                    diff = abs(value - expected[j])
                    max_abs = max(max_abs, diff)
                    require(diff <= atol, f"bootstrap mismatch: {replicate} {key}")
            yes = expected[METRIC_KEYS.index("pair_spillover|shared_native_claim_yes|all")]
            no = expected[METRIC_KEYS.index("pair_spillover|shared_native_claim_no|all")]
            expected_diff = yes - no if np.isfinite(yes) and np.isfinite(no) else math.nan
            value = float(row[PRIMARY_DIFFERENCE_KEY]) if row[PRIMARY_DIFFERENCE_KEY] != "" else math.nan
            distributions[PRIMARY_DIFFERENCE_KEY][replicate] = value
            if np.isnan(expected_diff):
                require(np.isnan(value), "bootstrap difference expected null")
            else:
                diff = abs(value - expected_diff)
                max_abs = max(max_abs, diff)
                require(diff <= atol, f"bootstrap difference mismatch: {replicate}")
        require(replicate + 1 == expected_replicates, "bootstrap row count mismatch")
    return distributions, max_abs


def validate_intervals(summary: dict, distributions: dict[str, np.ndarray], atol: float) -> float:
    max_abs = 0.0
    for key in METRIC_KEYS:
        values = distributions[key][np.isfinite(distributions[key])]
        cell = summary_cell(summary, key)
        expected_lower, expected_upper = np.quantile(values, [0.025, 0.975]) if len(values) else (math.nan, math.nan)
        for label, expected in (("ci95_lower", expected_lower), ("ci95_upper", expected_upper)):
            actual = cell[label]
            if np.isnan(expected):
                require(actual is None, f"expected null interval: {key}")
            else:
                diff = abs(float(actual) - float(expected))
                max_abs = max(max_abs, diff)
                require(diff <= atol, f"interval mismatch: {key} {label}")
    values = distributions[PRIMARY_DIFFERENCE_KEY][np.isfinite(distributions[PRIMARY_DIFFERENCE_KEY])]
    expected_lower, expected_upper = np.quantile(values, [0.025, 0.975])
    diff_cell = summary["primary_estimands"]["shared_minus_nonshared_rate_difference"]
    for label, expected in (("ci95_lower", expected_lower), ("ci95_upper", expected_upper)):
        diff = abs(float(diff_cell[label]) - float(expected))
        max_abs = max(max_abs, diff)
        require(diff <= atol, f"primary difference interval mismatch: {label}")
    return max_abs


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    artifact = run_dir / "artifact"
    config = load_json(run_dir / "CONFIG.json")
    status = load_json(run_dir / "STATUS.json")
    summary = load_json(artifact / "summary.json")
    manifest = load_json(artifact / "manifest.json")
    output = args.output or artifact / "TERMINAL_VALIDATION.json"
    output = output.resolve()
    require(not output.exists(), f"validation output already exists: {output}")

    core_files = {
        "selected_images.csv", "focal_events.csv", "nonfocal_events.csv",
        "per_image_sufficient_statistics.csv", "bootstrap_replicates.csv",
        "summary.json", "manifest.json",
    }
    actual_core = {path.name for path in artifact.iterdir() if path.is_file()}
    require(actual_core == core_files, f"unexpected pre-validation artifact set: {sorted(actual_core)}")
    config_hash = sha256(run_dir / "CONFIG.json")
    require(config_hash == status["config_sha256"], "CONFIG/STATUS hash mismatch")
    require(summary["status"] == manifest["status"] == "complete", "summary/manifest not complete")
    require(config["run_id"] == status["run_id"] == "20260901_spillover_formal300_v1", "run ID mismatch")
    require(manifest["protocol"] == config["protocol"], "protocol mismatch")
    require(manifest["stress"] == config["stress_mode"] == "equivalent_side", "stress mismatch")
    require(close(manifest["kappa"], config["kappa"], args.atol), "kappa mismatch")
    require(manifest["bootstrap_replicates"] == summary["bootstrap_replicates"] == config["bootstrap_replicates"] == 5000, "bootstrap count mismatch")
    require(manifest["bootstrap_seed"] == config["bootstrap_seed"], "bootstrap seed mismatch")
    require(manifest["source_selected_images"] == manifest["audited_selected_images"] == summary["selected_images"] == config["expected_selected_images"] == 300, "selected count mismatch")
    require(manifest["selection_mode"] == summary["selection_mode"] == "full", "selection mode mismatch")
    require(manifest["script_sha256"] == config["script_sha256"] == sha256(resolve(ROOT, config["script"])), "producer script drift")
    require(manifest["checkpoint_sha256"] == config["checkpoint_sha256"] == sha256(Path(config["checkpoint"])), "checkpoint drift")
    require(manifest["selected_images_source_sha256"] == config["selected_images_sha256"] == sha256(resolve(ROOT, config["selected_images"])), "selected source drift")
    gate_summary = ROOT / "runs" / config["gate_source_run"] / "artifact" / "summary.json"
    require(sha256(gate_summary) == config["gate_source_summary_sha256"], "pilot gate summary drift")
    data_path = resolve(ROOT, config["data"])
    data_hash = sha256(data_path)
    require(data_hash == args.expected_data_sha256.lower(), "contemporaneous data YAML hash drift")

    selected_rows, selected = read_selected(artifact / "selected_images.csv")
    require(len(selected_rows) == 300, "selected row count mismatch")
    validate_source_selected(config, selected)
    focal = validate_focal(artifact / "focal_events.csv", selected, config, args.atol)
    require(len(focal["records"]) == summary["focal_direction_events"] == manifest["focal_direction_events"], "focal row count mismatch")
    nonfocal = validate_nonfocal(artifact / "nonfocal_events.csv", selected, focal["records"], config, args.atol)
    require(nonfocal["rows"] == summary["nonfocal_pair_events"] == manifest["nonfocal_pair_events"], "nonfocal row count mismatch")
    require(nonfocal["primary_counts"] == summary["primary_relationship_raw_pair_counts"], "summary primary pair counts mismatch")
    require(nonfocal["secondary_counts"] == summary["relationship_raw_pair_counts"], "summary secondary pair counts mismatch")
    require(sum(nonfocal["primary_counts"].values()) == nonfocal["rows"], "primary pair partition incomplete")
    require(sum(nonfocal["secondary_counts"].values()) == nonfocal["rows"], "secondary pair partition incomplete")
    require(nonfocal["primary_spillovers"].get("shared_native_claim_yes", 0) == 52, "unexpected shared spillover count")
    require(nonfocal["primary_spillovers"].get("shared_native_claim_no", 0) == 0, "unexpected nonshared spillover count")

    numerators, denominators, image_ids, strata, observed = validate_sufficient(
        artifact / "per_image_sufficient_statistics.csv", selected, args.atol
    )
    max_summary_abs = validate_summary_from_sufficient(summary, observed, args.atol)
    distributions, max_bootstrap_abs = replay_bootstrap(
        artifact / "bootstrap_replicates.csv", numerators, denominators, image_ids, strata,
        observed["weights"], int(config["bootstrap_seed"]), int(config["bootstrap_replicates"]), args.atol,
    )
    max_interval_abs = validate_intervals(summary, distributions, args.atol)

    producer_outputs = set(manifest["outputs"])
    require(producer_outputs == core_files - {"manifest.json"}, "manifest outputs mismatch")
    report = {
        "status": "PASS",
        "protocol": "assignment_spillover_terminal_validation_v1",
        "run_id": config["run_id"],
        "classification": "validated_gated_discovery_output_not_confirmatory",
        "counts": {
            "selected_images": len(selected_rows),
            "focal_direction_events": len(focal["records"]),
            "nonfocal_pair_events": nonfocal["rows"],
            "sufficient_statistic_rows": observed["rows"],
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
            "shared_native_claim_yes_pairs": nonfocal["primary_counts"]["shared_native_claim_yes"],
            "shared_native_claim_no_pairs": nonfocal["primary_counts"]["shared_native_claim_no"],
            "shared_native_claim_yes_spillovers": nonfocal["primary_spillovers"]["shared_native_claim_yes"],
            "shared_native_claim_no_spillovers": nonfocal["primary_spillovers"].get("shared_native_claim_no", 0),
        },
        "numeric_checks": {
            "max_summary_rate_abs_diff": max_summary_abs,
            "max_bootstrap_replay_abs_diff": max_bootstrap_abs,
            "max_interval_abs_diff": max_interval_abs,
            "tolerance": args.atol,
        },
        "hashes": {
            "validator_script_sha256": sha256(Path(__file__)),
            "config_sha256": config_hash,
            "producer_script_sha256": manifest["script_sha256"],
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "selected_source_sha256": manifest["selected_images_source_sha256"],
            "pilot_gate_summary_sha256": config["gate_source_summary_sha256"],
            "data_yaml_sha256_contemporaneous": data_hash,
        },
        "validity_notes": [
            "The run configuration recorded the data-YAML path but did not prebind its hash.",
            "The contemporaneous preflight and post-run hash match, but this is not a prospective data-YAML hash binding.",
            "This artifact is gated discovery evidence only; it is neither causal nor confirmatory lockbox evidence.",
            "Zero observed nonshared spillovers do not establish equivalence or absence.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
