"""Offline same-stride sensitivity analysis for frozen O2O/O2M replay tables.

This analysis performs no model inference or training. Its strict object-level
estimand retains a ground truth only when every recorded legal replay direction
has a shifted active candidate on the same stride as the base candidate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SCALE_TINY = "t_8_16"
SCALE_SMALL = "s_16_32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-selected", type=Path, required=True)
    parser.add_argument("--ai-o2o", type=Path, required=True)
    parser.add_argument("--ai-o2m", type=Path, required=True)
    parser.add_argument("--vis-selected", type=Path, required=True)
    parser.add_argument("--vis-o2o", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def missing(value: str | None) -> bool:
    return value in (None, "", "None", "nan")


def selected_contract(path: Path) -> dict[str, dict[str, str | float]]:
    return {
        row["image_id"]: {
            "stratum": row["stratum"],
            "weight": float(row["sampling_weight"]),
        }
        for row in read_csv(path)
    }


def collapse_branch(
    path: Path,
    selected: dict[str, dict[str, str | float]],
    branch: str,
) -> list[dict]:
    if branch == "o2o":
        candidate_0, candidate_p = "o2o_candidate_0", "o2o_candidate_p"
        margin_0, flip = "o2o_margin_0", "o2o_flip"
    elif branch == "o2m":
        candidate_0, candidate_p = "o2m_rank_candidate_0", "o2m_rank_candidate_p"
        margin_0, flip = "o2m_rank_margin_0", "o2m_rank_flip"
    else:
        raise ValueError(branch)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["image_id"], row["gt_id"])].append(row)

    result = []
    for (image_id, gt_id), directions in grouped.items():
        first = directions[0]
        if image_id not in selected:
            raise RuntimeError(f"audit image absent from sampling contract: {image_id}")
        expected = int(first["valid_perturbations"])
        base_valid = (
            not missing(first[candidate_0])
            and not missing(first[margin_0])
            and not missing(first["stride_0"])
        )
        complete = expected > 0 and len(directions) == expected
        shifted_present = complete and all(
            not missing(row[candidate_p]) and not missing(row["stride_p"])
            for row in directions
        )
        strict_same_stride = bool(
            base_valid
            and shifted_present
            and all(row["stride_p"] == row["stride_0"] for row in directions)
        )
        same_directions = [
            row
            for row in directions
            if base_valid
            and not missing(row[candidate_p])
            and not missing(row["stride_p"])
            and row["stride_p"] == row["stride_0"]
        ]
        result.append(
            {
                "image_id": image_id,
                "gt_id": gt_id,
                "stratum": selected[image_id]["stratum"],
                "weight": float(selected[image_id]["weight"]),
                "size_bin": first["size_bin"],
                "base_valid": base_valid,
                "complete": complete,
                "strict_same_stride": strict_same_stride,
                "any_flip": float(any(int(row[flip]) for row in directions)),
                "any_cross_stride": float(
                    base_valid
                    and any(
                        not missing(row["stride_p"])
                        and row["stride_p"] != row["stride_0"]
                        for row in directions
                    )
                ),
                "any_shift_missing": float(
                    base_valid
                    and any(missing(row[candidate_p]) or missing(row["stride_p"]) for row in directions)
                ),
                "same_direction_n": len(same_directions),
                "same_direction_flips": sum(int(row[flip]) for row in same_directions),
            }
        )
    return result


def weighted_rate(rows: list[dict], outcome: str, size_bin: str, counts: dict[str, int] | None = None) -> float:
    numerator = denominator = 0.0
    for row in rows:
        if row["size_bin"] != size_bin:
            continue
        multiplicity = 1 if counts is None else counts.get(row["image_id"], 0)
        if not multiplicity:
            continue
        weight = row["weight"] * multiplicity
        numerator += weight * row[outcome]
        denominator += weight
    if denominator == 0:
        raise RuntimeError(f"empty scale subset: {size_bin}")
    return numerator / denominator


def weighted_direction_rate(rows: list[dict], size_bin: str, counts: dict[str, int] | None = None) -> float:
    numerator = denominator = 0.0
    for row in rows:
        if row["size_bin"] != size_bin or not row["base_valid"]:
            continue
        multiplicity = 1 if counts is None else counts.get(row["image_id"], 0)
        if not multiplicity:
            continue
        weight = row["weight"] * multiplicity
        numerator += weight * row["same_direction_flips"]
        denominator += weight * row["same_direction_n"]
    if denominator == 0:
        raise RuntimeError(f"empty same-stride directional subset: {size_bin}")
    return numerator / denominator


def bootstrap_contrast(
    rows: list[dict],
    selected: dict[str, dict[str, str | float]],
    replicates: int,
    seed: int,
    mode: str,
    outcome: str = "any_flip",
) -> list[float]:
    """Bootstrap a tiny-minus-small rate contrast from per-image sufficient statistics."""
    by_image = {image_id: np.zeros(4, dtype=float) for image_id in selected}
    for row in rows:
        if row["size_bin"] not in {SCALE_TINY, SCALE_SMALL}:
            continue
        offset = 0 if row["size_bin"] == SCALE_TINY else 2
        weight = row["weight"]
        if mode == "object":
            by_image[row["image_id"]][offset] += weight * row[outcome]
            by_image[row["image_id"]][offset + 1] += weight
        elif mode == "direction":
            if not row["base_valid"]:
                continue
            by_image[row["image_id"]][offset] += weight * row["same_direction_flips"]
            by_image[row["image_id"]][offset + 1] += weight * row["same_direction_n"]
        else:
            raise ValueError(mode)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for image_id, meta in selected.items():
        grouped[str(meta["stratum"])].append(by_image[image_id])
    matrices = [np.stack(grouped[key]) for key in sorted(grouped)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        total = np.zeros(4, dtype=float)
        for matrix in matrices:
            sampled = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            total += matrix[sampled].sum(axis=0)
        if total[1] > 0 and total[3] > 0:
            values.append(float(total[0] / total[1] - total[2] / total[3]))
    return values


def bootstrap_paired_contrasts(
    rows: list[dict],
    selected: dict[str, dict[str, str | float]],
    replicates: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    """Bootstrap paired O2O, O2M, and difference-in-scale-gradient contrasts."""
    by_image = {image_id: np.zeros(6, dtype=float) for image_id in selected}
    for row in rows:
        if row["size_bin"] not in {SCALE_TINY, SCALE_SMALL}:
            continue
        offset = 0 if row["size_bin"] == SCALE_TINY else 3
        weight = row["weight"]
        vector = by_image[row["image_id"]]
        vector[offset] += weight * row["any_o2o"]
        vector[offset + 1] += weight * row["any_o2m"]
        vector[offset + 2] += weight
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for image_id, meta in selected.items():
        grouped[str(meta["stratum"])].append(by_image[image_id])
    matrices = [np.stack(grouped[key]) for key in sorted(grouped)]
    rng = np.random.default_rng(seed)
    o2o_values, o2m_values, difference_values = [], [], []
    for _ in range(replicates):
        total = np.zeros(6, dtype=float)
        for matrix in matrices:
            sampled = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            total += matrix[sampled].sum(axis=0)
        if total[2] <= 0 or total[5] <= 0:
            continue
        o2o = total[0] / total[2] - total[3] / total[5]
        o2m = total[1] / total[2] - total[4] / total[5]
        o2o_values.append(float(o2o))
        o2m_values.append(float(o2m))
        difference_values.append(float(o2o - o2m))
    return o2o_values, o2m_values, difference_values


def percentile(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "bootstrap_median": float(np.median(array)),
        "ci_95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "replicates_ok": int(array.size),
    }


def coverage(rows: list[dict], predicate: str) -> dict:
    output = {}
    for size_bin in (SCALE_TINY, SCALE_SMALL):
        valid = [row for row in rows if row["size_bin"] == size_bin and row["base_valid"]]
        kept = [row for row in valid if row[predicate]]
        valid_weight = sum(row["weight"] for row in valid)
        kept_weight = sum(row["weight"] for row in kept)
        output[size_bin] = {
            "valid_gt": len(valid),
            "retained_gt": len(kept),
            "weighted_retention": kept_weight / valid_weight if valid_weight else None,
            "weighted_any_cross_stride": (
                sum(row["weight"] * row["any_cross_stride"] for row in valid) / valid_weight
                if valid_weight
                else None
            ),
            "weighted_any_shift_missing": (
                sum(row["weight"] * row["any_shift_missing"] for row in valid) / valid_weight
                if valid_weight
                else None
            ),
        }
    return output


def branch_summary(
    rows: list[dict], selected: dict[str, dict[str, str | float]], replicates: int, seed: int
) -> dict:
    strict = [row for row in rows if row["strict_same_stride"]]
    point_t = weighted_rate(strict, "any_flip", SCALE_TINY)
    point_s = weighted_rate(strict, "any_flip", SCALE_SMALL)
    point_direction_t = weighted_direction_rate(rows, SCALE_TINY)
    point_direction_s = weighted_direction_rate(rows, SCALE_SMALL)
    contrasts = bootstrap_contrast(strict, selected, replicates, seed, "object")
    directional_contrasts = bootstrap_contrast(rows, selected, replicates, seed + 100, "direction")
    return {
        "strict_object_level": {
            "definition": "valid base margin; every recorded legal replay has an active candidate on the base stride",
            "coverage": coverage(rows, "strict_same_stride"),
            "fragility_8_16": point_t,
            "fragility_16_32": point_s,
            "scale_contrast": point_t - point_s,
            "scale_contrast_bootstrap": percentile(contrasts),
        },
        "same_stride_direction_level": {
            "definition": "directional flip rate conditional on base and shifted active candidates sharing a stride",
            "flip_rate_8_16": point_direction_t,
            "flip_rate_16_32": point_direction_s,
            "scale_contrast": point_direction_t - point_direction_s,
            "scale_contrast_bootstrap": percentile(directional_contrasts),
        },
    }


def paired_summary(
    o2o: list[dict],
    o2m: list[dict],
    selected: dict[str, dict[str, str | float]],
    replicates: int,
    seed: int,
) -> dict:
    o2o_map = {(row["image_id"], row["gt_id"]): row for row in o2o}
    o2m_map = {(row["image_id"], row["gt_id"]): row for row in o2m}
    if set(o2o_map) != set(o2m_map):
        raise RuntimeError("paired O2O/O2M GT keys differ")
    rows = []
    for key in sorted(o2o_map):
        a, m = o2o_map[key], o2m_map[key]
        common_valid = bool(a["base_valid"] and m["base_valid"])
        strict_both = bool(common_valid and a["strict_same_stride"] and m["strict_same_stride"])
        rows.append(
            {
                "image_id": a["image_id"],
                "gt_id": a["gt_id"],
                "stratum": a["stratum"],
                "weight": a["weight"],
                "size_bin": a["size_bin"],
                "base_valid": common_valid,
                "strict_both": strict_both,
                "any_o2o": a["any_flip"],
                "any_o2m": m["any_flip"],
                "any_cross_stride": float(a["any_cross_stride"] or m["any_cross_stride"]),
                "any_shift_missing": float(a["any_shift_missing"] or m["any_shift_missing"]),
            }
        )
    strict = [row for row in rows if row["strict_both"]]

    def estimate(counts: dict[str, int] | None = None) -> tuple[float, float, float]:
        o2o_contrast = weighted_rate(strict, "any_o2o", SCALE_TINY, counts) - weighted_rate(
            strict, "any_o2o", SCALE_SMALL, counts
        )
        o2m_contrast = weighted_rate(strict, "any_o2m", SCALE_TINY, counts) - weighted_rate(
            strict, "any_o2m", SCALE_SMALL, counts
        )
        return o2o_contrast, o2m_contrast, o2o_contrast - o2m_contrast

    point = estimate()
    o2o_values, o2m_values, difference_values = bootstrap_paired_contrasts(
        strict, selected, replicates, seed
    )
    common = [row for row in rows if row["base_valid"]]
    valid_weight = sum(row["weight"] for row in common)
    strict_weight = sum(row["weight"] for row in strict)
    return {
        "definition": "valid margins in both branches; every legal replay remains on the base stride in both branches",
        "common_valid_gt": len(common),
        "strict_paired_gt": len(strict),
        "weighted_retention": strict_weight / valid_weight,
        "o2o_fragility_8_16": weighted_rate(strict, "any_o2o", SCALE_TINY),
        "o2o_fragility_16_32": weighted_rate(strict, "any_o2o", SCALE_SMALL),
        "o2m_fragility_8_16": weighted_rate(strict, "any_o2m", SCALE_TINY),
        "o2m_fragility_16_32": weighted_rate(strict, "any_o2m", SCALE_SMALL),
        "o2o_scale_contrast": point[0],
        "o2m_scale_contrast": point[1],
        "o2o_minus_o2m_specificity": point[2],
        "o2o_scale_contrast_bootstrap": percentile(o2o_values),
        "o2m_scale_contrast_bootstrap": percentile(o2m_values),
        "specificity_bootstrap": percentile(difference_values),
    }


def main() -> None:
    args = parse_args()
    sources = (args.ai_selected, args.ai_o2o, args.ai_o2m, args.vis_selected, args.vis_o2o)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.replicates <= 0:
        raise ValueError("invalid or pre-existing output directory")

    ai_selected = selected_contract(args.ai_selected)
    vis_selected = selected_contract(args.vis_selected)
    ai_o2o = collapse_branch(args.ai_o2o, ai_selected, "o2o")
    ai_o2m = collapse_branch(args.ai_o2m, ai_selected, "o2m")
    vis_o2o = collapse_branch(args.vis_o2o, vis_selected, "o2o")

    result = {
        "status": "complete",
        "protocol": "strict_same_stride_conditional_fragility_v1",
        "replicates": args.replicates,
        "seed": args.seed,
        "sources": {
            "ai_selected": {"file": args.ai_selected.name, "sha256": sha256(args.ai_selected)},
            "ai_o2o": {"file": args.ai_o2o.name, "sha256": sha256(args.ai_o2o)},
            "ai_o2m": {"file": args.ai_o2m.name, "sha256": sha256(args.ai_o2m)},
            "vis_selected": {"file": args.vis_selected.name, "sha256": sha256(args.vis_selected)},
            "vis_o2o": {"file": args.vis_o2o.name, "sha256": sha256(args.vis_o2o)},
        },
        "ai_tod_v2_o2o": branch_summary(ai_o2o, ai_selected, args.replicates, args.seed),
        "visdrone_o2o": branch_summary(vis_o2o, vis_selected, args.replicates, args.seed + 1),
        "ai_tod_v2_o2m": branch_summary(ai_o2m, ai_selected, args.replicates, args.seed + 2),
        "ai_tod_v2_paired_o2o_o2m": paired_summary(
            ai_o2o, ai_o2m, ai_selected, args.replicates, args.seed + 3
        ),
        "interpretation_boundary": [
            "Conditioning on observed same-stride replays changes the estimand and may induce selection.",
            "Shifted-active-missing cases have no shifted stride and are excluded from the strict subset.",
            "The result tests whether cross-stride switching is necessary for the scale gradient; it does not identify a causal mechanism.",
        ],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SAME_STRIDE_CONDITIONAL_ANALYSIS_PASS", args.output_dir)


if __name__ == "__main__":
    main()
