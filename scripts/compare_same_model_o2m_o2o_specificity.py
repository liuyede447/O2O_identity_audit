"""Compare YOLO26 O2M top-rank exchange with O2O loss-active identity changes."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--o2m-audit", type=Path, required=True)
    parser.add_argument("--o2m-models", type=Path, required=True)
    parser.add_argument("--o2m-descriptive", type=Path, required=True)
    parser.add_argument("--o2o-audit", type=Path, required=True)
    parser.add_argument("--o2o-models", type=Path, required=True)
    parser.add_argument("--o2o-descriptive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def candidate(value: str) -> int | None:
    return None if value in ("", "None") else int(value)


def weighted_mean(rows: list[dict], key: str) -> float:
    return sum(row["weight"] * row[key] for row in rows) / sum(row["weight"] for row in rows)


def summarize(rows: list[dict]) -> dict:
    return {
        "gt": len(rows),
        "base_candidate_agreement": weighted_mean(rows, "base_agreement"),
        "any_o2m_rank_flip": weighted_mean(rows, "any_o2m"),
        "any_o2o_identity_flip": weighted_mean(rows, "any_o2o"),
        "direction_only_o2m": weighted_mean(rows, "only_o2m"),
        "direction_only_o2o": weighted_mean(rows, "only_o2o"),
        "direction_both": weighted_mean(rows, "both"),
        "direction_neither": weighted_mean(rows, "neither"),
    }


def main() -> None:
    args = parse_args()
    paths = (args.selected_images, args.o2m_audit, args.o2m_models, args.o2m_descriptive, args.o2o_audit, args.o2o_models, args.o2o_descriptive)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with args.selected_images.open(newline="", encoding="utf-8") as handle:
        weights = {row["image_id"]: float(row["sampling_weight"]) for row in csv.DictReader(handle)}
    with args.o2m_audit.open(newline="", encoding="utf-8") as handle:
        o2m_rows = list(csv.DictReader(handle))
    with args.o2o_audit.open(newline="", encoding="utf-8") as handle:
        o2o_rows = list(csv.DictReader(handle))
    o2m = {(row["image_id"], row["gt_id"], row["perturbation"]): row for row in o2m_rows}
    o2o = {(row["image_id"], row["gt_id"], row["perturbation"]): row for row in o2o_rows}
    if set(o2m) != set(o2o):
        raise RuntimeError(f"directional replay keys differ: O2M={len(o2m)} O2O={len(o2o)}")
    grouped = defaultdict(list)
    for key in sorted(o2m):
        grouped[key[:2]].append((o2m[key], o2o[key]))
    rows = []
    for (image_id, gt_id), pairs in grouped.items():
        first_m, first_o = pairs[0]
        only_m = only_o = both = neither = 0
        for m, o in pairs:
            flip_m, flip_o = int(m["o2m_rank_flip"]), int(o["o2o_flip"])
            only_m += int(flip_m and not flip_o); only_o += int(flip_o and not flip_m)
            both += int(flip_m and flip_o); neither += int(not flip_m and not flip_o)
        n = len(pairs)
        rows.append({
            "image_id": image_id, "gt_id": gt_id, "size_bin": first_m["size_bin"], "weight": weights[image_id],
            "base_agreement": float(candidate(first_m["o2m_rank_candidate_0"]) == candidate(first_o["o2o_candidate_0"])),
            "any_o2m": float(any(int(m["o2m_rank_flip"]) for m, _ in pairs)),
            "any_o2o": float(any(int(o["o2o_flip"]) for _, o in pairs)),
            "only_o2m": only_m / n, "only_o2o": only_o / n, "both": both / n, "neither": neither / n,
        })
    by_size = defaultdict(list)
    for row in rows:
        by_size[row["size_bin"]].append(row)
    o2m_models = json.loads(args.o2m_models.read_text())
    o2o_models = json.loads(args.o2o_models.read_text())
    o2m_desc = {row["size_bin"]: row for row in json.loads(args.o2m_descriptive.read_text())["scale_rows"]}
    o2o_desc = {row["size_bin"]: row for row in json.loads(args.o2o_descriptive.read_text())["scale_rows"]}
    payload = {
        "status": "complete",
        "protocol": "same_checkpoint_same_forward_o2m_o2o_specificity_v1",
        "checkpoint_scope": "YOLO26s Native, identical images/GT/replays/candidate grid; separate learned branch scores and supervision semantics.",
        "estimand_separation": "O2M top-ranked candidate is not O2O loss-active identity; no pooled effect size.",
        "all": summarize(rows),
        "by_size": {key: summarize(by_size[key]) for key in sorted(by_size)},
        "model_summaries": {
            "o2m_rank_fragility": {
                "scale_8_16": o2m_desc["t_8_16"]["weighted_rank_fragility_rate_valid"],
                "scale_16_32": o2m_desc["s_16_32"]["weighted_rank_fragility_rate_valid"],
                "margin_OR": o2m_models["rank_margin_to_flip_any"]["bootstrap"]["margin_OR_median"],
                "margin_CI_95": o2m_models["rank_margin_to_flip_any"]["bootstrap"]["margin_OR_percentile_CI_95"],
            },
            "o2o_identity_fragility": {
                "scale_8_16": o2o_desc["t_8_16"]["weighted_fragility_rate_valid"],
                "scale_16_32": o2o_desc["s_16_32"]["weighted_fragility_rate_valid"],
                "margin_OR": o2o_models["margin_to_flip_any"]["bootstrap"]["margin_OR_median"],
                "margin_CI_95": o2o_models["margin_to_flip_any"]["bootstrap"]["margin_OR_percentile_CI_95"],
            },
        },
        "claim_rule": "Same-model evidence can localize scale-dependent amplification to unique O2O supervision semantics, but cannot prove uniqueness is the causal mechanism.",
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("SAME_MODEL_O2M_O2O_SPECIFICITY_PASS", json.dumps(payload))


if __name__ == "__main__":
    main()
