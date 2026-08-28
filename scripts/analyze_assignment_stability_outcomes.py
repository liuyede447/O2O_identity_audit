"""CPU-only outcome analysis for the frozen P3 assignment-stability audit."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "runs" / "audit" / "aitod_p3baseline_assignment_stability_v1"
PREDICTIONS = ROOT / "outputs" / "rechecks" / "yolo26s_aitodv2_p3baseline_native_b4_e100_s0" / "best" / "predictions.json"
DATA_YAML = ROOT / "configs" / "aitod_v2.yaml"
OUTPUT_DIR = AUDIT_DIR / "outcome_analysis_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Validate input mapping without writing analysis outputs.")
    return parser.parse_args()


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    inter_w, inter_h = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stability_group(flip_rate: float) -> str:
    if flip_rate == 0:
        return "stable"
    if flip_rate <= 0.5:
        return "mild"
    return "high"


def narrow_size_bin(area: float) -> str:
    side = math.sqrt(max(area, 0.0))
    if side < 4:
        return "lt4"
    if side < 6:
        return "4_6"
    if side < 8:
        return "6_8"
    return "ge8"


def main() -> None:
    args = parse_args()
    required = [AUDIT_DIR / "per_gt.csv", AUDIT_DIR / "summary.json", PREDICTIONS, DATA_YAML]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + ", ".join(missing))
    if OUTPUT_DIR.exists() and not args.smoke:
        raise FileExistsError(f"refusing to overwrite outcome analysis: {OUTPUT_DIR}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("stability_outcomes")
    from eval_size_ap import build_coco_gt, try_map_predictions

    with (AUDIT_DIR / "per_gt.csv").open(encoding="utf-8", newline="") as handle:
        perturb_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in perturb_rows:
        grouped[(row["image_id"], int(row["gt_id"]))].append(row)

    coco_gt, stem_to_id, name_to_id, image_count = build_coco_gt(DATA_YAML)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco_gt["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    with PREDICTIONS.open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    mapped_predictions = try_map_predictions(predictions, stem_to_id, name_to_id, image_count, len(coco_gt["categories"]))
    prediction_by_image_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for prediction in mapped_predictions:
        prediction_by_image_class[(int(prediction["image_id"]), int(prediction["category_id"]))].append(prediction)

    outcomes: list[dict] = []
    for (image_stem, gt_index), rows in grouped.items():
        image_id = stem_to_id.get(image_stem)
        if image_id is None or gt_index >= len(annotations_by_image[image_id]):
            continue
        gt = annotations_by_image[image_id][gt_index]
        candidates = prediction_by_image_class[(image_id, int(gt["category_id"]))]
        best = max((iou_xywh(gt["bbox"], pred["bbox"]) for pred in candidates), default=0.0)
        best_score = max((float(pred["score"]) for pred in candidates), default=0.0)
        flip_rate = mean([float(row["o2o_flip"]) for row in rows]) or 0.0
        stride_rate = mean([float(row["stride_flip"]) for row in rows]) or 0.0
        jaccard = mean([float(row["o2m_jaccard"]) for row in rows]) or 0.0
        rank_corr = mean([float(row["rank_corr"]) for row in rows if row["rank_corr"]])
        original_set = set(json.loads(rows[0]["original_o2m_candidate_ids"]))
        retention, symmetric_difference = [], []
        for row in rows:
            changed = set(json.loads(row["perturbed_o2m_candidate_ids"]))
            retention.append(len(original_set & changed) / len(original_set) if original_set else 1.0)
            symmetric_difference.append(len(original_set ^ changed))
        area = float(rows[0]["area"])
        outcomes.append({
            "image_id": image_stem,
            "gt_id": gt_index,
            "class_id": int(gt["category_id"]),
            "area": area,
            "size_bin": rows[0]["size_bin"],
            "narrow_size_bin": narrow_size_bin(area),
            "o2o_flip_rate": flip_rate,
            "stability_group": stability_group(flip_rate),
            "stride_flip_rate": stride_rate,
            "stride_competition": "cross_stride" if stride_rate > 0 else ("same_stride" if flip_rate > 0 else "stable"),
            "o2m_jaccard": jaccard,
            "o2m_intersection_retention": mean(retention),
            "o2m_symmetric_difference": mean(symmetric_difference),
            "o2m_positive_count": len(original_set),
            "o2o_rank_corr": rank_corr,
            "max_iou": best,
            "best_prediction_score": best_score,
            "fn_at_iou50": int(best < 0.5),
            "success_iou75": int(best >= 0.75),
            "valid_perturbations": int(rows[0]["valid_perturbations"]),
        })

    if args.smoke:
        logger.info("SMOKE_PASS joined_gt=%d", len(outcomes))
        return
    OUTPUT_DIR.mkdir(parents=True)
    logging.getLogger().addHandler(logging.FileHandler(OUTPUT_DIR / "analysis.log", encoding="utf-8"))
    columns = list(outcomes[0]) if outcomes else []
    write_csv(OUTPUT_DIR / "per_gt_outcomes.csv", outcomes, columns)

    def summarize(rows: list[dict], keys: list[str]) -> list[dict]:
        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[tuple(row[key] for key in keys)].append(row)
        result = []
        for key, subset in sorted(buckets.items()):
            result.append({
                **dict(zip(keys, key)), "n_gt": len(subset),
                "fn_rate": mean([row["fn_at_iou50"] for row in subset]),
                "mean_max_iou": mean([row["max_iou"] for row in subset]),
                "iou75_rate": mean([row["success_iou75"] for row in subset]),
                "mean_o2o_flip_rate": mean([row["o2o_flip_rate"] for row in subset]),
                "mean_o2m_jaccard": mean([row["o2m_jaccard"] for row in subset]),
                "mean_positive_count": mean([row["o2m_positive_count"] for row in subset]),
            })
        return result

    by_stability = summarize(outcomes, ["size_bin", "stability_group"])
    size_matched = summarize([row for row in outcomes if row["narrow_size_bin"] != "ge8"], ["narrow_size_bin", "stability_group"])
    stride_competition = summarize(outcomes, ["size_bin", "stride_competition"])
    write_csv(OUTPUT_DIR / "by_stability.csv", by_stability, list(by_stability[0]) if by_stability else ["size_bin", "stability_group"])
    write_csv(OUTPUT_DIR / "size_matched_control.csv", size_matched, list(size_matched[0]) if size_matched else ["narrow_size_bin", "stability_group"])
    write_csv(OUTPUT_DIR / "stride_competition.csv", stride_competition, list(stride_competition[0]) if stride_competition else ["size_bin", "stride_competition"])
    summary = {
        "status": "complete", "read_only": True, "audit_source": str(AUDIT_DIR), "prediction_source": str(PREDICTIONS),
        "joined_gt": len(outcomes), "outputs": ["per_gt_outcomes.csv", "by_stability.csv", "size_matched_control.csv", "stride_competition.csv", "analysis.log"],
        "note": "Top1-Top2 native assignment margin is not recoverable from audit v1 and requires a separate audit v2.",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("OUTCOME_ANALYSIS_COMPLETE joined_gt=%d", len(outcomes))


if __name__ == "__main__":
    main()
