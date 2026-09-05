"""Audit O2M rank--set decoupling under read-only focal-GT replay.

For every legal focal-GT shift, this script compares the native O2M assigned
positive sets and three explicitly separated rank identities: the legacy
global q>0 Top-1, the pre-conflict native Top-k Top-1, and the q Top-1 inside
the final assigned-positive set.  Set turnover is defined only by exact set
inequality; Jaccard and base retention remain continuous measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import legal_shift, trace_o2o
from run_reviewer_killer_controls import SelectedDataset, image_stratum, size_bin, stratified_cluster_samples


DIRECTIONS = (("left", -1.0, 0.0), ("right", 1.0, 0.0), ("up", 0.0, -1.0), ("down", 0.0, 1.0))
SIZE_BINS = ("t_8_16", "s_16_32")
SUMMARY_METRICS = (
    "jaccard",
    "base_retention",
    "base_set_size",
    "shift_set_size",
    "intersection_size",
    "union_size",
    "exact_set_equal",
    "set_turnover",
    "legacy_q_positive_top1_flip",
    "pre_conflict_topk_top1_flip",
    "assigned_set_q_top1_flip",
    "rank_only_exchange",
    "set_turnover_with_assigned_top1_flip",
    "set_turnover_without_assigned_top1_flip",
    "assigned_top1_defined_both",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stress", choices=("fixed_1px", "equivalent_side"), default="fixed_1px")
    parser.add_argument("--kappa", type=float, default=0.0625)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_set(state, gt: int) -> set[int]:
    mask = state.fg_mask[0] & state.target_gt_idx[0].eq(gt)
    return set(torch.where(mask)[0].tolist())


def q_top1(state, gt: int, candidates: set[int] | torch.Tensor) -> int | None:
    if isinstance(candidates, set):
        if not candidates:
            return None
        indices = torch.tensor(sorted(candidates), device=state.align.device, dtype=torch.long)
    else:
        indices = torch.where(candidates)[0]
        if indices.numel() == 0:
            return None
    scores = state.align[0, gt, indices]
    return int(indices[torch.argmax(scores)].item())


def legacy_q_positive_top1(state, gt: int) -> int | None:
    return q_top1(state, gt, state.align[0, gt].gt(0))


def set_metrics(base_set: set[int], shift_set: set[int]) -> dict:
    intersection = base_set & shift_set
    union = base_set | shift_set
    exact_equal = base_set == shift_set
    if not base_set and not shift_set:
        structural_state = "empty_empty"
    elif not base_set:
        structural_state = "empty_to_nonempty"
    elif not shift_set:
        structural_state = "nonempty_to_empty"
    else:
        structural_state = "nonempty_nonempty"
    return {
        "base_set_size": len(base_set),
        "shift_set_size": len(shift_set),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "set_structural_state": structural_state,
        "jaccard": len(intersection) / len(union) if union else None,
        "base_retention": len(intersection) / len(base_set) if base_set else None,
        "exact_set_equal": int(exact_equal),
        "set_turnover": int(not exact_equal),
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    weights = frame["sampling_weight"].to_numpy(float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return None
    return float(np.average(values[keep], weights=weights[keep]))


def estimate(frame: pd.DataFrame) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for bucket in ("all", *SIZE_BINS):
        part = frame if bucket == "all" else frame[frame["size_bin"].eq(bucket)]
        for metric in SUMMARY_METRICS:
            output[f"{bucket}.{metric}"] = weighted_mean(part, metric)
        flips = part[part["assigned_set_q_top1_flip"].eq(1)]
        output[f"{bucket}.rank_only_fraction_among_assigned_top1_flips"] = weighted_mean(flips, "rank_only_exchange")
    for metric in (*SUMMARY_METRICS, "rank_only_fraction_among_assigned_top1_flips"):
        tiny = output[f"t_8_16.{metric}"]
        small = output[f"s_16_32.{metric}"]
        output[f"t_8_16_minus_s_16_32.{metric}"] = None if tiny is None or small is None else tiny - small
    return output


def bootstrap_summary(frame: pd.DataFrame, reps: int, seed: int) -> dict:
    point = estimate(frame)
    if reps == 0:
        return {"point": point, "ci_95": {}, "bootstrap_replicates": 0}
    values = {key: [] for key in point}
    for indices in stratified_cluster_samples(frame, reps, seed):
        current = estimate(frame.iloc[indices])
        for key, value in current.items():
            if value is not None and math.isfinite(value):
                values[key].append(value)
    ci = {
        key: [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
        for key, sample in values.items()
        if sample
    }
    return {
        "point": point,
        "ci_95": ci,
        "bootstrap_replicates": reps,
        "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
    }


def validate_selection_manifest(path: Path, dataset, strata: dict[str, list[int]], smoke: bool) -> tuple[list[tuple[int, str]], list[dict]]:
    rows = pd.read_csv(path).to_dict("records")
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    if not rows or required.difference(rows[0]):
        raise ValueError("selected-images manifest is empty or incomplete")
    seen_indices, seen_images = set(), set()
    checked = []
    for row in rows:
        index = int(row["dataset_index"])
        image_id = str(row["image_id"])
        stratum = str(row["stratum"])
        probability = float(row["inclusion_probability"])
        weight = float(row["sampling_weight"])
        if index < 0 or index >= len(dataset) or stratum not in strata or index not in strata[stratum]:
            raise RuntimeError("selected-images manifest has an invalid dataset index or stratum")
        observed = Path(dataset.labels[index]["im_file"]).stem
        if observed != image_id:
            raise RuntimeError("selected-images manifest does not match the frozen dataset index")
        if index in seen_indices or image_id in seen_images:
            raise RuntimeError("selected-images manifest contains duplicate images")
        if not (0 < probability <= 1) or not math.isfinite(weight) or weight <= 0 or not math.isclose(weight, 1.0 / probability, rel_tol=1e-6):
            raise RuntimeError("selected-images manifest has an invalid inclusion probability or sampling weight")
        seen_indices.add(index)
        seen_images.add(image_id)
        checked.append({
            "dataset_index": index,
            "image_id": image_id,
            "stratum": stratum,
            "inclusion_probability": probability,
            "sampling_weight": weight,
        })
    checked.sort(key=lambda row: row["image_id"])
    if smoke:
        checked = checked[: min(3, len(checked))]
    return [(row["dataset_index"], row["stratum"]) for row in checked], checked


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data YAML or selected-images manifest is missing")
    if args.stress == "equivalent_side" and not (0 < args.kappa < 0.5):
        raise ValueError("kappa must be in (0, 0.5)")
    if args.bootstrap < 1 and not args.smoke:
        raise ValueError("bootstrap must be positive outside smoke mode")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("this protocol is frozen for the native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    if o2m.assigner.topk != 10 or o2m.assigner.topk2 != 10:
        raise RuntimeError("unexpected native O2M Top-10 contract")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    selected, selected_rows = validate_selection_manifest(args.selected_images, dataset, strata, args.smoke)
    loader = build_dataloader(SelectedDataset(dataset, [index for index, _ in selected]), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    stratum_by_image = {row["image_id"]: row["stratum"] for row in selected_rows}
    weight_by_image = {row["image_id"]: row["sampling_weight"] for row in selected_rows}

    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "selected_images.csv", selected_rows)
    manifest = {
        "status": "running",
        "protocol": "o2m_rank_set_decoupling_v1",
        "read_only": True,
        "script_sha256": sha256(Path(__file__)),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.expected_sha256.lower(),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "selected_images_source": str(args.selected_images.resolve()),
        "selected_images_source_sha256": sha256(args.selected_images),
        "selected_images_processed": len(selected),
        "smoke_manifest_prefix_only": bool(args.smoke),
        "stratum_population_counts": {key: len(value) for key, value in strata.items()},
        "stratum_sample_allocation": dict(Counter(stratum for _, stratum in selected)),
        "detector_contract": "YOLO26 native O2M Top-10 -> multi-GT conflict resolution -> Top-10",
        "rank_contracts": {
            "legacy_q_positive_top1": "q Top-1 among all candidates with q > 0",
            "pre_conflict_topk_top1": "q Top-1 inside the native pre-conflict Top-10 mask",
            "assigned_set_q_top1": "q Top-1 inside P_g = fg_mask & (target_gt_idx == focal_gt)",
        },
        "set_contract": "exact membership; no Jaccard or retention threshold",
        "empty_set_contract": {
            "empty_empty": "separate structural state; Jaccard and retention are undefined",
            "empty_to_nonempty": "separate structural state; Jaccard is zero and base retention is undefined",
            "nonempty_to_empty": "separate structural state; Jaccard and base retention are zero",
            "nonempty_nonempty": "Jaccard and base retention are calculated normally"
        },
        "stress": args.stress,
        "kappa": args.kappa if args.stress == "equivalent_side" else None,
        "bootstrap_replicates": 0 if args.smoke else args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw dual-branch output")
            raw = output[1]["one2many"]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_state = trace_o2o(o2m, raw, labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in SIZE_BINS:
                    continue
                delta = 1.0 if args.stress == "fixed_1px" else args.kappa * math.sqrt(area)
                base_set = candidate_set(base_state, gt)
                legacy0 = legacy_q_positive_top1(base_state, gt)
                pre_topk0 = q_top1(base_state, gt, base_state.pre[0, gt])
                assigned0 = q_top1(base_state, gt, base_set)
                for direction, unit_dx, unit_dy in DIRECTIONS:
                    shifted_box = legal_shift(box, unit_dx * delta, unit_dy * delta, float(image_w), float(image_h))
                    if shifted_box is None:
                        continue
                    altered = boxes.clone()
                    altered[0, gt] = shifted_box
                    shifted_state = trace_o2o(o2m, raw, labels, altered, mask)
                    shifted_set = candidate_set(shifted_state, gt)
                    legacy1 = legacy_q_positive_top1(shifted_state, gt)
                    pre_topk1 = q_top1(shifted_state, gt, shifted_state.pre[0, gt])
                    assigned1 = q_top1(shifted_state, gt, shifted_set)
                    metrics = set_metrics(base_set, shifted_set)
                    assigned_flip = int(assigned0 != assigned1)
                    exact_equal = bool(metrics["exact_set_equal"])
                    rank_only = int(exact_equal and assigned_flip)
                    turnover_with_flip = int(not exact_equal and assigned_flip)
                    turnover_without_flip = int(not exact_equal and not assigned_flip)
                    rows.append({
                        "image_id": image_id,
                        "gt_id": gt,
                        "class_id": int(labels[0, gt, 0].item()),
                        "size_bin": bucket,
                        "width": width,
                        "height": height,
                        "area": area,
                        "equivalent_side": math.sqrt(area),
                        "stratum": stratum_by_image[image_id],
                        "sampling_weight": weight_by_image[image_id],
                        "direction": direction,
                        "shift_pixels": delta,
                        "legacy_q_positive_top1_base": legacy0,
                        "legacy_q_positive_top1_shift": legacy1,
                        "legacy_q_positive_top1_flip": int(legacy0 != legacy1),
                        "pre_conflict_topk_top1_base": pre_topk0,
                        "pre_conflict_topk_top1_shift": pre_topk1,
                        "pre_conflict_topk_top1_flip": int(pre_topk0 != pre_topk1),
                        "assigned_set_q_top1_base": assigned0,
                        "assigned_set_q_top1_shift": assigned1,
                        "assigned_set_q_top1_flip": assigned_flip,
                        "assigned_top1_defined_both": int(assigned0 is not None and assigned1 is not None),
                        "base_positive_ids": ";".join(map(str, sorted(base_set))),
                        "shift_positive_ids": ";".join(map(str, sorted(shifted_set))),
                        **metrics,
                        "rank_only_exchange": rank_only,
                        "set_turnover_with_assigned_top1_flip": turnover_with_flip,
                        "set_turnover_without_assigned_top1_flip": turnover_without_flip,
                        "exact_classification": (
                            "rank_only_exchange" if rank_only else
                            "set_turnover_with_rank_exchange" if turnover_with_flip else
                            "set_turnover_without_rank_exchange" if turnover_without_flip else
                            "stable_exact_set_and_rank"
                        ),
                    })

    if not rows:
        raise RuntimeError("audit produced no legal focal-GT directional rows")
    write_rows(args.output_dir / "per_direction_rank_set.csv", rows)
    frame = pd.DataFrame(rows)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "estimand": "IPW directional O2M rank and exact assigned-positive-set stability for 8-16 px and 16-32 px focal GTs",
        "direction_rows": len(frame),
        "images": int(frame["image_id"].nunique()),
        "focal_gt": int(frame[["image_id", "gt_id"]].drop_duplicates().shape[0]),
        "direction_rows_by_size_bin": {str(key): int(value) for key, value in frame["size_bin"].value_counts().sort_index().items()},
        "classification_counts_unweighted": {str(key): int(value) for key, value in frame["exact_classification"].value_counts().sort_index().items()},
        "set_structural_state_counts_unweighted": {str(key): int(value) for key, value in frame["set_structural_state"].value_counts().sort_index().items()},
        **bootstrap_summary(frame, 0 if args.smoke else args.bootstrap, args.bootstrap_seed),
    }
    (args.output_dir / "rank_set_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({
        "status": summary["status"],
        "elapsed_sec": time.monotonic() - started,
        "direction_rows": len(frame),
        "focal_gt": summary["focal_gt"],
        "outputs": ["selected_images.csv", "per_direction_rank_set.csv", "rank_set_summary.json", "manifest.json"],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
