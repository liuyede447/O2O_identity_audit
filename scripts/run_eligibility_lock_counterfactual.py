"""Read-only native versus base-eligibility-lock replay on one frozen sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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

from margin_o2o_replay import O2OState, active_candidate_and_margin, legal_shift, trace_o2o
from run_reviewer_killer_controls import (
    SelectedDataset,
    image_stratum,
    o2m_count,
    percentile_ci,
    proportional_allocation,
    ranked_candidate_and_margin,
    size_bin,
    stratified_cluster_samples,
)
from ultralytics.utils.tal import make_anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=0.0625)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def trace_locked(criterion, raw, labels, boxes, mask, eligibility_override) -> O2OState:
    pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = raw["scores"].permute(0, 2, 1).contiguous()
    anchors, stride = make_anchors(raw["feats"], criterion.stride, 0.5)
    decoded = criterion.bbox_decode(anchors, pred_distri)
    assigner = criterion.assigner
    assigner.bs, assigner.n_max_boxes = pred_scores.shape[0], boxes.shape[1]
    eligible = eligibility_override.bool().clone()
    align, overlaps = assigner.get_box_metrics(
        pred_scores.detach().sigmoid(),
        (decoded.detach() * stride).type(boxes.dtype),
        labels,
        boxes,
        eligible * mask,
    )
    topk = assigner.select_topk_candidates(
        align, topk_mask=mask.expand(-1, -1, assigner.topk).bool()
    )
    pre = topk * eligible * mask
    conflict = pre.bool().clone()
    if conflict.sum(-2).max() > 1:
        multi = (conflict.sum(-2).unsqueeze(1) > 1).expand_as(conflict)
        winner = overlaps.argmax(1)
        selected = torch.zeros_like(conflict)
        selected.scatter_(1, winner.unsqueeze(1), 1)
        conflict = torch.where(multi, selected, conflict).bool()
    target, fg, post = assigner.select_highest_overlaps(pre.clone(), overlaps, assigner.n_max_boxes, align)
    return O2OState(eligible, pre.bool(), conflict, post.bool(), align, overlaps, target, fg.bool(), anchors, stride, decoded)


def weighted_rate(frame: pd.DataFrame, column: str, size: str) -> float:
    subset = frame[frame["size_bin"].eq(size)]
    return float(np.average(subset[column].astype(float), weights=subset["sampling_weight"].astype(float)))


def summary(frame: pd.DataFrame) -> dict:
    output = {}
    for stress in ("fixed_1px", "equivalent_side"):
        subset = frame[frame["stress"].eq(stress)]
        for arm in ("native", "eligibility_lock"):
            values = {}
            for branch in ("o2o", "o2m"):
                tiny = weighted_rate(subset, f"{arm}_{branch}_fragile", "t_8_16")
                small = weighted_rate(subset, f"{arm}_{branch}_fragile", "s_16_32")
                values[f"{branch}_8_16"] = tiny
                values[f"{branch}_16_32"] = small
                values[f"{branch}_gap"] = tiny - small
            values["paired_gap"] = values["o2o_gap"] - values["o2m_gap"]
            output[f"{stress}_{arm}"] = values
    output["fixed_o2o_gap_reduction"] = (
        output["fixed_1px_native"]["o2o_gap"] - output["fixed_1px_eligibility_lock"]["o2o_gap"]
    )
    output["fixed_o2o_gap_fraction_removed"] = output["fixed_o2o_gap_reduction"] / output["fixed_1px_native"]["o2o_gap"]
    output["normalized_o2o_gap_reduction"] = (
        output["equivalent_side_native"]["o2o_gap"] - output["equivalent_side_eligibility_lock"]["o2o_gap"]
    )
    return output


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not (0 < args.kappa < 0.5):
        raise ValueError("invalid input contract")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("expected native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    population = {key: len(value) for key, value in strata.items()}
    allocation = proportional_allocation(population, min(args.sample_images, sum(population.values())))
    rng = random.Random(args.selection_seed)
    selected = []
    for stratum in sorted(strata):
        selected.extend((index, stratum) for index in rng.sample(strata[stratum], allocation[stratum]))
    selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
    stratum_by_image, weight_by_image, selected_rows = {}, {}, []
    for index, stratum in selected:
        image_id = Path(dataset.labels[index]["im_file"]).stem
        probability = allocation[stratum] / population[stratum]
        stratum_by_image[image_id] = stratum
        weight_by_image[image_id] = 1.0 / probability
        selected_rows.append({"dataset_index": index, "image_id": image_id, "stratum": stratum, "inclusion_probability": probability, "sampling_weight": 1.0 / probability})
    loader = build_dataloader(SelectedDataset(dataset, [item[0] for item in selected]), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "selected_images.csv", selected_rows)
    manifest = {
        "status": "running", "protocol": "native_vs_base_eligibility_lock_v1", "read_only": True,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.expected_sha256,
        "selection_seed": args.selection_seed, "selected_images": len(selected),
        "bootstrap_replicates": args.bootstrap, "kappa": args.kappa,
        "definition": "shifted eligibility mask is replaced by the corresponding base-state branch mask before q, Top-k, conflict, and Top-1",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    per_direction, per_gt = [], []
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            base_m = trace_o2o(o2m, raw["one2many"], labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                active0, _, _ = active_candidate_and_margin(base_o, 0, gt)
                rank0, _, _ = ranked_candidate_and_margin(base_m, gt)
                object_rows = []
                for stress in ("fixed_1px", "equivalent_side"):
                    delta = 1.0 if stress == "fixed_1px" else args.kappa * math.sqrt(area)
                    directions = []
                    for direction, ux, uy in (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)):
                        shifted = legal_shift(box, ux * delta, uy * delta, float(image_w), float(image_h))
                        if shifted is None:
                            continue
                        altered = boxes.clone(); altered[0, gt] = shifted
                        native_o = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
                        native_m = trace_o2o(o2m, raw["one2many"], labels, altered, mask)
                        locked_o = trace_locked(o2o, raw["one2one"], labels, altered, mask, base_o.eligible)
                        locked_m = trace_locked(o2m, raw["one2many"], labels, altered, mask, base_m.eligible)
                        native_active, _, _ = active_candidate_and_margin(native_o, 0, gt)
                        locked_active, _, _ = active_candidate_and_margin(locked_o, 0, gt)
                        native_rank, _, _ = ranked_candidate_and_margin(native_m, gt)
                        locked_rank, _, _ = ranked_candidate_and_margin(locked_m, gt)
                        row = {
                            "image_id": image_id, "gt_id": gt, "direction": direction, "stress": stress,
                            "size_bin": bucket, "stratum": stratum_by_image[image_id], "sampling_weight": weight_by_image[image_id],
                            "native_o2o_flip": int(active0 != native_active), "eligibility_lock_o2o_flip": int(active0 != locked_active),
                            "native_o2m_flip": int(rank0 != native_rank), "eligibility_lock_o2m_flip": int(rank0 != locked_rank),
                            "eligibility_mask_equal_o2o": int(torch.equal(base_o.eligible, locked_o.eligible)),
                            "eligibility_mask_equal_o2m": int(torch.equal(base_m.eligible, locked_m.eligible)),
                        }
                        directions.append(row); per_direction.append(row)
                    if directions:
                        aggregate = {"image_id": image_id, "gt_id": gt, "stress": stress, "size_bin": bucket, "stratum": stratum_by_image[image_id], "sampling_weight": weight_by_image[image_id]}
                        for arm in ("native", "eligibility_lock"):
                            for branch in ("o2o", "o2m"):
                                aggregate[f"{arm}_{branch}_fragile"] = int(any(item[f"{arm}_{branch}_flip"] for item in directions))
                        object_rows.append(aggregate); per_gt.append(aggregate)

    write_rows(args.output_dir / "per_direction.csv", per_direction)
    write_rows(args.output_dir / "per_gt.csv", per_gt)
    frame = pd.DataFrame(per_gt)
    point = summary(frame)
    bootstrap_values = {key: [] for key in point if isinstance(point[key], (int, float))}
    nested_keys = [key for key, value in point.items() if isinstance(value, dict)]
    nested_boot = {key: {metric: [] for metric in point[key]} for key in nested_keys}
    # Resample unique images while keeping both stress contracts paired.
    unique = frame.drop_duplicates(["image_id", "stratum"])[["image_id", "stratum"]]
    groups = {key: values["image_id"].tolist() for key, values in unique.groupby("stratum")}
    row_groups = {key: values.index.to_numpy() for key, values in frame.groupby("image_id", sort=False)}
    boot_rng = np.random.default_rng(args.selection_seed + 901)
    for _ in range(args.bootstrap):
        indices = []
        for images in groups.values():
            for image in boot_rng.choice(images, size=len(images), replace=True):
                indices.extend(row_groups[image])
        value = summary(frame.loc[np.asarray(indices)])
        for key in bootstrap_values:
            bootstrap_values[key].append(value[key])
        for key in nested_keys:
            for metric in point[key]:
                nested_boot[key][metric].append(value[key][metric])
    payload = {
        "status": "complete", "protocol": manifest["protocol"], "point": point,
        "ci_95": {
            **{key: percentile_ci(values) for key, values in bootstrap_values.items()},
            **{key: {metric: percentile_ci(values) for metric, values in metrics.items()} for key, metrics in nested_boot.items()},
        },
        "bootstrap_replicates": args.bootstrap, "bootstrap_unit": "image cluster within outcome-blind stratum",
        "eligibility_lock_assertions": {
            "o2o_all_equal": bool(pd.DataFrame(per_direction)["eligibility_mask_equal_o2o"].eq(1).all()),
            "o2m_all_equal": bool(pd.DataFrame(per_direction)["eligibility_mask_equal_o2m"].eq(1).all()),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest.update({"status": "complete", "elapsed_sec": time.monotonic() - started, "per_gt_rows": len(per_gt), "per_direction_rows": len(per_direction), "outputs": ["selected_images.csv", "per_direction.csv", "per_gt.csv", "summary.json"]})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
