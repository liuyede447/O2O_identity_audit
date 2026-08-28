"""Compare a YOLOv8 detector before and after NMS on frozen stratified images.

The pre-NMS condition is an IoU-threshold-1.0 top-300 proxy after the same
confidence and class filtering. It is not the unbounded raw anchor tensor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "ultralytics_local"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
sys.path.insert(0, str(ROOT / "scripts"))

from assignment_stability_random_stratified import SelectedDataset, sha256, size_bin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--post-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def box_iou_one(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(1)
    area_first = (box[2:] - box[:2]).clamp(min=0).prod()
    area_second = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    return intersection / (area_first + area_second - intersection + 1e-12)


def detection_stats(box: torch.Tensor, class_id: int, detections: torch.Tensor) -> tuple[float, int]:
    if detections.numel() == 0:
        return 0.0, 0
    same_class = detections[detections[:, 5].long().eq(class_id)]
    if same_class.numel() == 0:
        return 0.0, 0
    overlaps = box_iou_one(box, same_class[:, :4])
    return float(overlaps.max().item()), int(overlaps.ge(0.5).sum().item())


def weighted_rate(rows: list[dict], key: str, denominator_key: str | None = None) -> float:
    numerator = sum(row["sampling_weight"] * float(row[key]) for row in rows)
    denominator = sum(row["sampling_weight"] * (float(row[denominator_key]) if denominator_key else 1.0) for row in rows)
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.data, args.selected_images):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.checkpoint) != args.expected_sha256 or args.output_dir.exists():
        raise RuntimeError("checkpoint hash mismatch or pre-existing output")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils import nms
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    wrapper = YOLO(str(args.checkpoint))
    model = wrapper.model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or model.end2end or head.end2end:
        raise RuntimeError("expected standard NMS-based Detect model")
    criterion = v8DetectionLoss(model, tal_topk=10)
    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))

    selected_rows = list(csv.DictReader(args.selected_images.open(newline="", encoding="utf-8")))
    indices = [int(row["dataset_index"]) for row in selected_rows]
    sampling = {row["image_id"]: {"stratum": row["stratum"], "sampling_weight": float(row["sampling_weight"])} for row in selected_rows}
    loader = build_dataloader(SelectedDataset(dataset, indices), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    gt_rows, image_rows = [], []
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            prediction = output[0] if isinstance(output, tuple) else output
            if not isinstance(prediction, torch.Tensor) or prediction.ndim != 3:
                raise RuntimeError("unexpected decoded prediction tensor")
            pre = nms.non_max_suppression(prediction.clone(), args.conf, 1.0, nc=criterion.nc, multi_label=True, agnostic=False, max_det=args.max_det, end2end=False)[0]
            post = nms.non_max_suppression(prediction.clone(), args.conf, args.post_iou, nc=criterion.nc, multi_label=True, agnostic=False, max_det=args.max_det, end2end=False)[0]

            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = criterion.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            image_id = Path(batch["im_file"][0]).stem
            sample = sampling[image_id]
            image_rows.append({
                "image_id": image_id, "stratum": sample["stratum"], "sampling_weight": sample["sampling_weight"],
                "pre_top300_predictions": len(pre), "post_nms_predictions": len(post),
                "suppressed_predictions": len(pre) - len(post),
            })
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                class_id = int(labels[0, gt, 0])
                pre_iou, pre_matches = detection_stats(box, class_id, pre)
                post_iou, post_matches = detection_stats(box, class_id, post)
                gt_rows.append({
                    "image_id": image_id, "gt_id": gt, "class_id": class_id, "size_bin": bucket, "area": area,
                    "stratum": sample["stratum"], "sampling_weight": sample["sampling_weight"],
                    "pre_max_iou": pre_iou, "post_max_iou": post_iou,
                    "pre_matches_iou50": pre_matches, "post_matches_iou50": post_matches,
                    "pre_fn": int(pre_iou < 0.5), "post_fn": int(post_iou < 0.5),
                    "pre_duplicate": int(pre_matches >= 2), "post_duplicate": int(post_matches >= 2),
                    "pre_excess_matches": max(pre_matches - 1, 0), "post_excess_matches": max(post_matches - 1, 0),
                    "nms_induced_fn": int(pre_iou >= 0.5 and post_iou < 0.5),
                })
    if not gt_rows or len(image_rows) != len(selected_rows):
        raise RuntimeError("NMS audit did not cover the selected sample")

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gt_rows[0])); writer.writeheader(); writer.writerows(gt_rows)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0])); writer.writeheader(); writer.writerows(image_rows)
    summary = {
        "status": "complete",
        "protocol": "yolov8_pre_top300_vs_nms_random_stratified_v1",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.expected_sha256,
        "model_contract": {"head": "Detect", "end2end": False, "reg_max": int(head.reg_max), "postprocess": "class-aware NMS"},
        "conditions": {
            "pre": {"description": "score-filtered top-300 with NMS IoU threshold 1.0", "conf": args.conf, "iou": 1.0, "max_det": args.max_det},
            "post": {"description": "formal class-aware NMS", "conf": args.conf, "iou": args.post_iou, "max_det": args.max_det},
        },
        "selected_images": len(image_rows), "focal_gt": len(gt_rows),
        "weighted": {
            "pre_fn_rate": weighted_rate(gt_rows, "pre_fn"), "post_fn_rate": weighted_rate(gt_rows, "post_fn"),
            "pre_recall_iou50": 1.0 - weighted_rate(gt_rows, "pre_fn"), "post_recall_iou50": 1.0 - weighted_rate(gt_rows, "post_fn"),
            "pre_duplicate_gt_rate": weighted_rate(gt_rows, "pre_duplicate"), "post_duplicate_gt_rate": weighted_rate(gt_rows, "post_duplicate"),
            "pre_mean_excess_matches": weighted_rate(gt_rows, "pre_excess_matches"), "post_mean_excess_matches": weighted_rate(gt_rows, "post_excess_matches"),
            "nms_induced_fn_rate": weighted_rate(gt_rows, "nms_induced_fn"),
            "mean_pre_predictions_per_image": weighted_rate(image_rows, "pre_top300_predictions"),
            "mean_post_predictions_per_image": weighted_rate(image_rows, "post_nms_predictions"),
            "mean_suppressed_predictions_per_image": weighted_rate(image_rows, "suppressed_predictions"),
        },
        "estimand_warning": "This comparison evaluates O2M+NMS inference specificity and is not an O2O loss-active identity audit.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("YOLOV8_NMS_EFFECT_AUDIT_PASS", json.dumps(summary))


if __name__ == "__main__":
    main()
