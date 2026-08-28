"""Random image-stratified active-identity O2O ambiguity audit.

Images are selected before model inference with a fixed seed and proportional
allocation across image strata containing 8--16 px, 16--32 px, or both focal
object scales.  Every 8--32 px GT in a selected image is audited.  Sampling
never uses margin, fragility, detections, or FN outcomes.
"""
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

import torch

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "ultralytics_local"
if LOCAL_ULTRALYTICS.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import (
    PERTURBATIONS,
    active_candidate_and_margin,
    legal_shift,
    pre_alignment_margin,
    trace_o2o,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wall-time-sec", type=int, default=21600)
    parser.add_argument("--assignment-contract", choices=("yolo26", "yolov10"), default="yolo26")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_bin(area: float) -> str:
    side = max(float(area), 0.0) ** 0.5
    return "vt_lt8" if side < 8 else "t_8_16" if side < 16 else "s_16_32" if side < 32 else "m_ge32"


def proportional_allocation(counts: dict[str, int], requested: int) -> dict[str, int]:
    total = sum(counts.values())
    if requested <= 0 or total <= 0:
        raise ValueError("sample size and eligible image count must be positive")
    requested = min(requested, total)
    exact = {key: requested * value / total for key, value in counts.items()}
    allocation = {key: min(value, int(math.floor(exact[key]))) for key, value in counts.items()}
    remaining = requested - sum(allocation.values())
    order = sorted(counts, key=lambda key: (exact[key] - allocation[key], counts[key], key), reverse=True)
    while remaining:
        changed = False
        for key in order:
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise AssertionError("proportional allocation could not satisfy requested sample")
    return allocation


def image_stratum(label: dict, imgsz: int) -> str | None:
    height, width = map(float, label["shape"])
    ratio = min(imgsz / max(height, 1.0), imgsz / max(width, 1.0))
    has_tiny, has_small = False, False
    for box in label["bboxes"]:
        side = max(float(box[2]) * width * ratio * float(box[3]) * height * ratio, 0.0) ** 0.5
        has_tiny |= 8.0 <= side < 16.0
        has_small |= 16.0 <= side < 32.0
    if has_tiny and has_small:
        return "t_and_s"
    if has_tiny:
        return "t_only"
    if has_small:
        return "s_only"
    return None


class SelectedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.collate_fn = dataset.collate_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def active_stride(state, batch_index: int, candidate: int | None) -> int | None:
    return None if candidate is None else int(state.stride[candidate].item())


def o2m_active_count(state, batch_index: int, gt_index: int) -> int:
    return int((state.fg_mask[batch_index] & state.target_gt_idx[batch_index].eq(gt_index)).sum())


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() or sha256(args.checkpoint) != args.expected_sha256:
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or args.output_dir.exists() or args.sample_images <= 0:
        raise ValueError("invalid or pre-existing audit target")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import E2EDetectLoss, v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))

    strata: dict[str, list[int]] = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    counts = {key: len(value) for key, value in strata.items()}
    allocation = proportional_allocation(counts, 3 if args.smoke else args.sample_images)
    rng = random.Random(args.seed)
    selected: list[tuple[int, str]] = []
    for stratum in sorted(strata):
        chosen = rng.sample(strata[stratum], allocation[stratum])
        selected.extend((index, stratum) for index in chosen)
    selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
    indices = [index for index, _ in selected]
    selected_dataset = SelectedDataset(dataset, indices)
    loader = build_dataloader(selected_dataset, batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    manifest_rows = []
    for index, stratum in selected:
        probability = allocation[stratum] / counts[stratum]
        manifest_rows.append({
            "dataset_index": index,
            "image_id": Path(dataset.labels[index]["im_file"]).stem,
            "stratum": stratum,
            "inclusion_probability": probability,
            "sampling_weight": 1.0 / probability,
        })

    args.output_dir.mkdir(parents=True)
    if args.assignment_contract == "yolo26":
        o2m = v8DetectionLoss(model, tal_topk=10)
        o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    else:
        if model.model[-1].__class__.__name__ != "v10Detect":
            raise RuntimeError("yolov10 assignment contract requires a v10Detect head")
        native_criterion = E2EDetectLoss(model)
        o2m, o2o = native_criterion.one2many, native_criterion.one2one
        if o2m.assigner.topk != 10 or o2o.assigner.topk != 1:
            raise RuntimeError("native YOLOv10 topk10/topk1 contract violated")

    manifest = {
        "status": "running",
        "protocol": "random_image_stratified_active_identity_v1",
        "selection_before_inference": True,
        "outcome_blind_selection": True,
        "seed": args.seed,
        "requested_images": 3 if args.smoke else args.sample_images,
        "dataset_total_images": len(dataset),
        "eligible_images": sum(counts.values()),
        "stratum_population_counts": counts,
        "stratum_sample_allocation": allocation,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256,
        "data": str(args.data),
        "imgsz": args.imgsz,
        "assignment_contract": args.assignment_contract,
        "o2m_topk": int(o2m.assigner.topk),
        "o2o_topk": int(o2o.assigner.topk),
        "o2o_topk2": int(o2o.assigner.topk2),
        "reg_max": int(o2o.reg_max),
        "use_dfl": bool(o2o.use_dfl),
    }
    (args.output_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.output_dir / "selected_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)

    rows = []
    focal_gt_counts, audited_gt_counts, no_legal_replay_counts = Counter(), Counter(), Counter()
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            if time.monotonic() - started > args.wall_time_sec:
                raise TimeoutError("random-stratified audit exceeded wall-time contract")
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw end-to-end output")
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_m = trace_o2o(o2m, raw["one2many"], labels, boxes, mask)
            base_o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                focal_gt_counts[bucket] += 1
                candidate, _, _ = active_candidate_and_margin(base_o, 0, gt)
                # A margin is valid only when the native loss has an active identity.
                # Candidate-deficient GTs remain explicit undefined-margin cases.
                margin = pre_alignment_margin(base_o, 0, gt) if candidate is not None else None
                stride = active_stride(base_o, 0, candidate)
                nplus = o2m_active_count(base_m, 0, gt)
                replay_rows = []
                for perturbation, dx, dy in PERTURBATIONS:
                    shifted = legal_shift(box, dx, dy, float(image_w), float(image_h))
                    if shifted is None:
                        continue
                    altered = boxes.clone(); altered[0, gt] = shifted
                    replay = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
                    candidate_p, _, _ = active_candidate_and_margin(replay, 0, gt)
                    stride_p = active_stride(replay, 0, candidate_p)
                    replay_rows.append({
                        "image_id": image_id,
                        "gt_id": gt,
                        "class_id": int(labels[0, gt, 0]),
                        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                        "width": width, "height": height, "area": area, "size_bin": bucket,
                        "perturbation": perturbation,
                        "o2m_post_count_0": nplus,
                        "o2o_candidate_0": candidate,
                        "o2o_candidate_p": candidate_p,
                        "o2o_flip": int(candidate != candidate_p),
                        "stride_0": stride,
                        "stride_p": stride_p,
                        "stride_flip": int(stride != stride_p),
                        "o2o_margin_0": margin,
                    })
                if replay_rows:
                    audited_gt_counts[bucket] += 1
                else:
                    no_legal_replay_counts[bucket] += 1
                for row in replay_rows:
                    row["valid_perturbations"] = len(replay_rows)
                    rows.append(row)

    if not rows:
        raise RuntimeError("random-stratified audit produced no focal records")
    with (args.output_dir / "per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest.update({
        "status": "complete",
        "selected_images": len(selected),
        "focal_gt_by_bin": dict(focal_gt_counts),
        "audited_gt_by_bin": dict(audited_gt_counts),
        "excluded_no_legal_replay_by_bin": dict(no_legal_replay_counts),
        "records": len(rows),
        "elapsed_sec": time.monotonic() - started,
    })
    (args.output_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("RANDOM_STRATIFIED_ACTIVE_IDENTITY_AUDIT_PASS", json.dumps(manifest))


if __name__ == "__main__":
    main()
