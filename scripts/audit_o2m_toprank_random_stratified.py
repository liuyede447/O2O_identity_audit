"""Random-stratified O2M top-ranked-candidate replay audit for an NMS-based detector.

This estimand is deliberately not an O2O loss-active identity. It tracks the
highest native O2M alignment candidate for each GT while retaining the full
top-k positive-set assignment for a secondary Jaccard stability measure.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from assignment_stability_random_stratified import SelectedDataset, image_stratum, proportional_allocation, sha256, size_bin
from margin_o2o_replay import PERTURBATIONS, legal_shift, trace_o2o


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
    parser.add_argument("--model-contract", choices=("yolov8_standard", "yolo26_o2m"), default="yolov8_standard")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ranked_candidate_and_margin(state, batch_index: int, gt_index: int) -> tuple[int | None, int | None, float | None]:
    scores = state.align[batch_index, gt_index]
    eligible = torch.where(scores > 0)[0]
    if eligible.numel() == 0:
        return None, None, None
    ordered = eligible[torch.argsort(scores[eligible], descending=True)]
    first = int(ordered[0].item())
    if ordered.numel() < 2:
        return first, None, None
    second = int(ordered[1].item())
    q1, q2 = float(scores[first].item()), float(scores[second].item())
    return first, second, (q1 - q2) / (q1 + 1e-12)


def positive_set(state, batch_index: int, gt_index: int) -> set[int]:
    ids = torch.where(state.fg_mask[batch_index] & state.target_gt_idx[batch_index].eq(gt_index))[0]
    return set(map(int, ids.tolist()))


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return 1.0 if not union else len(first & second) / len(union)


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
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    wrapper = YOLO(str(args.checkpoint))
    model = wrapper.model.to(device).eval()
    head = model.model[-1]
    if args.model_contract == "yolov8_standard":
        if head.__class__.__name__ != "Detect" or model.end2end or head.end2end:
            raise RuntimeError("expected a standard non-end2end Detect head")
    else:
        if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or head.reg_max != 1:
            raise RuntimeError("expected the YOLO26 end-to-end direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    criterion = v8DetectionLoss(model, tal_topk=10)
    if criterion.assigner.topk != 10:
        raise RuntimeError("expected native O2M topk10 assignment")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    counts = {key: len(value) for key, value in strata.items()}
    requested = 3 if args.smoke else args.sample_images
    allocation = proportional_allocation(counts, requested)
    rng = random.Random(args.seed)
    selected = []
    for stratum in sorted(strata):
        selected.extend((index, stratum) for index in rng.sample(strata[stratum], allocation[stratum]))
    selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
    loader = build_dataloader(SelectedDataset(dataset, [index for index, _ in selected]), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    sampling_rows = []
    for index, stratum in selected:
        probability = allocation[stratum] / counts[stratum]
        sampling_rows.append({"dataset_index": index, "image_id": Path(dataset.labels[index]["im_file"]).stem, "stratum": stratum, "inclusion_probability": probability, "sampling_weight": 1.0 / probability})

    args.output_dir.mkdir(parents=True)
    manifest = {
        "status": "running",
        "protocol": "random_image_stratified_o2m_toprank_replay_v1",
        "estimand": "native O2M top-ranked alignment candidate",
        "not_o2o_loss_active_identity": True,
        "selection_before_inference": True,
        "outcome_blind_selection": True,
        "seed": args.seed,
        "requested_images": requested,
        "dataset_total_images": len(dataset),
        "eligible_images": sum(counts.values()),
        "stratum_population_counts": counts,
        "stratum_sample_allocation": allocation,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256,
        "data": str(args.data),
        "imgsz": args.imgsz,
        "model_contract": args.model_contract,
        "head": "Detect",
        "end2end": bool(model.end2end),
        "reg_max": int(head.reg_max),
        "o2m_topk": int(criterion.assigner.topk),
        "inference_postprocess": "NMS" if args.model_contract == "yolov8_standard" else "O2O NMS-free; audited O2M branch is training-only",
    }
    (args.output_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "selected_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sampling_rows[0])); writer.writeheader(); writer.writerows(sampling_rows)

    rows = []
    focal_counts, audited_counts, no_legal_counts = Counter(), Counter(), Counter()
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            if time.monotonic() - started > args.wall_time_sec:
                raise TimeoutError("O2M top-rank audit exceeded wall-time contract")
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            raw = output[1] if isinstance(output, tuple) else output
            if args.model_contract == "yolo26_o2m":
                if not isinstance(raw, dict) or "one2many" not in raw:
                    raise RuntimeError("expected YOLO26 raw dual-branch output")
                raw = raw["one2many"]
            if not isinstance(raw, dict) or not {"boxes", "scores", "feats"}.issubset(raw):
                raise RuntimeError("expected raw standard Detect output")
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = criterion.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base = trace_o2o(criterion, raw, labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                bucket = size_bin(width * height)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                focal_counts[bucket] += 1
                candidate, runner_up, margin = ranked_candidate_and_margin(base, 0, gt)
                base_set = positive_set(base, 0, gt)
                base_stride = None if candidate is None else int(base.stride[candidate].item())
                replay_rows = []
                for perturbation, dx, dy in PERTURBATIONS:
                    shifted = legal_shift(box, dx, dy, float(image_w), float(image_h))
                    if shifted is None:
                        continue
                    altered = boxes.clone(); altered[0, gt] = shifted
                    replay = trace_o2o(criterion, raw, labels, altered, mask)
                    candidate_p, _, _ = ranked_candidate_and_margin(replay, 0, gt)
                    replay_set = positive_set(replay, 0, gt)
                    stride_p = None if candidate_p is None else int(replay.stride[candidate_p].item())
                    replay_rows.append({
                        "image_id": image_id, "gt_id": gt, "class_id": int(labels[0, gt, 0]),
                        "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                        "width": width, "height": height, "area": width * height, "size_bin": bucket,
                        "perturbation": perturbation,
                        "o2m_rank_candidate_0": candidate, "o2m_rank_runner_up_0": runner_up,
                        "o2m_rank_candidate_p": candidate_p, "o2m_rank_flip": int(candidate != candidate_p),
                        "o2m_rank_margin_0": margin,
                        "o2m_positive_count_0": len(base_set),
                        "o2m_positive_count_p": len(replay_set),
                        "o2m_set_jaccard": jaccard(base_set, replay_set),
                        "stride_0": base_stride, "stride_p": stride_p, "stride_flip": int(base_stride != stride_p),
                    })
                if replay_rows:
                    audited_counts[bucket] += 1
                else:
                    no_legal_counts[bucket] += 1
                for row in replay_rows:
                    row["valid_perturbations"] = len(replay_rows)
                    rows.append(row)

    if not rows:
        raise RuntimeError("O2M top-rank audit produced no focal records")
    with (args.output_dir / "per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest.update({
        "status": "complete", "selected_images": len(selected),
        "focal_gt_by_bin": dict(focal_counts), "audited_gt_by_bin": dict(audited_counts),
        "excluded_no_legal_replay_by_bin": dict(no_legal_counts), "records": len(rows),
        "elapsed_sec": time.monotonic() - started,
    })
    (args.output_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("O2M_TOPRANK_RANDOM_STRATIFIED_AUDIT_PASS", json.dumps(manifest))


if __name__ == "__main__":
    main()
