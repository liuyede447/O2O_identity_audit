"""Three-way parity on frozen real-image states.

Native path: TaskAlignedAssigner.forward.
Production path: margin_o2o_replay.trace_o2o.
Reference path: independent NumPy implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_native_alignment_path_anatomy import first_divergence, pair_changed
from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o
from reference_o2o_assigner_numpy import active_by_gt, assign_reference, first_divergence_reference
from run_reviewer_killer_controls import SelectedDataset, size_bin
from ultralytics.utils.tal import make_anchors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-threshold", type=int, default=170)
    parser.add_argument("--hash-modulus", type=int, default=10000)
    parser.add_argument("--selection-seed", type=int, default=20260831)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=0.0625)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_key(seed: int, image: str, gt: int, direction: str, stress: str, modulus: int, threshold: int) -> bool:
    digest = hashlib.sha256(f"{seed}|{image}|{gt}|{direction}|{stress}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus < threshold


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def reference_inputs(criterion, raw, labels, boxes, mask):
    pred_distri = raw["boxes"].permute(0, 2, 1).contiguous()
    logits = raw["scores"].permute(0, 2, 1).contiguous()
    anchors, stride = make_anchors(raw["feats"], criterion.stride, 0.5)
    decoded = criterion.bbox_decode(anchors, pred_distri)
    return {
        "scores_t": logits.sigmoid(),
        "boxes_t": decoded * stride,
        "anchors_t": anchors * stride,
        "labels_t": labels,
        "gt_boxes_t": boxes,
        "mask_t": mask,
        "scores": logits.sigmoid()[0].detach().cpu().numpy(),
        "boxes": (decoded * stride)[0].detach().cpu().numpy(),
        "anchors": (anchors * stride).detach().cpu().numpy(),
        "labels": labels[0, :, 0].long().detach().cpu().numpy(),
        "gt_boxes": boxes[0].detach().cpu().numpy(),
        "valid": mask[0, :, 0].detach().cpu().numpy().astype(bool),
    }


def native_final(criterion, inputs):
    _, _, _, fg, target = criterion.assigner(
        inputs["scores_t"], inputs["boxes_t"], inputs["anchors_t"],
        inputs["labels_t"], inputs["gt_boxes_t"], inputs["mask_t"],
    )
    return fg[0].detach().cpu().numpy(), target[0].detach().cpu().numpy()


def margin_reference(state, gt: int):
    active = active_by_gt(state)[gt]
    if active is None or state.align[gt, active] <= 1e-12:
        return active, None, None
    alternatives = np.where(state.align[gt] > 0)[0]
    alternatives = alternatives[alternatives != active]
    if len(alternatives) == 0:
        return active, None, None
    alternative = int(alternatives[np.argmax(state.align[gt, alternatives])])
    first, second = state.align[gt, active], state.align[gt, alternative]
    return active, alternative, float((first - second) / (first + 1e-12))


def compare_state(production, reference, native_fg, native_target) -> dict:
    checks = {
        "native_production_fg": np.array_equal(native_fg.astype(bool), production.fg_mask[0].detach().cpu().numpy()),
        "native_production_target": np.array_equal(native_target, production.target_gt_idx[0].detach().cpu().numpy()),
        "eligible": np.array_equal(production.eligible[0].detach().cpu().numpy(), reference.eligible),
        "topk": np.array_equal(production.pre[0].detach().cpu().numpy(), reference.topk),
        "conflict": np.array_equal(production.conflict[0].detach().cpu().numpy(), reference.conflict),
        "post": np.array_equal(production.post[0].detach().cpu().numpy(), reference.post),
        "fg": np.array_equal(production.fg_mask[0].detach().cpu().numpy(), reference.fg_mask),
        "target": np.array_equal(production.target_gt_idx[0].detach().cpu().numpy(), reference.target_gt_idx),
        "q": np.allclose(production.align[0].detach().cpu().numpy(), reference.align, atol=1e-6, rtol=2e-5),
        "ciou": np.allclose(production.overlaps[0].detach().cpu().numpy(), reference.overlaps, atol=1e-6, rtol=2e-5),
    }
    return {key: bool(value) for key, value in checks.items()}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint hash mismatch")
    selected = pd.read_csv(args.selected_images)
    indices = selected["dataset_index"].astype(int).tolist()

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    criterion = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    loader = build_dataloader(SelectedDataset(dataset, indices), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    args.output_dir.mkdir(parents=True)
    mismatch_rows, sampled_rows = [], []
    base_states = direction_states = 0
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor): batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            raw = model(batch["img"])[1]["one2one"]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            h, w = batch["img"].shape[-2:]
            targets = criterion.preprocess(targets, 1, torch.tensor([w, h, w, h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            image_id = Path(batch["im_file"][0]).stem
            production_base = trace_o2o(criterion, raw, labels, boxes, mask)
            inputs = reference_inputs(criterion, raw, labels, boxes, mask)
            reference_base = assign_reference(inputs["scores"], inputs["boxes"], inputs["anchors"], inputs["labels"], inputs["gt_boxes"], inputs["valid"])
            native_fg, native_target = native_final(criterion, inputs)
            base_checks = compare_state(production_base, reference_base, native_fg, native_target)
            base_states += 1
            if not all(base_checks.values()):
                mismatch_rows.append({"image_id": image_id, "gt_id": -1, "direction": "base", "stress": "base", "mismatch": ";".join(key for key, value in base_checks.items() if not value)})

            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                if size_bin(area) not in {"t_8_16", "s_16_32"}: continue
                for stress in ("fixed_1px", "equivalent_side"):
                    delta = 1.0 if stress == "fixed_1px" else args.kappa * math.sqrt(area)
                    for direction, ux, uy in (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)):
                        if not selected_key(args.selection_seed, image_id, gt, direction, stress, args.hash_modulus, args.hash_threshold):
                            continue
                        shifted = legal_shift(box, ux * delta, uy * delta, float(w), float(h))
                        if shifted is None: continue
                        altered = boxes.clone(); altered[0, gt] = shifted
                        production = trace_o2o(criterion, raw, labels, altered, mask)
                        shifted_inputs = reference_inputs(criterion, raw, labels, altered, mask)
                        reference = assign_reference(shifted_inputs["scores"], shifted_inputs["boxes"], shifted_inputs["anchors"], shifted_inputs["labels"], shifted_inputs["gt_boxes"], shifted_inputs["valid"])
                        shifted_native_fg, shifted_native_target = native_final(criterion, shifted_inputs)
                        checks = compare_state(production, reference, shifted_native_fg, shifted_native_target)
                        prod_active, prod_runner, prod_margin = active_candidate_and_margin(production, 0, gt)
                        ref_active, ref_runner, ref_margin = margin_reference(reference, gt)
                        checks["active"] = prod_active == ref_active
                        checks["runner"] = prod_runner == ref_runner
                        checks["margin_domain"] = (prod_margin is None) == (ref_margin is None)
                        checks["margin_value"] = prod_margin is None or bool(np.isclose(prod_margin, ref_margin, atol=1e-6, rtol=2e-6))

                        active0, _, _ = active_candidate_and_margin(production_base, 0, gt)
                        eligibility_changed = pair_changed(production_base.eligible[0, gt], production.eligible[0, gt], active0, prod_active)
                        topk_changed = pair_changed(production_base.pre[0, gt], production.pre[0, gt], active0, prod_active)
                        conflict_changed = pair_changed(production_base.conflict[0, gt], production.conflict[0, gt], active0, prod_active)
                        rank_reversal = False
                        if active0 is not None and prod_active is not None and active0 != prod_active:
                            rank_reversal = bool(production_base.align[0, gt, active0] > production_base.align[0, gt, prod_active] and production.align[0, gt, active0] < production.align[0, gt, prod_active])
                        prod_label = first_divergence(eligibility_changed=eligibility_changed, topk_changed=topk_changed, conflict_changed=conflict_changed, rank_reversal=rank_reversal, active_shift_missing=prod_active is None) if active0 != prod_active else "stable"
                        ref_label = first_divergence_reference(reference_base, reference, gt)
                        checks["first_divergence"] = prod_label == ref_label
                        direction_states += 1
                        sampled_rows.append({
                            "image_id": image_id, "gt_id": gt, "direction": direction, "stress": stress,
                            "production_label": prod_label, "reference_label": ref_label,
                            "production_margin": prod_margin, "reference_margin": ref_margin,
                            "margin_abs_diff": None if prod_margin is None or ref_margin is None else abs(prod_margin - ref_margin),
                            "pass": int(all(checks.values())),
                        })
                        if not all(checks.values()):
                            mismatch_rows.append({"image_id": image_id, "gt_id": gt, "direction": direction, "stress": stress, "mismatch": ";".join(key for key, value in checks.items() if not value)})

    write_rows(args.output_dir / "sampled_states.csv", sampled_rows)
    write_rows(args.output_dir / "mismatches.csv", mismatch_rows)
    payload = {
        "status": "PASS" if not mismatch_rows else "FAIL",
        "protocol": "native_production_numpy_reference_parity_v1",
        "selected_images": len(indices), "base_states": base_states, "sampled_direction_states": direction_states,
        "mismatches": len(mismatch_rows), "hash_sampling": {"seed": args.selection_seed, "threshold": args.hash_threshold, "modulus": args.hash_modulus},
        "tolerances": {"q_atol": 1e-6, "q_rtol": 2e-5, "margin_atol": 1e-6, "margin_rtol": 2e-6},
        "elapsed_sec": time.monotonic() - started,
        "source_sha256": {
            "production": sha256(ROOT / "scripts" / "margin_o2o_replay.py"),
            "reference": sha256(ROOT / "scripts" / "reference_o2o_assigner_numpy.py"),
            "validator": sha256(Path(__file__)),
            "selected_images": sha256(args.selected_images),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
