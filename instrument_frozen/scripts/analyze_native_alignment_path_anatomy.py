"""Read-only native alignment and O2O assignment-path anatomy.

The same frozen forward output is replayed after moving only one focal GT.
Classification scores are therefore candidate-specific static offsets.  This
script localises the earliest native stage at which base and shifted states
diverge and computes the exact same-pair q-gap decomposition only when the
base and shifted active candidates remain comparable through conflict
resolution.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o
from run_reviewer_killer_controls import SelectedDataset, image_stratum, proportional_allocation, size_bin
from validate_frozen_extraction_contract import validate_extraction_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=300)
    parser.add_argument("--selected-images", type=Path, help="Frozen outcome-blind selection manifest; overrides random sampling")
    parser.add_argument("--selection-seed", type=int, default=20260823)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stress", choices=("fixed_1px", "equivalent_side"), default="fixed_1px")
    parser.add_argument("--kappa", type=float, default=0.0625)
    parser.add_argument("--sealed-extraction", action="store_true", help="Write raw lockbox rows and hashes only; do not compute or print pathway summaries")
    parser.add_argument("--selected-manifest", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--instrument-manifest", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_changed(mask0: torch.Tensor, mask1: torch.Tensor, a: int | None, b: int | None) -> bool:
    candidates = {candidate for candidate in (a, b) if candidate is not None}
    return any(bool(mask0[candidate]) != bool(mask1[candidate]) for candidate in candidates)


def first_divergence(
    *,
    eligibility_changed: bool,
    topk_changed: bool,
    conflict_changed: bool,
    rank_reversal: bool,
    active_shift_missing: bool,
) -> str:
    """Mutually exclusive, prespecified first-divergence taxonomy."""
    if eligibility_changed:
        return "eligibility_boundary"
    if topk_changed:
        return "topk_membership_transition"
    if conflict_changed:
        return "conflict_reassignment"
    if rank_reversal:
        return "within_set_geometry_rank_reversal"
    if active_shift_missing:
        return "active_disappearance"
    return "compound_or_other"


def weighted_summary(rows: list[dict]) -> dict:
    result = {}
    for bucket in ("all", "t_8_16", "s_16_32"):
        subset = rows if bucket == "all" else [row for row in rows if row["size_bin"] == bucket]
        total_weight = sum(float(row["sampling_weight"]) for row in subset)
        flips = [row for row in subset if int(row["o2o_flip"]) == 1]
        flip_weight = sum(float(row["sampling_weight"]) for row in flips)
        pathways = Counter()
        clipping_weight = 0.0
        for row in flips:
            weight = float(row["sampling_weight"])
            pathways[row["first_divergence"]] += weight
            clipping_weight += weight * int(row["geometry_clipping_transition"])
        result[bucket] = {
            "direction_rows": len(subset),
            "weighted_flip_rate": flip_weight / total_weight if total_weight else None,
            "flip_rows": len(flips),
            "pathway_weighted_fraction_among_flips": {
                key: value / flip_weight for key, value in sorted(pathways.items())
            } if flip_weight else {},
            "geometry_clipping_weighted_fraction_among_flips": clipping_weight / flip_weight if flip_weight else None,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not (0 < args.kappa < 0.5):
        raise ValueError("invalid input contract")
    frozen_contract = None
    if args.sealed_extraction:
        required = {
            "--selected-images": args.selected_images,
            "--selected-manifest": args.selected_manifest,
            "--preregistration": args.preregistration,
            "--instrument-manifest": args.instrument_manifest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"sealed extraction requires: {', '.join(missing)}")
        role = "fixed_anatomy" if args.stress == "fixed_1px" else "normalized_anatomy"
        frozen_contract = validate_extraction_contract(
            role=role,
            current_script=Path(__file__).resolve(),
            checkpoint=args.checkpoint,
            expected_checkpoint_sha256=args.expected_sha256,
            data=args.data,
            selected_images=args.selected_images,
            selected_manifest=args.selected_manifest,
            preregistration=args.preregistration,
            instrument_manifest=args.instrument_manifest,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            detector_contract="yolo26",
            stress_mode=args.stress,
            normalization="equivalent_side" if args.stress == "equivalent_side" else None,
            kappa=args.kappa if args.stress == "equivalent_side" else None,
        )

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    if args.sealed_extraction and (device.type != "cuda" or device.index not in (None, 0)):
        raise RuntimeError("sealed anatomy runtime requires CUDA device 0")
    if args.sealed_extraction and any(
        parameter.dtype != torch.float32 or parameter.device.type != "cuda"
        for parameter in model.parameters()
    ):
        raise RuntimeError("sealed anatomy model parameters must be CUDA float32")
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("this anatomy protocol is frozen for the native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if float(o2o.assigner.alpha) != 0.5 or float(o2o.assigner.beta) != 6.0:
        raise RuntimeError("unexpected native alignment exponents")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    population = {key: len(value) for key, value in strata.items()}
    if args.selected_images:
        import pandas as pd
        selected_rows = pd.read_csv(args.selected_images).to_dict("records")
        required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
        if not selected_rows or required.difference(selected_rows[0]):
            raise ValueError("selected-images manifest is empty or incomplete")
        selected = [(int(row["dataset_index"]), str(row["stratum"])) for row in selected_rows]
        for row, (index, stratum) in zip(selected_rows, selected):
            observed = Path(dataset.labels[index]["im_file"]).stem
            if observed != str(row["image_id"]) or index not in strata[stratum]:
                raise RuntimeError("selected-images manifest does not match the frozen dataset index")
        selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
        selected_rows.sort(key=lambda row: str(row["image_id"]))
        requested = len(selected)
        allocation = Counter(stratum for _, stratum in selected)
        stratum_by_image = {str(row["image_id"]): str(row["stratum"]) for row in selected_rows}
        weight_by_image = {str(row["image_id"]): float(row["sampling_weight"]) for row in selected_rows}
    else:
        requested = 3 if args.smoke else args.sample_images
        allocation = proportional_allocation(population, requested)
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
        "status": "running",
        "protocol": "native_alignment_and_first_divergence_anatomy_v1",
        "read_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256,
        "data": str(args.data),
        "detector_contract": "YOLO26 native Top-7 -> conflict resolution -> Top-1",
        "alignment": "q = p^0.5 * max(CIoU,0)^6",
        "stress": args.stress,
        "kappa": args.kappa if args.stress == "equivalent_side" else None,
        "selection_seed": args.selection_seed,
        "selected_images_source": str(args.selected_images) if args.selected_images else None,
        "selected_images": requested,
        "stratum_population_counts": population,
        "stratum_sample_allocation": allocation,
    }
    if frozen_contract is not None:
        manifest["frozen_contract"] = frozen_contract
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    rows, q_rows = [], []
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
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            class_scores = raw["one2one"]["scores"].permute(0, 2, 1).contiguous().sigmoid()
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                active0, runner0, margin0 = active_candidate_and_margin(base, 0, gt)
                delta = 1.0 if args.stress == "fixed_1px" else args.kappa * math.sqrt(area)
                for direction, unit_dx, unit_dy in (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)):
                    shifted_box = legal_shift(box, unit_dx * delta, unit_dy * delta, float(image_w), float(image_h))
                    if shifted_box is None:
                        continue
                    altered = boxes.clone()
                    altered[0, gt] = shifted_box
                    shifted = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
                    active1, runner1, margin1 = active_candidate_and_margin(shifted, 0, gt)
                    flip = int(active0 != active1)
                    eligibility_changed = pair_changed(base.eligible[0, gt], shifted.eligible[0, gt], active0, active1)
                    topk_changed = pair_changed(base.pre[0, gt], shifted.pre[0, gt], active0, active1)
                    conflict_changed = pair_changed(base.conflict[0, gt], shifted.conflict[0, gt], active0, active1)
                    rank_reversal = False
                    clipping = False
                    q_record = None
                    if active0 is not None and active1 is not None and active0 != active1:
                        a, b = active0, active1
                        qa0, qb0 = float(base.align[0, gt, a]), float(base.align[0, gt, b])
                        qa1, qb1 = float(shifted.align[0, gt, a]), float(shifted.align[0, gt, b])
                        rank_reversal = qa0 > qb0 and qa1 < qb1
                        ca0, cb0 = float(base.overlaps[0, gt, a]), float(base.overlaps[0, gt, b])
                        ca1, cb1 = float(shifted.overlaps[0, gt, a]), float(shifted.overlaps[0, gt, b])
                        clipping = min(ca0, cb0, ca1, cb1) <= 0
                        class_id = int(labels[0, gt, 0].item())
                        pa = float(class_scores[0, a, class_id])
                        pb = float(class_scores[0, b, class_id])
                        comparable = not eligibility_changed and not topk_changed and not conflict_changed and rank_reversal
                        q_record = {
                            "image_id": image_id, "gt_id": gt, "direction": direction, "size_bin": bucket,
                            "sampling_weight": weight_by_image[image_id], "candidate_a": a, "candidate_b": b,
                            "p_a": pa, "p_b": pb, "static_class_offset": 0.5 * math.log(pa / pb),
                            "c_a_base": ca0, "c_b_base": cb0, "c_a_shift": ca1, "c_b_shift": cb1,
                            "q_a_base": qa0, "q_b_base": qb0, "q_a_shift": qa1, "q_b_shift": qb1,
                            "comparable_simple_reversal": int(comparable), "geometry_clipping_transition": int(clipping),
                        }
                        if comparable and not clipping:
                            geometry_base = 6.0 * math.log(ca0 / cb0)
                            geometry_shift = 6.0 * math.log(ca1 / cb1)
                            q_record.update({
                                "geometry_gap_base": geometry_base,
                                "geometry_gap_shift": geometry_shift,
                                "delta_geometry_gap": geometry_shift - geometry_base,
                                "pairwise_log_q_gap_base": math.log(qa0 / qb0),
                                "pairwise_log_q_gap_shift": math.log(qa1 / qb1),
                                "delta_pairwise_log_q_gap": math.log(qa1 / qb1) - math.log(qa0 / qb0),
                                "exact_residual_base": math.log(qa0 / qb0) - (0.5 * math.log(pa / pb) + geometry_base),
                                "exact_residual_shift": math.log(qa1 / qb1) - (0.5 * math.log(pa / pb) + geometry_shift),
                                "delta_classification_term": 0.0,
                            })
                        else:
                            q_record.update({key: None for key in ("geometry_gap_base", "geometry_gap_shift", "delta_geometry_gap", "pairwise_log_q_gap_base", "pairwise_log_q_gap_shift", "delta_pairwise_log_q_gap", "exact_residual_base", "exact_residual_shift", "delta_classification_term")})
                        q_rows.append(q_record)
                    pathway = first_divergence(
                        eligibility_changed=eligibility_changed,
                        topk_changed=topk_changed,
                        conflict_changed=conflict_changed,
                        rank_reversal=rank_reversal,
                        active_shift_missing=active1 is None,
                    ) if flip else "stable"
                    flags = sum((eligibility_changed, topk_changed, conflict_changed, rank_reversal, active1 is None))
                    rows.append({
                        "image_id": image_id, "gt_id": gt, "direction": direction, "size_bin": bucket,
                        "width": width, "height": height, "area": area, "stratum": stratum_by_image[image_id],
                        "sampling_weight": weight_by_image[image_id], "stress": args.stress, "shift_pixels": delta,
                        "active_base": active0, "active_shift": active1, "runner_base": runner0, "runner_shift": runner1,
                        "margin_base": margin0, "margin_shift": margin1, "o2o_flip": flip,
                        "eligibility_pair_changed": int(eligibility_changed), "topk_pair_changed": int(topk_changed),
                        "conflict_pair_changed": int(conflict_changed), "pair_rank_reversal": int(rank_reversal),
                        "active_shift_missing": int(active1 is None), "geometry_clipping_transition": int(clipping),
                        "compound_flag": int(flags > 1), "first_divergence": pathway,
                    })
            if args.smoke and len(rows) > 40:
                break

    write_rows(args.output_dir / "per_direction_anatomy.csv", rows)
    write_rows(args.output_dir / "pairwise_q_gap.csv", q_rows)
    if args.sealed_extraction:
        raw_outputs = ["selected_images.csv", "per_direction_anatomy.csv", "pairwise_q_gap.csv"]
        manifest.update({
            "status": "sealed_extraction_smoke_complete" if args.smoke else "sealed_extraction_complete",
            "sealed_outcomes_not_summarized": True,
            "outcome_access_permitted": False,
            "elapsed_sec": time.monotonic() - started,
            "direction_rows": len(rows),
            "q_pair_rows": len(q_rows),
            "outputs": raw_outputs,
            "artifact_sha256": {name: sha256(args.output_dir / name) for name in raw_outputs},
        })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": manifest["status"],
            "selected_images": manifest["selected_images"],
            "direction_rows": manifest["direction_rows"],
            "q_pair_rows": manifest["q_pair_rows"],
            "artifact_sha256": manifest["artifact_sha256"],
        }, indent=2))
        return
    finite = [row for row in q_rows if row.get("exact_residual_base") is not None]
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "taxonomy": "prespecified mutually exclusive first-divergence order: eligibility -> Top-k -> conflict -> within-set q-rank reversal -> active disappearance -> compound/other",
        "classification_change_under_replay": 0.0,
        "weighted": weighted_summary(rows),
        "simple_pairwise_reversals": len(finite),
        "max_abs_exact_residual": max((max(abs(row["exact_residual_base"]), abs(row["exact_residual_shift"])) for row in finite), default=None),
        "direction_rows": len(rows),
        "q_pair_rows": len(q_rows),
    }
    (args.output_dir / "anatomy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({"status": summary["status"], "elapsed_sec": time.monotonic() - started, "direction_rows": len(rows), "q_pair_rows": len(q_rows), "outputs": ["selected_images.csv", "per_direction_anatomy.csv", "pairwise_q_gap.csv", "anatomy_summary.json"]})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
