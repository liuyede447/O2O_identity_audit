"""Read-only fixed-K counterfactual replay for frozen native O2M states.

The model is forwarded exactly once per image.  Native O2M and O2O assignment
states are then replayed from the frozen raw tensors while only the focal GT
centre is shifted.  For each native final O2M assigned-positive set, the script
records the complete q ordering and constructs post-hoc Top-K sets of exactly
K members whenever both the base and shifted native sets contain at least K
members.  This is a descriptive counterfactual truncation, not native O2M and
not a training intervention.
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
from run_reviewer_killer_controls import SelectedDataset, image_stratum, size_bin


DIRECTIONS = (("left", -1.0, 0.0), ("right", 1.0, 0.0), ("up", 0.0, -1.0), ("down", 0.0, 1.0))
SIZE_BINS = ("t_8_16", "s_16_32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--kappa", type=float, default=0.0625)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--smoke", action="store_true", help="Process one low-GT selected image and skip bootstrap")
    parser.add_argument("--smoke-image-id")
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


def resolve_device(value: str) -> torch.device:
    if value == "0":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resized_focal_count(label: dict, imgsz: int) -> int:
    height, width = map(float, label["shape"])
    ratio = min(imgsz / max(height, 1.0), imgsz / max(width, 1.0))
    count = 0
    for box in label["bboxes"]:
        area = float(box[2]) * width * ratio * float(box[3]) * height * ratio
        if size_bin(area) in SIZE_BINS:
            count += 1
    return count


def validate_selection(
    path: Path,
    dataset,
    strata: dict[str, list[int]],
    smoke: bool,
    smoke_image_id: str | None,
    imgsz: int,
) -> tuple[list[tuple[int, str]], list[dict]]:
    rows = pd.read_csv(path).to_dict("records")
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    if not rows or required.difference(rows[0]):
        raise ValueError("selected-images manifest is empty or incomplete")
    checked: list[dict] = []
    seen_indices, seen_images = set(), set()
    for row in rows:
        index = int(row["dataset_index"])
        image_id = str(row["image_id"])
        stratum = str(row["stratum"])
        probability = float(row["inclusion_probability"])
        weight = float(row["sampling_weight"])
        if index < 0 or index >= len(dataset) or stratum not in strata or index not in strata[stratum]:
            raise RuntimeError("selection contains invalid dataset index or stratum")
        if Path(dataset.labels[index]["im_file"]).stem != image_id:
            raise RuntimeError("selection image ID does not match dataset index")
        if index in seen_indices or image_id in seen_images:
            raise RuntimeError("selection contains duplicate image")
        if not (0 < probability <= 1) or not math.isclose(weight, 1.0 / probability, rel_tol=1e-6):
            raise RuntimeError("selection has invalid inclusion probability or weight")
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
        if smoke_image_id is not None:
            checked = [row for row in checked if row["image_id"] == smoke_image_id]
            if not checked:
                raise ValueError("--smoke-image-id is absent from selected-images")
        else:
            checked = [min(
                checked,
                key=lambda row: (resized_focal_count(dataset.labels[row["dataset_index"]], imgsz), row["image_id"]),
            )]
    return [(row["dataset_index"], row["stratum"]) for row in checked], checked


def relative_top2_margin(state, gt: int) -> float | None:
    scores = state.align[0, gt]
    ids = torch.where(scores > 0)[0]
    if ids.numel() < 2:
        return None
    values = sorted(
        ((int(idx), float(scores[idx].item())) for idx in ids.tolist()),
        key=lambda item: (-item[1], item[0]),
    )
    return (values[0][1] - values[1][1]) / (values[0][1] + 1e-12)


def q_ranking(state, gt: int, assigned_only: bool) -> list[tuple[int, float]]:
    scores = state.align[0, gt]
    if assigned_only:
        mask = state.fg_mask[0] & state.target_gt_idx[0].eq(gt)
    else:
        mask = scores > 0
    ids = torch.where(mask)[0].tolist()
    return sorted(
        ((int(idx), float(scores[idx].item())) for idx in ids),
        key=lambda item: (-item[1], item[0]),
    )


def serialize_ids(ranking: list[tuple[int, float]]) -> str:
    return ";".join(str(candidate) for candidate, _ in ranking)


def serialize_scores(ranking: list[tuple[int, float]]) -> str:
    return ";".join(format(score, ".17g") for _, score in ranking)


def fixed_k_metrics(base: list[tuple[int, float]], shifted: list[tuple[int, float]], k: int) -> dict[str, object]:
    eligible = len(base) >= k and len(shifted) >= k
    if not eligible:
        return {
            f"k{k}_eligible": 0,
            f"k{k}_top1_base": None,
            f"k{k}_top1_shift": None,
            f"k{k}_top1_flip": None,
            f"k{k}_base_ids": "",
            f"k{k}_shift_ids": "",
            f"k{k}_set_turnover": None,
            f"k{k}_jaccard": None,
        }
    base_ids = [candidate for candidate, _ in base[:k]]
    shifted_ids = [candidate for candidate, _ in shifted[:k]]
    base_set, shifted_set = set(base_ids), set(shifted_ids)
    union = base_set | shifted_set
    return {
        f"k{k}_eligible": 1,
        f"k{k}_top1_base": base_ids[0],
        f"k{k}_top1_shift": shifted_ids[0],
        f"k{k}_top1_flip": int(base_ids[0] != shifted_ids[0]),
        f"k{k}_base_ids": ";".join(map(str, base_ids)),
        f"k{k}_shift_ids": ";".join(map(str, shifted_ids)),
        f"k{k}_set_turnover": int(base_set != shifted_set),
        f"k{k}_jaccard": len(base_set & shifted_set) / len(union),
    }


def build_object_rows(directions: pd.DataFrame, fixed_k: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["image_id", "gt_id", "class_id", "size_bin", "stratum", "sampling_weight", "common_valid_margin"]
    for values, part in directions.groupby(keys, sort=False, dropna=False):
        item = dict(zip(keys, values))
        item["legal_directions"] = len(part)
        for k in fixed_k:
            strict = bool(part[f"k{k}_eligible"].eq(1).all())
            item[f"k{k}_strict_eligible"] = int(strict)
            item[f"k{k}_top1_fragile"] = int(part[f"k{k}_top1_flip"].eq(1).any()) if strict else None
            item[f"k{k}_set_turnover_any"] = int(part[f"k{k}_set_turnover"].eq(1).any()) if strict else None
        rows.append(item)
    return pd.DataFrame(rows)


def weighted_rate(rows: pd.DataFrame, outcome: str) -> float | None:
    values = pd.to_numeric(rows[outcome], errors="coerce").to_numpy(float)
    weights = rows["sampling_weight"].to_numpy(float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep])) if np.any(keep) else None


def outcome_summary(frame: pd.DataFrame, eligible: str, outcome: str) -> dict[str, object]:
    result: dict[str, object] = {}
    rates: dict[str, float | None] = {}
    for size in SIZE_BINS:
        all_rows = frame[frame["size_bin"].eq(size)]
        rows = all_rows[all_rows[eligible].eq(1)]
        all_weight = float(all_rows["sampling_weight"].sum())
        rates[size] = weighted_rate(rows, outcome)
        result[f"{size}_rate"] = rates[size]
        result[f"{size}_rows"] = int(len(rows))
        result[f"{size}_weighted_coverage"] = (
            float(rows["sampling_weight"].sum()) / all_weight if all_weight else None
        )
    tiny, small = rates[SIZE_BINS[0]], rates[SIZE_BINS[1]]
    result["gap_tiny_minus_small"] = None if tiny is None or small is None else tiny - small
    return result


def estimate_all(directions: pd.DataFrame, objects: pd.DataFrame, fixed_k: list[int]) -> dict[str, object]:
    output: dict[str, object] = {}
    for k in fixed_k:
        output[str(k)] = {
            "direction_top1": outcome_summary(directions, f"k{k}_eligible", f"k{k}_top1_flip"),
            "direction_set_turnover": outcome_summary(directions, f"k{k}_eligible", f"k{k}_set_turnover"),
            "object_top1": outcome_summary(objects, f"k{k}_strict_eligible", f"k{k}_top1_fragile"),
            "object_set_turnover": outcome_summary(objects, f"k{k}_strict_eligible", f"k{k}_set_turnover_any"),
        }
    return output


def flatten_rates(estimates: dict[str, object]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for k, levels in estimates.items():
        for level, metrics in levels.items():
            for metric in ("t_8_16_rate", "s_16_32_rate", "gap_tiny_minus_small"):
                value = metrics[metric]
                if value is not None and math.isfinite(float(value)):
                    flattened[f"K{k}.{level}.{metric}"] = float(value)
    return flattened


def sampled_indices(frame: pd.DataFrame, chosen_images: list[str]) -> np.ndarray:
    groups = {image: rows.index.to_numpy() for image, rows in frame.groupby("image_id", sort=False)}
    pieces = [groups[image] for image in chosen_images if image in groups]
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=int)


def bootstrap_summary(
    directions: pd.DataFrame,
    objects: pd.DataFrame,
    selected_images: list[dict],
    fixed_k: list[int],
    reps: int,
    seed: int,
) -> dict[str, object]:
    point = estimate_all(directions, objects, fixed_k)
    if reps == 0:
        return {"point": point, "ci_95": {}, "bootstrap_replicates": 0}
    by_stratum = {
        stratum: rows["image_id"].astype(str).tolist()
        for stratum, rows in pd.DataFrame(selected_images).groupby("stratum", sort=True)
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(reps):
        chosen: list[str] = []
        for images in by_stratum.values():
            chosen.extend(rng.choice(images, size=len(images), replace=True).tolist())
        d_idx = sampled_indices(directions, chosen)
        o_idx = sampled_indices(objects, chosen)
        current = flatten_rates(estimate_all(directions.loc[d_idx], objects.loc[o_idx], fixed_k))
        for key, value in current.items():
            samples.setdefault(key, []).append(value)
    ci = {
        key: {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
        for key, values in samples.items()
    }
    return {
        "point": point,
        "ci_95": ci,
        "bootstrap_replicates": reps,
        "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
    }


def main() -> None:
    args = parse_args()
    fixed_k = sorted(set(int(k) for k in args.fixed_k))
    if not fixed_k or fixed_k[0] < 1 or fixed_k[-1] > 10:
        raise ValueError("--fixed-k must contain unique integers in [1, 10]")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data YAML or selected-images manifest is missing")
    if not (0 < args.kappa < 0.5):
        raise ValueError("--kappa must be in (0, 0.5)")
    if args.bootstrap < 1 and not args.smoke:
        raise ValueError("--bootstrap must be positive outside smoke mode")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = resolve_device(args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("protocol requires the frozen YOLO26 end-to-end direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if o2m.assigner.topk != 10 or o2m.assigner.topk2 != 10:
        raise RuntimeError("unexpected native O2M Top-10 contract")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={
        "task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers,
        "rect": False, "cache": False, "fraction": 1.0,
    })
    dataset = build_yolo_dataset(
        cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32)
    )
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    selected, selected_rows = validate_selection(
        args.selected_images, dataset, strata, args.smoke, args.smoke_image_id, args.imgsz
    )
    loader = build_dataloader(
        SelectedDataset(dataset, [index for index, _ in selected]), batch=1, workers=args.workers,
        shuffle=False, rank=-1, drop_last=False, pin_memory=False,
    )
    stratum_by_image = {row["image_id"]: row["stratum"] for row in selected_rows}
    weight_by_image = {row["image_id"]: row["sampling_weight"] for row in selected_rows}

    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "selected_images.csv", selected_rows)
    manifest = {
        "status": "running",
        "protocol": "o2m_fixed_k_post_native_counterfactual_v1",
        "read_only": True,
        "counterfactual_not_native_o2m": True,
        "script_sha256": sha256(Path(__file__)),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.expected_sha256.lower(),
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "selected_images_source": str(args.selected_images.resolve()),
        "selected_images_source_sha256": sha256(args.selected_images),
        "selected_images_processed": len(selected),
        "smoke": bool(args.smoke),
        "device": str(device),
        "forward_contract": "exactly one model forward per processed image; all GT shifts replay frozen raw tensors",
        "native_o2m_contract": "Top-10 get_pos_mask -> native multi-GT conflict resolution -> final assigned-positive set",
        "fixed_k_contract": (
            "sort each final native assigned-positive set by q descending with candidate-ID tie break, then retain Top-K; "
            "a row is eligible only when base and shifted sets both contain at least K, so both counterfactual states have exactly K members"
        ),
        "analysis_support": "common-valid base O2O/O2M Top-2 margins; K-specific availability additionally required",
        "full_q_ranking_outputs": {
            "base": "all q-positive candidates and final native assigned-positive candidates",
            "shift": "all q-positive candidates and final native assigned-positive candidates for every legal direction",
        },
        "fixed_k": fixed_k,
        "stress": "equivalent-area-side",
        "kappa": args.kappa,
        "bootstrap_replicates": 0 if args.smoke else args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    base_rows: list[dict] = []
    direction_rows: list[dict] = []
    forward_passes = 0
    started = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            forward_passes += 1
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw dual-branch output")
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            image_scale = torch.tensor(
                [image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype
            )
            targets = o2m.preprocess(targets, 1, image_scale)
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
                if bucket not in SIZE_BINS:
                    continue
                base_assigned = q_ranking(base_m, gt, assigned_only=True)
                base_all_q = q_ranking(base_m, gt, assigned_only=False)
                o2m_margin = relative_top2_margin(base_m, gt)
                o2o_margin = relative_top2_margin(base_o, gt)
                common_valid = int(o2m_margin is not None and o2o_margin is not None)
                base_rows.append({
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
                    "o2o_base_margin": o2o_margin,
                    "o2m_base_margin": o2m_margin,
                    "common_valid_margin": common_valid,
                    "all_q_positive_count": len(base_all_q),
                    "all_q_positive_ids": serialize_ids(base_all_q),
                    "all_q_positive_scores": serialize_scores(base_all_q),
                    "native_assigned_count": len(base_assigned),
                    "native_assigned_ids_q_order": serialize_ids(base_assigned),
                    "native_assigned_scores_q_order": serialize_scores(base_assigned),
                })
                delta = args.kappa * math.sqrt(area)
                for direction, unit_dx, unit_dy in DIRECTIONS:
                    shifted_box = legal_shift(
                        box, unit_dx * delta, unit_dy * delta, float(image_w), float(image_h)
                    )
                    if shifted_box is None:
                        continue
                    altered = boxes.clone()
                    altered[0, gt] = shifted_box
                    shifted_m = trace_o2o(o2m, raw["one2many"], labels, altered, mask)
                    shifted_assigned = q_ranking(shifted_m, gt, assigned_only=True)
                    shifted_all_q = q_ranking(shifted_m, gt, assigned_only=False)
                    row = {
                        "image_id": image_id,
                        "gt_id": gt,
                        "class_id": int(labels[0, gt, 0].item()),
                        "size_bin": bucket,
                        "stratum": stratum_by_image[image_id],
                        "sampling_weight": weight_by_image[image_id],
                        "direction": direction,
                        "shift_pixels": delta,
                        "common_valid_margin": common_valid,
                        "base_all_q_positive_count": len(base_all_q),
                        "shift_all_q_positive_count": len(shifted_all_q),
                        "shift_all_q_positive_ids": serialize_ids(shifted_all_q),
                        "shift_all_q_positive_scores": serialize_scores(shifted_all_q),
                        "base_native_assigned_count": len(base_assigned),
                        "shift_native_assigned_count": len(shifted_assigned),
                        "shift_native_assigned_ids_q_order": serialize_ids(shifted_assigned),
                        "shift_native_assigned_scores_q_order": serialize_scores(shifted_assigned),
                    }
                    for k in fixed_k:
                        row.update(fixed_k_metrics(base_assigned, shifted_assigned, k))
                    direction_rows.append(row)

    if forward_passes != len(selected):
        raise RuntimeError("one-forward-per-image contract failed")
    if not base_rows or not direction_rows:
        raise RuntimeError("fixed-K replay produced no focal-GT rows")
    base_frame = pd.DataFrame(base_rows)
    direction_frame = pd.DataFrame(direction_rows)
    analysis_directions = direction_frame[direction_frame["common_valid_margin"].eq(1)].copy()
    object_frame = build_object_rows(analysis_directions, fixed_k)
    reps = 0 if args.smoke else args.bootstrap
    analysis = bootstrap_summary(
        analysis_directions, object_frame, selected_rows, fixed_k, reps, args.bootstrap_seed
    )

    write_rows(args.output_dir / "base_q_rankings.csv", base_rows)
    write_rows(args.output_dir / "per_direction_fixed_k.csv", direction_rows)
    object_frame.to_csv(args.output_dir / "per_gt_fixed_k.csv", index=False)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "estimand": (
            "sampling-weighted fixed-K post-native O2M counterfactual on common-valid base-margin support; "
            "all reported gaps are 8-16 px minus 16-32 px"
        ),
        "counterfactual_not_native_o2m": True,
        "source_gt_rows": len(base_frame),
        "analysis_gt_rows": int(base_frame["common_valid_margin"].eq(1).sum()),
        "source_direction_rows": len(direction_frame),
        "analysis_direction_rows": len(analysis_directions),
        "analysis_object_rows_with_legal_direction": len(object_frame),
        "images": int(base_frame["image_id"].nunique()),
        "forward_passes": forward_passes,
        **analysis,
    }
    (args.output_dir / "fixed_k_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({
        "status": summary["status"],
        "elapsed_sec": time.monotonic() - started,
        "forward_passes": forward_passes,
        "base_q_ranking_rows": len(base_frame),
        "direction_rows": len(direction_frame),
        "outputs": [
            "selected_images.csv", "base_q_rankings.csv", "per_direction_fixed_k.csv",
            "per_gt_fixed_k.csv", "fixed_k_summary.json", "manifest.json",
        ],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
