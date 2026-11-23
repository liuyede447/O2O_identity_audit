"""Run normalized replay and margin incremental-value controls.

This is a read-only sensitivity analysis. It never updates model parameters or
training targets. The normalized replay shifts each focal GT centre by a fixed
fraction of its equivalent side length, while keeping image pixels, features,
candidate grids, other GTs, and model parameters fixed.
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
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o
from validate_frozen_extraction_contract import validate_extraction_contract


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outcomes", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sample-images", type=int, default=300)
    p.add_argument("--selected-images", type=Path, help="Frozen outcome-blind selection manifest; overrides random sampling")
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--kappa", type=float, default=0.0625)
    p.add_argument("--normalization", choices=("equivalent_side", "axis"), default="equivalent_side")
    p.add_argument("--stress-mode", choices=("normalized", "fixed_1px"), default="normalized")
    p.add_argument("--detector-contract", choices=("yolo26", "yolov10"), default="yolo26")
    p.add_argument("--checkpoint-role", default="native sensitivity checkpoint")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--normalized-only", action="store_true")
    p.add_argument("--sealed-extraction", action="store_true", help="Write raw lockbox rows and hashes only; do not compute or print outcome summaries")
    p.add_argument("--selected-manifest", type=Path)
    p.add_argument("--preregistration", type=Path)
    p.add_argument("--instrument-manifest", type=Path)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def size_bin(area: float) -> str:
    side = math.sqrt(max(float(area), 0.0))
    return "vt_lt8" if side < 8 else "t_8_16" if side < 16 else "s_16_32" if side < 32 else "m_ge32"


def image_stratum(label: dict, imgsz: int) -> str | None:
    height, width = map(float, label["shape"])
    ratio = min(imgsz / max(height, 1.0), imgsz / max(width, 1.0))
    has_tiny = has_small = False
    for box in label["bboxes"]:
        side = math.sqrt(max(float(box[2]) * width * ratio * float(box[3]) * height * ratio, 0.0))
        has_tiny |= 8.0 <= side < 16.0
        has_small |= 16.0 <= side < 32.0
    if has_tiny and has_small:
        return "t_and_s"
    if has_tiny:
        return "t_only"
    if has_small:
        return "s_only"
    return None


def proportional_allocation(counts: dict[str, int], requested: int) -> dict[str, int]:
    total = sum(counts.values())
    requested = min(requested, total)
    exact = {key: requested * value / total for key, value in counts.items()}
    out = {key: min(value, int(math.floor(exact[key]))) for key, value in counts.items()}
    remaining = requested - sum(out.values())
    order = sorted(counts, key=lambda key: (exact[key] - out[key], counts[key], key), reverse=True)
    while remaining:
        for key in order:
            if out[key] < counts[key]:
                out[key] += 1
                remaining -= 1
                if remaining == 0:
                    break
    return out


class SelectedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.collate_fn = dataset.collate_fn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def ranked_candidate_and_margin(state, gt: int) -> tuple[int | None, int | None, float | None]:
    scores = state.align[0, gt]
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


def o2m_count(state, gt: int) -> int:
    return int((state.fg_mask[0] & state.target_gt_idx[0].eq(gt)).sum().item())


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_rate(df: pd.DataFrame, column: str, bucket: str) -> float:
    part = df[df["size_bin"].eq(bucket)]
    return weighted_mean(part[column].to_numpy(float), part["sampling_weight"].to_numpy(float))


def normalized_contrasts(df: pd.DataFrame) -> dict:
    o2o_t = weighted_rate(df, "o2o_fragile", "t_8_16")
    o2o_s = weighted_rate(df, "o2o_fragile", "s_16_32")
    o2m_t = weighted_rate(df, "o2m_fragile", "t_8_16")
    o2m_s = weighted_rate(df, "o2m_fragile", "s_16_32")
    o2o_gap, o2m_gap = o2o_t - o2o_s, o2m_t - o2m_s
    return {
        "o2o_8_16": o2o_t,
        "o2o_16_32": o2o_s,
        "o2o_scale_contrast": o2o_gap,
        "o2m_8_16": o2m_t,
        "o2m_16_32": o2m_s,
        "o2m_scale_contrast": o2m_gap,
        "paired_difference_in_scale_gradients": o2o_gap - o2m_gap,
    }


def stratified_cluster_samples(df: pd.DataFrame, reps: int, seed: int):
    image_strata = df[["image_id", "stratum"]].drop_duplicates()
    by_stratum = {key: values["image_id"].tolist() for key, values in image_strata.groupby("stratum")}
    row_groups = {key: values.index.to_numpy() for key, values in df.groupby("image_id", sort=False)}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        indices = []
        for images in by_stratum.values():
            chosen = rng.choice(images, size=len(images), replace=True)
            indices.extend(row_groups[image] for image in chosen)
        yield np.concatenate(indices)


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def weighted_standardize(train: np.ndarray, test: np.ndarray, weights: np.ndarray):
    mean = np.average(train, axis=0, weights=weights)
    var = np.average((train - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(var, 1e-12))
    return (train - mean) / scale, (test - mean) / scale


def metric_set(y: np.ndarray, prob: np.ndarray, weight: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, prob, sample_weight=weight)),
        "pr_auc": float(average_precision_score(y, prob, sample_weight=weight)),
        "log_loss": float(log_loss(y, prob, sample_weight=weight, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob, sample_weight=weight)),
    }


def fit_oof_models(df: pd.DataFrame, seed: int):
    model_features = {
        "M0_area_coverage": ["log_area", "log1p_o2m_count"],
        "M1_plus_log_q_active": ["log_area", "log1p_o2m_count", "log_q_active"],
        "M2_plus_active_ciou": ["log_area", "log1p_o2m_count", "log_q_active", "active_ciou"],
        "M3_plus_margin": ["log_area", "log1p_o2m_count", "log_q_active", "active_ciou", "margin"],
    }
    y = df["fn_at_iou50"].to_numpy(int)
    groups = df["image_id"].astype(str).to_numpy()
    weights = df["sampling_weight"].to_numpy(float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = {name: np.zeros(len(df), dtype=float) for name in model_features}
    fold_id = np.full(len(df), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups)):
        fold_id[test_idx] = fold
        for name, features in model_features.items():
            train = df.iloc[train_idx][features].to_numpy(float)
            test = df.iloc[test_idx][features].to_numpy(float)
            x_train, x_test = weighted_standardize(train, test, weights[train_idx])
            model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            model.fit(x_train, y[train_idx], sample_weight=weights[train_idx])
            predictions[name][test_idx] = model.predict_proba(x_test)[:, 1]
    if np.any(fold_id < 0):
        raise RuntimeError("grouped OOF split did not cover every row")
    metrics = {name: metric_set(y, prob, weights) for name, prob in predictions.items()}
    return model_features, predictions, metrics, fold_id


def likelihood_ratio(df: pd.DataFrame, features_small: list[str], features_large: list[str]):
    y = df["fn_at_iou50"].to_numpy(int)
    w = df["sampling_weight"].to_numpy(float)
    w = w / np.mean(w)

    def fit(features):
        x = df[features].to_numpy(float)
        x, _ = weighted_standardize(x, x, w)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
        model.fit(x, y, sample_weight=w)
        prob = np.clip(model.predict_proba(x)[:, 1], 1e-12, 1 - 1e-12)
        ll = float(np.sum(w * (y * np.log(prob) + (1 - y) * np.log(1 - prob))))
        return ll

    ll_small, ll_large = fit(features_small), fit(features_large)
    statistic = 2 * (ll_large - ll_small)
    return {"weighted_lr_statistic": statistic, "df": len(features_large) - len(features_small), "p_value": float(chi2.sf(statistic, len(features_large) - len(features_small)))}


def write_rows(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or digest(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or (args.stress_mode == "normalized" and not (0 < args.kappa < 0.5)):
        raise ValueError("invalid data or kappa")
    if not args.normalized_only and (args.outcomes is None or not args.outcomes.is_file()):
        raise ValueError("incremental analysis requires a frozen outcomes CSV")
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
        role = "fixed_branch" if args.stress_mode == "fixed_1px" else "normalized_branch"
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
            detector_contract=args.detector_contract,
            stress_mode=args.stress_mode,
            normalization=args.normalization,
            kappa=args.kappa if args.stress_mode == "normalized" else None,
        )

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    if args.sealed_extraction and (device.type != "cuda" or device.index not in (None, 0)):
        raise RuntimeError("sealed audit runtime requires CUDA device 0")
    if args.sealed_extraction and any(
        parameter.dtype != torch.float32 or parameter.device.type != "cuda"
        for parameter in model.parameters()
    ):
        raise RuntimeError("sealed audit model parameters must be CUDA float32")
    head = model.model[-1]
    if args.detector_contract == "yolo26":
        if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
            raise RuntimeError("expected YOLO26 end-to-end direct-LTRB Detect contract")
        o2o_topk = 7
    else:
        if head.__class__.__name__ != "v10Detect" or not model.end2end or not head.end2end or int(head.reg_max) <= 1:
            raise RuntimeError("expected native YOLOv10 end-to-end DFL v10Detect contract")
        o2o_topk = 1
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=o2o_topk, tal_topk2=1)

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)
    counts = {key: len(value) for key, value in strata.items()}
    if args.selected_images:
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
        strata_by_image = {str(row["image_id"]): str(row["stratum"]) for row in selected_rows}
        weights_by_image = {str(row["image_id"]): float(row["sampling_weight"]) for row in selected_rows}
    else:
        requested = 3 if args.smoke else args.sample_images
        allocation = proportional_allocation(counts, requested)
        rng = random.Random(args.seed)
        selected = []
        for stratum in sorted(strata):
            selected.extend((index, stratum) for index in rng.sample(strata[stratum], allocation[stratum]))
        selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
        selected_rows = []
        strata_by_image = {}
        weights_by_image = {}
        for index, stratum in selected:
            image_id = Path(dataset.labels[index]["im_file"]).stem
            probability = allocation[stratum] / counts[stratum]
            selected_rows.append({"dataset_index": index, "image_id": image_id, "stratum": stratum, "inclusion_probability": probability, "sampling_weight": 1.0 / probability})
            strata_by_image[image_id] = stratum
            weights_by_image[image_id] = 1.0 / probability
    loader = build_dataloader(SelectedDataset(dataset, [x[0] for x in selected]), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    args.output_dir.mkdir(parents=True)
    manifest = {
        "status": "running",
        "protocol": "normalized_o2m_o2o_replay_and_incremental_value_v1",
        "read_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256,
        "checkpoint_role": args.checkpoint_role,
        "detector_contract": args.detector_contract,
        "native_o2m_topk": 10,
        "native_o2o_topk": o2o_topk,
        "data": str(args.data),
        "selection_seed": args.seed,
        "selected_images_source": str(args.selected_images) if args.selected_images else None,
        "requested_images": requested,
        "stratum_population_counts": counts,
        "stratum_sample_allocation": allocation,
        "stress_mode": args.stress_mode,
        "normalized_shift": (
            "fixed one-input-pixel centre shift"
            if args.stress_mode == "fixed_1px"
            else "delta_x = delta_y = kappa * sqrt(width * height)"
            if args.normalization == "equivalent_side"
            else "delta_x = kappa * width; delta_y = kappa * height"
        ),
        "normalization": args.normalization,
        "kappa": args.kappa,
        "bootstrap_replicates": 0 if args.smoke else args.bootstrap,
    }
    if frozen_contract is not None:
        manifest["frozen_contract"] = frozen_contract
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_rows(args.output_dir / "selected_images.csv", selected_rows)

    per_direction, per_gt = [], []
    focal_counts, audited_counts = Counter(), Counter()
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
            base_m = trace_o2o(o2m, raw["one2many"], labels, boxes, mask)
            base_o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            pred_scores_o = raw["one2one"]["scores"].permute(0, 2, 1).contiguous().sigmoid()
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt]
                width, height = float(box[2] - box[0]), float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket not in {"t_8_16", "s_16_32"}:
                    continue
                focal_counts[bucket] += 1
                active, runner, margin = active_candidate_and_margin(base_o, 0, gt)
                m_rank, m_runner, m_margin = ranked_candidate_and_margin(base_m, gt)
                equivalent_delta = 1.0 if args.stress_mode == "fixed_1px" else args.kappa * math.sqrt(area)
                shift_x = 1.0 if args.stress_mode == "fixed_1px" else equivalent_delta if args.normalization == "equivalent_side" else args.kappa * width
                shift_y = 1.0 if args.stress_mode == "fixed_1px" else equivalent_delta if args.normalization == "equivalent_side" else args.kappa * height
                class_id = int(labels[0, gt, 0].item())
                base = {
                    "image_id": image_id,
                    "gt_id": gt,
                    "class_id": class_id,
                    "width": width,
                    "height": height,
                    "area": area,
                    "equivalent_side": math.sqrt(area),
                    "size_bin": bucket,
                    "stratum": strata_by_image[image_id],
                    "sampling_weight": weights_by_image[image_id],
                    "normalized_kappa": args.kappa if args.stress_mode == "normalized" else None,
                    "shift_pixels": equivalent_delta,
                    "shift_pixels_x": shift_x,
                    "shift_pixels_y": shift_y,
                    "o2o_active": active,
                    "o2o_runner": runner,
                    "o2o_margin": margin,
                    "o2m_rank": m_rank,
                    "o2m_runner": m_runner,
                    "o2m_margin": m_margin,
                    "o2m_positive_count": o2m_count(base_m, gt),
                }
                if active is not None:
                    base.update({
                        "q_active": float(base_o.align[0, gt, active].item()),
                        "active_ciou": float(base_o.overlaps[0, gt, active].item()),
                        "active_class_score": float(pred_scores_o[0, active, class_id].item()),
                        "active_stride": int(base_o.stride[active].item()),
                    })
                else:
                    base.update({"q_active": None, "active_ciou": None, "active_class_score": None, "active_stride": None})
                rows = []
                for direction, unit_dx, unit_dy in (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)):
                    shifted = legal_shift(box, unit_dx * shift_x, unit_dy * shift_y, float(image_w), float(image_h))
                    if shifted is None:
                        continue
                    altered = boxes.clone(); altered[0, gt] = shifted
                    replay_o = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
                    replay_m = trace_o2o(o2m, raw["one2many"], labels, altered, mask)
                    active_p, _, _ = active_candidate_and_margin(replay_o, 0, gt)
                    rank_p, _, _ = ranked_candidate_and_margin(replay_m, gt)
                    row = dict(base)
                    row.update({
                        "direction": direction,
                        "o2o_active_shift": active_p,
                        "o2o_flip": int(active != active_p),
                        "o2m_rank_shift": rank_p,
                        "o2m_flip": int(m_rank != rank_p),
                    })
                    rows.append(row)
                    per_direction.append(row)
                if rows:
                    audited_counts[bucket] += 1
                    aggregate = dict(base)
                    aggregate.update({
                        "legal_directions": len(rows),
                        "o2o_fragile": int(any(row["o2o_flip"] for row in rows)),
                        "o2m_fragile": int(any(row["o2m_flip"] for row in rows)),
                        "o2o_flip_rate": sum(row["o2o_flip"] for row in rows) / len(rows),
                        "o2m_flip_rate": sum(row["o2m_flip"] for row in rows) / len(rows),
                        "common_valid_margin": int(margin is not None and m_margin is not None),
                    })
                    per_gt.append(aggregate)
            if args.smoke and len(per_gt) > 10:
                break

    if not per_gt:
        raise RuntimeError("audit produced no per-GT rows")
    write_rows(args.output_dir / "per_direction.csv", per_direction)
    write_rows(args.output_dir / "per_gt.csv", per_gt)
    manifest.update({
        "status": "smoke_complete" if args.smoke else "audit_complete",
        "selected_images": len(selected),
        "focal_gt_by_bin": dict(focal_counts),
        "audited_gt_by_bin": dict(audited_counts),
        "per_gt_rows": len(per_gt),
        "per_direction_rows": len(per_direction),
        "audit_elapsed_sec": time.monotonic() - started,
    })

    if args.sealed_extraction:
        raw_outputs = ["selected_images.csv", "per_direction.csv", "per_gt.csv"]
        manifest.update({
            "status": "sealed_extraction_smoke_complete" if args.smoke else "sealed_extraction_complete",
            "sealed_outcomes_not_summarized": True,
            "outcome_access_permitted": False,
            "outputs": raw_outputs,
            "artifact_sha256": {name: digest(args.output_dir / name) for name in raw_outputs},
        })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": manifest["status"],
            "selected_images": manifest["selected_images"],
            "per_gt_rows": manifest["per_gt_rows"],
            "per_direction_rows": manifest["per_direction_rows"],
            "artifact_sha256": manifest["artifact_sha256"],
        }, indent=2))
        return

    if args.smoke:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    gt_df = pd.DataFrame(per_gt)
    common = gt_df[gt_df["common_valid_margin"].eq(1)].copy().reset_index(drop=True)
    point = normalized_contrasts(common)
    boot_values = {key: [] for key in point}
    for indices in stratified_cluster_samples(common, args.bootstrap, args.seed + 11):
        stats = normalized_contrasts(common.iloc[indices])
        for key, value in stats.items():
            boot_values[key].append(value)
    normalized_summary = {
        "estimand": "common-valid O2M/O2O margins; population-weighted 8-16 minus 16-32 rates",
        "n_gt": len(common),
        "n_images": int(common["image_id"].nunique()),
        "point": point,
        "ci_95": {key: percentile_ci(values) for key, values in boot_values.items()},
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
    }
    (args.output_dir / "normalized_perturbation_summary.json").write_text(json.dumps(normalized_summary, indent=2) + "\n", encoding="utf-8")

    if args.normalized_only:
        manifest.update({
            "status": "complete",
            "common_valid_normalized_gt": len(common),
            "total_elapsed_sec": time.monotonic() - started,
            "outputs": ["per_direction.csv", "per_gt.csv", "normalized_perturbation_summary.json"],
        })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": manifest, "normalized": normalized_summary}, indent=2))
        return

    outcomes = pd.read_csv(args.outcomes)
    joined = gt_df.merge(outcomes[["image_id", "gt_id", "fn_at_iou50"]], on=["image_id", "gt_id"], how="inner", validate="one_to_one")
    incremental = joined[
        joined["o2o_margin"].notna()
        & joined["q_active"].notna()
        & joined["active_ciou"].notna()
        & joined["fn_at_iou50"].notna()
    ].copy().reset_index(drop=True)
    incremental["log_area"] = np.log(incremental["area"].to_numpy(float))
    incremental["log1p_o2m_count"] = np.log1p(incremental["o2m_positive_count"].to_numpy(float))
    incremental["log_q_active"] = np.log(np.maximum(incremental["q_active"].to_numpy(float), 1e-12))
    incremental["margin"] = incremental["o2o_margin"].to_numpy(float)
    features, predictions, metrics, folds = fit_oof_models(incremental, args.seed + 21)
    for name, values in predictions.items():
        incremental[f"prob_{name}"] = values
    incremental["oof_fold"] = folds
    incremental.to_csv(args.output_dir / "incremental_value_oof_predictions.csv", index=False)

    delta_point = {
        "delta_roc_auc": metrics["M3_plus_margin"]["roc_auc"] - metrics["M2_plus_active_ciou"]["roc_auc"],
        "delta_pr_auc": metrics["M3_plus_margin"]["pr_auc"] - metrics["M2_plus_active_ciou"]["pr_auc"],
        "delta_log_loss": metrics["M3_plus_margin"]["log_loss"] - metrics["M2_plus_active_ciou"]["log_loss"],
        "delta_brier": metrics["M3_plus_margin"]["brier"] - metrics["M2_plus_active_ciou"]["brier"],
    }
    delta_boot = {key: [] for key in delta_point}
    y = incremental["fn_at_iou50"].to_numpy(int)
    weights = incremental["sampling_weight"].to_numpy(float)
    p2 = incremental["prob_M2_plus_active_ciou"].to_numpy(float)
    p3 = incremental["prob_M3_plus_margin"].to_numpy(float)
    for indices in stratified_cluster_samples(incremental, args.bootstrap, args.seed + 31):
        m2 = metric_set(y[indices], p2[indices], weights[indices])
        m3 = metric_set(y[indices], p3[indices], weights[indices])
        delta_boot["delta_roc_auc"].append(m3["roc_auc"] - m2["roc_auc"])
        delta_boot["delta_pr_auc"].append(m3["pr_auc"] - m2["pr_auc"])
        delta_boot["delta_log_loss"].append(m3["log_loss"] - m2["log_loss"])
        delta_boot["delta_brier"].append(m3["brier"] - m2["brier"])
    lr = likelihood_ratio(incremental, features["M2_plus_active_ciou"], features["M3_plus_margin"])
    incremental_summary = {
        "outcome": "FN@IoU50",
        "n_gt": len(incremental),
        "n_images": int(incremental["image_id"].nunique()),
        "models": features,
        "oof_metrics": metrics,
        "M3_minus_M2_point": delta_point,
        "M3_minus_M2_ci_95": {key: percentile_ci(values) for key, values in delta_boot.items()},
        "likelihood_ratio_full_weighted_fit": lr,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
        "interpretation_rule": "margin adds useful prediction only if discrimination/calibration deltas improve with intervals excluding no improvement",
    }
    (args.output_dir / "incremental_value_summary.json").write_text(json.dumps(incremental_summary, indent=2) + "\n", encoding="utf-8")

    manifest.update({
        "status": "complete",
        "common_valid_normalized_gt": len(common),
        "incremental_joined_gt": len(joined),
        "incremental_analysis_gt": len(incremental),
        "total_elapsed_sec": time.monotonic() - started,
        "outputs": ["per_direction.csv", "per_gt.csv", "normalized_perturbation_summary.json", "incremental_value_oof_predictions.csv", "incremental_value_summary.json"],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "normalized": normalized_summary, "incremental": incremental_summary}, indent=2))


if __name__ == "__main__":
    main()
