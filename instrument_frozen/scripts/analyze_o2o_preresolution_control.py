"""Compare native pre-conflict Top-1 and final loss-active O2O identities.

This is a read-only replay control.  The primary pre-resolution identity is
defined narrowly as the highest-q candidate inside the native pre-conflict
Top-7 mask (``trace_o2o(...).pre``).  It is not the highest candidate over all
positive q values.  The latter can be emitted only as an explicitly labelled
secondary analysis.

The script requires a frozen image-selection manifest and a checkpoint digest.
It never updates model parameters, features, predictions, or training targets.
Only the focal ground-truth centre is replayed under the requested stress
contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
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

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o


DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))
PRIMARY_DEFINITION = "highest-q candidate inside the native pre-conflict Top-7 mask"
SECONDARY_DEFINITION = "highest-q candidate among all candidates with q>0 (secondary only)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stress-mode", choices=("fixed_1px", "equivalent_side"), required=True)
    parser.add_argument("--kappa", type=float, default=0.0625)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--checkpoint-role", default="frozen native O2O audit checkpoint")
    parser.add_argument("--include-all-positive-secondary", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="audit at most three selected images and skip bootstrap")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_bin(area: float) -> str:
    side = math.sqrt(max(float(area), 0.0))
    if 8.0 <= side < 16.0:
        return "t_8_16"
    if 16.0 <= side < 32.0:
        return "s_16_32"
    return "outside_primary_bins"


def image_stratum(label: dict, imgsz: int) -> str | None:
    height, width = map(float, label["shape"])
    ratio = min(imgsz / max(height, 1.0), imgsz / max(width, 1.0))
    has_tiny = False
    has_small = False
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


class SelectedDataset:
    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.collate_fn = dataset.collate_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_preconflict_top1(state, gt_index: int) -> int | None:
    """Highest q within the native pre-conflict Top-7 mask."""
    candidates = torch.where(state.pre[0, gt_index])[0]
    if candidates.numel() == 0:
        return None
    scores = state.align[0, gt_index, candidates]
    return int(candidates[torch.argmax(scores)].item())


def all_positive_top1(state, gt_index: int) -> int | None:
    """Secondary-only highest q among all q>0 candidates."""
    scores = state.align[0, gt_index]
    candidates = torch.where(scores > 0)[0]
    if candidates.numel() == 0:
        return None
    return int(candidates[torch.argmax(scores[candidates])].item())


def focal_conflict_changed(state, gt_index: int) -> bool:
    """Whether conflict resolution changed this focal GT's candidate mask."""
    return bool(torch.any(state.pre[0, gt_index] != state.conflict[0, gt_index]).item())


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= 0:
        raise ValueError("non-positive weight total")
    return float(np.sum(values * weights) / denominator)


def weighted_rate(df: pd.DataFrame, column: str, size: str | None = None) -> float:
    part = df if size is None else df[df["size_bin"].eq(size)]
    if part.empty:
        raise ValueError(f"empty analysis cell for {column!r}, size={size!r}")
    return weighted_mean(part[column].to_numpy(float), part["sampling_weight"].to_numpy(float))


def contrast_statistics(df: pd.DataFrame, include_secondary: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    methods = (("preconflict_top7", "pre_fragile"), ("loss_active", "active_fragile"))
    if include_secondary:
        methods += (("all_positive_secondary", "all_positive_fragile"),)
    for method, column in methods:
        tiny = weighted_rate(df, column, "t_8_16")
        small = weighted_rate(df, column, "s_16_32")
        out[f"{method}_fragility_8_16"] = tiny
        out[f"{method}_fragility_16_32"] = small
        out[f"{method}_scale_gap_8_16_minus_16_32"] = tiny - small
    out["loss_active_minus_preconflict_scale_gap"] = (
        out["loss_active_scale_gap_8_16_minus_16_32"]
        - out["preconflict_top7_scale_gap_8_16_minus_16_32"]
    )
    out["any_direction_fragility_disagreement_rate"] = weighted_rate(df, "fragility_disagreement")
    out["any_direction_conflict_involved_rate"] = weighted_rate(df, "any_conflict_involved")
    out["fragility_disagreement_with_conflict_rate"] = weighted_rate(df, "disagreement_with_conflict")
    return out


def stratified_cluster_samples(df: pd.DataFrame, replicates: int, seed: int):
    image_strata = df[["image_id", "stratum"]].drop_duplicates()
    if image_strata["image_id"].duplicated().any():
        raise RuntimeError("an image appears in multiple strata")
    by_stratum = {
        stratum: group["image_id"].tolist()
        for stratum, group in image_strata.groupby("stratum", sort=True)
    }
    row_groups = {image: group.index.to_numpy() for image, group in df.groupby("image_id", sort=False)}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        chunks = []
        for images in by_stratum.values():
            sampled = rng.choice(images, size=len(images), replace=True)
            chunks.extend(row_groups[image] for image in sampled)
        yield np.concatenate(chunks)


def percentile_ci(values: list[float]) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def directional_conflict_summary(df: pd.DataFrame) -> dict:
    disagreement = df["flip_disagreement"].eq(1)
    identity_disagreement = df["pre_active_identity_disagreement"].eq(1)
    conflict = df["conflict_involved"].eq(1)

    def rate(mask: pd.Series, denominator: pd.Series | None = None) -> float | None:
        selected = df if denominator is None else df[denominator]
        if selected.empty:
            return None
        values = mask.loc[selected.index].to_numpy(float)
        return weighted_mean(values, selected["sampling_weight"].to_numpy(float))

    unexplained = disagreement & ~conflict
    return {
        "directional_rows": int(len(df)),
        "directional_pre_vs_active_flip_disagreement_count": int(disagreement.sum()),
        "directional_pre_vs_active_flip_disagreement_weighted_rate": rate(disagreement),
        "conflict_involved_weighted_rate": rate(conflict),
        "conflict_involved_among_flip_disagreements_weighted_rate": rate(conflict, disagreement),
        "flip_disagreement_without_recorded_conflict_count": int(unexplained.sum()),
        "pre_vs_active_identity_disagreement_weighted_rate": rate(identity_disagreement),
        "conflict_involved_among_identity_disagreements_weighted_rate": rate(conflict, identity_disagreement),
        "definition": (
            "conflict_involved means the focal GT's native pre-conflict Top-7 mask differs from its "
            "post-conflict mask at base or shifted state; it localises but does not establish causality"
        ),
    }


def load_selected_manifest(path: Path, dataset, imgsz: int, smoke: bool) -> tuple[list[dict], list[int]]:
    rows = pd.read_csv(path).to_dict("records")
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    if not rows or required.difference(rows[0]):
        raise ValueError("selected-images manifest is empty or lacks required columns")
    rows.sort(key=lambda row: str(row["image_id"]))
    if smoke:
        # Exercise both prespecified size bins when the frozen manifest has the
        # usual t_only/s_only/t_and_s allocation.  This is selection by frozen
        # stratum only, never by an audit outcome.
        smoke_rows = []
        for stratum in ("t_only", "s_only", "t_and_s"):
            match = next((row for row in rows if str(row["stratum"]) == stratum), None)
            if match is not None:
                smoke_rows.append(match)
        rows = smoke_rows or rows[:3]
    seen: set[str] = set()
    indices: list[int] = []
    for row in rows:
        index = int(row["dataset_index"])
        if index < 0 or index >= len(dataset):
            raise IndexError(f"selected dataset index out of range: {index}")
        image_id = Path(dataset.labels[index]["im_file"]).stem
        observed_stratum = image_stratum(dataset.labels[index], imgsz)
        if image_id != str(row["image_id"]) or observed_stratum != str(row["stratum"]):
            raise RuntimeError("selected-images manifest does not match frozen dataset index/stratum")
        if image_id in seen:
            raise RuntimeError(f"duplicate selected image: {image_id}")
        probability = float(row["inclusion_probability"])
        weight = float(row["sampling_weight"])
        if not (0 < probability <= 1) or not math.isclose(weight, 1.0 / probability, rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError(f"invalid inclusion probability/weight for {image_id}")
        seen.add(image_id)
        indices.append(index)
    return rows, indices


def artifact_hashes(output_dir: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256(output_dir / name) for name in names}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    observed_checkpoint_hash = sha256(args.checkpoint)
    if observed_checkpoint_hash.lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data YAML or selected-images manifest is missing")
    if args.stress_mode == "equivalent_side" and not (0 < args.kappa < 0.5):
        raise ValueError("equivalent-side replay requires 0 < kappa < 0.5")
    if args.bootstrap < 1 and not args.smoke:
        raise ValueError("bootstrap must be positive outside smoke mode")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    started = time.monotonic()
    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("this Top-7 control requires the frozen native YOLO26 end-to-end direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    criterion = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if int(criterion.assigner.topk) != 7:
        raise RuntimeError("native pre-conflict contract is not Top-7")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": args.imgsz,
            "batch": 1,
            "workers": args.workers,
            "rect": False,
            "cache": False,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        data["val"],
        batch=1,
        data=data,
        mode="val",
        stride=max(int(model.stride.max()), 32),
    )
    selected_rows, selected_indices = load_selected_manifest(args.selected_images, dataset, args.imgsz, args.smoke)
    selected_by_image = {str(row["image_id"]): row for row in selected_rows}
    loader = build_dataloader(
        SelectedDataset(dataset, selected_indices),
        batch=1,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
    )

    args.output_dir.mkdir(parents=True)
    selection_copy = args.output_dir / "selected_images.csv"
    write_csv(selection_copy, selected_rows)
    manifest = {
        "status": "running",
        "protocol": "o2o_preresolution_top7_vs_loss_active_v1",
        "read_only": True,
        "primary_pre_resolution_definition": PRIMARY_DEFINITION,
        "secondary_definition": SECONDARY_DEFINITION if args.include_all_positive_secondary else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": observed_checkpoint_hash,
        "checkpoint_role": args.checkpoint_role,
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "selected_images_source": str(args.selected_images.resolve()),
        "selected_images_source_sha256": sha256(args.selected_images),
        "selected_images": len(selected_rows),
        "stress_mode": args.stress_mode,
        "stress_definition": (
            "one input pixel in each cardinal direction"
            if args.stress_mode == "fixed_1px"
            else "kappa times focal-GT equivalent side in each cardinal direction"
        ),
        "kappa": args.kappa if args.stress_mode == "equivalent_side" else None,
        "imgsz": args.imgsz,
        "native_o2o_topk": 7,
        "native_o2o_topk2": 1,
        "bootstrap_replicates": 0 if args.smoke else args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "image cluster within frozen outcome-blind sampling stratum",
        "ipw": True,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    directional_rows: list[dict] = []
    gt_rows: list[dict] = []
    focal_counts: Counter[str] = Counter()
    with torch.no_grad():
        for batch in loader:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            batch["img"] = batch["img"].float() / 255
            output = model(batch["img"])
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise RuntimeError("expected raw dual-branch output")
            raw = output[1]["one2one"]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            image_h, image_w = batch["img"].shape[-2:]
            targets = criterion.preprocess(
                targets,
                1,
                torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype),
            )
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_state = trace_o2o(criterion, raw, labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            selection = selected_by_image[image_id]
            for gt_index in torch.where(mask[0, :, 0])[0].tolist():
                box = boxes[0, gt_index]
                width = float(box[2] - box[0])
                height = float(box[3] - box[1])
                area = width * height
                bucket = size_bin(area)
                if bucket == "outside_primary_bins":
                    continue
                focal_counts[bucket] += 1
                shift = 1.0 if args.stress_mode == "fixed_1px" else args.kappa * math.sqrt(area)
                base_pre = native_preconflict_top1(base_state, gt_index)
                base_active, _, _ = active_candidate_and_margin(base_state, 0, gt_index)
                base_all = all_positive_top1(base_state, gt_index) if args.include_all_positive_secondary else None
                base_conflict = focal_conflict_changed(base_state, gt_index)
                object_directions: list[dict] = []
                for direction, unit_dx, unit_dy in DIRECTIONS:
                    shifted_box = legal_shift(
                        box,
                        unit_dx * shift,
                        unit_dy * shift,
                        float(image_w),
                        float(image_h),
                    )
                    if shifted_box is None:
                        continue
                    shifted_boxes = boxes.clone()
                    shifted_boxes[0, gt_index] = shifted_box
                    shifted_state = trace_o2o(criterion, raw, labels, shifted_boxes, mask)
                    shifted_pre = native_preconflict_top1(shifted_state, gt_index)
                    shifted_active, _, _ = active_candidate_and_margin(shifted_state, 0, gt_index)
                    shifted_all = all_positive_top1(shifted_state, gt_index) if args.include_all_positive_secondary else None
                    shifted_conflict = focal_conflict_changed(shifted_state, gt_index)
                    pre_flip = int(base_pre != shifted_pre)
                    active_flip = int(base_active != shifted_active)
                    conflict_involved = int(base_conflict or shifted_conflict)
                    row = {
                        "image_id": image_id,
                        "gt_id": gt_index,
                        "direction": direction,
                        "stratum": str(selection["stratum"]),
                        "sampling_weight": float(selection["sampling_weight"]),
                        "inclusion_probability": float(selection["inclusion_probability"]),
                        "width": width,
                        "height": height,
                        "area": area,
                        "equivalent_side": math.sqrt(area),
                        "size_bin": bucket,
                        "shift_pixels": shift,
                        "preconflict_top7_base": base_pre,
                        "preconflict_top7_shift": shifted_pre,
                        "preconflict_top7_flip": pre_flip,
                        "loss_active_base": base_active,
                        "loss_active_shift": shifted_active,
                        "loss_active_flip": active_flip,
                        "pre_active_identity_disagreement_base": int(base_pre != base_active),
                        "pre_active_identity_disagreement_shift": int(shifted_pre != shifted_active),
                        "pre_active_identity_disagreement": int(base_pre != base_active or shifted_pre != shifted_active),
                        "conflict_changed_base": int(base_conflict),
                        "conflict_changed_shift": int(shifted_conflict),
                        "conflict_involved": conflict_involved,
                        "flip_disagreement": int(pre_flip != active_flip),
                        "flip_disagreement_with_conflict": int(pre_flip != active_flip and conflict_involved),
                    }
                    if args.include_all_positive_secondary:
                        row.update(
                            {
                                "all_positive_secondary_base": base_all,
                                "all_positive_secondary_shift": shifted_all,
                                "all_positive_secondary_flip": int(base_all != shifted_all),
                            }
                        )
                    directional_rows.append(row)
                    object_directions.append(row)
                if not object_directions:
                    continue
                pre_fragile = int(any(row["preconflict_top7_flip"] for row in object_directions))
                active_fragile = int(any(row["loss_active_flip"] for row in object_directions))
                any_conflict = int(any(row["conflict_involved"] for row in object_directions))
                aggregate = {
                    "image_id": image_id,
                    "gt_id": gt_index,
                    "stratum": str(selection["stratum"]),
                    "sampling_weight": float(selection["sampling_weight"]),
                    "inclusion_probability": float(selection["inclusion_probability"]),
                    "width": width,
                    "height": height,
                    "area": area,
                    "equivalent_side": math.sqrt(area),
                    "size_bin": bucket,
                    "shift_pixels": shift,
                    "legal_directions": len(object_directions),
                    "pre_fragile": pre_fragile,
                    "active_fragile": active_fragile,
                    "fragility_disagreement": int(pre_fragile != active_fragile),
                    "any_conflict_involved": any_conflict,
                    "disagreement_with_conflict": int(pre_fragile != active_fragile and any_conflict),
                }
                if args.include_all_positive_secondary:
                    aggregate["all_positive_fragile"] = int(
                        any(row["all_positive_secondary_flip"] for row in object_directions)
                    )
                gt_rows.append(aggregate)

    if not gt_rows or not directional_rows:
        raise RuntimeError("audit produced no primary-bin rows")
    gt_df = pd.DataFrame(gt_rows)
    direction_df = pd.DataFrame(directional_rows)
    if set(gt_df["size_bin"].unique()) != {"t_8_16", "s_16_32"} and not args.smoke:
        raise RuntimeError("full analysis requires both prespecified size bins")
    write_csv(args.output_dir / "per_direction.csv", directional_rows)
    write_csv(args.output_dir / "per_gt.csv", gt_rows)

    point = contrast_statistics(gt_df, args.include_all_positive_secondary)
    bootstrap_rows: list[dict] = []
    bootstrap_ci: dict[str, list[float]] = {}
    if not args.smoke:
        values = {key: [] for key in point}
        for replicate, indices in enumerate(
            stratified_cluster_samples(gt_df, args.bootstrap, args.bootstrap_seed), start=1
        ):
            stats = contrast_statistics(gt_df.iloc[indices], args.include_all_positive_secondary)
            bootstrap_rows.append({"replicate": replicate, **stats})
            for key, value in stats.items():
                values[key].append(value)
        bootstrap_ci = {key: percentile_ci(samples) for key, samples in values.items()}
        write_csv(args.output_dir / "bootstrap_replicates.csv", bootstrap_rows)

    conflict_summary = directional_conflict_summary(direction_df)
    gt_disagreement = gt_df["fragility_disagreement"].eq(1)
    if gt_disagreement.any():
        disagreement_part = gt_df[gt_disagreement]
        conflict_summary["conflict_involved_among_any_direction_fragility_disagreements_weighted_rate"] = weighted_rate(
            disagreement_part, "any_conflict_involved"
        )
    else:
        conflict_summary["conflict_involved_among_any_direction_fragility_disagreements_weighted_rate"] = None
    conflict_summary["any_direction_fragility_disagreement_count"] = int(gt_disagreement.sum())
    conflict_summary["any_direction_fragility_disagreement_without_recorded_conflict_count"] = int(
        (gt_disagreement & gt_df["any_conflict_involved"].eq(0)).sum()
    )

    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "estimand": (
            "IPW any-direction identity fragility for 8-16 px versus 16-32 px focal GTs; "
            "paired comparison of native pre-conflict Top-7 q winner and final loss-active identity"
        ),
        "primary_pre_resolution_definition": PRIMARY_DEFINITION,
        "secondary_definition": SECONDARY_DEFINITION if args.include_all_positive_secondary else None,
        "n_gt": int(len(gt_df)),
        "n_images": int(gt_df["image_id"].nunique()),
        "n_directional_replays": int(len(direction_df)),
        "focal_gt_by_bin": dict(focal_counts),
        "point": point,
        "ci_95": bootstrap_ci,
        "bootstrap_replicates": 0 if args.smoke else args.bootstrap,
        "bootstrap_unit": "image cluster within frozen outcome-blind sampling stratum",
        "conflict_localisation": conflict_summary,
        "interpretation_guardrail": (
            "Conflict-stage coincidence localises disagreement between the two identity definitions; "
            "it is not a causal effect estimate."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    output_names = ["selected_images.csv", "per_direction.csv", "per_gt.csv", "summary.json"]
    if not args.smoke:
        output_names.append("bootstrap_replicates.csv")
    manifest.update(
        {
            "status": "smoke_complete" if args.smoke else "complete",
            "focal_gt_by_bin": dict(focal_counts),
            "per_gt_rows": len(gt_rows),
            "per_direction_rows": len(directional_rows),
            "elapsed_sec": time.monotonic() - started,
            "outputs": output_names,
            "artifact_sha256": artifact_hashes(args.output_dir, output_names),
        }
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_names = ["manifest.json", *output_names]
    checksum_text = "".join(f"{sha256(args.output_dir / name)}  {name}\n" for name in checksum_names)
    (args.output_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
