"""Read-only nonfocal assignment-spillover audit on frozen O2O outputs.

For every audited tiny/small focal GT and cardinal replay direction, only the
focal GT centre is shifted.  The script then records whether the native O2O
loss-active identity of every other GT in the image changes.  The primary
structural relationship is whether the focal and nonfocal GT share any native
pre-conflict Top-k claim in the base or shifted state (yes/no).  Continuous
normalized centre distance is recorded without thresholding.

For backward-compatible secondary description only, nonfocal GTs are also
classified as ``shared_conflict``, ``local_nonsharing``, or
``distant_nonsharing``.  The local/distant split uses the CLI-specified cutoff
and never changes the primary shared-vs-nonshared estimates.

Both pair-level spillover and focal-event ``any spillover`` rates are estimated
with the frozen image sampling weights.  Percentile intervals use a stratified
image-cluster bootstrap, resampling selected images within selection strata.
The detector forward output is frozen; there is no training or postprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from run_reviewer_killer_controls import SelectedDataset, image_stratum, size_bin


DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))
PRIMARY_RELATIONSHIPS = ("shared_native_claim_yes", "shared_native_claim_no")
SECONDARY_RELATIONSHIPS = ("shared_conflict", "local_nonsharing", "distant_nonsharing")
FOCAL_GROUPS = ("all", "focal_flip", "focal_stable")
PRIMARY_DIFFERENCE_KEY = "pair_spillover|shared_native_claim_yes_minus_no|all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selected-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stress", choices=("fixed_1px", "equivalent_side"), required=True)
    parser.add_argument("--kappa", type=float, default=0.0625)
    parser.add_argument(
        "--local-distance-threshold",
        type=float,
        default=2.0,
        help="Local/nonlocal cutoff for the prespecified normalized base-centre distance",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--max-images",
        type=int,
        help="Outcome-blind gated-pilot size selected by seeded SHA-256 rank",
    )
    selection_group.add_argument("--smoke", action="store_true", help="Audit only the first selected image")
    parser.add_argument("--subset-seed", type=int, default=20260831)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_image_subset(rows: list[dict], max_images: int, seed: int) -> list[dict]:
    """Select an outcome-blind pilot subset, independent of manifest row order."""
    if not (0 < max_images <= len(rows)):
        raise ValueError("max-images must be between 1 and the source selected-image count")

    def rank(row: dict) -> tuple[str, str]:
        image_id = str(row["image_id"])
        digest = hashlib.sha256(f"{seed}|{image_id}".encode("utf-8")).hexdigest()
        return digest, image_id

    return sorted(rows, key=rank)[:max_images]


def ordered_image_ids_sha256(rows: list[dict]) -> str:
    payload = "".join(f"{row['image_id']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def active_identity(state, gt: int) -> int | None:
    active, _, _ = active_candidate_and_margin(state, 0, gt)
    return active


def centre_and_equivalent_side(box: torch.Tensor) -> tuple[float, float, float]:
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    return float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2), math.sqrt(width * height)


def normalized_centre_distance(box_a: torch.Tensor, box_b: torch.Tensor) -> tuple[float, float]:
    ax, ay, side_a = centre_and_equivalent_side(box_a)
    bx, by, side_b = centre_and_equivalent_side(box_b)
    distance = math.hypot(ax - bx, ay - by)
    normalizer = (side_a + side_b) / 2.0
    if normalizer <= 0:
        raise RuntimeError("encountered a non-positive GT equivalent-side normalizer")
    return distance, distance / normalizer


def shared_claim_ids(state, focal_gt: int, neighbor_gt: int) -> set[int]:
    shared = state.pre[0, focal_gt] & state.pre[0, neighbor_gt]
    return set(map(int, torch.where(shared)[0].tolist()))


def pair_owner(state, candidate: int, focal_gt: int, neighbor_gt: int) -> str:
    focal_owns = bool(state.conflict[0, focal_gt, candidate])
    neighbor_owns = bool(state.conflict[0, neighbor_gt, candidate])
    if focal_owns and neighbor_owns:
        raise RuntimeError("native conflict state assigns one candidate to both pair members")
    if focal_owns:
        return "focal"
    if neighbor_owns:
        return "neighbor"
    return "other_or_none"


def relation_metrics(
    base,
    shifted,
    boxes: torch.Tensor,
    shifted_boxes: torch.Tensor,
    focal_gt: int,
    neighbor_gt: int,
    local_threshold: float,
) -> dict:
    shared_base = shared_claim_ids(base, focal_gt, neighbor_gt)
    shared_shift = shared_claim_ids(shifted, focal_gt, neighbor_gt)
    shared_union = shared_base | shared_shift
    distance_base_px, distance_base_norm = normalized_centre_distance(boxes[0, focal_gt], boxes[0, neighbor_gt])
    distance_shift_px, distance_shift_norm = normalized_centre_distance(shifted_boxes[0, focal_gt], shifted_boxes[0, neighbor_gt])
    if shared_union:
        secondary_relationship = "shared_conflict"
    elif distance_base_norm < local_threshold:
        secondary_relationship = "local_nonsharing"
    else:
        secondary_relationship = "distant_nonsharing"
    shared_native_claim = int(bool(shared_union))
    primary_relationship = "shared_native_claim_yes" if shared_native_claim else "shared_native_claim_no"

    base_focal_wins = base_neighbor_wins = shift_focal_wins = shift_neighbor_wins = transfers = 0
    for candidate in shared_union:
        owner_base = pair_owner(base, candidate, focal_gt, neighbor_gt)
        owner_shift = pair_owner(shifted, candidate, focal_gt, neighbor_gt)
        base_focal_wins += int(owner_base == "focal")
        base_neighbor_wins += int(owner_base == "neighbor")
        shift_focal_wins += int(owner_shift == "focal")
        shift_neighbor_wins += int(owner_shift == "neighbor")
        transfers += int((owner_base, owner_shift) in {("focal", "neighbor"), ("neighbor", "focal")})

    return {
        "shared_native_claim": shared_native_claim,
        "primary_relationship": primary_relationship,
        "secondary_relationship": secondary_relationship,
        "relationship": secondary_relationship,
        "centre_distance_px_base": distance_base_px,
        "normalized_centre_distance_base": distance_base_norm,
        "centre_distance_px_shift": distance_shift_px,
        "normalized_centre_distance_shift": distance_shift_norm,
        "shared_claim_base": int(bool(shared_base)),
        "shared_claim_shift": int(bool(shared_shift)),
        "shared_candidate_count_base": len(shared_base),
        "shared_candidate_count_shift": len(shared_shift),
        "shared_candidate_count_union": len(shared_union),
        "focal_won_shared_count_base": base_focal_wins,
        "neighbor_won_shared_count_base": base_neighbor_wins,
        "focal_won_shared_count_shift": shift_focal_wins,
        "neighbor_won_shared_count_shift": shift_neighbor_wins,
        "pair_ownership_transfer_count": transfers,
    }


def new_cluster(image_id: str, stratum: str, sampling_weight: float) -> dict:
    return {
        "image_id": image_id,
        "stratum": stratum,
        "sampling_weight": sampling_weight,
        "metrics": defaultdict(lambda: [0, 0]),
    }


def add_metric(cluster: dict, family: str, relationship: str, focal_group: str, numerator: int, denominator: int) -> None:
    key = f"{family}|{relationship}|{focal_group}"
    cluster["metrics"][key][0] += int(numerator)
    cluster["metrics"][key][1] += int(denominator)


def metric_keys() -> list[str]:
    return [
        f"{family}|{relationship}|{focal_group}"
        for family in ("pair_spillover", "focal_any_spillover")
        for relationship in ("overall", *PRIMARY_RELATIONSHIPS, *SECONDARY_RELATIONSHIPS)
        for focal_group in FOCAL_GROUPS
    ]


def aggregate_metrics(clusters: list[dict], multiplicity: dict[str, int] | None = None) -> dict[str, dict]:
    result = {}
    for key in metric_keys():
        weighted_numerator = weighted_denominator = 0.0
        raw_numerator = raw_denominator = 0
        for cluster in clusters:
            copies = 1 if multiplicity is None else multiplicity.get(cluster["image_id"], 0)
            if copies == 0:
                continue
            numerator, denominator = cluster["metrics"].get(key, (0, 0))
            raw_numerator += copies * numerator
            raw_denominator += copies * denominator
            factor = copies * float(cluster["sampling_weight"])
            weighted_numerator += factor * numerator
            weighted_denominator += factor * denominator
        result[key] = {
            "rate": weighted_numerator / weighted_denominator if weighted_denominator else None,
            "weighted_numerator": weighted_numerator,
            "weighted_denominator": weighted_denominator,
            "raw_numerator": raw_numerator,
            "raw_denominator": raw_denominator,
        }
    return result


def bootstrap_metrics(clusters: list[dict], replicates: int, seed: int) -> tuple[list[dict], dict[str, tuple[float | None, float | None]]]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        by_stratum[cluster["stratum"]].append(cluster["image_id"])
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    distributions: dict[str, list[float]] = {key: [] for key in (*metric_keys(), PRIMARY_DIFFERENCE_KEY)}
    for replicate in range(replicates):
        multiplicity: Counter = Counter()
        for stratum in sorted(by_stratum):
            image_ids = by_stratum[stratum]
            sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
            multiplicity.update(map(str, sampled.tolist()))
        estimates = aggregate_metrics(clusters, dict(multiplicity))
        row = {"replicate": replicate}
        for key in metric_keys():
            value = estimates[key]["rate"]
            row[key] = value
            if value is not None:
                distributions[key].append(float(value))
        shared_rate = estimates["pair_spillover|shared_native_claim_yes|all"]["rate"]
        nonshared_rate = estimates["pair_spillover|shared_native_claim_no|all"]["rate"]
        difference = None if shared_rate is None or nonshared_rate is None else shared_rate - nonshared_rate
        row[PRIMARY_DIFFERENCE_KEY] = difference
        if difference is not None:
            distributions[PRIMARY_DIFFERENCE_KEY].append(float(difference))
        rows.append(row)
    intervals = {}
    for key, values in distributions.items():
        if values:
            intervals[key] = (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
        else:
            intervals[key] = (None, None)
    return rows, intervals


def nested_summary(observed: dict[str, dict], intervals: dict[str, tuple[float | None, float | None]]) -> dict:
    summary: dict[str, dict] = {}
    for key in metric_keys():
        family, relationship, focal_group = key.split("|")
        lower, upper = intervals[key]
        cell = dict(observed[key])
        cell.update({"ci95_lower": lower, "ci95_upper": upper})
        summary.setdefault(family, {}).setdefault(relationship, {})[focal_group] = cell
    return summary


def primary_summary(observed: dict[str, dict], intervals: dict[str, tuple[float | None, float | None]]) -> dict:
    shared_key = "pair_spillover|shared_native_claim_yes|all"
    nonshared_key = "pair_spillover|shared_native_claim_no|all"
    shared = dict(observed[shared_key])
    nonshared = dict(observed[nonshared_key])
    shared.update({"ci95_lower": intervals[shared_key][0], "ci95_upper": intervals[shared_key][1]})
    nonshared.update({"ci95_lower": intervals[nonshared_key][0], "ci95_upper": intervals[nonshared_key][1]})
    difference = None if shared["rate"] is None or nonshared["rate"] is None else shared["rate"] - nonshared["rate"]
    return {
        "estimand": "IPW pair-level nonfocal spillover rate by any shared native claim in base or shift",
        "shared_native_claim_yes": shared,
        "shared_native_claim_no": nonshared,
        "shared_minus_nonshared_rate_difference": {
            "estimate": difference,
            "ci95_lower": intervals[PRIMARY_DIFFERENCE_KEY][0],
            "ci95_upper": intervals[PRIMARY_DIFFERENCE_KEY][1],
            "bootstrap_unit": "image cluster within frozen selection stratum",
        },
        "continuous_normalized_centre_distance": {
            "analysis_role": "primary continuous structural measure; no cutoff is applied",
            "pair_level_columns": [
                "normalized_centre_distance_base",
                "normalized_centre_distance_shift",
            ],
            "output": "nonfocal_events.csv",
        },
    }


NONFOCAL_FIELDS = [
    "image_id", "stratum", "sampling_weight", "focal_gt_id", "neighbor_gt_id", "direction", "stress",
    "shift_pixels", "focal_size_bin", "focal_width", "focal_height", "focal_area", "neighbor_width",
    "neighbor_height", "neighbor_area", "local_distance_threshold", "shared_native_claim",
    "primary_relationship", "secondary_relationship", "relationship", "centre_distance_px_base",
    "normalized_centre_distance_base", "centre_distance_px_shift", "normalized_centre_distance_shift",
    "shared_claim_base", "shared_claim_shift", "shared_candidate_count_base", "shared_candidate_count_shift",
    "shared_candidate_count_union", "focal_won_shared_count_base", "neighbor_won_shared_count_base",
    "focal_won_shared_count_shift", "neighbor_won_shared_count_shift", "pair_ownership_transfer_count",
    "focal_active_base", "focal_active_shift", "focal_flip", "neighbor_active_base", "neighbor_active_shift",
    "neighbor_flip",
]


FOCAL_FIELDS = [
    "image_id", "stratum", "sampling_weight", "focal_gt_id", "direction", "stress", "shift_pixels",
    "focal_size_bin", "focal_width", "focal_height", "focal_area", "focal_active_base", "focal_active_shift",
    "focal_flip", "nonfocal_neighbor_count", "nonfocal_spillover_count", "any_nonfocal_spillover",
    "shared_native_claim_yes_neighbor_count", "shared_native_claim_yes_spillover_count",
    "shared_native_claim_yes_any_spillover", "shared_native_claim_no_neighbor_count",
    "shared_native_claim_no_spillover_count", "shared_native_claim_no_any_spillover",
    "shared_conflict_neighbor_count", "shared_conflict_spillover_count", "shared_conflict_any_spillover",
    "local_nonsharing_neighbor_count", "local_nonsharing_spillover_count", "local_nonsharing_any_spillover",
    "distant_nonsharing_neighbor_count", "distant_nonsharing_spillover_count", "distant_nonsharing_any_spillover",
]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.checkpoint.is_file() or sha256(args.checkpoint).lower() != args.expected_sha256.lower():
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or not args.selected_images.is_file():
        raise FileNotFoundError("data or selected-images manifest missing")
    if not (0 < args.kappa < 0.5):
        raise ValueError("kappa must be in (0, 0.5)")
    if args.local_distance_threshold <= 0:
        raise ValueError("local-distance-threshold must be positive")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("spillover protocol is frozen for the native YOLO26 direct-LTRB contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    if int(o2o.assigner.topk) != 7 or float(o2o.assigner.alpha) != 0.5 or float(o2o.assigner.beta) != 6.0:
        raise RuntimeError("unexpected native O2O assignment contract")

    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": args.imgsz, "batch": 1, "workers": args.workers, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, args.imgsz)
        if stratum is not None:
            strata[stratum].append(index)

    with args.selected_images.open(newline="", encoding="utf-8-sig") as stream:
        selected_rows = list(csv.DictReader(stream))
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    if not selected_rows or required.difference(selected_rows[0]):
        raise ValueError("selected-images manifest is empty or incomplete")
    seen_indices, seen_ids = set(), set()
    for row in selected_rows:
        index, stratum = int(row["dataset_index"]), str(row["stratum"])
        image_id = str(row["image_id"])
        if index in seen_indices or image_id in seen_ids:
            raise RuntimeError("selected-images manifest contains duplicates")
        if not (0 < float(row["inclusion_probability"]) <= 1 and float(row["sampling_weight"]) > 0):
            raise RuntimeError("selected-images manifest contains invalid probability or weight")
        observed = Path(dataset.labels[index]["im_file"]).stem
        if image_id != observed or stratum not in strata or index not in strata[stratum]:
            raise RuntimeError("selected-images manifest does not match the frozen dataset index")
        seen_indices.add(index)
        seen_ids.add(image_id)
    source_selected_count = len(selected_rows)
    if args.max_images is not None:
        selected_rows = deterministic_image_subset(selected_rows, args.max_images, args.subset_seed)
        selection_mode = "gated_pilot_sha256"
    else:
        selected_rows.sort(key=lambda row: str(row["image_id"]))
        selection_mode = "smoke" if args.smoke else "full"
    if args.smoke:
        selected_rows = selected_rows[:1]
    selected_indices = [int(row["dataset_index"]) for row in selected_rows]
    selection_by_image = {str(row["image_id"]): row for row in selected_rows}
    loader = build_dataloader(SelectedDataset(dataset, selected_indices), batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)

    args.output_dir.mkdir(parents=True)
    selected_output_rows = [
        {
            "dataset_index": int(row["dataset_index"]),
            "image_id": str(row["image_id"]),
            "stratum": str(row["stratum"]),
            "inclusion_probability": float(row["inclusion_probability"]),
            "sampling_weight": float(row["sampling_weight"]),
        }
        for row in selected_rows
    ]
    write_rows(args.output_dir / "selected_images.csv", selected_output_rows)
    selected_images_output_sha256 = sha256(args.output_dir / "selected_images.csv")
    manifest = {
        "status": "running",
        "protocol": "read_only_native_o2o_assignment_spillover_v2",
        "read_only": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.expected_sha256.lower(),
        "data": str(args.data),
        "selected_images_source": str(args.selected_images),
        "selected_images_source_sha256": sha256(args.selected_images),
        "source_selected_images": source_selected_count,
        "audited_selected_images": len(selected_rows),
        "selection_mode": selection_mode,
        "gated_pilot_subset": {
            "enabled": args.max_images is not None,
            "source_selected_images": source_selected_count,
            "pilot_selected_images": len(selected_rows) if args.max_images is not None else None,
            "max_images": args.max_images,
            "subset_seed": args.subset_seed if args.max_images is not None else None,
            "ranking_rule": "ascending SHA256(f'{subset_seed}|{image_id}') with image_id as deterministic collision tie-break",
            "selection_timing": "after frozen selected-manifest validation and before detector forward or outcome access",
            "selected_subset_image_ids_sha256": ordered_image_ids_sha256(selected_rows) if args.max_images is not None else None,
            "selected_subset_hash_payload": "UTF-8 image_id values in SHA-256 rank order, each followed by newline",
            "selected_images_csv_sha256": selected_images_output_sha256 if args.max_images is not None else None,
        },
        "stress": args.stress,
        "kappa": args.kappa if args.stress == "equivalent_side" else None,
        "directions": [direction for direction, _, _ in DIRECTIONS],
        "focal_population": "valid 8--16 px and 16--32 px GTs in frozen selected images",
        "nonfocal_population": "all other valid GTs in the same image",
        "spillover_event": "change in native post-conflict loss-active O2O identity of a nonfocal GT",
        "primary_estimands": {
            "structural_relationship": "any shared native pre-conflict Top-k claim in base or shifted state: yes versus no",
            "rate_comparison": "IPW pair-level nonfocal spillover rates and shared-minus-nonshared difference",
            "continuous_measure": "base and shifted centre distance normalized by the arithmetic mean equivalent side length",
            "uses_local_distance_threshold": False,
        },
        "secondary_descriptive_estimands": {
            "relationship_order": list(SECONDARY_RELATIONSHIPS),
            "shared_conflict_definition": "at least one candidate jointly claimed in native pre-conflict Top-k by the focal and nonfocal GT in base or shifted state",
            "local_nonsharing_definition": "no shared claim and normalized base-centre distance < local_distance_threshold",
            "distant_nonsharing_definition": "no shared claim and normalized base-centre distance >= local_distance_threshold",
            "local_distance_threshold": args.local_distance_threshold,
            "threshold_source": "CLI --local-distance-threshold; must be fixed independently of these outcomes",
            "affects_primary_estimands": False,
        },
        "relationship_order": list(SECONDARY_RELATIONSHIPS),
        "relationship_order_role": "backward-compatible secondary descriptive classification",
        "shared_conflict_definition": "at least one candidate jointly claimed in native pre-conflict Top-k by the focal and nonfocal GT in base or shifted state",
        "local_nonsharing_definition": "no shared claim and normalized base-centre distance < local_distance_threshold",
        "distant_nonsharing_definition": "no shared claim and normalized base-centre distance >= local_distance_threshold",
        "distance_normalizer": "arithmetic mean of focal and nonfocal equivalent side lengths sqrt(GT area)",
        "distance_state_for_relationship": "base GT boxes; shifted distance is recorded descriptively but does not define the threshold class",
        "local_distance_threshold": args.local_distance_threshold,
        "weighting": "inverse inclusion probability from the selected-images manifest",
        "bootstrap": "stratified image-cluster resampling with replacement within frozen selection strata",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "script_sha256": sha256(Path(__file__)),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    focal_rows: list[dict] = []
    clusters_by_image: dict[str, dict] = {}
    nonfocal_row_count = 0
    primary_relationship_raw_counts: Counter = Counter()
    relationship_raw_counts: Counter = Counter()
    started = time.monotonic()
    nonfocal_path = args.output_dir / "nonfocal_events.csv"
    with torch.no_grad(), nonfocal_path.open("w", newline="", encoding="utf-8") as nonfocal_stream:
        nonfocal_writer = csv.DictWriter(nonfocal_stream, fieldnames=NONFOCAL_FIELDS, extrasaction="raise")
        nonfocal_writer.writeheader()
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
            targets = o2o.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            valid_gts = list(map(int, torch.where(mask[0, :, 0])[0].tolist()))
            base = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            base_active = {gt: active_identity(base, gt) for gt in valid_gts}
            image_id = Path(batch["im_file"][0]).stem
            selection = selection_by_image[image_id]
            stratum = str(selection["stratum"])
            sampling_weight = float(selection["sampling_weight"])
            cluster = new_cluster(image_id, stratum, sampling_weight)
            clusters_by_image[image_id] = cluster

            for focal_gt in valid_gts:
                focal_box = boxes[0, focal_gt]
                focal_width = float(focal_box[2] - focal_box[0])
                focal_height = float(focal_box[3] - focal_box[1])
                focal_area = focal_width * focal_height
                focal_size_bin = size_bin(focal_area)
                if focal_size_bin not in {"t_8_16", "s_16_32"}:
                    continue
                delta = 1.0 if args.stress == "fixed_1px" else args.kappa * math.sqrt(focal_area)
                for direction, unit_dx, unit_dy in DIRECTIONS:
                    shifted_box = legal_shift(focal_box, unit_dx * delta, unit_dy * delta, float(image_w), float(image_h))
                    if shifted_box is None:
                        continue
                    shifted_boxes = boxes.clone()
                    shifted_boxes[0, focal_gt] = shifted_box
                    shifted = trace_o2o(o2o, raw["one2one"], labels, shifted_boxes, mask)
                    shifted_active = {gt: active_identity(shifted, gt) for gt in valid_gts}
                    focal_flip = int(base_active[focal_gt] != shifted_active[focal_gt])
                    focal_group = "focal_flip" if focal_flip else "focal_stable"
                    primary_counts = {relationship: 0 for relationship in PRIMARY_RELATIONSHIPS}
                    primary_spillovers = {relationship: 0 for relationship in PRIMARY_RELATIONSHIPS}
                    secondary_counts = {relationship: 0 for relationship in SECONDARY_RELATIONSHIPS}
                    secondary_spillovers = {relationship: 0 for relationship in SECONDARY_RELATIONSHIPS}
                    grouped_relationships = [
                        (relationship, primary_counts, primary_spillovers)
                        for relationship in PRIMARY_RELATIONSHIPS
                    ] + [
                        (relationship, secondary_counts, secondary_spillovers)
                        for relationship in SECONDARY_RELATIONSHIPS
                    ]

                    for neighbor_gt in valid_gts:
                        if neighbor_gt == focal_gt:
                            continue
                        neighbor_box = boxes[0, neighbor_gt]
                        neighbor_width = float(neighbor_box[2] - neighbor_box[0])
                        neighbor_height = float(neighbor_box[3] - neighbor_box[1])
                        neighbor_area = neighbor_width * neighbor_height
                        relation = relation_metrics(base, shifted, boxes, shifted_boxes, focal_gt, neighbor_gt, args.local_distance_threshold)
                        primary_relationship = relation["primary_relationship"]
                        secondary_relationship = relation["secondary_relationship"]
                        neighbor_flip = int(base_active[neighbor_gt] != shifted_active[neighbor_gt])
                        primary_counts[primary_relationship] += 1
                        primary_spillovers[primary_relationship] += neighbor_flip
                        secondary_counts[secondary_relationship] += 1
                        secondary_spillovers[secondary_relationship] += neighbor_flip
                        primary_relationship_raw_counts[primary_relationship] += 1
                        relationship_raw_counts[secondary_relationship] += 1
                        nonfocal_writer.writerow({
                            "image_id": image_id,
                            "stratum": stratum,
                            "sampling_weight": sampling_weight,
                            "focal_gt_id": focal_gt,
                            "neighbor_gt_id": neighbor_gt,
                            "direction": direction,
                            "stress": args.stress,
                            "shift_pixels": delta,
                            "focal_size_bin": focal_size_bin,
                            "focal_width": focal_width,
                            "focal_height": focal_height,
                            "focal_area": focal_area,
                            "neighbor_width": neighbor_width,
                            "neighbor_height": neighbor_height,
                            "neighbor_area": neighbor_area,
                            "local_distance_threshold": args.local_distance_threshold,
                            **relation,
                            "focal_active_base": base_active[focal_gt],
                            "focal_active_shift": shifted_active[focal_gt],
                            "focal_flip": focal_flip,
                            "neighbor_active_base": base_active[neighbor_gt],
                            "neighbor_active_shift": shifted_active[neighbor_gt],
                            "neighbor_flip": neighbor_flip,
                        })
                        nonfocal_row_count += 1
                        for group in ("all", focal_group):
                            add_metric(cluster, "pair_spillover", "overall", group, neighbor_flip, 1)
                            add_metric(cluster, "pair_spillover", primary_relationship, group, neighbor_flip, 1)
                            add_metric(cluster, "pair_spillover", secondary_relationship, group, neighbor_flip, 1)

                    neighbor_count = sum(primary_counts.values())
                    spillover_count = sum(primary_spillovers.values())
                    if neighbor_count:
                        for group in ("all", focal_group):
                            add_metric(cluster, "focal_any_spillover", "overall", group, int(spillover_count > 0), 1)
                            for relationship, counts, spillovers in grouped_relationships:
                                if counts[relationship]:
                                    add_metric(
                                        cluster,
                                        "focal_any_spillover",
                                        relationship,
                                        group,
                                        int(spillovers[relationship] > 0),
                                        1,
                                    )
                    focal_rows.append({
                        "image_id": image_id,
                        "stratum": stratum,
                        "sampling_weight": sampling_weight,
                        "focal_gt_id": focal_gt,
                        "direction": direction,
                        "stress": args.stress,
                        "shift_pixels": delta,
                        "focal_size_bin": focal_size_bin,
                        "focal_width": focal_width,
                        "focal_height": focal_height,
                        "focal_area": focal_area,
                        "focal_active_base": base_active[focal_gt],
                        "focal_active_shift": shifted_active[focal_gt],
                        "focal_flip": focal_flip,
                        "nonfocal_neighbor_count": neighbor_count,
                        "nonfocal_spillover_count": spillover_count,
                        "any_nonfocal_spillover": int(spillover_count > 0),
                        **{
                            f"{relationship}_{suffix}": value
                            for relationship, counts, spillovers in grouped_relationships
                            for suffix, value in (
                                ("neighbor_count", counts[relationship]),
                                ("spillover_count", spillovers[relationship]),
                                ("any_spillover", int(spillovers[relationship] > 0)),
                            )
                        },
                    })

    write_rows(args.output_dir / "focal_events.csv", focal_rows, FOCAL_FIELDS)
    clusters = [clusters_by_image[image_id] for image_id in sorted(clusters_by_image)]
    sufficient_rows = []
    for cluster in clusters:
        for key in metric_keys():
            numerator, denominator = cluster["metrics"].get(key, (0, 0))
            family, relationship, focal_group = key.split("|")
            sufficient_rows.append({
                "image_id": cluster["image_id"],
                "stratum": cluster["stratum"],
                "sampling_weight": cluster["sampling_weight"],
                "metric_family": family,
                "estimand_role": (
                    "primary" if relationship in PRIMARY_RELATIONSHIPS
                    else "secondary_descriptive" if relationship in SECONDARY_RELATIONSHIPS
                    else "overall"
                ),
                "relationship": relationship,
                "focal_group": focal_group,
                "raw_numerator": numerator,
                "raw_denominator": denominator,
            })
    write_rows(args.output_dir / "per_image_sufficient_statistics.csv", sufficient_rows)

    observed = aggregate_metrics(clusters)
    bootstrap_rows, intervals = bootstrap_metrics(clusters, args.bootstrap_replicates, args.bootstrap_seed)
    bootstrap_fields = ["replicate", *metric_keys(), PRIMARY_DIFFERENCE_KEY]
    write_rows(args.output_dir / "bootstrap_replicates.csv", bootstrap_rows, bootstrap_fields)
    estimates = nested_summary(observed, intervals)
    summary = {
        "status": "smoke_complete" if args.smoke else "complete",
        "selection_mode": selection_mode,
        "selected_images": len(selected_rows),
        "focal_direction_events": len(focal_rows),
        "nonfocal_pair_events": nonfocal_row_count,
        "primary_relationship_raw_pair_counts": dict(sorted(primary_relationship_raw_counts.items())),
        "relationship_raw_pair_counts": dict(sorted(relationship_raw_counts.items())),
        "bootstrap_replicates": args.bootstrap_replicates,
        "primary_estimands": primary_summary(observed, intervals),
        "secondary_descriptive_estimands": {
            "status": "secondary descriptive only; not a primary relationship",
            "local_distance_threshold": args.local_distance_threshold,
            "threshold_source": "CLI --local-distance-threshold",
            "affects_primary_estimands": False,
            "relationships": {
                relationship: {
                    family: estimates[family][relationship]
                    for family in ("pair_spillover", "focal_any_spillover")
                }
                for relationship in SECONDARY_RELATIONSHIPS
            },
        },
        "estimates": estimates,
        "elapsed_sec": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest.update({
        "status": summary["status"],
        "elapsed_sec": summary["elapsed_sec"],
        "focal_direction_events": len(focal_rows),
        "nonfocal_pair_events": nonfocal_row_count,
        "outputs": [
            "selected_images.csv",
            "focal_events.csv",
            "nonfocal_events.csv",
            "per_image_sufficient_statistics.csv",
            "bootstrap_replicates.csv",
            "summary.json",
        ],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
