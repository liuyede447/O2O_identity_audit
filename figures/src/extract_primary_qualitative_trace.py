"""Extract a figure-only qualitative trace from the primary frozen checkpoint.

Cases are selected from the frozen fixed-replay table, then replayed once from
the primary checkpoint. Every candidate identity is checked against that table.
No aggregate estimate or scientific result is computed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
sys.path.insert(0, str(ROOT / "scripts"))

from margin_o2o_replay import active_candidate_and_margin, legal_shift, trace_o2o  # noqa: E402
from run_reviewer_killer_controls import SelectedDataset  # noqa: E402


CHECKPOINT = ROOT / "release_candidates" / "O2O_identity_e300_checkpoints_20260901" / "checkpoints" / "yolo26s_aitodv2_e300_seed0_best.pt"
CHECKPOINT_SHA = "4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa"
DATA = ROOT / "configs" / "aitod_v2_local_g.yaml"
SELECTION = ROOT / "evidence" / "artifacts" / "results" / "five_experiment_upgrade_20260830" / "anatomy_fixed_1px_v2" / "selected_images.csv"
FIXED_ROWS = ROOT / "evidence" / "artifacts" / "results" / "measurement_validation_20260831" / "fixed_branch_replay_v1" / "per_direction.csv"
FIXED_ROWS_SHA = "930a50dadb0abdfdcede0de344cb93ed2f68c0c5a4c1f183a2f70ddde8be87f8"
IMAGE_ID = "0000170_00401_d_0000001__160_0"
OUT = ROOT / "figures" / "source_data" / "figs1_primary_trace"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_payload(state, gt: int, candidate: int | None) -> dict | None:
    if candidate is None:
        return None
    stride = float(state.stride[candidate].item())
    decoded = (state.decoded[0, candidate] * stride).detach().cpu().tolist()
    anchor = (state.anchors[candidate] * stride).detach().cpu().tolist()
    return {
        "candidate": int(candidate),
        "alignment_score": float(state.align[0, gt, candidate].item()),
        "stride": int(round(stride)),
        "anchor_xy": [float(value) for value in anchor],
        "decoded_xyxy": [float(value) for value in decoded],
    }


def select_cases(rows: pd.DataFrame) -> list[dict]:
    grouped = []
    for gt_id, group in rows.groupby("gt_id", sort=True):
        base_active = group["o2o_active"].dropna().iloc[0] if group["o2o_active"].notna().any() else None
        base_runner = group["o2o_runner"].dropna().iloc[0] if group["o2o_runner"].notna().any() else None
        margin = group["o2o_margin"].dropna().iloc[0] if group["o2o_margin"].notna().any() else None
        grouped.append({"gt_id": int(gt_id), "group": group, "active": base_active, "runner": base_runner, "margin": margin})
    stable = max(
        (item for item in grouped if item["active"] is not None and item["runner"] is not None and not item["group"]["o2o_flip"].astype(bool).any() and item["margin"] is not None and item["margin"] >= 0),
        key=lambda item: item["margin"],
    )
    fragile = min(
        (item for item in grouped if item["active"] is not None and item["runner"] is not None and item["group"]["o2o_flip"].astype(bool).any() and item["margin"] is not None and item["margin"] >= 0),
        key=lambda item: item["margin"],
    )
    undefined = next(item for item in grouped if item["active"] is not None and item["runner"] is None)
    result = []
    for label, item in (("Stable", stable), ("Fragile", fragile), ("Undefined", undefined)):
        if label == "Stable":
            row = item["group"].sort_values("direction").iloc[0]
        elif label == "Fragile":
            row = item["group"][item["group"]["o2o_flip"].astype(bool)].sort_values("direction").iloc[0]
        else:
            missing = item["group"][item["group"]["o2o_active_shift"].isna()]
            row = (missing if not missing.empty else item["group"]).sort_values("direction").iloc[0]
        result.append({"case": label, "gt_id": item["gt_id"], "direction": str(row["direction"]), "expected": row.to_dict()})
    return result


def main() -> None:
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("primary checkpoint SHA mismatch")
    if sha256(FIXED_ROWS) != FIXED_ROWS_SHA:
        raise RuntimeError(f"fixed replay table SHA mismatch: {sha256(FIXED_ROWS)}")
    frozen = pd.read_csv(FIXED_ROWS)
    frozen = frozen[frozen["image_id"].eq(IMAGE_ID)].copy()
    cases = select_cases(frozen)
    selection = pd.read_csv(SELECTION)
    selection_row = selection[selection.image_id.eq(IMAGE_ID)]
    if len(selection_row) != 1:
        raise RuntimeError("qualitative image is not uniquely present in frozen selection")
    dataset_index = int(selection_row.iloc[0]["dataset_index"])

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(CHECKPOINT)).model.to(device).eval()
    head = model.model[-1]
    if head.__class__.__name__ != "Detect" or not model.end2end or not head.end2end or int(head.reg_max) != 1:
        raise RuntimeError("checkpoint does not satisfy the primary YOLO26 contract")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    data = check_det_dataset(str(DATA))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    if Path(dataset.labels[dataset_index]["im_file"]).stem != IMAGE_ID:
        raise RuntimeError("dataset index/image identity mismatch")
    loader = build_dataloader(SelectedDataset(dataset, [dataset_index]), batch=1, workers=0, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    batch = next(iter(loader))
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    source_pixels = batch["img"][0].permute(1, 2, 0).cpu().numpy()
    source_image = Image.fromarray(source_pixels)
    batch["img"] = batch["img"].float() / 255
    with torch.no_grad():
        output = model(batch["img"])
        raw = output[1]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        image_h, image_w = batch["img"].shape[-2:]
        targets = o2m.preprocess(targets, 1, torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=batch["img"].dtype))
        labels, boxes = targets.split((1, 4), 2)
        mask = boxes.sum(2, keepdim=True).gt_(0)
        base = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)

        records = []
        direction_delta = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}
        for spec in cases:
            gt = spec["gt_id"]
            dx, dy = direction_delta[spec["direction"]]
            shifted_box = legal_shift(boxes[0, gt], dx, dy, float(image_w), float(image_h))
            if shifted_box is None:
                raise RuntimeError("selected qualitative shift is illegal")
            altered = boxes.clone()
            altered[0, gt] = shifted_box
            shifted = trace_o2o(o2o, raw["one2one"], labels, altered, mask)
            active, runner, margin = active_candidate_and_margin(base, 0, gt)
            active_s, runner_s, margin_s = active_candidate_and_margin(shifted, 0, gt)
            expected = spec["expected"]
            expected_active = None if pd.isna(expected["o2o_active"]) else int(expected["o2o_active"])
            expected_runner = None if pd.isna(expected["o2o_runner"]) else int(expected["o2o_runner"])
            expected_shift = None if pd.isna(expected["o2o_active_shift"]) else int(expected["o2o_active_shift"])
            if (active, runner, active_s) != (expected_active, expected_runner, expected_shift):
                raise RuntimeError(f"frozen identity mismatch for GT {gt}: {(active, runner, active_s)} != {(expected_active, expected_runner, expected_shift)}")
            records.append(
                {
                    "case": spec["case"],
                    "gt_id": gt,
                    "class_id": int(labels[0, gt, 0].item()),
                    "direction": spec["direction"],
                    "gt_xyxy": [float(value) for value in boxes[0, gt].detach().cpu().tolist()],
                    "shift_gt_xyxy": [float(value) for value in shifted_box.detach().cpu().tolist()],
                    "base": {"active": candidate_payload(base, gt, active), "runner": candidate_payload(base, gt, runner), "margin": margin},
                    "shift": {"active": candidate_payload(shifted, gt, active_s), "runner": candidate_payload(shifted, gt, runner_s), "margin": margin_s},
                    "frozen_row": {"active": expected_active, "runner": expected_runner, "shift_active": expected_shift, "o2o_flip": int(expected["o2o_flip"])},
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    image_path = OUT / f"{IMAGE_ID}.png"
    source_image.save(image_path)
    payload = {
        "status": "PASS",
        "artifact_role": "figure_only_primary_checkpoint_trace_not_scientific_estimate",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "fixed_replay_source": str(FIXED_ROWS.relative_to(ROOT)).replace("\\", "/"),
        "fixed_replay_source_sha256": FIXED_ROWS_SHA,
        "selection_source": str(SELECTION.relative_to(ROOT)).replace("\\", "/"),
        "selection_source_sha256": sha256(SELECTION),
        "image_id": IMAGE_ID,
        "dataset_index": dataset_index,
        "input_hw": [int(image_h), int(image_w)],
        "source_image": str(image_path.relative_to(ROOT)).replace("\\", "/"),
        "source_image_sha256": sha256(image_path),
        "case_selection": "within one frozen selected image: maximum nonnegative-margin stable case, minimum nonnegative-margin fragile case, first base-active runner-undefined case",
        "records": records,
        "scientific_results_recomputed": False,
    }
    trace_path = OUT / "primary_qualitative_trace.json"
    trace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    validation = {"status": "PASS", "checkpoint_sha256": CHECKPOINT_SHA, "records": len(records), "identity_checks": len(records) * 3, "trace_sha256": sha256(trace_path), "image_sha256": sha256(image_path)}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
