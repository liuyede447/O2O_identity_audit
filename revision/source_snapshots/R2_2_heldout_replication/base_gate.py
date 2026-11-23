"""Base O2O identity per GT for one checkpoint on one selection (parameterised base_gate_all.py).

Same code path as run_eligibility_lock_counterfactual.py (trace_o2o + active_candidate_and_margin).
Also records the Table-2 support flag (O2O margin and O2M margin both defined at base), used only for
the secondary descriptive support.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from margin_o2o_replay import active_candidate_and_margin, trace_o2o  # noqa: E402
from run_reviewer_killer_controls import SelectedDataset, ranked_candidate_and_margin  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--selected-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != a.expected_sha256:
        raise RuntimeError("checkpoint hash mismatch")
    if a.out.exists():
        raise RuntimeError(f"{a.out} exists")

    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0")
    data = check_det_dataset(str(a.data))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    model = YOLO(str(a.checkpoint)).model.to(device).eval()
    o2m = v8DetectionLoss(model, tal_topk=10)
    o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    sel = pd.read_csv(a.selected_images)
    for idx, image_id in zip(sel["dataset_index"], sel["image_id"]):
        if Path(dataset.labels[int(idx)]["im_file"]).stem != image_id:
            raise RuntimeError("selection does not match dataset index")
    loader = build_dataloader(SelectedDataset(dataset, sel["dataset_index"].astype(int).tolist()), batch=1, workers=0,
                              shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            batch["img"] = batch["img"].float() / 255
            raw = model(batch["img"])[1]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            h, w = batch["img"].shape[-2:]
            targets = o2o.preprocess(targets, 1, torch.tensor([w, h, w, h], device=device, dtype=batch["img"].dtype))
            labels, boxes = targets.split((1, 4), 2)
            mask = boxes.sum(2, keepdim=True).gt_(0)
            if not mask.any():
                continue
            base_o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
            base_m = trace_o2o(o2m, raw["one2many"], labels, boxes, mask)
            image_id = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0, :, 0])[0].tolist():
                active0, _, margin = active_candidate_and_margin(base_o, 0, gt)
                _, _, m_margin = ranked_candidate_and_margin(base_m, gt)
                rows.append({"image_id": image_id, "gt_id": gt, "o2o_active": active0,
                             "common_valid_margin": int(margin is not None and m_margin is not None)})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} GT rows to {a.out}")


if __name__ == "__main__":
    main()
