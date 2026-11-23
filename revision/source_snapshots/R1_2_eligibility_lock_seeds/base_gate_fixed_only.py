"""Base O2O defined-ness for GTs that appear only in the fixed_1px lock rows.

The eq-area per_gt.csv used as the base gate omits GTs whose equivalent-area shift is
illegal. Base O2O identity is stress-independent, so it is recomputed here with the
same code path as run_eligibility_lock_counterfactual.py (trace_o2o + active_candidate_and_margin).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from margin_o2o_replay import active_candidate_and_margin, trace_o2o  # noqa: E402
from run_reviewer_killer_controls import SelectedDataset  # noqa: E402

CK = PR / "release_candidates" / "O2O_assignment_state_audit_v3.0.0" / "release_assets"
LOCK_OUT = Path(r"G:\revision_20261120\eligibility_lock")
TARGETS = {("362__0_1200", 5), ("9999982_00000_d_0000175__0_250", 16)}


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0")
    data = check_det_dataset(str(PR / "configs" / "aitod_v2_local_g.yaml"))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    rows = []
    for seed in (0, 1, 2, 3):
        model = YOLO(str(CK / f"yolo26s_aitodv2_seed{seed}_best_epoch280.pt")).model.to(device).eval()
        o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
        dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
        sel = pd.read_csv(LOCK_OUT / f"seed{max(seed, 1)}" / "selected_images.csv")
        idx = sel.loc[sel["image_id"].isin({t[0] for t in TARGETS}), "dataset_index"].tolist()
        loader = build_dataloader(SelectedDataset(dataset, idx), batch=1, workers=0, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
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
                base_o = trace_o2o(o2o, raw["one2one"], labels, boxes, mask)
                image_id = Path(batch["im_file"][0]).stem
                for gt in torch.where(mask[0, :, 0])[0].tolist():
                    if (image_id, gt) in TARGETS:
                        active0, _, _ = active_candidate_and_margin(base_o, 0, gt)
                        rows.append({"seed": seed, "image_id": image_id, "gt_id": gt, "o2o_active": active0})
    out = Path(__file__).resolve().parent / "base_gate_fixed_only.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
