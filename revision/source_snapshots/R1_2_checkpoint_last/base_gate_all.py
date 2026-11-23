"""Base O2O defined-ness for every focal GT in the 300 discovery images, per checkpoint.

Base O2O identity is stress-independent. It is recomputed with the same code path as
run_eligibility_lock_counterfactual.py (trace_o2o + active_candidate_and_margin), for both the
released epoch-280 checkpoints and the final-epoch last.pt checkpoints of seeds 0-3.
The epoch-280 rows serve as a gate: they must reproduce the gates used in
../R1_2_eligibility_lock_seeds/aggregate_lock_seeds.py.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
B = Path(r"E:\two_paper\beifen\archive_backup")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from margin_o2o_replay import active_candidate_and_margin, trace_o2o  # noqa: E402
from run_reviewer_killer_controls import SelectedDataset  # noqa: E402

CK = PR / "release_candidates" / "O2O_assignment_state_audit_v3.0.0" / "release_assets"
SELECTED = Path(r"G:\revision_20261120\eligibility_lock\seed1\selected_images.csv")
CHECKPOINTS = {
    (0, "e280"): (CK / "yolo26s_aitodv2_seed0_best_epoch280.pt", "4b57787f7351"),
    (1, "e280"): (CK / "yolo26s_aitodv2_seed1_best_epoch280.pt", "d9432d90e9b7"),
    (2, "e280"): (CK / "yolo26s_aitodv2_seed2_best_epoch280.pt", "d83f86f57039"),
    (3, "e280"): (CK / "yolo26s_aitodv2_seed3_best_epoch280.pt", "28fee07d38f6"),
    (0, "last"): (B / "artifacts/analysis_runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s0/weights/last.pt", "46ff9c348e4e"),
    (1, "last"): (B / "server_A/runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s1/weights/last.pt", "1c689b3e6ae7"),
    (2, "last"): (B / "server_A/runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s2/weights/last.pt", "b47855abfb33"),
    (3, "last"): (B / "artifacts/consolidated_training_20260801/active_main/server_B/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s3/weights/last.pt", "326a5bc8ca16"),
}


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss

    device = torch.device("cuda:0")
    data = check_det_dataset(str(PR / "configs" / "aitod_v2_local_g.yaml"))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    idx = pd.read_csv(SELECTED)["dataset_index"].tolist()
    rows = []
    for (seed, role), (path, sha_prefix) in CHECKPOINTS.items():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if not sha.startswith(sha_prefix):
            raise RuntimeError(f"seed {seed} {role}: sha {sha[:12]} != {sha_prefix}")
        model = YOLO(str(path)).model.to(device).eval()
        o2o = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
        dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
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
                    active0, _, _ = active_candidate_and_margin(base_o, 0, gt)
                    rows.append({"seed": seed, "checkpoint": role, "checkpoint_sha256": sha,
                                 "image_id": image_id, "gt_id": gt, "o2o_active": active0})
        print(f"seed {seed} {role}: {sum(r['seed'] == seed and r['checkpoint'] == role for r in rows)} GT rows", flush=True)
        del model, o2o
        torch.cuda.empty_cache()
    out = Path(__file__).resolve().parent / "base_gate_all.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
