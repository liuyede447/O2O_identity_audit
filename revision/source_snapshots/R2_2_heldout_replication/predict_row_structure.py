"""Model-free prediction of which (image, GT, stress) rows the lock replay can produce.

Legal shifts depend only on GT geometry and image bounds, so the row structure of the
replication output (before the model-dependent base gate) is known from annotations alone.
Reports GTs and images that exist under only one stress contract: the failure class that
terminated the historical lockbox analyzer. Run on the discovery list (check against the real
lock output) and on the preregistered 600-image list. No model is loaded.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import torch

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from margin_o2o_replay import legal_shift  # noqa: E402
from run_reviewer_killer_controls import size_bin  # noqa: E402

KAPPA, IMGSZ = 0.0625, 800
CASES = {
    "discovery": (PR / "configs" / "aitod_v2_local_g.yaml", Path(r"G:\revision_20261120\eligibility_lock\seed1\selected_images.csv"),
                  Path(r"G:\revision_20261120\checkpoint_last\seed0\per_gt.csv")),
    "replication": (Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO\data.yaml"),
                    Path(__file__).resolve().parent / "selected_images_preregistered.csv", None),
}


def rows_for(data_yaml: Path, selected_csv: Path) -> pd.DataFrame:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": IMGSZ, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)
    sel = pd.read_csv(selected_csv)
    out = []
    for idx, image_id in zip(sel["dataset_index"], sel["image_id"]):
        item = dataset[int(idx)]  # same transform path as the replay dataloader
        if Path(item["im_file"]).stem != image_id:
            raise RuntimeError(f"index/image mismatch at {idx}")
        h_img, w_img = item["img"].shape[-2:]
        # Mirror v8DetectionLoss.preprocess: scale normalized xywh, then xywh2xyxy.
        b = item["bboxes"].float().mul(torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32))
        dw, dh = b[:, 2] / 2, b[:, 3] / 2
        xyxy = torch.stack((b[:, 0] - dw, b[:, 1] - dh, b[:, 0] + dw, b[:, 1] + dh), 1)
        for gt, box in enumerate(xyxy):
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            bucket = size_bin(w * h)
            if bucket not in {"t_8_16", "s_16_32"}:
                continue
            for stress in ("fixed_1px", "equivalent_side"):
                delta = 1.0 if stress == "fixed_1px" else KAPPA * math.sqrt(w * h)
                legal = sum(legal_shift(box, ux * delta, uy * delta, float(IMGSZ), float(IMGSZ)) is not None
                            for ux, uy in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                if legal:
                    out.append({"image_id": image_id, "gt_id": gt, "stress": stress, "size_bin": bucket, "legal": legal})
    return pd.DataFrame(out)


def report(name: str, df: pd.DataFrame) -> None:
    per_gt = df.groupby(["image_id", "gt_id"])["stress"].nunique()
    one = per_gt[per_gt.eq(1)].reset_index()[["image_id", "gt_id"]].merge(df, on=["image_id", "gt_id"])
    per_img = df.groupby("image_id")["stress"].nunique()
    print(f"[{name}] rows {len(df)} | by stress {df['stress'].value_counts().to_dict()} | images {df['image_id'].nunique()}")
    print(f"[{name}] GTs under one stress only: {len(one)} {one[['image_id', 'gt_id', 'stress', 'size_bin']].to_dict('records')[:10]}")
    print(f"[{name}] images under one stress only: {int(per_img.eq(1).sum())}")


def main() -> None:
    for name, (yaml, sel, real) in CASES.items():
        df = rows_for(yaml, sel)
        report(name, df)
        if real is not None:
            r = pd.read_csv(real)[["image_id", "gt_id", "stress"]]
            pred = df[["image_id", "gt_id", "stress"]]
            a = set(map(tuple, pred.astype(str).to_numpy())); b = set(map(tuple, r.astype(str).to_numpy()))
            print(f"[{name}] predicted vs real lock rows: predicted {len(a)}, real {len(b)}, only predicted {len(a - b)}, only real {len(b - a)}")


if __name__ == "__main__":
    main()
