"""Post hoc mechanism check for 8-16 px targets whose short side is ~8 px (validation sample).

For each GT, rebuild the float32 xyxy box exactly as the replay does (predict_row_structure.py),
then for the base box and every legal fixed 1 px shift evaluate the native eligibility pieces of
tal.py:504-517 on the stride-8 grid:
  - rule flag: xyxy2xywh width/height < 8 (strict) -> that side becomes 16 for the eligibility test
  - number of stride-8 anchor centres strictly inside the (possibly widened) box, per axis
Classify each GT: does any legal shift toggle the rule flag, and does the stride-8 eligible anchor
count change? Join with the replay's native fixed-pixel O2O fragility.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REP = Path(__file__).resolve().parent.parent / "R2_2_heldout_replication"
sys.path.insert(0, str(REP))
import predict_row_structure as prs  # noqa: E402  (sets up ultralytics_local paths)
from margin_o2o_replay import legal_shift  # noqa: E402

FORMAL = Path(r"G:\revision_20261120\heldout_replication\formal")
DATA = Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO\data.yaml")
SEL = REP / "selected_images_preregistered.csv"
EPS = 1e-9


def eligibility(box: torch.Tensor) -> tuple[bool, bool, int, int]:
    """Mirror tal.select_candidates_in_gts for one box on the stride-8 grid (800 px input)."""
    x1, y1, x2, y2 = box.tolist()
    b = box.clone().float()
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = b[2] - b[0], b[3] - b[1]
    rule_w, rule_h = bool(w < 8), bool(h < 8)
    w = torch.tensor(16.0) if rule_w else w
    h = torch.tensor(16.0) if rule_h else h
    lx, rx = float(cx - w / 2), float(cx + w / 2)
    ly, ry = float(cy - h / 2), float(cy + h / 2)
    centres = np.arange(4.0, 800.0, 8.0, dtype=np.float32)
    nx = int(((centres - lx > EPS) & (rx - centres > EPS)).sum())
    ny = int(((centres - ly > EPS) & (ry - centres > EPS)).sum())
    return rule_w, rule_h, nx, ny


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(DATA))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)
    sel = pd.read_csv(SEL)
    rows = []
    for idx, image_id in zip(sel["dataset_index"], sel["image_id"]):
        item = dataset[int(idx)]
        h_img, w_img = item["img"].shape[-2:]
        b = item["bboxes"].float().mul(torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32))
        dw, dh = b[:, 2] / 2, b[:, 3] / 2
        xyxy = torch.stack((b[:, 0] - dw, b[:, 1] - dh, b[:, 0] + dw, b[:, 1] + dh), 1)
        for gt, box in enumerate(xyxy):
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            if prs.size_bin(w * h) != "t_8_16":
                continue
            base = eligibility(box)
            toggles, count_change, n_legal = False, False, 0
            for ux, uy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                s = legal_shift(box, float(ux), float(uy), 800.0, 800.0)
                if s is None:
                    continue
                n_legal += 1
                e = eligibility(s)
                toggles |= (e[0], e[1]) != (base[0], base[1])
                count_change |= (e[2] * e[3]) != (base[2] * base[3])
            rows.append({"image_id": image_id, "gt_id": gt, "w": w, "h": h, "short": min(w, h),
                         "base_rule": base[0] or base[1], "rule_toggles": toggles,
                         "s8_count_changes": count_change, "base_s8_count": base[2] * base[3], "n_legal": n_legal})
    g = pd.DataFrame(rows)
    lock = pd.read_csv(FORMAL / "lock" / "per_gt.csv")
    lock = lock[lock["stress"].eq("fixed_1px")]
    gate = pd.read_csv(FORMAL / "base_gate.csv", usecols=["image_id", "gt_id", "o2o_active"])
    m = lock.merge(gate, on=["image_id", "gt_id"]).merge(g, on=["image_id", "gt_id"], how="left")
    m = m[m["o2o_active"].notna() & m["size_bin"].eq("t_8_16")]
    if m["short"].isna().any():
        raise RuntimeError("unmatched GTs")
    for col in ("base_rule", "rule_toggles", "s8_count_changes"):
        m[col] = m[col].astype(bool)
    m["band"] = np.select([m["short"] < 8 - 1e-3, (m["short"] - 8).abs() < 1e-3], ["short<8", "short=8"], default="short>8")
    out = {}
    for band, c in m.groupby("band"):
        w = c["sampling_weight"]
        def wr(mask, col="native_o2o_fragile"):
            cc = c[mask]
            return float(100 * (cc["sampling_weight"] * cc[col]).sum() / cc["sampling_weight"].sum()) if len(cc) else None
        out[band] = {
            "n": int(len(c)),
            "share_rule_toggles": float(100 * w[c["rule_toggles"]].sum() / w.sum()),
            "share_s8_count_changes": float(100 * w[c["s8_count_changes"]].sum() / w.sum()),
            "share_base_rule": float(100 * w[c["base_rule"]].sum() / w.sum()),
            "native_fragile_all": wr(c["n_legal"] >= 0),
            "native_fragile_if_toggle": wr(c["rule_toggles"]),
            "native_fragile_if_no_toggle": wr(~c["rule_toggles"]),
            "native_fragile_if_s8_change": wr(c["s8_count_changes"]),
            "native_fragile_if_no_s8_change": wr(~c["s8_count_changes"]),
            "locked_fragile_all": wr(c["n_legal"] >= 0, "eligibility_lock_o2o_fragile"),
            "short_side_examples": [float(x) for x in c["short"].head(5)],
        }
    m.to_csv(Path(__file__).with_name("boundary8_per_gt.csv"), index=False)
    Path(__file__).with_name("boundary8_mechanism.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
