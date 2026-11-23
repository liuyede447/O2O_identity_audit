"""Tie test for short-side = 8 px targets (validation sample, fixed 1 px, stride-8 grid).

For every 8-16 px GT, in the base box and each legal shift, record per axis the signed distance from
each box edge to the nearest stride-8 anchor centre (4 + 8k) after the replay's float32 rebuild.
An eligibility "tie" is an edge within 1e-3 px of an anchor centre, where the strict test
(distance > 1e-9) is decided by float32 rounding. Report the share of stride-8 count changes that
involve a tie, and whether tied anchors were counted inside or outside.
Also decomposes the validation tiny-group lock drop by short-side band.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import boundary8_mechanism as bm

TIE = 1e-3


def edge_info(box: torch.Tensor):
    x1, y1, x2, y2 = [float(v) for v in box.float()]
    out = {}
    for ax, (lo, hi) in (("x", (x1, x2)), ("y", (y1, y2))):
        d = []
        for e in (lo, hi):
            k = round((e - 4.0) / 8.0)
            d.append(e - (4.0 + 8.0 * k))
        out[ax] = d
    return out


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(bm.DATA))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)
    sel = pd.read_csv(bm.SEL)
    ev = []
    for idx, image_id in zip(sel["dataset_index"], sel["image_id"]):
        item = dataset[int(idx)]
        h_img, w_img = item["img"].shape[-2:]
        b = item["bboxes"].float().mul(torch.tensor([w_img, h_img, w_img, h_img], dtype=torch.float32))
        dw, dh = b[:, 2] / 2, b[:, 3] / 2
        xyxy = torch.stack((b[:, 0] - dw, b[:, 1] - dh, b[:, 0] + dw, b[:, 1] + dh), 1)
        for gt, box in enumerate(xyxy):
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            if bm.prs.size_bin(w * h) != "t_8_16":
                continue
            short = min(w, h)
            band = "short<8" if short < 8 - 1e-3 else ("short=8" if abs(short - 8) < 1e-3 else "short>8")
            base = bm.eligibility(box)
            bi = edge_info(box)
            for ux, uy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                s = bm.legal_shift(box, float(ux), float(uy), 800.0, 800.0)
                if s is None:
                    continue
                e = bm.eligibility(s)
                si = edge_info(s)
                ties_shift = [d for ax in ("x", "y") for d in si[ax] if abs(d) < TIE]
                ties_base = [d for ax in ("x", "y") for d in bi[ax] if abs(d) < TIE]
                ev.append({"image_id": image_id, "gt_id": gt, "band": band,
                           "count_change": (e[2] * e[3]) != (base[2] * base[3]),
                           "tie_any": bool(ties_shift or ties_base),
                           "tie_max_abs": max([abs(d) for d in ties_shift + ties_base], default=np.nan),
                           "tie_exact_zero": any(d == 0.0 for d in ties_shift + ties_base),
                           "base_count": base[2] * base[3], "shift_count": e[2] * e[3]})
    d = pd.DataFrame(ev)
    out = {}
    for band, c in d.groupby("band"):
        cc = c[c["count_change"]]
        out[band] = {"directions": int(len(c)), "count_change_directions": int(len(cc)),
                     "share_count_changes_with_tie": float(100 * cc["tie_any"].mean()) if len(cc) else None,
                     "share_all_directions_with_tie": float(100 * c["tie_any"].mean()),
                     "ties_exactly_zero_share": float(100 * c.loc[c["tie_any"], "tie_exact_zero"].mean()) if c["tie_any"].any() else None,
                     "max_abs_tie_distance": float(c["tie_max_abs"].max()) if c["tie_any"].any() else None,
                     "count_change_without_tie": int((c["count_change"] & ~c["tie_any"]).sum())}
    # Lock-drop decomposition for the tiny group (fixed 1 px, validation, base-defined support).
    g = pd.read_csv(Path(__file__).with_name("boundary8_per_gt.csv"))
    W = g["sampling_weight"].sum()
    dec = {}
    for band, c in g.groupby("band"):
        w = c["sampling_weight"]
        dec[band] = {"weight_share": float(100 * w.sum() / W),
                     "contribution_to_tiny_drop_pp": float(100 * (w * (c["native_o2o_fragile"] - c["eligibility_lock_o2o_fragile"])).sum() / W)}
    out["tiny_drop_decomposition"] = dec
    d.to_csv(Path(__file__).with_name("boundary8_tie_events.csv"), index=False)
    Path(__file__).with_name("boundary8_ties.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
