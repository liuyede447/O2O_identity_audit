"""Per-direction sensitivity check on the 5,701 common base-nonempty GTs (RFLA and ATSS).

Uses the exact stratified image-cluster bootstrap ordering of assigner_audit.py
(seed 20261123, 5,000 replicates; strata sorted alphabetically, image IDs in
appearance order within each stratum via pandas groupby), pre-aggregating
per-image numerators and denominators for fast exact evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

AUD = Path(r"G:\revision_20261120\assigner_audit_v1")
FOCAL = Path(r"E:\two_paper\beifen\publication_reproducible\results\reviewer_controls_20260829\formal_s0e300_k00625\per_gt.csv")
FIXED_GT = Path(r"E:\two_paper\beifen\修复\post_submission_audit_20260908\inputs\fixed_branch_per_gt.csv")
KEYS = ["image_id", "gt_id"]
COLS = ["rfla_set_change", "rfla_top1_change", "atss_set_change", "atss_top1_change"]
SEED = 20261123
REPS = 5000


def summarise_exact_fast(df: pd.DataFrame, cols: list[str], reps: int, seed: int) -> dict:
    df = df.reset_index(drop=True)
    image_strata = df[["image_id", "stratum"]].drop_duplicates()
    by_stratum = {key: values["image_id"].tolist() for key, values in image_strata.groupby("stratum")}
    all_images = [img for imgs in by_stratum.values() for img in imgs]
    img_to_idx = {img: i for i, img in enumerate(all_images)}
    n_img = len(all_images)

    wt = np.zeros(n_img, dtype=np.float64)
    ws = np.zeros(n_img, dtype=np.float64)
    nt = {c: np.zeros(n_img, dtype=np.float64) for c in cols}
    ns = {c: np.zeros(n_img, dtype=np.float64) for c in cols}

    img_idx = df["image_id"].map(img_to_idx).to_numpy()
    w = df["sampling_weight"].to_numpy(dtype=np.float64)
    is_t = (df["size_bin"] == "t_8_16").to_numpy()
    is_s = (df["size_bin"] == "s_16_32").to_numpy()

    np.add.at(wt, img_idx[is_t], w[is_t])
    np.add.at(ws, img_idx[is_s], w[is_s])
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64) * w
        np.add.at(nt[c], img_idx[is_t], v[is_t])
        np.add.at(ns[c], img_idx[is_s], v[is_s])

    rng = np.random.default_rng(seed)
    boot_indices = []
    offset = 0
    for imgs in by_stratum.values():
        m = len(imgs)
        chosen = rng.choice(np.arange(offset, offset + m), size=(reps, m), replace=True)
        boot_indices.append(chosen)
        offset += m
    boot = np.hstack(boot_indices)

    wt_b = wt[boot].sum(axis=1)
    ws_b = ws[boot].sum(axis=1)
    point, ci_95 = {}, {}
    for c in cols:
        p8 = float(nt[c].sum() / wt.sum())
        p16 = float(ns[c].sum() / ws.sum())
        gap = p8 - p16
        b8 = nt[c][boot].sum(axis=1) / wt_b
        b16 = ns[c][boot].sum(axis=1) / ws_b
        bgap = b8 - b16
        point[f"{c}|8_16"] = p8
        point[f"{c}|16_32"] = p16
        point[f"{c}|gap"] = gap
        ci_95[f"{c}|8_16"] = [float(np.quantile(b8, 0.025)), float(np.quantile(b8, 0.975))]
        ci_95[f"{c}|16_32"] = [float(np.quantile(b16, 0.025)), float(np.quantile(b16, 0.975))]
        ci_95[f"{c}|gap"] = [float(np.quantile(bgap, 0.025)), float(np.quantile(bgap, 0.975))]
    return {
        "n_gt": int(df[KEYS].drop_duplicates().shape[0]),
        "n_directions": int(len(df)),
        "n_images": int(df["image_id"].nunique()),
        "point": point,
        "ci_95": ci_95,
    }


def main() -> None:
    d_all = pd.read_csv(AUD / "per_direction.csv")
    yolo_src = {"fixed_1px": FIXED_GT, "equivalent_side": FOCAL}
    out = {"seed": SEED, "replicates": REPS, "sampling_order": "assigner_audit.stratified_cluster_samples (strata sorted alphabetically, images in appearance order)"}
    for stress in ("fixed_1px", "equivalent_side"):
        obj = pd.read_csv(AUD / f"per_gt_{stress}.csv")
        y = pd.read_csv(yolo_src[stress], usecols=KEYS + ["common_valid_margin"])
        y = y.loc[y["common_valid_margin"].eq(1), KEYS]
        j_gt = obj.loc[obj["rfla_base_size"].gt(0) & obj["atss_base_size"].gt(0), KEYS].merge(y, on=KEYS)
        dd = d_all.loc[d_all["stress"].eq(stress)].merge(j_gt, on=KEYS)
        res = summarise_exact_fast(dd, COLS, REPS, SEED)
        out[stress] = res
        print(f"=== {stress} (n_gt={res['n_gt']}, n_dirs={res['n_directions']}, n_img={res['n_images']}) ===")
        for c in COLS:
            p8 = 100 * res["point"][f"{c}|8_16"]
            p16 = 100 * res["point"][f"{c}|16_32"]
            g = 100 * res["point"][f"{c}|gap"]
            lo, hi = [100 * x for x in res["ci_95"][f"{c}|gap"]]
            print(f"  {c:18s}: 8-16={p8:5.2f}%, 16-32={p16:5.2f}%, gap={g:+6.2f} [{lo:.2f}, {hi:.2f}]")
    out_path = Path(__file__).with_name("per_direction_sensitivity.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
