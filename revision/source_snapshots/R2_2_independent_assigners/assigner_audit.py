"""R2.2: GT-centre stress audit of two prediction-independent native assigners.

  * RFLA  (HieAssigner + RFLA_FCOSHead receptive fields; config aitodv2_fcos_r50_rfla_kld_1x.py)
  * ATSS  (ATSSAssigner topk=9 + AnchorGenerator of the standard MMDetection 2.13 ATSS-R50-FPN config)

Both assigners read only GT boxes and fixed priors (receptive-field boxes / anchors); neither
reads classification scores, box predictions or proposals. The audit therefore needs no checkpoint:
the assignment computed here is exactly the assignment these detectors receive at any training step.

The original MMDetection-2.13/RFLA source files are executed unmodified except for import lines,
which are rewritten to small local stubs (registry, AssignResult base class). SHA-256 of every
original file is recorded in the manifest.

Deviation from the original training code, documented on purpose: RFLA's anchor_rescale() writes into
the shared receptive-field tensor, so in a multi-image batch later images see shrunken fields. We pass a
fresh copy to every assign() call, i.e. the first-image (batch-size-1) behaviour.

Same 300 images, same focal GTs (8-16 / 16-32 px bins), same four cardinal directions, same legal-shift
rule (never clip), same fixed-1px and equivalent-area (kappa=0.0625) stress contracts, same inverse-
inclusion weights and stratified image-cluster bootstrap as the YOLO audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIRD = Path(r"G:\revision_20261120\third_party\mmdet-rfla")
PR = Path(r"E:\two_paper\beifen\publication_reproducible")
DATA = Path(r"G:\data\processed\AI-TOD-v2-YOLO\val")
SEL = PR / "results/five_experiment_upgrade_20260830/anatomy_fixed_1px_v2/selected_images.csv"
FOCAL = PR / "results/reviewer_controls_20260829/formal_s0e300_k00625/per_gt.csv"
IMG = 800.0
STRIDES = [8, 16, 32, 64, 128]
KAPPA = 0.0625
DIRECTIONS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))
SRC = {
    "assign_result": "mmdet/core/bbox/assigners/assign_result.py",
    "base_assigner": "mmdet/core/bbox/assigners/base_assigner.py",
    "hierarchical_assigner": "mmdet/core/bbox/assigners/hierarchical_assigner.py",
    "atss_assigner": "mmdet/core/bbox/assigners/atss_assigner.py",
    "metric_calculator": "mmdet/core/bbox/iou_calculators/metric_calculator.py",
    "iou2d_calculator": "mmdet/core/bbox/iou_calculators/iou2d_calculator.py",
    "anchor_generator": "mmdet/core/anchor/anchor_generator.py",
    "rfla_fcos_head": "mmdet/models/dense_heads/rfla_fcos_head.py",
    "rfla_config": "configs/rfla/aitodv2_fcos_r50_rfla_kld_1x.py",
}


# ----------------------------------------------------------------------------- loading original code
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Registry:
    def register_module(self, *a, **k):
        return lambda cls: cls


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _exec_original(key: str, name: str, replacements: list[tuple[str, str]]) -> types.ModuleType:
    text = (THIRD / SRC[key]).read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"{key}: expected import line not found: {old!r}")
        text = text.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = str(THIRD / SRC[key])
    sys.modules[name] = mod
    exec(compile(text, str(THIRD / SRC[key]), "exec"), mod.__dict__)
    return mod


def load_original():
    reg = _Registry()
    _module("audit_stub_registry", BBOX_ASSIGNERS=reg, IOU_CALCULATORS=reg, PRIOR_GENERATORS=reg)
    _module("audit_stub_mixins", util_mixins=types.SimpleNamespace(NiceRepr=object))
    _module("mmcv", is_tuple_of=lambda x, t: isinstance(x, tuple) and all(isinstance(i, t) for i in x))
    R = "from audit_stub_registry import "
    ar = _exec_original("assign_result", "audit_assign_result",
                        [("from mmdet.utils import util_mixins", "from audit_stub_mixins import util_mixins")])
    ba = _exec_original("base_assigner", "audit_base_assigner", [])
    mc = _exec_original("metric_calculator", "audit_metric_calculator", [("from .builder import ", R)])
    io = _exec_original("iou2d_calculator", "audit_iou2d_calculator", [("from .builder import ", R)])
    calculators = {"BboxDistanceMetric": mc.BboxDistanceMetric, "BboxOverlaps2D": io.BboxOverlaps2D}
    _module("audit_iou_builder", build_iou_calculator=lambda cfg: calculators[cfg["type"]]())
    common = [("from ..builder import ", R),
              ("from ..iou_calculators import build_iou_calculator", "from audit_iou_builder import build_iou_calculator"),
              ("from .assign_result import AssignResult", "from audit_assign_result import AssignResult"),
              ("from .base_assigner import BaseAssigner", "from audit_base_assigner import BaseAssigner")]
    hie = _exec_original("hierarchical_assigner", "audit_hie",
                         common + [("from turtle import back\n", "\n")])  # unused GUI import
    atss = _exec_original("atss_assigner", "audit_atss", common)
    ag = _exec_original("anchor_generator", "audit_anchor_generator", [("from .builder import ", R)])
    return hie.HieAssigner, atss.ATSSAssigner, ag.AnchorGenerator, mc.BboxDistanceMetric, io.BboxOverlaps2D


def rfla_gen_trf():
    """Verbatim body of RFLA_FCOSHead.gen_trf (rfla_fcos_head.py, lines 653-685)."""
    j_i = [1]
    for i in range(7):
        j = j_i[i] * 2
        j_i.append(j)
    r0 = 1
    r1 = r0 + (7 - 1) * j_i[0]
    r2 = r1 + (3 - 1) * j_i[1]
    trf_p2 = r2 + (3 - 1) * j_i[2] * 3
    r3 = trf_p2 + (3 - 1) * j_i[2]
    trf_p3 = r3 + (3 - 1) * j_i[3] * 3
    r4 = trf_p3 + (3 - 1) * j_i[3]
    trf_p4 = r4 + (3 - 1) * j_i[4] * 5
    r5 = trf_p4 + (3 - 1) * j_i[4]
    trf_p5 = r5 + (3 - 1) * j_i[5] * 2
    trf_p6 = trf_p5 + (3 - 1) * j_i[6]
    trf_p7 = trf_p6 + (3 - 1) * j_i[7]
    return [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]


def featmap_sizes() -> list[tuple[int, int]]:
    # ResNet-50-FPN on an 800x800 input padded to /32: P3..P5 = 800/s; P6, P7 by stride-2 3x3 pad-1 convs.
    p3, p4, p5 = 100, 50, 25
    p6 = (p5 + 2 - 3) // 2 + 1
    p7 = (p6 + 2 - 3) // 2 + 1
    return [(p3, p3), (p4, p4), (p5, p5), (p6, p6), (p7, p7)]


def rfla_priors(device, fraction=0.5, fpn_layer="p3"):
    """RFLA_FCOSHead._get_points_single + get_targets receptive-field construction (lines 477-541)."""
    trfs = rfla_gen_trf()
    fields = []
    for num, ((h, w), stride) in enumerate(zip(featmap_sizes(), STRIDES)):
        x = torch.arange(w, device=device).float()
        y = torch.arange(h, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack((xx.reshape(-1) * stride, yy.reshape(-1) * stride), dim=-1) + stride // 2
        rfnum = num + 1 if fpn_layer == "p3" else num
        rf = trfs[min(rfnum, 5)] * fraction
        fields.append(torch.stack((points[:, 0] - rf / 2, points[:, 1] - rf / 2,
                                   points[:, 0] + rf / 2, points[:, 1] + rf / 2), dim=1))
    return torch.cat(fields, 0)


def atss_priors(AnchorGenerator, device):
    gen = AnchorGenerator(strides=STRIDES, ratios=[1.0], octave_base_scale=8, scales_per_octave=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        levels = gen.grid_anchors(featmap_sizes(), device=device)
    return torch.cat(levels, 0), [lvl.shape[0] for lvl in levels]


# ----------------------------------------------------------------------------- labels
def load_boxes(image_id: str) -> torch.Tensor:
    """Ultralytics label semantics: float32 xywhn * 800, exact-duplicate rows removed via np.unique."""
    lb = np.loadtxt(DATA / "labels" / f"{image_id}.txt", ndmin=2).astype(np.float32)
    _, idx = np.unique(lb, axis=0, return_index=True)
    if len(idx) < len(lb):
        lb = lb[idx]
    xywh = lb[:, 1:5] * np.float32(IMG)
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    return torch.from_numpy(xyxy.astype(np.float32))


def legal_shift(box, dx, dy):
    s = box.clone()
    s[[0, 2]] += dx
    s[[1, 3]] += dy
    if s[0] < 0 or s[1] < 0 or s[2] > IMG or s[3] > IMG:
        return None
    return s


# ----------------------------------------------------------------------------- per-assigner state
def rfla_state(assigner, metric, priors, gts, focal):
    """Final positive set of `focal`, its stage-1 (top-3, KL) set and Top-1 by KL similarity."""
    p = priors.clone()                               # fresh copy: see module docstring
    res = assigner.assign(p, gts, gt_bboxes_ignore=None)
    final = torch.nonzero(res.gt_inds == focal + 1).flatten()
    kl = metric(gts[focal:focal + 1], priors, mode="kl")[0]
    stage1 = assigner.assign_wrt_ranking(metric(gts, priors, mode="kl"), assigner.topk[0], None)
    s1 = torch.nonzero(stage1 == focal + 1).flatten()
    top1 = int(final[kl[final].argmax()]) if final.numel() else -1
    return {"final": set(final.tolist()), "stage1": set(s1.tolist()), "top1": top1}


def atss_state(assigner, iou_calc, priors, nlev, gts, focal):
    """Final positive set, plus the ATSS stage sets of `focal`: distance candidates, IoU >= mean+std,
    centre-inside-GT (eligibility gate), final after multi-GT conflict. Top-1 by IoU."""
    res = assigner.assign(priors, nlev, gts, gt_bboxes_ignore=None)
    final = torch.nonzero(res.gt_inds == focal + 1).flatten()
    g = gts[focal]
    iou = iou_calc(priors, g[None])[:, 0]
    cxy = (priors[:, :2] + priors[:, 2:]) / 2
    gc = (g[:2] + g[2:]) / 2
    dist = (cxy - gc).pow(2).sum(-1).sqrt()
    cand, start = [], 0
    for n in nlev:
        k = min(assigner.topk, n)
        cand.append(dist[start:start + n].topk(k, largest=False).indices + start)
        start += n
    cand = torch.cat(cand)
    thr = iou[cand].mean() + iou[cand].std()
    passed = cand[iou[cand] >= thr]
    c = cxy[passed]
    inside = passed[torch.stack([c[:, 0] - g[0], c[:, 1] - g[1], g[2] - c[:, 0], g[3] - c[:, 1]], 1).min(1).values > 0.01]
    top1 = int(final[iou[final].argmax()]) if final.numel() else -1
    return {"final": set(final.tolist()), "candidates": set(cand.tolist()), "threshold": set(passed.tolist()),
            "inside": set(inside.tolist()), "top1": top1}


STAGES = {"rfla": ["stage1", "final"], "atss": ["candidates", "threshold", "inside", "final"]}


# ----------------------------------------------------------------------------- statistics
def stratified_cluster_samples(df, reps, seed):
    """Verbatim from scripts/run_reviewer_killer_controls.py."""
    image_strata = df[["image_id", "stratum"]].drop_duplicates()
    by_stratum = {key: values["image_id"].tolist() for key, values in image_strata.groupby("stratum")}
    row_groups = {key: values.index.to_numpy() for key, values in df.groupby("image_id", sort=False)}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        indices = []
        for images in by_stratum.values():
            chosen = rng.choice(images, size=len(images), replace=True)
            indices.extend(row_groups[image] for image in chosen)
        yield np.concatenate(indices)


def wrate(df, col, b):
    part = df[df["size_bin"].eq(b)]
    w = part["sampling_weight"].to_numpy(float)
    return float(np.sum(part[col].to_numpy(float) * w) / np.sum(w))


def contrasts(df, cols):
    out = {}
    for c in cols:
        t, s = wrate(df, c, "t_8_16"), wrate(df, c, "s_16_32")
        out[f"{c}|8_16"], out[f"{c}|16_32"], out[f"{c}|gap"] = t, s, t - s
    return out


def summarise(obj: pd.DataFrame, cols, reps, seed):
    obj = obj.reset_index(drop=True)
    point = contrasts(obj, cols)
    boots = {k: [] for k in point}
    for idx in stratified_cluster_samples(obj, reps, seed):
        st = contrasts(obj.iloc[idx], cols)
        for k, v in st.items():
            boots[k].append(v)
    ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in boots.items()}
    return {"n_gt": len(obj), "n_images": int(obj["image_id"].nunique()), "point": point, "ci_95": ci}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--limit-images", type=int, default=0, help="smoke test")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    dev = torch.device(args.device)
    t0 = time.time()

    HieAssigner, ATSSAssigner, AnchorGenerator, BboxDistanceMetric, BboxOverlaps2D = load_original()
    hie = HieAssigner(ignore_iof_thr=-1, gpu_assign_thr=256, iou_calculator=dict(type="BboxDistanceMetric"),
                      assign_metric="kl", topk=[3, 1], ratio=0.9)
    atss = ATSSAssigner(topk=9)
    kl_metric, iou_calc = BboxDistanceMetric(), BboxOverlaps2D()
    rf = rfla_priors(dev)
    an, nlev = atss_priors(AnchorGenerator, dev)

    sel = pd.read_csv(SEL)
    focal_df = pd.read_csv(FOCAL)
    if args.limit_images:
        sel = sel.head(args.limit_images)
    weights = dict(zip(sel.image_id, sel.sampling_weight))
    strata = dict(zip(sel.image_id, sel.stratum))

    rows, width_check = [], []
    for n_img, image_id in enumerate(sel.image_id):
        gts_cpu = load_boxes(image_id)
        gts = gts_cpu.to(dev)
        focal_rows = focal_df[focal_df.image_id == image_id]
        for _, fr in focal_rows.iterrows():
            f = int(fr.gt_id)
            box = gts_cpu[f]
            w, h = float(box[2] - box[0]), float(box[3] - box[1])
            width_check.append(abs(w - fr.width) + abs(h - fr.height))
            base = {"rfla": rfla_state(hie, kl_metric, rf, gts, f),
                    "atss": atss_state(atss, iou_calc, an, nlev, gts, f)}
            for stress in ("fixed_1px", "equivalent_side"):
                delta = 1.0 if stress == "fixed_1px" else KAPPA * math.sqrt(w * h)
                for dname, ux, uy in DIRECTIONS:
                    sb = legal_shift(box, ux * delta, uy * delta)
                    if sb is None:
                        continue
                    alt = gts.clone()
                    alt[f] = sb.to(dev)
                    shifted = {"rfla": rfla_state(hie, kl_metric, rf, alt, f),
                               "atss": atss_state(atss, iou_calc, an, nlev, alt, f)}
                    row = {"image_id": image_id, "gt_id": f, "size_bin": fr.size_bin, "stratum": strata[image_id],
                           "sampling_weight": weights[image_id], "stress": stress, "direction": dname,
                           "shift_pixels": delta, "width": w, "height": h}
                    for a in ("rfla", "atss"):
                        b, s = base[a], shifted[a]
                        row[f"{a}_base_size"] = len(b["final"])
                        row[f"{a}_shift_size"] = len(s["final"])
                        row[f"{a}_set_change"] = int(b["final"] != s["final"])
                        row[f"{a}_top1_change"] = int(b["top1"] != s["top1"])
                        first = "none"
                        for st in STAGES[a]:
                            if b[st] != s[st]:
                                first = st
                                break
                        row[f"{a}_first_divergence"] = first
                        for st in STAGES[a]:
                            row[f"{a}_{st}_change"] = int(b[st] != s[st])
                    rows.append(row)
        if n_img % 25 == 0:
            print(f"[{time.time() - t0:7.1f}s] image {n_img + 1}/{len(sel)} rows={len(rows)}", flush=True)

    per_dir = pd.DataFrame(rows)
    per_dir.to_csv(args.output_dir / "per_direction.csv", index=False)
    max_width_err = float(max(width_check)) if width_check else float("nan")

    measures = [f"{a}_{m}" for a in ("rfla", "atss") for m in ("set_change", "top1_change")] + \
               [f"atss_{st}_change" for st in STAGES["atss"]] + [f"rfla_{st}_change" for st in STAGES["rfla"]]
    keys = ["image_id", "gt_id"]
    summary = {"protocol": "R2.2 prediction-independent assigner stress audit", "max_box_size_mismatch_px": max_width_err}
    yolo_cv = focal_df[focal_df.common_valid_margin.eq(1)][keys]
    for stress in ("fixed_1px", "equivalent_side"):
        d = per_dir[per_dir.stress.eq(stress)]
        obj = d.groupby(keys, sort=False).agg(
            **{m: (m, "max") for m in measures},
            rfla_base_size=("rfla_base_size", "first"), atss_base_size=("atss_base_size", "first"),
            size_bin=("size_bin", "first"), stratum=("stratum", "first"),
            sampling_weight=("sampling_weight", "first")).reset_index()
        obj["rfla_base_empty"] = (obj.rfla_base_size == 0).astype(int)
        obj["atss_base_empty"] = (obj.atss_base_size == 0).astype(int)
        cols = measures + ["rfla_base_empty", "atss_base_empty"]
        obj.to_csv(args.output_dir / f"per_gt_{stress}.csv", index=False)
        summary[stress] = {
            "all_focal": summarise(obj, cols, args.bootstrap, 20261121),
            "yolo_table1_common_valid": summarise(obj.merge(yolo_cv, on=keys), cols, args.bootstrap, 20261122),
        }
        fd = {}
        for a in ("rfla", "atss"):
            ch = d[d[f"{a}_set_change"].eq(1)]
            fd[a] = (ch[f"{a}_first_divergence"].value_counts(normalize=True)).to_dict()
        summary[stress]["first_divergence_share_among_set_changes_unweighted"] = fd
    summary["elapsed_sec"] = time.time() - t0
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "read_only": True, "device": str(dev), "images": int(len(sel)), "per_direction_rows": len(per_dir),
        "third_party_repo": "https://github.com/Chasel-Tsui/mmdet-rfla", "third_party_commit": "198977b5e198005fec783e81f10f9363946d2ef8",
        "third_party_sha256": {k: sha256(THIRD / v) for k, v in SRC.items()},
        "script_sha256": sha256(Path(__file__)),
        "rfla_config": {"fraction": 0.5, "fpn_layer": "p3", "strides": STRIDES, "topk": [3, 1], "ratio": 0.9,
                        "assign_metric": "kl", "trf": rfla_gen_trf()},
        "atss_config": {"topk": 9, "anchor": "AnchorGenerator(ratios=[1.0], octave_base_scale=8, scales_per_octave=1, strides=[8,16,32,64,128])"},
        "featmap_sizes": featmap_sizes(), "kappa": KAPPA, "selected_images_sha256": sha256(SEL),
        "max_box_size_mismatch_px": max_width_err, "elapsed_sec": summary["elapsed_sec"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("max_box_size_mismatch_px", "elapsed_sec")}))


if __name__ == "__main__":
    main()
