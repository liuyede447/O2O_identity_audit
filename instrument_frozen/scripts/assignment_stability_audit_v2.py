"""Read-only margin, coverage, and fixed-K assignment audit for frozen YOLO26 checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics_local"))
DATA_YAML = ROOT / "configs" / "aitod_v2.template.yaml"  # configure an absolute dataset path per docs/REPRODUCIBILITY.md
PERTURBATIONS = (("left_1px", -1.0, 0.0), ("right_1px", 1.0, 0.0), ("up_1px", 0.0, -1.0), ("down_1px", 0.0, 1.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA_YAML)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-per-bin", type=int, default=250)
    p.add_argument("--max-images", type=int, default=1500)
    p.add_argument("--fixed-k", type=int, default=4)
    p.add_argument("--wall-time-sec", type=int, default=10800)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_bin(area: float) -> str:
    side = max(area, 0.0) ** 0.5
    return "vt_lt8" if side < 8 else "t_8_16" if side < 16 else "s_16_32" if side < 32 else "m_ge32"


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def rank_corr(left: torch.Tensor, right: torch.Tensor) -> float | None:
    valid = (left > 0) | (right > 0)
    left, right = left[valid].float(), right[valid].float()
    if left.numel() < 2:
        return None
    a = torch.argsort(torch.argsort(left, stable=True), stable=True).float()
    b = torch.argsort(torch.argsort(right, stable=True), stable=True).float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.square().sum().sqrt() * b.square().sum().sqrt()
    return None if denom.item() == 0 else float((a * b).sum().div(denom).item())


def trace(criterion, raw, labels, boxes, mask):
    from ultralytics.utils.tal import make_anchors
    pd, scores = raw["boxes"].permute(0, 2, 1).contiguous(), raw["scores"].permute(0, 2, 1).contiguous()
    anchors, stride = make_anchors(raw["feats"], criterion.stride, 0.5)
    decoded = criterion.bbox_decode(anchors, pd)
    assigner = criterion.assigner
    assigner.bs, assigner.n_max_boxes = scores.shape[0], boxes.shape[1]
    pre, align, overlaps = assigner.get_pos_mask(scores.detach().sigmoid(), (decoded.detach() * stride).type(boxes.dtype), labels, boxes, anchors * stride, mask)
    _, _, post = assigner.select_highest_overlaps(pre, overlaps, assigner.n_max_boxes, align)
    return pre.bool(), post.bool(), align, stride.squeeze(-1)


def summary(pre, post, align, stride, gt, fixed_k):
    pre_ids = torch.where(pre[0, gt])[0]
    post_ids = torch.where(post[0, gt])[0]
    scores = align[0, gt]
    eligible = torch.where(scores > 0)[0]
    ordered = eligible[torch.argsort(scores[eligible], descending=True)] if eligible.numel() else eligible
    fixed = ordered[:fixed_k]
    top = int(ordered[0].item()) if ordered.numel() else None
    second = float(scores[ordered[1]].item()) if ordered.numel() > 1 else None
    first = float(scores[ordered[0]].item()) if ordered.numel() else None
    margin = ((first - second) / (first + 1e-12)) if second is not None and first is not None else None
    o2o = int(post_ids[scores[post_ids].argmax()].item()) if post_ids.numel() else None
    return {"pre": set(map(int, pre_ids.cpu().tolist())), "post": set(map(int, post_ids.cpu().tolist())), "fixed": set(map(int, fixed.cpu().tolist())), "o2o": o2o, "stride": int(stride[o2o].item()) if o2o is not None else None, "rank": scores.detach(), "margin": margin, "pre_count": int(pre_ids.numel()), "post_count": int(post_ids.numel())}


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file() or digest(args.checkpoint) != args.expected_sha256:
        raise RuntimeError("checkpoint missing or SHA-256 mismatch")
    if not args.data.is_file() or args.max_per_bin <= 0 or args.fixed_k <= 0:
        raise ValueError("invalid audit arguments")
    if args.output_dir.exists() and not args.smoke:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("audit_v2")
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    device = torch.device("cuda:0" if args.device == "0" and torch.cuda.is_available() else args.device)
    model = YOLO(str(args.checkpoint)).model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    data = check_det_dataset(str(args.data))
    cfg = get_cfg(overrides={"task":"detect","imgsz":args.imgsz,"batch":1,"workers":args.workers,"rect":False,"cache":False,"single_cls":False,"classes":None,"fraction":1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=max(int(model.stride.max()), 32))
    loader = build_dataloader(dataset, batch=1, workers=args.workers, shuffle=False, rank=-1, drop_last=False, pin_memory=False)
    o2m, o2o = v8DetectionLoss(model, tal_topk=10), v8DetectionLoss(model, tal_topk=7, tal_topk2=1)
    rows, counts, images, started = [], defaultdict(int), 0, time.monotonic()
    with torch.no_grad():
        for batch in loader:
            if images >= args.max_images or time.monotonic() - started >= args.wall_time_sec: break
            images += 1
            for k,v in list(batch.items()):
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            batch["img"] = batch["img"].float()/255
            output = model(batch["img"])
            if not isinstance(output, tuple) or not isinstance(output[1], dict): raise RuntimeError("expected raw end-to-end output")
            raw = output[1]
            targets = torch.cat((batch["batch_idx"].view(-1,1),batch["cls"].view(-1,1),batch["bboxes"]),1)
            image_h,image_w=batch["img"].shape[-2:]
            targets = o2m.preprocess(targets,1,torch.tensor([image_w,image_h,image_w,image_h],device=device,dtype=batch["img"].dtype))
            labels, boxes = targets.split((1,4),2); mask = boxes.sum(2,keepdim=True).gt_(0)
            if not mask.any(): continue
            base_m, base_o = trace(o2m,raw["one2many"],labels,boxes,mask), trace(o2o,raw["one2one"],labels,boxes,mask)
            image = Path(batch["im_file"][0]).stem
            for gt in torch.where(mask[0,:,0])[0].tolist():
                b=boxes[0,gt]; w,h=float((b[2]-b[0]).item()),float((b[3]-b[1]).item()); area=w*h; bucket=size_bin(area)
                if counts[bucket] >= args.max_per_bin: continue
                counts[bucket]+=1; bm=summary(*base_m,gt,args.fixed_k); bo=summary(*base_o,gt,args.fixed_k); valid=[]
                for name,dx,dy in PERTURBATIONS:
                    shifted=b.clone(); shifted[[0,2]]+=dx; shifted[[1,3]]+=dy
                    if shifted[0]<0 or shifted[1]<0 or shifted[2]>image_w or shifted[3]>image_h: continue
                    altered=boxes.clone(); altered[0,gt]=shifted
                    cm,co=summary(*trace(o2m,raw["one2many"],labels,altered,mask),gt,args.fixed_k),summary(*trace(o2o,raw["one2one"],labels,altered,mask),gt,args.fixed_k)
                    valid.append({"image_id":image,"gt_id":gt,"class_id":int(labels[0,gt,0]),"x1":float(b[0]),"y1":float(b[1]),"x2":float(b[2]),"y2":float(b[3]),"width":w,"height":h,"area":area,"size_bin":bucket,"perturbation":name,"o2m_pre_count_0":bm["pre_count"],"o2m_post_count_0":bm["post_count"],"o2m_pre_count_p":cm["pre_count"],"o2m_post_count_p":cm["post_count"],"o2m_jaccard_post":jaccard(bm["post"],cm["post"]),"o2m_retention_post":len(bm["post"]&cm["post"])/len(bm["post"]) if bm["post"] else 1.0,"o2m_symmetric_difference":len(bm["post"]^cm["post"]),"fixedk_jaccard":jaccard(bm["fixed"],cm["fixed"]),"o2o_candidate_0":bo["o2o"],"o2o_candidate_p":co["o2o"],"o2o_flip":int(bo["o2o"]!=co["o2o"]),"stride_0":bo["stride"],"stride_p":co["stride"],"stride_flip":int(bo["stride"]!=co["stride"]),"o2o_margin_0":bo["margin"],"o2o_rank_corr":rank_corr(bo["rank"],co["rank"])})
                for r in valid: r["valid_perturbations"]=len(valid); rows.append(r)
                if args.smoke: break
            if args.smoke or all(counts[x]>=args.max_per_bin for x in ("vt_lt8","t_8_16","s_16_32","m_ge32")): break
    if args.smoke: log.info("SMOKE_PASS rows=%d",len(rows)); return
    args.output_dir.mkdir(parents=True); (args.output_dir/"audit.log").write_text("read-only audit v2\n",encoding="utf-8")
    cols=list(rows[0]);
    with (args.output_dir/"per_gt.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    audit_summary={"status":"complete","read_only":True,"checkpoint":str(args.checkpoint),"checkpoint_sha256":args.expected_sha256,"data":str(args.data),"imgsz":args.imgsz,"images_seen":images,"sampled_gt_by_bin":dict(counts),"records":len(rows),"fixed_k":args.fixed_k,"elapsed_sec":time.monotonic()-started}
    (args.output_dir/"summary.json").write_text(json.dumps(audit_summary,indent=2),encoding="utf-8")
    log.info("AUDIT_V2_COMPLETE %s",audit_summary)

if __name__=="__main__": main()
