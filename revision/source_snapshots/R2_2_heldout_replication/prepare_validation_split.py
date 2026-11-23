"""Prepare the official AI-TOD-v2 validation split for the pre-specified replication.

Step 1 (gate): re-convert 100 randomly chosen official test images with the unchanged
production conversion code and require byte-identical labels versus the existing
G:/data/processed/AI-TOD-v2-YOLO/val/labels (which holds the official test split).
Step 2: convert the 2,804 validation images with the same code (convert_split, unchanged),
copying images, and write a data YAML plus per-file SHA-256 manifests.
No model is loaded and no replay output is produced.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

CONVERTER = Path(r"E:\two_paper\SWD-MicroDet\scripts\prepare_aitod_v2_yolo.py")
CONVERTER_SHA = "ab77bb788422759bac3607adb26b8325af85e88f11a13a6cf4e3fff4d211a519"
TEST_JSON = Path(r"G:\data\raw\AI-TOD-v2\aitodv2_test.json")
TEST_LABELS = Path(r"G:\data\processed\AI-TOD-v2-YOLO\val\labels")
ANN_DIR = Path(r"G:\revision_20261120\heldout_replication\extracted\annotations")
VAL_JSON_SHA = "d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf"
VAL_IMAGES = Path(r"G:\two_paper\SWD-MicroDet\third_party\AI-TOD-official\aitodtoolkit\aitod\images\val")
OUT = Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO")
HERE = Path(__file__).resolve().parent
CHECK_SEED, CHECK_N = 20261121, 100


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(CONVERTER) != CONVERTER_SHA:
        raise RuntimeError("converter changed")
    spec = importlib.util.spec_from_file_location("conv", CONVERTER)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    # Step 1: conversion gate on official test images (same label-writing logic as convert_split).
    data = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = sorted(data["images"], key=lambda im: im["file_name"])
    picked = random.Random(CHECK_SEED).sample(images, CHECK_N)
    mismatches = []
    for image in picked:
        lines = [conv.yolo_line(a, image, cat_index) for a in by_image.get(image["id"], [])
                 if not a.get("iscrowd", 0) and a["bbox"][2] > 0 and a["bbox"][3] > 0]
        # Write exactly as convert_split does (Path.write_text, platform newline translation).
        tmp = HERE / "_gate_tmp.txt"
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        expected = tmp.read_bytes()
        tmp.unlink()
        existing = (TEST_LABELS / f"{Path(image['file_name']).stem}.txt").read_bytes()
        if expected != existing:
            mismatches.append(image["file_name"])
    print(f"conversion gate: {CHECK_N - len(mismatches)}/{CHECK_N} byte-identical")
    if mismatches:
        raise RuntimeError(f"conversion gate failed: {mismatches[:5]}")

    # Step 2: convert the validation split with the unchanged convert_split.
    if sha256(ANN_DIR / "aitod_val_v2.json") != VAL_JSON_SHA:
        raise RuntimeError("validation annotation hash mismatch")
    if OUT.exists():
        raise RuntimeError(f"{OUT} exists; refusing to overwrite")
    summary = conv.convert_split(ANN_DIR, OUT, "val",
                                 {"json": ("aitod_val_v2.json",), "images": (str(VAL_IMAGES),)},
                                 "copy", allow_missing=False)
    names = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(cats))
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: val/images\nval: val/images\n\nnames:\n{names}\n", encoding="utf-8")
    manifest = {
        "converter": str(CONVERTER), "converter_sha256": CONVERTER_SHA,
        "annotation": str(ANN_DIR / "aitod_val_v2.json"), "annotation_sha256": VAL_JSON_SHA,
        "source_images": str(VAL_IMAGES), "conversion_gate": f"{CHECK_N}/{CHECK_N} test labels byte-identical (seed {CHECK_SEED})",
        "images": summary["images"], "annotations_kept": summary["annotations"],
        "missing_images": summary["missing_images"], "classes": summary["classes"],
    }
    (HERE / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (HERE / "validation_files_sha256.txt").open("w", encoding="utf-8") as f:
        for sub in ("images", "labels"):
            for p in sorted((OUT / "val" / sub).iterdir()):
                f.write(f"{sha256(p)}  val/{sub}/{p.name}\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "classes"}, indent=1))


if __name__ == "__main__":
    main()
