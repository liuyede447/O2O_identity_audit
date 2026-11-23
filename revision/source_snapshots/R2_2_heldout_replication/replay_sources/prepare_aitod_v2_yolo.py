"""Convert AI-TOD-v2 COCO annotations to a YOLO-format dataset.

AI-TOD-v2 reuses AI-TOD images and provides revised COCO annotations. Put the
v2 annotation JSON files under data/raw/AI-TOD-v2/annotations, then point this
script at the existing image directories.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_SPLITS = {
    "train": {
        "json": (
            "small_train_v2_1.0.json",
            "aitodv2_train.json",
            "aitod_train_v2.json",
        ),
        "images": (
            "third_party/AI-TOD-official/aitodtoolkit/aitod/images/train",
            "data/raw/AI-TOD_wo_xview/aitod_wo_xview_train_imgs/images",
            "data/raw/AI-TOD_wo_xview/aitod_wo_xview_trainval_imgs/images",
        ),
    },
    "val": {
        "json": (
            "small_val_v2_1.0.json",
            "small_test_v2_2.0.json",
            "aitodv2_val.json",
            "aitodv2_test.json",
            "aitod_val_v2.json",
        ),
        "images": (
            "third_party/AI-TOD-official/aitodtoolkit/aitod/images/val",
            "third_party/AI-TOD-official/aitodtoolkit/aitod/images/test",
            "data/raw/AI-TOD_wo_xview/aitod_wo_xview_test_imgs/images",
            "data/raw/AI-TOD_wo_xview/aitod_wo_xview_val_img/images",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-dir",
        type=Path,
        default=Path("data/raw/AI-TOD-v2/annotations"),
        help="Directory containing AI-TOD-v2 COCO annotation JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/AI-TOD-v2-YOLO"),
        help="Output directory for YOLO-format AI-TOD-v2.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place images in the processed dataset.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow conversion to continue when annotation images are missing.",
    )
    return parser.parse_args()


def make_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def yolo_line(annotation: dict, image: dict, category_to_index: dict[int, int]) -> str:
    x, y, w, h = annotation["bbox"]
    cls = category_to_index[annotation["category_id"]]
    cx = (x + w / 2) / image["width"]
    cy = (y + h / 2) / image["height"]
    return f"{cls} {cx:.6f} {cy:.6f} {w / image['width']:.6f} {h / image['height']:.6f}"


def resolve_annotation_file(ann_dir: Path, candidates: str | tuple[str, ...]) -> Path:
    names = (candidates,) if isinstance(candidates, str) else candidates
    for name in names:
        path = ann_dir / name
        if path.exists():
            return path
    expected = "\n".join(f"  {ann_dir / name}" for name in names)
    raise FileNotFoundError(
        "Missing AI-TOD-v2 annotation file. Expected one of:\n"
        f"{expected}\n"
        "Download the official AI-TOD-v2 annotations. The images can remain in "
        "data/raw/AI-TOD_wo_xview."
    )


def resolve_image_dirs(candidates: str | tuple[str, ...]) -> list[Path]:
    names = (candidates,) if isinstance(candidates, str) else candidates
    paths = [Path(name) for name in names if Path(name).exists()]
    if paths:
        return paths
    expected = "\n".join(f"  {name}" for name in names)
    raise FileNotFoundError(f"Missing image directory. Expected one of:\n{expected}")


def find_image(image_dirs: list[Path], filename: str) -> Path | None:
    for image_dir in image_dirs:
        path = image_dir / filename
        if path.exists():
            return path
    return None


def convert_split(
    ann_dir: Path,
    out_dir: Path,
    split: str,
    spec: dict,
    mode: str,
    allow_missing: bool,
) -> dict:
    annotation_file = resolve_annotation_file(ann_dir, spec["json"])
    image_dirs = resolve_image_dirs(spec["images"])

    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda item: item["id"])
    category_to_index = {item["id"]: idx for idx, item in enumerate(categories)}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    missing_images = []
    image_sources = {}
    for image in data["images"]:
        src_image = find_image(image_dirs, image["file_name"])
        if src_image is None:
            missing_images.append(image["file_name"])
            continue
        image_sources[image["id"]] = src_image

    if missing_images and not allow_missing:
        sample = ", ".join(missing_images[:10])
        raise FileNotFoundError(
            f"{split}: {len(missing_images)} of {len(data['images'])} images are missing.\n"
            f"Sample missing files: {sample}\n"
            "This usually means only AI-TOD_wo_xview is present. The current route needs the "
            "full AI-TOD-v2 image set generated with the xView training images."
        )

    image_out = out_dir / split / "images"
    label_out = out_dir / split / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    kept_annotations = 0
    for image in data["images"]:
        src_image = image_sources.get(image["id"])
        if src_image is None:
            continue
        make_image(src_image, image_out / image["file_name"], mode)

        lines = []
        for ann in annotations_by_image.get(image["id"], []):
            if ann.get("iscrowd", 0):
                continue
            if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
                continue
            lines.append(yolo_line(ann, image, category_to_index))
        kept_annotations += len(lines)
        (label_out / f"{Path(image['file_name']).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    return {
        "split": split,
        "images": len(data["images"]),
        "annotations": kept_annotations,
        "missing_images": missing_images,
        "classes": [item["name"] for item in categories],
    }


def write_dataset_yaml(out_dir: Path, classes: list[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    text = f"""path: {out_dir.resolve().as_posix()}
train: train/images
val: val/images

names:
{names}
"""
    (out_dir / "aitod_v2.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        convert_split(args.ann_dir, args.out_dir, split, spec, args.copy_mode, args.allow_missing)
        for split, spec in DEFAULT_SPLITS.items()
    ]
    write_dataset_yaml(args.out_dir, summaries[0]["classes"])

    for summary in summaries:
        print(
            f"{summary['split']}: {summary['images']} images, "
            f"{summary['annotations']} annotations, "
            f"{len(summary['missing_images'])} missing images"
        )
        if summary["missing_images"]:
            print("  sample missing:", ", ".join(summary["missing_images"][:5]))
    print(f"dataset yaml: {(args.out_dir / 'aitod_v2.yaml').resolve()}")


if __name__ == "__main__":
    main()
