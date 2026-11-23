"""Draw the 600-image replication sample without loading a model.

Uses the replay script's own dataset construction, image_stratum and proportional_allocation,
so the formal run (same data YAML, --sample-images 600, same --selection-seed) must regenerate
this list exactly. Only annotation sizes are read; no prediction or outcome exists yet.
"""
from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from run_reviewer_killer_controls import image_stratum, proportional_allocation  # noqa: E402

DATA = Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO\data.yaml")
SAMPLE_IMAGES = 600
SELECTION_SEED = 20261122
IMGSZ = 800
HERE = Path(__file__).resolve().parent


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(DATA))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": IMGSZ, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        stratum = image_stratum(label, IMGSZ)
        if stratum is not None:
            strata[stratum].append(index)
    population = {k: len(v) for k, v in strata.items()}
    allocation = proportional_allocation(population, min(SAMPLE_IMAGES, sum(population.values())))
    rng = random.Random(SELECTION_SEED)
    selected = []
    for stratum in sorted(strata):
        selected.extend((index, stratum) for index in rng.sample(strata[stratum], allocation[stratum]))
    selected.sort(key=lambda item: Path(dataset.labels[item[0]]["im_file"]).stem)
    out = HERE / "selected_images_preregistered.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"])
        for index, stratum in selected:
            p = allocation[stratum] / population[stratum]
            w.writerow([index, Path(dataset.labels[index]["im_file"]).stem, stratum, p, 1.0 / p])
    print("dataset images", len(dataset.labels))
    print("population", population, "allocation", allocation)
    print("selected", len(selected), "sha256", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
