"""No-model check that both S2 scripts accept the preregistered list on the replication dataset.

Rank-set script: calls its own validate_selection_manifest(). Killer-controls script: repeats its
inline manifest check (required columns, index/image/stratum agreement). No model, no inference.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

PR = Path(r"E:\two_paper\beifen\publication_reproducible")
sys.path.insert(0, str(PR / "ultralytics_local"))
sys.path.insert(0, str(PR / "scripts"))
from run_reviewer_killer_controls import image_stratum  # noqa: E402

DATA = Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO\data.yaml")
SEL = Path(__file__).resolve().parent / "selected_images_preregistered.csv"


def main() -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(DATA))
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 800, "batch": 1, "workers": 0, "rect": False, "cache": False, "fraction": 1.0})
    dataset = build_yolo_dataset(cfg, data["val"], batch=1, data=data, mode="val", stride=32)
    strata = {"t_only": [], "s_only": [], "t_and_s": []}
    for index, label in enumerate(dataset.labels):
        s = image_stratum(label, 800)
        if s is not None:
            strata[s].append(index)

    spec = importlib.util.spec_from_file_location("rankset", PR / "scripts" / "analyze_o2m_rank_set_decoupling.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    selected, rows = rs.validate_selection_manifest(SEL, dataset, strata, False)
    print(f"rank-set validate_selection_manifest: accepted {len(selected)} images")

    rows_k = pd.read_csv(SEL).to_dict("records")
    required = {"dataset_index", "image_id", "stratum", "inclusion_probability", "sampling_weight"}
    assert rows_k and not required.difference(rows_k[0])
    for row in rows_k:
        index, stratum = int(row["dataset_index"]), str(row["stratum"])
        if Path(dataset.labels[index]["im_file"]).stem != str(row["image_id"]) or index not in strata[stratum]:
            raise RuntimeError("killer-controls manifest check failed")
    print(f"killer-controls manifest check: accepted {len(rows_k)} images")


if __name__ == "__main__":
    main()
