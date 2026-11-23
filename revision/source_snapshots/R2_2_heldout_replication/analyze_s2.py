"""S2: comparable O2O / O2M set-change measure (R2.1 estimator, unchanged) on one S2 output root.

Reuses run_seed() from R2_1_comparable_measure/comparable_measure.py by pointing its SEEDS table at
<root>/normalized and <root>/rank_set. Reported estimand: object-level O2M positive-set change minus
O2O change among 8-16 px objects (set_minus_o2o|8_16) on the Table-1 common-valid support, with the
built-in gate (per_gt rates reproduce the run's own summary) and join checks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "R2_1_comparable_measure" / "comparable_measure.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("cm", SRC)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    cm.SEEDS = {0: (str((a.root / "normalized").resolve()), str((a.root / "rank_set").resolve()))}
    res = cm.run_seed(0)
    if not res["gate"]["pass"]:
        raise RuntimeError(f"S2 gate failed: {res['gate']}")
    c = res["checks"]
    bad = {k: c[k] for k in ("per_direction_dup_keys", "rank_set_dup_keys", "join_left_only", "join_right_only",
                             "weight_mismatch", "size_bin_mismatch", "obj_o2o_vs_per_gt_mismatch", "obj_o2m_vs_per_gt_mismatch") if c[k]}
    if bad:
        raise RuntimeError(f"S2 checks failed: {bad}")
    a.out.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    p, ci = res["object_level"]["point"], res["object_level"]["ci_95"]
    for k in ("o2o_change|8_16", "o2m_set_change|8_16", "set_minus_o2o|8_16", "set_minus_o2o|16_32"):
        print(f"{k}: {100 * p[k]:.2f} [{100 * ci[k][0]:.2f}, {100 * ci[k][1]:.2f}]")
    print("objects", c["objects_used"], "gate n_gt", res["gate"]["n_gt"])


if __name__ == "__main__":
    main()
