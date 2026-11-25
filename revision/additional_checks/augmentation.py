"""Recompute the 1,000-image augmentation summary from the derived replay records."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from verify import compare

def summarize_frame(g: pd.DataFrame) -> dict:
    # Restrict to base-defined GTs with legal shifts under both stresses for clean pairing
    f = g[g["o2o_active"].notna() & g["fixed_native"].notna() & g["scaled_native"].notna()].copy()
    tiny_all = g[g["size_bin"].eq("t_8_16")]
    tiny_f = f[f["size_bin"].eq("t_8_16")]
    small_f = f[f["size_bin"].eq("s_16_32")]

    def rates(sub: pd.DataFrame) -> dict:
        if not len(sub):
            return {"n": 0}
        return {
            "n": int(len(sub)),
            "fixed_native_pct": float(100.0 * sub["fixed_native"].mean()),
            "fixed_lock_pct": float(100.0 * sub["fixed_lock"].mean()),
            "fixed_removed_pp": float(100.0 * (sub["fixed_native"] - sub["fixed_lock"]).mean()),
            "scaled_native_pct": float(100.0 * sub["scaled_native"].mean()),
            "scaled_lock_pct": float(100.0 * sub["scaled_lock"].mean()),
            "scaled_removed_pp": float(100.0 * (sub["scaled_native"] - sub["scaled_lock"]).mean()),
        }

    rt = rates(tiny_f)
    rs = rates(small_f)
    gap_fn = (rt["fixed_native_pct"] - rs["fixed_native_pct"]) if rt["n"] and rs["n"] else None
    gap_fl = (rt["fixed_lock_pct"] - rs["fixed_lock_pct"]) if rt["n"] and rs["n"] else None
    gap_sn = (rt["scaled_native_pct"] - rs["scaled_native_pct"]) if rt["n"] and rs["n"] else None
    gap_sl = (rt["scaled_lock_pct"] - rs["scaled_lock_pct"]) if rt["n"] and rs["n"] else None

    return {
        "n_focal_all": int(len(g)),
        "n_base_defined_both_legal": int(len(f)),
        "geometry_diagnostics_t_8_16": {
            "n": int(len(tiny_all)),
            "integer_box_pct": float(100.0 * tiny_all["is_integer_box"].mean()),
            "s8_edge_coincidence_pct": float(100.0 * tiny_all["has_s8_edge_coincidence"].mean()),
            "short_eq8_pct": float(100.0 * tiny_all["is_short_eq8"].mean()),
            "short_lt8_pct": float(100.0 * tiny_all["is_short_lt8"].mean()),
            "short_gt8_pct": float(100.0 * tiny_all["is_short_gt8"].mean()),
            "rule_toggles_1px_pct": float(100.0 * tiny_all["rule_toggles_1px"].mean()),
            "s8_changes_1px_pct": float(100.0 * tiny_all["s8_changes_1px"].mean()),
            "s8_changes_scaled_pct": float(100.0 * tiny_all["s8_changes_scaled"].mean()),
        },
        "size_bins": {
            "t_8_16": rt,
            "s_16_32": rs,
            "fixed_native_gap_pp": gap_fn,
            "fixed_lock_gap_pp": gap_fl,
            "fixed_gap_attenuation_pp": None if gap_fn is None else gap_fn - gap_fl,
            "scaled_native_gap_pp": gap_sn,
            "scaled_lock_gap_pp": gap_sl,
        },
        "t_8_16_by_short_band": {
            "short_lt8": rates(tiny_f[tiny_f["is_short_lt8"].eq(1)]),
            "short_eq8": rates(tiny_f[tiny_f["is_short_eq8"].eq(1)]),
            "short_gt8": rates(tiny_f[tiny_f["is_short_gt8"].eq(1)]),
        },
        "t_8_16_by_mechanism_1px": {
            "rule_toggles_1px_yes": rates(tiny_f[tiny_f["rule_toggles_1px"].eq(1)]),
            "s8_changes_1px_only": rates(tiny_f[(tiny_f["rule_toggles_1px"].eq(0)) & (tiny_f["s8_changes_1px"].eq(1))]),
            "no_s8_change_1px": rates(tiny_f[tiny_f["s8_changes_1px"].eq(0)]),
        },
    }


here = Path(__file__).resolve().parent
inputs = here / "inputs" / "augmentation"
expected = json.loads((here / "expected" / "exp2_augmented_replay_summary.json").read_text(encoding="utf-8"))
compare(summarize_frame(pd.read_csv(inputs / "augmented_train_per_gt.csv")), expected["augmented_train"], "augmentation")
