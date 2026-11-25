"""Summarise the discovery and official-validation direction records."""
import json
from pathlib import Path

import pandas as pd

PATHWAYS = (
    "eligibility_boundary",
    "topk_membership_transition",
    "conflict_reassignment",
    "within_set_geometry_rank_reversal",
    "active_disappearance",
    "compound_or_other",
)

def summarize_first_div(df: pd.DataFrame) -> dict:
    res = {}
    for st_name, sub in df.groupby("stress"):
        # Primary manuscript contract: base-defined identity-exchange directions
        fl = sub[(sub["o2o_flip"] == 1) & sub["active_base"].notna() & sub["size_bin"].isin(["t_8_16", "s_16_32"])].copy()
        w = fl["sampling_weight"].to_numpy(float)
        wtot = float(w.sum())
        fd_shares = {p: float(100.0 * w[fl["first_divergence"].eq(p).to_numpy()].sum() / wtot) for p in PATHWAYS}
        e = fl["eligibility_pair_changed"].to_numpy(bool)
        r = fl["pair_rank_reversal"].to_numpy(bool)
        # By size bin
        by_bin = {}
        for sb in ("t_8_16", "s_16_32"):
            fb = fl[fl["size_bin"].eq(sb)]
            wb = fb["sampling_weight"].to_numpy(float)
            wtb = float(wb.sum())
            by_bin[sb] = {
                "n_exchanges": int(len(fb)),
                "first_div_pct": {p: float(100.0 * wb[fb["first_divergence"].eq(p).to_numpy()].sum() / wtb) if wtb > 0 else None for p in PATHWAYS},
                "anystage_elig_pct": float(100.0 * wb[fb["eligibility_pair_changed"].to_numpy(bool)].sum() / wtb) if wtb > 0 else None,
                "anystage_rev_pct": float(100.0 * wb[fb["pair_rank_reversal"].to_numpy(bool)].sum() / wtb) if wtb > 0 else None,
            }
        # All flips (including base-undefined) for comparison
        fl_all = sub[(sub["o2o_flip"] == 1) & sub["size_bin"].isin(["t_8_16", "s_16_32"])]
        wa = fl_all["sampling_weight"].to_numpy(float)
        res[st_name] = {
            "n_directions": int(len(sub)),
            "n_base_defined_exchanges": int(len(fl)),
            "first_divergence_pct": fd_shares,
            "anystage_eligibility_pct": float(100.0 * w[e].sum() / wtot),
            "anystage_within_set_rank_pct": float(100.0 * w[r].sum() / wtot),
            "anystage_both_pct": float(100.0 * w[e & r].sum() / wtot),
            "all_flips_first_div_elig_pct": float(100.0 * wa[fl_all["first_divergence"].eq("eligibility_boundary").to_numpy()].sum() / float(wa.sum())),
            "by_size_bin": by_bin,
        }
    return res



if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    inputs = here / "inputs" / "first_divergence"
    cases = {
        "discovery_300": {f"seed{s}": inputs / f"discovery_seed{s}_per_direction_anatomy.csv" for s in range(1, 4)},
        "heldout_600": {
            "seed0": inputs / "heldout600_seed0_per_direction_anatomy.csv",
            "seed0_last": inputs / "heldout600_seed0_last_per_direction_anatomy.csv",
        },
    }
    cases["discovery_300"]["seed0_published_ref"] = inputs / "discovery_seed0_published.csv"
    results = {group: {name: summarize_first_div(pd.read_csv(path, dtype={"image_id": str}))
                      for name, path in tables.items()} for group, tables in cases.items()}
    out = here / "out"
    out.mkdir(exist_ok=True)
    (out / "first_divergence.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
