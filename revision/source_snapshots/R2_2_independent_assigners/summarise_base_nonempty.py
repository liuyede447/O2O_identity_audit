"""Post-hoc summary of the RFLA/ATSS audit on the analogue of YOLO's base-defined population.

YOLO Table 1 conditions on a defined base O2O identity. The analogue here is a non-empty
base positive set for the assigner being read. Each assigner is filtered on its own base
state. Uses summarise() from assigner_audit.py unchanged (same bootstrap seeds).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AUD = Path(r"G:\revision_20261120\assigner_audit_v1")
FOCAL = Path(r"E:\two_paper\beifen\publication_reproducible\results\reviewer_controls_20260829\formal_s0e300_k00625\per_gt.csv")

spec = importlib.util.spec_from_file_location("aud", HERE / "assigner_audit.py")
aud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aud)

KEYS = ["image_id", "gt_id"]
COLS = {
    "rfla": ["rfla_set_change", "rfla_top1_change"],
    "atss": ["atss_set_change", "atss_top1_change", "atss_candidates_change", "atss_threshold_change", "atss_inside_change"],
}


def main() -> None:
    focal = pd.read_csv(FOCAL)
    cv = focal.loc[focal["common_valid_margin"].eq(1), KEYS].drop_duplicates()
    out = {"population": "base-nonempty positive set for the assigner read; YOLO analogue of base-defined O2O identity"}
    for stress in ("fixed_1px", "equivalent_side"):
        obj = pd.read_csv(AUD / f"per_gt_{stress}.csv")
        out[stress] = {}
        for a, cols in COLS.items():
            ne = obj.loc[obj[f"{a}_base_size"].gt(0)]
            out[stress][a] = {
                "base_nonempty": aud.summarise(ne, cols, 5000, 20261121),
                "base_nonempty_and_common_valid": aud.summarise(ne.merge(cv, on=KEYS), cols, 5000, 20261122),
            }
            print(stress, a, "done", flush=True)
    # Joint population: one GT set for every assigner, YOLO included.
    # = YOLO Table 1 common-valid AND RFLA base non-empty AND ATSS base non-empty (seed 0).
    yolo_src = {
        "fixed_1px": Path(r"E:\two_paper\beifen\修复\post_submission_audit_20260908\inputs\fixed_branch_per_gt.csv"),
        "equivalent_side": FOCAL,
    }
    out["joint"] = {}
    for stress in ("fixed_1px", "equivalent_side"):
        obj = pd.read_csv(AUD / f"per_gt_{stress}.csv")
        y = pd.read_csv(yolo_src[stress], usecols=KEYS + ["o2o_fragile", "o2m_fragile", "common_valid_margin"])
        y = y.loc[y["common_valid_margin"].eq(1)].rename(columns={"o2o_fragile": "yolo_o2o_change", "o2m_fragile": "yolo_o2m_top1_change"})
        j = obj.loc[obj["rfla_base_size"].gt(0) & obj["atss_base_size"].gt(0)].merge(y.drop(columns="common_valid_margin"), on=KEYS)
        cols = ["yolo_o2o_change", "yolo_o2m_top1_change", "rfla_set_change", "rfla_top1_change", "atss_set_change", "atss_top1_change"]
        out["joint"][stress] = aud.summarise(j, cols, 5000, 20261123)
        r = out["joint"][stress]
        for k in r["point"]:
            if k.endswith("gap"):
                c = r["ci_95"][k]
                print(f"JOINT {stress:16s} n={r['n_gt']} {k:28s} {100*r['point'][k]:7.2f} [{100*c[0]:.2f}, {100*c[1]:.2f}]")
    # first-divergence share for ATSS set changes, by stress and size, base-nonempty, direction level
    d = pd.read_csv(AUD / "per_direction.csv")
    d = d.loc[d["atss_base_size"].gt(0) & d["atss_set_change"].eq(1)]
    fd = {}
    for stress in ("fixed_1px", "equivalent_side"):
        for b in ("t_8_16", "s_16_32"):
            x = d.loc[d["stress"].eq(stress) & d["size_bin"].eq(b)]
            fd[f"{stress}|{b}"] = (x.groupby("atss_first_divergence")["sampling_weight"].sum() / x["sampling_weight"].sum()).to_dict()
    out["atss_first_divergence_weighted_share_base_nonempty"] = fd
    (HERE / "summary_base_nonempty.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for stress in ("fixed_1px", "equivalent_side"):
        for a in COLS:
            for pop, r in out[stress][a].items():
                for k, v in r["point"].items():
                    if k.endswith("gap") or "|8_16" in k:
                        c = r["ci_95"][k]
                        print(f"{stress:16s} {pop:32s} n={r['n_gt']:5d} {k:32s} {100*v:7.2f} [{100*c[0]:.2f}, {100*c[1]:.2f}]")
    print(json.dumps(fd, indent=1))


if __name__ == "__main__":
    main()
