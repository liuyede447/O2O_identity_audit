"""Pre-run rehearsal of analyze_replication.py on discovery outputs and synthetic edge cases.

D1  seed-0 epoch-280 discovery output -> must equal LOCK_CORRECTED_ESTIMATES.json (points and CIs, 1e-12).
D2  seed-0 last.pt discovery output   -> comparator numbers for the replication (same checkpoint).
T*  synthetic frames covering the failure classes of the historical lockbox analyzer.
No replication image is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_replication as ar

HERE = Path(__file__).resolve().parent
GATES = HERE.parent / "R1_2_checkpoint_last" / "base_gate_all.csv"
FROZEN = Path(r"E:\two_paper\beifen\修复\post_submission_audit_20260908\outputs\LOCK_CORRECTED_ESTIMATES.json")
LOCK = Path(r"G:\revision_20261120")
results: dict = {}


def gate_for(seed: int, role: str) -> pd.DataFrame:
    g = pd.read_csv(GATES)
    return g.loc[g["seed"].eq(seed) & g["checkpoint"].eq(role), ["image_id", "gt_id", "o2o_active"]]


def run_dir(lock_dir: Path, seed: int, role: str) -> dict:
    v, sel, info = ar.validate(pd.read_csv(lock_dir / "per_gt.csv"), pd.read_csv(lock_dir / "selected_images.csv"),
                               gate_for(seed, role), None)
    res = ar.analyze(v, sel)
    res["decision"], res["validation"] = ar.decide(res), info
    return res


# D1: exact reproduction of the published seed-0 corrected estimates.
d1 = run_dir(LOCK / "eligibility_lock" / "seed0_rerun", 0, "e280")
frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
worst = 0.0
for stress in ar.STRESSES:
    for k, val in frozen["estimates"][stress]["point"].items():
        if k == "reduction_fraction":
            continue
        worst = max(worst, abs(d1["point"][f"{stress}|{k}"] - val))
    for k, (lo, hi) in frozen["estimates"][stress]["ci_95"].items():
        if k == "reduction_fraction":
            continue
        a, b = d1["ci_95"][f"{stress}|{k}"]
        worst = max(worst, abs(a - lo), abs(b - hi))
results["D1_max_abs_diff_vs_published"] = worst
assert worst < 1e-12, f"D1 failed: {worst}"
assert not any(d1["undefined_replicates"].values())

# D2: comparator on the same checkpoint as the replication.
d2 = run_dir(LOCK / "checkpoint_last" / "seed0", 0, "last")
results["D2_seed0_last_discovery"] = {k: [round(100 * d2["point"][k], 2)] + [round(100 * x, 2) for x in d2["ci_95"][k]]
                                      for k in d2["point"]}
results["D2_counts"] = d2["counts"]
results["D2_decision"] = d2["decision"]


# Synthetic tests.
def synth(n_img=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, sel = [], []
    strata = ["t_only"] * 5 + ["s_only"] * 3 + ["t_and_s"] * 4
    for i in range(n_img):
        img = f"img{i:02d}"
        sel.append({"dataset_index": i, "image_id": img, "stratum": strata[i], "inclusion_probability": 0.1, "sampling_weight": 10.0})
        for g in range(4):
            size = "s_16_32" if strata[i] == "s_only" or (strata[i] == "t_and_s" and g % 2) else "t_8_16"
            for s in ar.STRESSES:
                rows.append({"image_id": img, "gt_id": g, "stress": s, "size_bin": size, "stratum": strata[i], "sampling_weight": 10.0,
                             "native_o2o_fragile": int(rng.random() < 0.3), "eligibility_lock_o2o_fragile": int(rng.random() < 0.1)})
    per_gt, sel = pd.DataFrame(rows), pd.DataFrame(sel)
    gate = per_gt[["image_id", "gt_id"]].drop_duplicates().assign(o2o_active=1.0)
    return per_gt, sel, gate


def expect_ok(name, per_gt, sel, gate, check=None):
    v, s, info = ar.validate(per_gt, sel, gate, None)
    res = ar.analyze(v, s, replicates=300)
    if check:
        check(res, info)
    results[name] = "pass"


def expect_fail(name, per_gt, sel, gate):
    try:
        v, s, _ = ar.validate(per_gt, sel, gate, None)
        ar.analyze(v, s, replicates=50)
    except ar.AnalysisError as e:
        results[name] = f"pass (aborted: {e})"
        return
    raise AssertionError(f"{name}: expected abort")


p, s, g = synth()
expect_ok("T0_baseline", p, s, g)

# T1 image present under one stress only.
p1 = p.loc[~((p["image_id"] == "img00") & (p["stress"] == "equivalent_side"))]
expect_ok("T1_single_stress_image", p1, s, g, lambda r, i: (i["single_stress_images"] == ["img00"]) or (_ for _ in ()).throw(AssertionError("T1")))

# T2 image whose GTs all fail the base gate.
g2 = g.copy(); g2.loc[g2["image_id"] == "img01", "o2o_active"] = np.nan
expect_ok("T2_all_gated_out_image", p, s, g2, lambda r, i: ("img01" in i["zero_row_images"]) or (_ for _ in ()).throw(AssertionError("T2")))

# T3 zero-row image in the selection (no per_gt rows at all; e.g. box outside image bounds).
p3 = p.loc[p["image_id"] != "img02"]
expect_ok("T3_zero_row_image", p3, s, g, lambda r, i: ("img02" in i["zero_row_images"]) or (_ for _ in ()).throw(AssertionError("T3")))

# T4 small stratum of 3 images (s_only) is resampled without error; covered by baseline. All-zero outcomes:
p4 = p.assign(native_o2o_fragile=0, eligibility_lock_o2o_fragile=0)
expect_ok("T4_all_zero_outcomes", p4, s, g, lambda r, i: (abs(r["point"]["fixed_1px|native_gap"]) == 0) or (_ for _ in ()).throw(AssertionError("T4")))

# T5 GT present under one stress only (asymmetric keys).
p5 = p.loc[~((p["image_id"] == "img03") & (p["gt_id"] == 0) & (p["stress"] == "equivalent_side"))]
expect_ok("T5_single_stress_gt", p5, s, g, lambda r, i: (i["single_stress_gts"] == 1) or (_ for _ in ()).throw(AssertionError("T5")))

# T6 duplicate keys -> abort.
expect_fail("T6_duplicate_keys", pd.concat([p, p.iloc[[0]]]), s, g)
# T7 missing base-gate row -> abort.
expect_fail("T7_missing_gate_row", p, s, g.iloc[1:])
# T8 stratum disagreement -> abort.
p8 = p.copy(); p8.loc[p8.index[0], "stratum"] = "s_only"
expect_fail("T8_stratum_mismatch", p8, s, g)
# T9 merge order: shuffled rows give identical estimates.
r_a = ar.analyze(*ar.validate(p, s, g, None)[:2], replicates=200)
r_b = ar.analyze(*ar.validate(p.sample(frac=1, random_state=3), s.sample(frac=1, random_state=4), g.sample(frac=1, random_state=5), None)[:2], replicates=200)
assert max(abs(r_a["point"][k] - r_b["point"][k]) for k in r_a["point"]) < 1e-12
assert max(abs(r_a["ci_95"][k][j] - r_b["ci_95"][k][j]) for k in r_a["ci_95"] for j in (0, 1)) < 1e-12
results["T9_row_order_invariance"] = "pass"
# T10 no small objects under fixed 1 px -> primary point undefined -> structural failure (abort).
p10 = p.loc[p["size_bin"].eq("t_8_16")]
try:
    ar.analyze(*ar.validate(p10, s, g, None)[:2], replicates=10)
    raise AssertionError("T10 expected failure")
except ar.AnalysisError:
    results["T10_primary_empty_size_cell"] = "pass (structural failure reported)"
# T11 secondary block undefined (no small objects under equivalent-area) while primary is defined:
# primary estimates and intervals must equal those of a run where the secondary block is defined.
p11 = p.loc[~(p["stress"].eq("equivalent_side") & p["size_bin"].eq("s_16_32"))]
r11 = ar.analyze(*ar.validate(p11, s, g, None)[:2], replicates=300)
r0 = ar.analyze(*ar.validate(p, s, g, None)[:2], replicates=300)
assert all(np.isnan(r11["point"][f"equivalent_side|{k}"]) for k in ("native_gap", "reduction_pp"))
assert np.isnan(r11["point"]["S1|fixed_minus_equivalent_native_gap"])
assert all(abs(r11["point"][k] - r0["point"][k]) < 1e-12 and max(abs(x - y) for x, y in zip(r11["ci_95"][k], r0["ci_95"][k])) < 1e-12
           for k in r0["point"] if k.startswith("fixed_1px|"))
assert r11["undefined_replicates"]["fixed_1px|reduction_pp"] == 0
results["T11_secondary_failure_isolated"] = "pass (primary unchanged, secondary NaN)"
# T12 a resample that loses every small object under fixed 1 px: only those replicates are undefined.
s12 = s.copy()
p12 = p.loc[~(p["stress"].eq("fixed_1px") & p["size_bin"].eq("s_16_32") & ~p["image_id"].eq("img05"))]
r12 = ar.analyze(*ar.validate(p12, s12, g, None)[:2], replicates=300)
results["T12_partial_undefined_replicates"] = f"pass ({r12['undefined_replicates']['fixed_1px|native_gap']} of 300 primary replicates undefined, counted and excluded)"
assert 0 < r12["undefined_replicates"]["fixed_1px|native_gap"] < 300

(HERE / "rehearsal_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=1))
