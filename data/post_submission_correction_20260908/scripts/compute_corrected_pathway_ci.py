from pathlib import Path
import csv
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "PATHWAY_BASE_DEFINED_CORRECTION"
OUT.mkdir(parents=True, exist_ok=True)
REPS, SEED = 5000, 20260830
PATHS = [("Fixed 1 px", ROOT / "inputs/anatomy_fixed_per_direction.csv"), ("Equivalent-area side", ROOT / "inputs/anatomy_equivalent_per_direction.csv")]
LABELS = ["eligibility_boundary", "topk_membership_transition", "conflict_reassignment", "within_set_geometry_rank_reversal", "active_disappearance", "compound_or_other"]

def shares(frame):
    f = frame[frame.o2o_flip.eq(1)].copy()
    w = f.sampling_weight.to_numpy(float)
    den = w.sum()
    first = {k: float((w * f.first_divergence.eq(k).to_numpy()).sum() / den) for k in LABELS}
    anyv = {"eligibility": float((w * f.eligibility_pair_changed.to_numpy(int)).sum() / den), "within_set": float((w * f.pair_rank_reversal.to_numpy(int)).sum() / den), "eligibility_within": float((w * (f.eligibility_pair_changed.eq(1) & f.pair_rank_reversal.eq(1)).to_numpy()).sum() / den), "geometry_clipping": float((w * f.geometry_clipping_transition.to_numpy(int)).sum() / den)}
    return {**first, **anyv}

def bootstrap(frame, seed):
    f = frame[frame.o2o_flip.eq(1)].copy()
    groups = {s: [g for _, g in x.groupby("image_id")] for s, x in f.groupby("stratum")}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(REPS):
        pieces = []
        for gs in groups.values():
            pieces.extend(gs[i] for i in rng.integers(0, len(gs), len(gs)))
        values.append(shares(pd.concat(pieces, ignore_index=True)))
    return {k: [float(np.percentile([v[k] for v in values], q)) for q in (2.5, 97.5)] for k in values[0]}

records = []
for display, path in PATHS:
    data = pd.read_csv(path)
    data = data[data.active_base.notna()].copy()
    records.append((display, data, shares(data), bootstrap(data, SEED + (0 if display.startswith("Fixed") else 100))))

rows = []
for display, data, point, ci in records:
    row = {"contract": display, "flip_directions": int(data.o2o_flip.eq(1).sum()), "images": int(data[data.o2o_flip.eq(1)].image_id.nunique())}
    for key, value in point.items():
        row[key] = value
        row[key + "_ci_low"] = ci[key][0]
        row[key + "_ci_high"] = ci[key][1]
    rows.append(row)
with (OUT / "FIRST_DIVERGENCE_ANY_STAGE_CORRECTED_CI.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

def pct(x):
    return f"{100*x:.2f}"
def pctci(x):
    return f"[{100*x[0]:.2f}, {100*x[1]:.2f}]"

fixed, _, pf, cf = records[0]
equiv, _, pe, ce = records[1]
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Pathway & Fixed 1 px & Equivalent side & Equivalent minus fixed \\", r"\midrule"]
for key in LABELS:
    lines.append(f"{key.replace('_',' ').title()} & {pct(pf[key])} {pctci(cf[key])} & {pct(pe[key])} {pctci(ce[key])} & n/a \\\\")
lines += [f"Geometry-clipping diagnostic & {pct(pf['geometry_clipping'])} {pctci(cf['geometry_clipping'])} & {pct(pe['geometry_clipping'])} {pctci(ce['geometry_clipping'])} & descriptive \\\\", r"\midrule", f"Directional flip rows & {int(records[0][1].o2o_flip.eq(1).sum())} & {int(records[1][1].o2o_flip.eq(1).sum())} & n/a \\\\", f"Images with at least one exchange & {records[0][1][records[0][1].o2o_flip.eq(1)].image_id.nunique()} & {records[1][1][records[1][1].o2o_flip.eq(1)].image_id.nunique()} & n/a \\\\", r"\bottomrule", r"\end{tabular}"]
(OUT / "first_divergence_corrected_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Metric & Fixed 1 px & Equivalent side \\", r"\midrule"]
for key, label in [("eligibility", "Any eligibility"), ("within_set", "Any within-set rank"), ("eligibility_within", r"Eligibility $\cap$ within-set")]:
    lines.append(f"{label} & {pct(pf[key])} {pctci(cf[key])} & {pct(pe[key])} {pctci(ce[key])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "any_stage_corrected_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
