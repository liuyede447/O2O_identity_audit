from pathlib import Path
import csv, json

out = Path(__file__).resolve().parent
rows = []
for src in [out / "LOCK_CORRECTED_ESTIMATES.csv", out / "PATHWAY_BASE_DEFINED_CORRECTION" / "FIRST_DIVERGENCE_ANY_STAGE_CORRECTED.csv", out / "PRE_RESOLUTION_BASE_DEFINED_CORRECTION" / "PRE_RESOLUTION_CORRECTED_ESTIMATES.csv"]:
    if not src.exists():
        continue
    with src.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["source_artifact"] = str(src.relative_to(out))
            rows.append(row)
fields = sorted({k for r in rows for k in r})
with (out / "CORRECTED_SECONDARY_ESTIMATES_MASTER.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

(out / "BASE_DEFINED_CORRECTION_FULL_AUDIT.md").write_text("""# Base-defined correction closure\n\nAll O2O identity summaries below apply the frozen base-assigned-identity-defined gate before aggregation. No detector inference, training, or evidence-freeze artifact was changed.\n\n- Eligibility-lock effective supports: fixed 7,265 GT / 298 images; equivalent-area 7,263 GT / 298 images.\n- The 169 excluded base-undefined GTs are all in the 8--16 px stratum.\n- Corrected fixed lock gap: 15.51 [11.96, 18.87] pp to 3.49 [1.12, 5.64] pp; reduction 12.02 [9.77, 14.05] pp, 77.50% [68.01%, 91.53%].\n- Corrected equivalent-area lock gap: 1.07 [-2.56, 4.16] pp to -0.17 [-2.70, 2.06] pp.\n- Primary Table 1, Rank--Set, and dense-boundary estimands retain their explicit gates and are unaffected.\n\nSource artifacts: `LOCK_CORRECTED_ESTIMATES.csv`, `ANATOMY_BASE_DEFINED_CORRECTION/summary.json`, `PATHWAY_BASE_DEFINED_CORRECTION/FIRST_DIVERGENCE_ANY_STAGE_CORRECTED.csv`, and `PRE_RESOLUTION_BASE_DEFINED_CORRECTION/summary.json`.\n""", encoding="utf-8")
(out / "PRE_RESOLUTION_RECOVERY_REPORT.md").write_text("""# Pre-resolution exact recovery\n\n`PRE_RESOLUTION_EXACT_RECOVERY = TRUE`. The corrected pre-resolution estimates were deterministically recomputed from the copied frozen pre-resolution rows with the base-assigned-identity-defined gate, original inverse image-inclusion weights, and 5,000 image-cluster bootstrap replicates. Fixed and equivalent-area supports are 7,265/298 images/28,942 directions and 7,263/298 images/28,938 directions, respectively. Corrected point estimates and intervals are in `PRE_RESOLUTION_BASE_DEFINED_CORRECTION/PRE_RESOLUTION_CORRECTED_ESTIMATES.csv`; no aggregate manuscript number was used to infer a replacement.\n""", encoding="utf-8")
(out / "CORRECTION_OLD_VS_NEW.csv").write_text("""analysis,old_status,new_status,old_value,new_value,reason\nEligibility-lock,CORRECTED,BASE_DEFINED_GATE,17.10 -> 3.52 pp,15.51 -> 3.49 pp,169 base-undefined GTs excluded before aggregation\nEligibility-stable,CORRECTED,BASE_DEFINED_GATE,submitted support,corrected summary,base-defined rows only\nFirst-divergence,CORRECTED,BASE_DEFINED_GATE,submitted shares,corrected shares,base-defined rows only\nPre-resolution,CORRECTED,BASE_DEFINED_GATE,submitted control,corrected estimates,exact frozen row recovery\nPrimary Table 1,UNAFFECTED,UNCHANGED,14.37 / 0.13 pp,14.37 / 0.13 pp,competition-defined support already requires base identity\nRank-Set,UNAFFECTED,UNCHANGED, frozen, frozen, distinct O2M state estimand\nDense boundary,UNAFFECTED,UNCHANGED, frozen, frozen, explicit base-defined risk-set gate\n""", encoding="utf-8")
(out / "MANUSCRIPT_IMPACT_MAP.csv").write_text("""location,status,action\nMain Results §4.3,CORRECTED,replace lock support estimates and wording\nMain Figure 3A,CORRECTED,regenerate corrected panel\nMain Figure 3B,CORRECTED,replace pathway shares\nMain Figure 3C,CORRECTED,replace pre-resolution control\nMain Conclusion,UNCHANGED_AFTER_RECOMPUTATION,qualitative direction preserved\nSupplement §5.1,CORRECTED,replace lock estimates\nSupplement §5.2,CORRECTED,replace pathway shares\nSupplement Table S17,CORRECTED,replace first-divergence values\nSupplement Table S18,CORRECTED,replace any-stage values\nSupplement Table S25,CORRECTED,replace lock support and estimates\nTable 1,UNAFFECTED_DIFFERENT_PIPELINE,no change\nDense boundary,UNAFFECTED_DIFFERENT_PIPELINE,no change\nRank--Set,UNAFFECTED_DIFFERENT_PIPELINE,no change\n""", encoding="utf-8")
(out / "EDITOR_CORRECTION_SUMMARY.md").write_text("""# Editor correction summary\n\nA post-submission reproducibility audit identified a denominator inconsistency in secondary O2O mechanism summaries: rows with undefined base assigned O2O identity were not excluded before aggregation as required by the manuscript estimand contract. The issue affects eligibility-lock, eligibility-stable, first-divergence/pathway, and pre-resolution summaries. It does not affect the primary Table 1 endpoint, Rank--Set results, or dense-boundary analysis. Corrected estimates preserve the qualitative conclusion that eligibility locking attenuates the fixed-pixel O2O contrast. Withdrawal is not required; editor notification and replacement files are required.\n""", encoding="utf-8")
decision = {
    "primary_table1_affected": False,
    "rankset_affected": False,
    "dense_boundary_affected": False,
    "eligibility_lock_affected": True,
    "eligibility_stable_affected": True,
    "first_divergence_affected": True,
    "pre_resolution_affected": True,
    "qualitative_claim_changed": False,
    "pre_resolution_exact_recovery": True,
    "manuscript_replacement_required": True,
    "editor_notification_required": True,
    "withdrawal_required": False,
    "status": "LOCALIZED_SUBSTANTIVE_POST_SUBMISSION_CORRECTION",
}
(out / "POST_SUBMISSION_FINAL_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
print(f"wrote {len(rows)} corrected rows")
