# Superseded pre-correction values in `generated_final_v2`

`generated_final_v2` is the frozen manuscript-generation snapshot. It is kept byte-for-byte so that its ledger and hash references stay valid. The post-submission audit found that a few secondary O2O identity summaries kept rows whose base O2O identity was undefined (169 ground truths, all 8–16 px). The corrected article reports those summaries from the base-defined recomputation in `data/post_submission_correction_20260908/`, and the entries below are **not** used by the corrected article.

| Location | Entry | Pre-correction value | Corrected source |
|---|---|---|---|
| `tables/first_divergence_cooccurrence.tex` | first-divergence / any-stage table | 67.09 / 51.01 first eligibility, … | `outputs/PATHWAY_BASE_DEFINED_CORRECTION/first_divergence_corrected_table.tex`, `any_stage_corrected_table.tex` |
| `evidence_values.tex`, `claim_values.json` | `FixedEligibilityPct` | 67.09 | `FIRST_DIVERGENCE_ANY_STAGE_CORRECTED_CI.csv` (63.91) |
| `evidence_values.tex`, `claim_values.json` | `FixedWithinPct` | 30.13 | same (34.23) |
| `evidence_values.tex`, `claim_values.json` | `NormalizedEligibilityPct` | 51.01 | same (40.18) |
| `evidence_values.tex`, `claim_values.json` | `NormalizedWithinPct` | 45.37 | same (56.98) |
| `evidence_values.tex`, `claim_values.json` | `EligibilityLockFixedGapPP` | 13.58 | `LOCK_CORRECTED_ESTIMATES.json` (reduction 12.02 pp) |
| `evidence_values.tex`, `claim_values.json` | `EligibilityLockFixedFractionPct` | 79.44 | same (77.50%) |
| `evidence_values.tex`, `claim_values.json` | `EligibilityLockNormalizedGapPP` | 3.15 | same (equivalent-area reduction 1.24 pp) |

Paths in the last column are relative to `data/post_submission_correction_20260908/outputs/`. The eligibility-stable subset and the pre-resolution control were corrected in the same way (`ANATOMY_BASE_DEFINED_CORRECTION/`, `PRE_RESOLUTION_BASE_DEFINED_CORRECTION/`). The primary Table 1 endpoint, the Rank-Set analyses, and the dense-boundary analysis already required a defined base identity, so the audit found them unaffected.

The article does not call `LockboxIntersectionGT` (7243). It is a historical Lockbox-v1 governance count and is not part of the correction.
