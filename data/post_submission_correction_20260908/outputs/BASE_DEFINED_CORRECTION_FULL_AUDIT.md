# Base-defined correction closure

All O2O identity summaries below apply the frozen base-assigned-identity-defined gate before aggregation. No detector inference, training, or evidence-freeze artifact was changed.

- Eligibility-lock effective supports: fixed 7,265 GT / 298 images; equivalent-area 7,263 GT / 298 images.
- The 169 excluded base-undefined GTs are all in the 8--16 px stratum.
- Corrected fixed lock gap: 15.51 [11.96, 18.87] pp to 3.49 [1.12, 5.64] pp; reduction 12.02 [9.77, 14.05] pp, 77.50% [68.01%, 91.53%].
- Corrected equivalent-area lock gap: 1.07 [-2.56, 4.16] pp to -0.17 [-2.70, 2.06] pp.
- Primary Table 1, Rank--Set, and dense-boundary estimands retain their explicit gates and are unaffected.
- Corrected first-divergence bootstrap CIs use 5,000 stratified image-cluster replicates on 1,931 fixed and 988 equivalent-area identity-exchange directions; any-stage shares use this same exchange-direction denominator.

Source artifacts: `LOCK_CORRECTED_ESTIMATES.csv`, `ANATOMY_BASE_DEFINED_CORRECTION/summary.json`, `PATHWAY_BASE_DEFINED_CORRECTION/FIRST_DIVERGENCE_ANY_STAGE_CORRECTED.csv`, and `PRE_RESOLUTION_BASE_DEFINED_CORRECTION/summary.json`.
