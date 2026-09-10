# Frozen-support sensitivity analysis (2026-09-09, bug-fixed rerun)

## Bug-fix notice

The original 2026-09-09 run (`support_sensitivity_20260909`, retained as
`support_sensitivity_20260909_BUGGY_PREBUGFIX_DO_NOT_USE`) had a coding error
in `dense_rmsr()` inside `inputs/run_support_sensitivity.py`: it called
`bootstrap_multiplicities(meta, 5000, 20260831)` directly and treated index 0
of the resulting 5,000-row array as the point estimate. That function returns
only random multinomial resamples — it does **not** prepend an identity
(all-ones) row the way the production estimator
(`analyze_assignment_boundary_geometry.py`, which this script otherwise
reuses) does. So the "point estimate" published for the common-trajectory
RMCBD contrast (Supplementary Table S37 / main-text results) was actually the
**first of 5,000 random bootstrap replicates**, not the true full-sample
statistic, and the reported 95% interval was built from only 4,999 of the
5,000 replicates (the mislabeled one having been consumed as the "point").

This rerun fixes `dense_rmsr()` to prepend the identity row before calling
`evaluate_process`, exactly matching the production convention, and is
otherwise byte-identical in logic to the original script. Inputs are
unchanged (see `MANIFEST.json`; all three input SHA-256 values match the
original run exactly — this is a pure estimator fix, not a different
dataset or a different pipeline). `RUNNER_FREE_ENDPOINT_SENSITIVITY.csv` and
`DENSE_COMMON_RISKSET_COUNTS.csv` are byte-identical to the original run
(diffed and confirmed): the bug only affected `DENSE_COMMON_TRAJECTORY_RMCBD.csv`,
i.e. only the common-trajectory RMCBD numbers used for Table S37, nothing
else in this analysis or in Table S36.

## Scope and provenance

This report is a read-only reanalysis of existing frozen row-level artifacts. No detector inference, replay, training, checkpoint selection, or new model-derived outcome was run. The endpoint inputs are the canonical fixed 1-pixel and equivalent-area (κ=0.0625) `per_gt.csv` files; the dense input is the frozen four-cardinal-ray `per_direction_boundary_events.csv` file. SHA-256 values and row counts are recorded in `MANIFEST.json`.

## Endpoint support definitions

* **O2O natural support:** `o2o_active` is defined; no runner requirement.
* **O2M natural support:** `o2m_rank` is defined; no runner requirement.
* **State-common runner-free support:** both `o2o_active` and `o2m_rank` are defined; no runner requirement.
* **Competition-defined common support:** `common_valid_margin == 1`, which was checked to be exactly equivalent to both state fields and both runner fields being defined.

The fixed endpoint contains 7,265 O2O-natural rows, 7,322 O2M-natural rows, and 7,263 state-common runner-free rows; the competition-defined common support is 6,378 rows. The corresponding equivalent-area counts are 7,263, 7,320, 7,261, and 6,377. Thus the runner-free state-common sensitivity recovers 885 (fixed) and 884 (equivalent-area) rows that are excluded only by the competition-defined runner restriction. The 2-row gap between each O2O-natural and state-common support is due to O2M state non-definition, not runner availability.

`RUNNER_FREE_ENDPOINT_SENSITIVITY.csv` reports branch estimates on each named support. On O2O-natural or O2M-natural rows, the other branch estimate is evaluated on the rows where that branch's state is defined; these are descriptive branch-specific estimates, not paired common-support contrasts. The paired endpoint comparison is interpretable on state-common or competition-defined supports only. (Unaffected by the bug fix.)

For the state-common runner-free support, the additional 5,000-replicate image-cluster bootstrap gives the following scale gaps (8--16 minus 16--32 percentage points): fixed O2O **15.48 [11.96, 18.91]**, fixed O2M **−2.63 [−7.52, 2.61]**, paired **18.11 [11.42, 24.85]**; equivalent-area O2O **1.06 [−2.60, 4.16]**, equivalent-area O2M **−21.95 [−27.76, −15.79]**, paired **23.01 [16.05, 30.07]**. The component rates and their intervals are in `runner_free_bootstrap_fixed/RUNNER_FREE_ENDPOINT_BOOTSTRAP_CI.csv` and `runner_free_bootstrap_equivalent/RUNNER_FREE_ENDPOINT_BOOTSTRAP_CI.csv`. (Unaffected by the bug fix; computed by a separate, correctly-written script that already isolates the point estimate from the bootstrap draws.)

Relative to the frozen competition-defined Table 1 values (fixed O2O 14.37 [10.84, 17.78], fixed O2M 1.42 [−3.18, 6.31], paired 12.95 [6.52, 19.16]; equivalent O2O 0.13 [−3.41, 3.15], equivalent O2M −17.45 [−22.72, −11.71], paired 17.58 [11.06, 23.98]), all runner-free intervals overlap the corresponding primary intervals. The fixed O2O and equivalent-area paired contrasts remain positive; the equivalent-area O2O absolute gap remains compatible with zero. This is support sensitivity, not a replacement of the primary estimand.

## Dense common-risk-set counts

The dense file has 29,724 trajectories (7,431 focal GTs × 4 cardinal directions). Initial state definition is O2O for 29,048 trajectories, O2M assigned-positive for 28,996, and both states for 28,992. Through τ=0.125, the common O2O risk set has 2,496 first events, 191 geometric censors before τ, 0 administrative/rmax censors before τ, and 24,951 trajectories observable through τ. The corresponding common O2M counts are 10,067, 167, 0, and 23,632. Counts by size stratum are provided without collapsing event, censoring, and observability categories into a mutually exclusive endpoint outcome. (Unaffected by the bug fix.)

## Common-trajectory RMCBD sensitivity (corrected)

Using the frozen weighted Kaplan–Meier integral implementation, τ=0.125, 5,000 image-cluster bootstrap replicates, and seed 20260831, with the identity-row point-estimate fix applied, the common-trajectory O2O-minus-O2M RMCBD difference is:

| support | all | tiny (8–16 px) | small (16–32 px) |
|---|---:|---:|---:|
| common trajectories (corrected) | 0.01962 [0.01795, 0.02138] | 0.01747 [0.01593, 0.01899] | 0.03319 [0.02937, 0.03655] |
| common trajectories (original, buggy) | 0.02037 [0.01795, 0.02138] | 0.01707 [0.01593, 0.01899] | 0.03452 [0.02937, 0.03655] |

The 95% intervals are essentially unchanged (they were already built from ~4,999-5,000 genuine bootstrap draws either way); only the point estimates move. `DENSE_COMMON_RISKSET_COUNTS.csv` shows this holds only for the larger stratum, not both: for 16-32 px, common-initial-defined (3,964) equals both O2O-natural (3,964) and O2M-natural (3,964), so the common-trajectory restriction excludes zero rows there. For 8-16 px, common-initial-defined (25,028) is smaller than O2O-natural (25,084, a 56-row exclusion) and smaller than O2M-natural (25,032, a 4-row exclusion) — this stratum is *not* row-restriction-free. Consistent with that split, the corrected common-trajectory point estimate for 16-32 px matches the main-text natural-support RMCBD contrast (0.03319) to five decimal places, while the 8-16 px estimate (0.01747) is close to but not identical to its natural-support counterpart (0.01745), reflecting the small residual row exclusion in that stratum. `dense_rmsr()` reuses the same frozen weighted Kaplan-Meier/RMCBD estimator functions as the primary dense-grid pipeline (imported directly from `analyze_assignment_boundary_geometry.py`, confirmed at the source level, not merely by numeric proximity), so this sensitivity supports the robustness of the primary dense-grid result to the common-trajectory row restriction; it is not evidence of, nor does it require invoking, a separately estimated or differently processed quantity.

These are descriptive sensitivity estimates on the same trajectory rows for both branches. They do not establish a causal branch effect, a two-dimensional boundary, or a universal law. Full branch-specific estimates and confidence intervals are in `DENSE_COMMON_TRAJECTORY_RMCBD.csv`.

## Interpretation

The runner-free state-common analysis separates intrinsic state definition from the competition-defined margin restriction. The dense common-trajectory analysis keeps O2O and O2M on identical image/GT/direction rows and retains the first-departure censoring rules. The outputs therefore address support sensitivity and comparability; they do not add a new model outcome or alter the frozen primary estimands.

No manuscript, figure, table, GitHub release, or Zenodo record was modified by this run itself; the manuscript files were updated separately using the corrected numbers reported above.
