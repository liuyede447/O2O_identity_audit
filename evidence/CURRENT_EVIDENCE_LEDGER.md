# Current Evidence Ledger

Status: **FINAL EVIDENCE FROZEN**  
Updated: 2026-09-01 11:10 +08:00

The normative JSON ledger uses schema version 1.1 and passed `validate_current_evidence_ledger.py` for all 24 current entries. This Markdown file is a navigation summary, not the machine-readable authority.

Authority order:

1. Frozen result artifact
2. Current Evidence Ledger
3. Historical 2026-08-22 SSOT
4. Manuscript

The manuscript is not a fact source. Each accepted quantitative claim must later receive a complete claim record containing dataset, checkpoint, seed, unit, denominator, estimand, stress contract, sampling, weighting, uncertainty, multiplicity status, artifact path, and hashes.

## Validated instrument evidence

- `EV-INSTR-ORACLE-V1`: analytical oracle 6/6 and property invariants 7/7 passed.
- `EV-INSTR-MUTATION-V1`: 6/6 prespecified mutations were killed.
- `EV-INSTR-PARITY-975-V1`: zero native/production/NumPy mismatches across 300 base states and 975 deterministic directional states.
- `EV-INSTR-PARITY-5251-V1`: zero native/production/NumPy mismatches across 300 base states and 5,251 deterministic directional states.

## Validated discovery evidence currently registered

- `EV-ELIGIBILITY-LOCK-V1`
- `EV-ELIGIBILITY-STABLE-V2`
- `EV-PATHWAY-ANATOMY-V2`
- `EV-SAME-STRIDE-V1`
- `EV-RANK-SET-FIXED-V1`
- `EV-RANK-SET-NORMALIZED-V1`
- `EV-PRERESOLUTION-FIXED-V1`
- `EV-PRERESOLUTION-NORMALIZED-V1`
- `EV-LEARNING-DYNAMICS-V1`
- `EV-YOLOV10-NORMALIZED-V1`
- `EV-BOUNDARY-GEOMETRY-V2`
- `EV-MARGIN-RHOR-CONSTRUCT-V2`
- `EV-SPILLOVER-FORMAL300-V1`
- `EV-STRESS-CONTRACT-PRIMARY-V1`
- `EV-STRESS-CONTRACT-KAPPA-DATASET-V1`

## Validated supplementary diagnostic evidence

- `EV-MARGIN-SHAPE-SENSITIVITY-V1`: five-fold image-grouped OOF linear-versus-spline sensitivity with training-fold-only spline knots and 5,000 image-cluster bootstrap replicates per outcome. It supports only the absence of stable metric-wide spline improvement in this analysis; it is not causal, confirmatory, or an equivalence result.
- `EV-SAMPLE-EFFICIENCY-1000-V1`: 1,000 outcome-blind stratified subsamples per budget. It supports reduced-budget dominant-signature screening while documenting that 100-image effect estimates can still have material absolute error; it is not a power analysis or precision guarantee.
- `EV-PRECISION-DESIGN-V1`: discovery-based 300-image expected-precision planning for H1-H4. It is used only to document expected bootstrap variability before preregistration; it is not post-hoc power and does not access lockbox outcomes.

## Post-lockbox corrective exploratory evidence

- `EV-LOCKBOX-CORRECTIVE-EXPLORATORY-V1`: the only prospective analyzer invocation was invalidated before a verdict. The corrective analysis retained all 300 clusters, disclosed the 7,243-key common-support intersection and three fixed-only keys, and independently reproduced 5,000 bootstrap replicates. Its H1-H5 analogues are exploratory only; no confirmatory PASS/FAIL exists.

## Validated formal boundary instrument evidence

- `EV-BOUNDARY-REFERENCE-FORMAL-V2`: 30 image forwards and 4,408 independent NumPy trajectories matched production with zero mismatch and zero event-radius difference; its recursive 37-file inventory passed.

The JSON ledger contains authoritative artifact paths and SHA-256 values.

## Not evidence

- The exhaustive parity attempt in `parity_exhaustive_v1` is explicitly incomplete.
- Smoke tests and analyzer self-tests validate implementation plumbing only; they are not scientific results.

## Completed raw core artifact

- `20260831T174200_boundary_discovery_v2` completed on 300 images with 7,431 audited focal GTs and 29,724 directional trajectories. Its counts, keys, config, checkpoint, and SHA-256 manifest were reconciled. Formal independent reference and boundary-geometry analysis are now complete; claims are registered under the IDs above.

## Pending gates

Instrument/estimand freeze, H1-H5 preregistration, one-time lockbox selection/extraction, the invalidated prospective outcome access, and the post-lockbox corrective exploratory analysis are complete. The next gate is final evidence freeze before manuscript or figure updates.
