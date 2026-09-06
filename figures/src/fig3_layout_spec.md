# Figure 3 layout specification

## Scientific role

Figure 3 asks where the fixed-pixel O2O response is localised along the native
assignment path.  It reports sensitivity and descriptive localisation, not a
causal decomposition.  Every plotted value is generated from the SHA-bound
Figure Evidence Manifest; no value is copied from an earlier figure or typed
into the plotting layer.

## Physical layout

- Final canvas: 178 mm wide by 130 mm high, white background.
- Grid: two panels across the first row; the key pathway panel occupies the
  left two-thirds of the second row and the same-branch control the right third.
- Panel c is itself split into two axes: exclusive first-observed composition
  on the left and non-exclusive any-stage occurrence on the right.
- All visible text is at least 7 pt at final size.  Text is black or dark grey.
- Statistical panels have no box, gradient, shadow, 3D effect, or full grid.

## Panel contracts

### a. Eligibility-lock sensitivity

A two-row dumbbell compares the O2O 8–16 minus 16–32 px scale gap under the
native and base-eligibility-locked definitions for fixed 1 px and
equivalent-area-side replay.  The fixed-row annotation reports the frozen
fraction attenuated, explicitly on lock support.  This is not a causal
contribution fraction; support differs from the primary common-valid endpoint.

### b. Eligibility-stable observational subset

An open-marker forest reports the object-level and direction-level scale gaps
after post-replay eligibility-stable conditioning for both stress contracts.
Open markers distinguish observational conditioning from panel a's
counterfactual lock.  Conditioning changes the estimand.

### c. First-observed versus any-stage localisation

The left sub-axis is a 100% stacked composition because the first-observed
labels are mutually exclusive.  Categories are Eligibility, Top-k, Conflict,
Within-set rank, and Other/reserved.  The reserved category combines the two
prespecified zero fallback labels.

The right sub-axis is a horizontal point-range plot because any-stage flags are
non-exclusive.  It shows Eligibility, Top-k, Conflict, Within-set rank, Active
missing, Geometry clipping, and Eligibility intersect Within-set rank.  Filled
and open markers encode fixed and equivalent-area-side replay.  All labels are
descriptive; first-observed labels depend on execution order and none are
causal contribution fractions.

### d. Pre-resolution same-branch control

A forest compares the pre-conflict Top-7 winner scale gap, final post-conflict assigned
scale gap, their paired difference, and fragility-definition disagreement for
each stress contract.  This is a same-branch descriptive control, not a
conflict-mediation test.

## Frozen inputs

Input paths and expected SHA-256 values are resolved from
`FIGURE_EVIDENCE_MANIFEST.json`.  Panel c additionally requires
`ADD-FIG-COOCCURRENCE-V1` in `FIGURE_EVIDENCE_ADDENDUM.json`.  The script exits
before writing source data or figures if any source is missing, its hash differs,
or the addendum authority/scope is not the declared figure-only source.

## Outputs

- `source_data/fig3_source.csv`
- `source_data/fig3_source.json`
- `draft/Fig3.svg` with editable text
- `draft/Fig3.pdf` with TrueType/type-42 fonts
- `draft/Fig3.png` at 600 dpi
- `qa/Fig3.validation.json`
