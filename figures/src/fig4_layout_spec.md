# Figure 4 layout specification

## Scientific role

Figure 4 reports grid-resolved cardinal boundary geometry from the frozen dense
1/1024 scan.  It does not depict a full two-dimensional boundary or stability
radius.  Base-undefined trajectories remain excluded and right censoring is
retained.  Only the final O2M assigned-positive Top-1 rank-state definition is
used; legacy and pre-Top-k aliases are prohibited.

## Physical layout

- Canvas: 178 mm by 132 mm, white, double-column width.
- Top left: panel a with separate 8–16 px and 16–32 px survival facets.
- Top right: panel b with overall O2O Aalen–Johansen first-cause curves.
- Bottom left: panel c with separate absolute and paired RMCBD axes.
- Bottom right: panel d with the six prespecified truncation horizons.
- A dedicated inter-panel gutter prevents long labels and legends from entering
  another panel.  All visible text is at least 7 pt at final size.

## Panel contracts

### a. First-departure survival

Kaplan–Meier estimates are drawn as unsmoothed post-step curves with low-opacity
95% stratified image-cluster bootstrap bands.  O2O is the post-conflict assigned identity;
O2M is the assigned-set Top-1 rank state, not an O2M identity or unique positive.
The two size facets are 8–16 px (tiny) and 16–32 px (small).

### b. First O2O boundary cause

Aalen–Johansen cause-specific cumulative-incidence post-step curves are drawn
for Eligibility, Within-set, Top-k, Conflict, and Other.  Minor and zero causes
remain on the common scale and are not enlarged.

### c. Restricted mean cardinal boundary distance

For All, 8–16 px, and 16–32 px, the left axis reports absolute O2O and final O2M
RMCBD and the right axis reports the paired O2O minus O2M difference.  Values are
shown to three decimals; source data retains full precision.  Branch-specific
supports are retained and recorded separately.

### d. Truncation-horizon sensitivity

The paired O2O minus O2M RMCBD is shown at tau = 0.05, 0.0625, 0.10, 0.125,
0.20, and 0.25 for All, 8–16 px, and 16–32 px.  Lines connect only the tested
horizons.  A pale vertical guide marks the primary tau = 0.125.  The panel
supports branch ordering across prespecified horizons, not magnitude invariance.

## Frozen inputs and fail-closed rules

All input paths and expected SHA-256 values are resolved from
`FIGURE_EVIDENCE_MANIFEST.json`.  Numeric inputs must be under
`runs/20260903_boundary_full_grid_1024_postprocess_v2`, carry evidence ID
`EV-BOUNDARY-GEOMETRY-V3`, and match the terminal validation manifest.  The
script exits before writing if any hash, terminal status, source-binding, tau
set, estimand name, or support value differs.  Rows containing `o2m_legacy` or
`o2m_pre_topk` are never copied into figure source data.

## Outputs

- `source_data/fig4_source.csv`
- `source_data/fig4_source.json`
- `draft/Fig4.svg`, `draft/Fig4.pdf`, `draft/Fig4.png`
- `qa/Fig4.validation.json`
