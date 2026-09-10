# Supplementary Figure S1 layout specification

## Scientific role

Supplementary Figure S1 is a qualitative audit view of three real-image O2O
assignment states under the frozen primary checkpoint.  It is not an aggregate
estimate and the overlaid candidate boxes are audited assignment states, not
final detector outputs.

## Authoritative inputs

Only the following inputs are permitted:

- `figures/source_data/figs1_primary_trace/primary_qualitative_trace.json`
- its adjacent `validation.json`
- the source image named by the trace
- `results/measurement_validation_20260831/fixed_branch_replay_v1/per_direction.csv`

All four paths and SHA-256 values are resolved from the current Figure Evidence
Manifest and cross-checked against the trace and validation sidecar.  The old
qualitative pack and its non-primary checkpoint are forbidden.

## Case identity

The rows are fixed and fail closed:

1. Stable: GT 40, down, assigned O2O candidate 5552 to 5552.
2. Fragile: GT 28, up, assigned O2O candidate 11582 to 6264.
3. Identity-disappearance departure: GT 17, right, assigned O2O candidate 3930 to no assigned identity.

These identities are checked against frozen `per_direction.csv`; no scientific
result is recomputed.

## Layout

- Canvas: 178 mm by 170 mm, white background.
- Grid: three rows by three columns: Context, Base state, Shift state.
- Context uses a fixed 180-by-180 input-pixel crop; Base and Shift use the same
  fixed 72-by-72 crop within a row. Crops are centred from trace coordinates,
  then constrained to the source image's automatically detected non-black
  content extent, so letterbox padding can never enter a displayed panel.
- Row headings and all semantic labels sit outside the image axes.
- No in-image text, callout, arrow, gradient, shadow, or 3D effect.

## Overlay grammar

- Focal/base GT: green solid rectangle.
- Shifted GT: vermilion/red dashed rectangle.
- Assigned O2O candidate: dark blue solid rectangle.
- Runner candidate: purple dashed rectangle.
- The shift panel retains the original focal GT as a reference and adds the
  exact shifted GT box.  Missing active/runner states are not imputed.

Every rectangle is drawn directly from `gt_xyxy`, `shift_gt_xyxy`, or
`decoded_xyxy`.  Coordinate transforms are exact subtraction of the displayed
crop origin; no resizing-dependent offsets are introduced.

## Interpretation guard

The figure footer states: “Overlays are audited assignment states, not final
detections.”  It also notes that image pixels are unchanged and only the focal
GT centre is shifted by one input pixel.

## Outputs

- `source_data/figs1_source.csv`
- `source_data/figs1_source.json`
- `draft/FigS1.svg`, `draft/FigS1.pdf`, `draft/FigS1.png`
- `qa/FigS1.validation.json`
