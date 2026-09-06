# Figure 1 final layout specification

Figure 1 defines one measurement object through a single left-to-right narrative. It does not summarize downstream results.

- Final canvas: 178 mm wide, approximately 94.5 mm high, white background.
- Typography: Arial, black text, 7 pt minimum; lowercase 8 pt bold panel letters and 7.6 pt compact headings.
- Visual separation: two 0.5 pt light-gray vertical rules only. No quadrant borders, cards, pills, dashboard tiles, gradients, shadows, or pseudo-3D.
- Palette: O2O deep blue, O2M orange, focal GT green, change/boundary burgundy, neutral gray. Purple is not used.

## a. Controlled focal-GT replay

- A 500×375-source-pixel AI-TOD-v2 crop from source extent `[120,165,620,540]`, displayed at approximately 41.7×31.3 mm so the embedded raster itself remains above 300 ppi.
- GT 28 is the focal box; GT 17 and GT 40 are non-focal context. IDs remain in provenance and the caption, not the artwork.
- Four small cardinal arrows summarize the center shifts.
- Stress contracts are shown as `Fixed: δ=1 px` and `Eq.-area: δ=κs_g, s_g=√(w_g h_g)`.
- The only prose statement is `Only the focal-GT centre moves`; frozen-output details remain in the caption.

## b. Native assignment-state extraction

- No backbone/FPN is shown.
- One header states that branch-specific predictions are frozen under the same focal-GT replay; it does not imply shared learned scores or decoded boxes.
- O2O: `Eligible → Top-7 → Conflict → A`, where A is the post-conflict assigned O2O ID.
- O2M: `Eligible → Top-10 → Conflict → {A,B,C} → A`, where the last A is assigned-set Top-1.
- O2M Top-1 is a rank readout rather than a uniquely supervised positive; this qualification is carried by the caption.

## c. Base-to-shift comparison

- Three rows show `A/A Stable`, `A/B Identity change`, and `A/— Identity-disappearance departure`; a base-undefined state is described in the caption rather than conflated with a legal shifted disappearance.
- A small one-ray inset marks the first observed cardinal boundary with ρ.
- Instrument checks, pathway causes, Rank–Set decomposition, margin, and downstream readouts are excluded from Figure 1 and remain in Methods, captions, or later figures.
