# Numerical repeatability contract

Status: **pre-lockbox instrument contract**.

## Normative execution

- The native detector and production replay use PyTorch float32 on the recorded CUDA environment.
- Native `TaskAlignedAssigner` stage outputs are normative for assignment identity and tie behaviour.
- The independent NumPy parity implementation uses explicit float32 for contract parity. Float64 may be used only as a labelled sensitivity check, never to silently replace native results.

## Numerical comparisons

- Alignment `q`: absolute tolerance `1e-6`, relative tolerance `2e-5`.
- Margin: absolute tolerance `1e-6`, relative tolerance `2e-6`.
- Positive alignment domain: `q > 0`; an active score at or below `1e-12` has no meaningful relative margin.
- Exact float32 base A/B ties are explicit numerical/undefined states, not substantive rank-boundary events.
- Candidate-order invariance is required only away from exact ties; native candidate ordering remains normative at a tie.

## Boundary radius

- Radius units are focal-GT equivalent side.
- Discovery coarse step is `1/64`; fine step and maximum reporting resolution are `1/512`.
- Production/reference radius or bracket differences up to `0.001953126` are within one fine-step contract; all larger differences are mismatches.
- No result may be reported with precision finer than the frozen fine step.
- Boundary search is forward coarse-to-forward fine and does not assume monotonicity.

## Legal geometry

- GT boxes and candidate geometry are never clipped to manufacture a legal replay.
- A continuous radius trajectory is defined only when the radius-zero focal GT box is fully inside the model input. Base-outside cases are counted and excluded.
- If an exact terminal image-boundary censor point alone rounds outside under native float32 arithmetic, the endpoint is moved inward by the first legal guard in `8,16,32,64,128 × eps_float32 × max(image dimension)`. The applied pixel guard is recorded per trajectory.

## Disagreement handling

- Stage or identity disagreement is never silently dropped.
- A disagreement confined to a prespecified numerical tie/tolerance is labelled a numerical-boundary case.
- Other disagreement fails instrument qualification and blocks freeze.

Qualification evidence includes the analytical oracle, property and mutation suites, 300 base plus 5,251 deterministic directional three-way parity states with zero mismatches, byte-identical repeated primary boundary outputs, and independent boundary reference checks.
