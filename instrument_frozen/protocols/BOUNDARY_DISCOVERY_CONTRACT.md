# Boundary discovery contract

Status: **formally qualified discovery contract; not yet the lockbox freeze**.

The discovery scan uses equivalent-side radius, `rmax=0.25`, a forward coarse step of `1/64`, a forward fine step of `1/512`, four cardinal directions, and RMSR truncation at `tau=0.125`. It does not assume monotonic identity response and does not use binary search.

The fixed base active/runner pair defines `rho_R`. Eligibility, pre-Top-k, or conflict comparability loss is treated as a competing censor before a rank crossing. Missing pairs and exact native float32 ties remain explicit structural/numerical states.

A continuous legal trajectory is defined only when the radius-zero focal GT box is fully inside the model input. Base-outside boxes are counted and excluded without clipping. A deterministic float32 inward guard applies only to a terminal image-boundary censor point that would otherwise round outside; the applied pixel guard is recorded.

The production smoke was byte-identical across two runs for event and curve CSVs. An independent NumPy smoke recovered all 48 production trajectories with zero mismatches and zero event-radius difference.

Formal production then completed on 300 images, 7,431 audited focal GTs, and 29,724 directional trajectories. The deterministic 30-image independent NumPy reference reproduced all 4,408 production trajectories with zero mismatch and zero event-radius difference; its recursive 37-file inventory passed. Formal geometry used 5,000 stratified image-cluster bootstrap replicates, `tau=0.125`, and endpoints `0.03125`, `0.0625`, and `0.125`; all eight outputs and the recursive inventory passed their SHA-256 checks.

These values may change only before instrument freeze for a documented numerical or algorithmic reason followed by repeated validation. They may not change after lockbox outcome access.
