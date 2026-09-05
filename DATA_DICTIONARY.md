# Data dictionary

## `selected_images.csv`

- `dataset_index`: immutable dataset row index.
- `image_id`: dataset image identifier; no pixels are included.
- `stratum`: outcome-blind sampling stratum.
- `inclusion_probability`: image inclusion probability.
- `sampling_weight`: inverse inclusion probability.

## `per_direction_boundary_events.csv`

One row per focal-GT × legal cardinal direction. Identity fields describe native base states; `*_event_observed`, `*_radius`, censoring and bracket fields describe first sampled state changes on the 1/1024 grid. `o2m_assigned_positive_*` is the post-conflict assigned-positive Top-1 rank summary. `o2m_positive_count_base` concerns the full native positive set. `rho_R_*` describes the fixed base active/runner rank-crossing construct and competing censoring. `o2o_first_divergence` is an ordered first-localisation label, not a causal share.

## Separate `o2m_rank_set_radius_curve.csv` release asset

One row per evaluated radius. Keys are `image_id`, `gt_id`, `direction`, `radius` and `evaluation_phase`. The table records O2O active identity, three O2M rank summaries, positive-set count/intersection/union/Jaccard/retention/structural state and fixed-pair q-gap comparability. It has 7,593,288 rows and is not stored in Git history.

## Six-τ post-processing tables

- `survival_cif.csv`: Kaplan--Meier survival and Aalen--Johansen cause-specific incidence.
- `rmsr.csv`, `rmsr_contrasts.csv`: restricted mean cardinal boundary distances and contrasts through τ.
- `return_transition.csv`: observed return-transition counts/proportions.
- `o2m_endpoint_rank_set.csv`: O2M rank/set endpoint summaries.
- `margin_construct.csv`, `rho_R_status.csv`: margin--ρR constructs and support/censoring summaries.

## Figure source tables

`figures/source_data/fig2_source.csv` through `fig6_source.csv` contain panel labels, evidence IDs, source paths/hashes, estimates and intervals. Empty cells mean not applicable, not zero. See each JSON sidecar for semantics and orientation.
