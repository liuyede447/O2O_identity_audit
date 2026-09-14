# Base-undefined propagation audit

This is a static-and-row-level audit of the specific bug pattern: a base O2O identity of `None` being retained in an identity-fragility summary.

| Analysis | Evidence | Gate status | Disposition |
|---|---|---|---|
| Primary O2O/O2M endpoint | `run_reviewer_killer_controls.py` retains `common_valid_margin == 1`; an O2O margin cannot exist without a unique active identity. | Effective gate present through the estimand definition. | Not affected by this incident. |
| Eligibility lock | Original source calls `int(active0 != native_active)` and writes every row; copied row data contain 169 base-O2O-undefined GTs. | Missing. | Affected; corrected estimates produced. |
| Eligibility-stable | `analyze_eligibility_stable_and_pathway_contrasts.py` loads the anatomy rows and has no `active_base` gate. Copied frozen anatomy rows contain 674 base-undefined directions / 169 GTs for each contract. | Missing. | Affected; re-estimation required. |
| First-divergence | The same anatomy rows are restricted only to `o2o_flip == 1`; 263 fixed and 253 equivalent base-undefined directional rows are counted as flips and have first-divergence labels. | Missing. | Affected; re-estimation required. |
| Pre-resolution O2O control | `analyze_o2o_preresolution_control.py` calculates `int(base_active != shifted_active)` and aggregates it without a base-active gate. The stored full-release artifact was not located in this pass, so the submitted point estimates must be recomputed before quantifying impact. | Missing. | Affected code path; exact impact pending re-estimation. |
| Dense O2O boundary | `analyze_assignment_boundary_geometry.py` defines `base_defined()` and filters `event_arrays()` to that gate, reporting excluded counts. | Explicit gate present. | Not affected by this incident. |

## Frozen-row evidence

- Fixed anatomy: 29,616 directions; 674 directions / 169 GTs have missing `active_base`; 263 of those directions carry `o2o_flip = 1`.
- Equivalent anatomy: 29,612 directions; 674 directions / 169 GTs have missing `active_base`; 253 of those directions carry `o2o_flip = 1`.
- All 169 affected GTs are in the 8--16 px stratum. They cannot be treated as valid O2O identity-fragility observations under the submitted state-inclusion matrix.

This audit does not infer that analyses with a missing gate share the same numerical correction. Each affected estimand requires its own copied-row recomputation.
