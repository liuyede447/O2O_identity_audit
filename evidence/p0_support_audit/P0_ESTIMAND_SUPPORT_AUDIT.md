# P0 Final Primary-Estimand Support Audit

**Read-only status: `PASS_CLARIFICATION_ONLY`**

No detector, replay, training, analyzer, or new scientific estimate was run. Counts below were read from frozen row-level CSVs, manifests, and existing dense-grid event/curve files. No manuscript file was edited.

## Primary answer

**YES, indirectly in the Table 1 inclusion logic; NO for the intrinsic O2O identity-fragility event definition.** The executable endpoint writer computes `o2o_fragile` from identity comparisons, but Table 1 filters to `common_valid_margin==1`. That flag is written as `int(margin is not None and m_margin is not None)`, so an O2O runner and an O2M runner (positive alternative) are required for inclusion in the common-valid support. This is a support restriction, not a claim that a runner is needed to define an O2O identity change.

## Support flow

| contract | focal eligible | audited endpoint rows | base O2O unavailable | base O2O present | runner/positive alternative unavailable | common-valid support |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 7444 | 7434 | 169 | 7265 | 887 | 6378 |
| equivalent | 7444 | 7432 | 169 | 7263 | 886 | 6377 |

For fixed replay, 7,444 focal GTs become 7,434 rows after 10 no-legal-shift cases; 169 rows lack a base assigned O2O identity; 887 of the remaining rows lack at least one required runner/alternative, leaving 6,378 common-valid GTs. For equivalent-area replay, the corresponding sequence is 7,444 -> 7,432 (12 no-legal-shift), -> 7,263 base-O2O-present, -> 6,377 after 886 runner/alternative exclusions. The counts are mutually exclusive in this order.

`shifted identity unavailable` and `O2M positive_count==0` are reported as diagnostics in `P0_SUPPORT_FLOW.csv`; they are not inserted into the mutually exclusive Table 1 support flow.

## State answers

* **O2O identity fragility:** runner not intrinsically required; shifted identity null with a defined base and legal shift is an identity-departure/fragility event. Table 1 nevertheless inherits runner dependence through common-valid support.
* **O2M assigned-set Top-1 fragility:** the base Top-1 must be defined; a shifted Top-1 null differs from the base and is a fragility event. Runner is not intrinsic to the comparison, but Table 1 common-valid support requires the O2M runner.
* **Dense O2O first-departure:** requires only a defined base O2O identity and legal trajectory; runner is not required. The dense estimator code defines O2O rows from `o2o_base_active` alone.
* **Margin:** runner/positive alternative is required by definition; absent runner makes margin undefined.

For `base identity present + shifted identity absent`, the classification is **fragility/event**, not structural undefined, provided the shift is legal. Base identity unavailable is structural undefined/excluded; no legal shift is excluded because no comparison is observable.

## O2M empty-set distinction

The artifact has both `o2m_rank` and `o2m_positive_count`. `o2m_rank` is null for 112 rows in each contract, while `o2m_positive_count==0` occurs in 182 rows. They are not identical in the frozen output (including rows where positive count is zero but a rank probe is present), so they must not be conflated.

## Dense-grid descriptive risk set

`P0_DENSE_RISKSET_COUNTS.csv` reports counts at tau=0.125 from the existing 29,724 four-cardinal trajectories. The 2,511 first events, 195 geometric censors strictly before tau, 22,429 base-defined trajectories right-censored at rmax overall (the scan-limit label appears on 28,792 base-defined trajectories, including earlier-event trajectories), 28,853 exact endpoint-observable base-defined trajectories, and 14 return transitions are descriptive row counts only; no KM/AJ/RMCBD sensitivity was recomputed.

## Evidence and decision

The implementation and frozen summaries are internally consistent with the declared **common-valid O2O/O2M** Table 1 support. The only required action is clarification that runner availability enters through the support restriction, while O2O identity fragility itself is runner-free. Therefore the final status is **`PASS_CLARIFICATION_ONLY`**, not a scientific re-estimation request.

Primary source artifacts:

* `results/measurement_validation_20260831/fixed_branch_replay_v1/per_gt.csv` and `manifest.json`
* `results/reviewer_controls_20260829/formal_s0e300_k00625/per_gt.csv` and `manifest.json`
* `scripts/run_reviewer_killer_controls.py` and `scripts/margin_o2o_replay.py`
* `runs/20260902_boundary_full_grid_1024_population_v1/per_direction_boundary_events.csv` and `o2m_rank_set_radius_curve.csv`
* `evidence_freeze/final_evidence_v4/CURRENT_EVIDENCE_LEDGER.json` (EV-STRESS-CONTRACT-PRIMARY-V1)

Frozen dense-grid hashes: event CSV `ee33d0927ebbb9953a8e0e3f6c02956ef02cb6570beb3660e03ab172527e897d`; curve CSV `3622652665b30f727258aa0528436e33ab402255e29935dcc8168a32ea5f6b33`.
