# P0 Runner / positive-alternative definition (read-only audit)

## Exact executable definitions

The endpoint writer is `scripts/run_reviewer_killer_controls.py`; it imports `active_candidate_and_margin` and `legal_shift` from `scripts/margin_o2o_replay.py`.

* **O2O assigned identity:** `active = where(fg_mask & target_gt_idx == gt)`. It is defined only when exactly one native post-conflict assigned anchor exists.
* **O2O positive alternative / runner:** `eligible = where(scores > 0)` for the same GT; remove the native assigned anchor; select the remaining candidate with maximal alignment score. If the native alignment score is `<=1e-12` or no alternative remains, runner and margin are `None`.
* **O2M assigned-set Top-1 probe:** `eligible = where(scores > 0)` for the same GT, ordered descending; first is `o2m_rank`, second is `o2m_runner`; fewer than two eligible candidates makes the runner/margin `None`.
* **Common-valid support:** endpoint code writes `common_valid_margin = int(margin is not None and m_margin is not None)`. Therefore Table 1's common-valid O2O/O2M support requires an O2O runner and an O2M runner, even though the intrinsic identity-fragility booleans only compare base and shifted identities.

Source hashes:

* `results\measurement_validation_20260831\fixed_branch_replay_v1\per_gt.csv` SHA-256 `0596d59486c5da70a577a6928f9b14fddc6c1ba115511e20486175dabada0d7a`
* `results\reviewer_controls_20260829\formal_s0e300_k00625\per_gt.csv` SHA-256 `bf6d2f87ec2b0922427870218d3341fb0d33a45b8d8794591e927bc0f35993de`
* `scripts/run_reviewer_killer_controls.py` was inspected; no detector/replay was executed in this audit.
