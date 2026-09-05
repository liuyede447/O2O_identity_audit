# Repaired analyzer / Lockbox-v1 corrective parity

Status: **PASS**.

The repaired aggregation and bootstrap implementation reproduced all five frozen post-access exploratory point estimates, confidence intervals, empirical p-values, 5,000 bootstrap trajectories, and support counts within an absolute tolerance of 1e-15.

The historical outcome-access receipt and its sidecar were byte-hash unchanged. No confirmatory CLI was invoked and no `LOCKBOX_VERDICT.json` was created. Lockbox v1 remains invalidated and the comparison is software QA only.
