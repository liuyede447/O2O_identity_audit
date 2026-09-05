# Synthetic lockbox analyzer CLI rehearsal

Status: **PASS**.

This post-failure software-QA rehearsal used only generated fixtures. It exercised the real analyzer CLI from sealed manifests and raw tables through zero-fill, common-support intersection, clustered bootstrap, Tier-A gatekeeping, Tier-B Holm correction, and `LOCKBOX_VERDICT.json`.

It retained 300 selected clusters, including zero-row and empty-stratum clusters; handled fixed-only and normalized-only GT keys; reproduced the expected synthetic verdict; produced identical scientific outputs after raw-row reversal; and rejected a sealed duplicate-key scenario.

No real outcome, checkpoint, project lockbox receipt, or scientific claim was accessed or changed. This does not validate or restore Lockbox v1.
