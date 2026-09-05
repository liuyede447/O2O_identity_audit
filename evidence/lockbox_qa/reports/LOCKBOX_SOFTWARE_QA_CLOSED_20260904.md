# Lockbox software-QA closure — 2026-09-04

## Status

The repaired Lockbox analyzer software chain is closed for release-candidate QA. This record does not create scientific evidence, does not restore the failed prospective Lockbox v1, and does not authorise a Lockbox v2.

## Repaired analyzer

- Script: `scripts/analyze_confirmatory_lockbox.py`
- SHA-256: `3e587b048dc1b2c0465d4ef0215fb5a4012d479af6bdfd19d30db250d7c46905`

The authoritative full SHA-256 is recorded in the artifacts below; the analyzer self-test reports 33/33 PASS. The repaired output payload now records the selected-image cluster count and fixed/normalized common-support audit.

## Synthetic CLI end-to-end rehearsal

- Artifact: `results/measurement_validation_20260904/lockbox_analyzer_synthetic_cli_e2e_v2`
- Status: PASS
- Role: synthetic post-failure software QA; not scientific evidence
- Selected synthetic clusters: 300
- Bootstrap replicates: 99
- Support: fixed 597; normalized 597; intersection 594; fixed-only 3; normalized-only 3

The single harness command constructed isolated Git-sealed instrument and selection repositories, four raw-extraction roles, and a readiness artifact, then invoked the repaired analyzer through its real CLI. It covered a selected zero-row cluster, an empty stratum, an all-zero outcome cluster, unequal raw rosters, fixed-only and normalized-only GT keys, raw-row reversal, and duplicate-key rejection. The expected synthetic verdict was reproduced, row-reversal scientific payloads were identical, and the duplicate-key case created only its case-local synthetic receipt before failing closed without a verdict bundle.

No project data, checkpoint, Lockbox-v1 raw outcome, or Lockbox-v1 receipt was accessed by this rehearsal.

## Historical sealed-bundle parity

- Artifact: `results/measurement_validation_20260904/lockbox_repaired_analyzer_post_access_parity_v1`
- Status: PASS
- Role: post-access software QA; not a normative verdict
- Bootstrap replicates: 5,000; seed 20260831
- Support: fixed 7,246; normalized 7,243; intersection 7,243; fixed-only 3; normalized-only 0
- Maximum point difference: 0
- Maximum interval difference: 0
- Maximum empirical-p difference: 0
- Maximum bootstrap-trajectory difference: `9.71445146547012e-17`

The parity validator imported the repaired aggregation and bootstrap implementation without invoking the confirmatory CLI. It reproduced the frozen post-access corrective artifact within the prespecified `1e-15` tolerance. The historical outcome-access receipt SHA-256 remained `8a5eaee6703c75fec23ec287cf77db2f833a4d85540f21ac2c147b78a0464c83` before and after; its sidecar was also unchanged. No `LOCKBOX_VERDICT.json` was created.

## Scientific-status boundary

- Prospective Lockbox v1 remains invalidated due to the support-contract failure after irreversible outcome access.
- The post-access correction remains exploratory and excluded from the scientific claim ledger.
- No Lockbox v2 was selected, preregistered, or run.
- A future v2 would require separate author authorisation and a prior untouched-pool provenance audit.
