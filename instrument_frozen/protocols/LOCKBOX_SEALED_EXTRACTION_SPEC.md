# Sealed lockbox extraction specification

Status: **pre-freeze implementation requirement**.

The prospective lockbox is not opened when its image manifest is generated or when raw replay rows are extracted. Outcome access begins only when the preregistered confirmatory analyzer reads all completed raw branch and anatomy artifacts.

## Permitted extraction output

- progress counts;
- image and row counts;
- errors;
- source and artifact hashes;
- a manifest status of `sealed_extraction_complete`.

## Prohibited before the single analyzer run

- effect estimates;
- confidence intervals or empirical P values;
- hypothesis verdicts;
- branch or scale summaries;
- pathway proportions;
- manual inspection of outcome columns.

The fixed/normalized branch extractors must emit `per_direction.csv` and `per_gt.csv` only. The fixed/normalized anatomy extractors must emit `per_direction_anatomy.csv` and `pairwise_q_gap.csv` only. Summary computation and all H1--H5 decisions belong exclusively to the preregistered analyzer.

Before selection, the source-only instrument commit must carry `audit_instrument_v1.0`, and the subsequent clean commit containing the finalized preregistration/tool-manifest bytes must carry `audit_preregistration_v1.0`. The selection bundle is committed immediately and tagged `audit_lockbox_selection_v1.0`. All four sealed extraction manifests must be complete and hashed before outcome access. `validate_sealed_lockbox_bundle.py` must revalidate all three Git seals, the instrument, preregistration, selection, extraction manifests, and exact raw-artifact hashes, then write `SEALED_BUNDLE_READY.json` plus `SEALED_BUNDLE_READY.json.sha256`. The confirmatory analyzer is required to consume that ready artifact, its sidecar, the selection manifest, and the instrument manifest; it must rebuild and compare the entire non-interpretive hash chain before reading the first raw outcome CSV. After all preflight checks it must create `OUTCOME_ACCESS_RECEIPT.json` atomically with exclusive-create semantics. The receipt is irreversible: if the analyzer crashes afterward, the lockbox is consumed and cannot be rerun prospectively. Only then may the analyzer read raw outcomes and write `LOCKBOX_VERDICT.json` and `LOCKBOX_REPORT.md`. Any later estimator-affecting code correction invalidates the prospective status of the affected estimand; the original output remains retained as `invalidated`, and the correction is labelled post-lockbox exploratory.
