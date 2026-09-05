# Lockbox execution checklist

This checklist is normative after instrument freeze.

## Before selection

- [ ] Core discovery and instrument qualification are complete.
- [ ] `audit_instrument_v1.0` source commit and tag exist in the isolated freeze repository.
- [ ] `TOOL_FREEZE_MANIFEST.json` and its SHA-256 sidecar validate.
- [ ] `CONFIRMATORY_AUDIT_PLAN.json` and its SHA-256 sidecar validate with the frozen analyzer.
- [ ] `audit_preregistration_v1.0` resolves to the clean commit containing the exact preregistration and tool-manifest bytes.
- [ ] Tier A and Tier B families, bootstrap seed/replicates, alpha, and gatekeeping procedures are frozen.
- [ ] Any secondary family contains no more than two explicit hypotheses; otherwise it is absent.

## Selection

- [ ] The selector validates the instrument and preregistration hashes before reading the dataset roster.
- [ ] Exactly 300 outcome-blind images are drawn with the frozen stratum allocation.
- [ ] The selected image IDs have zero overlap with the discovery manifest.
- [ ] The selection manifest, selected CSV, and hashes are committed before extraction.
- [ ] `audit_lockbox_selection_v1.0` resolves to the clean selection repository HEAD.

## Sealed extraction

- [ ] Fixed branch raw extraction completed with `--sealed-extraction`.
- [ ] Normalized branch raw extraction completed with `--sealed-extraction`.
- [ ] Fixed anatomy raw extraction completed with `--sealed-extraction`.
- [ ] Normalized anatomy raw extraction completed with `--sealed-extraction`.
- [ ] Each manifest reports `sealed_extraction_complete`, raw row counts, and artifact hashes only.
- [ ] No effect estimate, interval, P value, or verdict has been printed or inspected.
- [ ] `validate_sealed_lockbox_bundle.py` writes `SEALED_BUNDLE_READY.json` and its SHA-256 sidecar only after revalidating all four raw hashes and the frozen instrument/preregistration/selection chain.

## Single outcome access

- [ ] All four raw manifest hashes are recorded.
- [ ] The analyzer requires `--sealed-ready`, `--selected-manifest`, and `--instrument-manifest`; bypassing the ready artifact is impossible.
- [ ] The ready sidecar and a freshly rebuilt sealed-chain payload match before any raw CSV is parsed.
- [ ] The analyzer creates one exclusive `OUTCOME_ACCESS_RECEIPT.json` before the first raw CSV read; an existing receipt blocks every later invocation regardless of output directory.
- [ ] One UTC outcome-access timestamp is recorded.
- [ ] The confirmatory analyzer validates all frozen gates before reading raw outcomes.
- [ ] The analyzer writes one immutable verdict bundle and retains every H1--H5 result regardless of PASS/FAIL.
- [ ] A Tier A gate failure leaves later estimates descriptive, not silently discarded.
- [ ] Tier B and any secondary family use the frozen Holm procedure.

## After outcome access

- [ ] No estimator-affecting code or definition is modified under a prospective label.
- [ ] Any defect invalidates the original affected result and makes corrections post-lockbox exploratory.
- [ ] Lockbox failure triggers claim downgrading, never resampling.
