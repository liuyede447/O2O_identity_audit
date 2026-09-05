# Native assignment-state boundary audit

Reproducibility package for **Measuring Native Assignment-State Boundary Geometry in Aerial Tiny-Object Detection**.

Version: `3.0.0`. Git tag: `v3.0.0`. No DOI is asserted until Zenodo mints one.

This package implements a read-only audit of native O2O and assigned-positive O2M states. It moves only a focal ground-truth centre while detector outputs, parameters, candidate grids, object extent and all other ground truths remain frozen. It does not retrain a detector or identify a causal effect.

## Evidence status

- **Discovery:** primary mechanistic estimates use the audited AI-TOD-v2 seed-0 checkpoint.
- **Sensitivity/replication:** cross-dataset, archived-stage, compatible detector-contract and three-seed analyses have those narrower roles; they are not pooled into the primary estimate.
- **Instrument qualification:** analytical oracles, property tests, mutations, native-path parity and exact full-grid replay validate the measurement implementation.
- **Boundary evidence v3:** 300 selected images, 7,431 focal GTs, 29,724 directional trajectories and 7,593,288 curve rows on a complete legal `1/1024` cardinal grid. Cardinal-ray distance is not a two-dimensional minimum radius.

The O2M assigned-positive Top-1 state is a rank summary, not the full positive set. Native margin is a local coordinate and is not presented as a calibrated predictor.

## Lockbox boundary

The only prospective analyzer invocation produced an irreversible access receipt and then terminated before H1--H5 estimates or decisions. Therefore the Lockbox has **no prospective PASS/FAIL verdict** and was not rerun. Later synthetic CLI rehearsal and old-bundle parity are **post-access software QA only**; they do not repair, replace or upgrade the original confirmatory evidence.

## Repository contents

- `instrument_frozen/`: the frozen v1.0-v3 measurement snapshot, including the modified Ultralytics 8.4.51 source tree; bundled sample images are excluded.
- `scripts/`: manuscript-synchronised analysis, dense-grid and post-access QA code.
- `evidence/final_evidence_v3/`: the active authority and evidence maps.
- `evidence/artifacts/`: hash-verified compact source artifacts cited by the evidence map.
- `data/dense_grid_v3/`: event-level table and six-τ post-processing outputs; the 1.411 GB curve is a separate release asset.
- `figures/`: source data, code and image-free rendered Figures 2--6.

Raw AI-TOD-v2/VisDrone images and annotations, model checkpoints, raw predictions and the dense curve are not in the ordinary repository. See `docs/RELEASE_ASSETS.md` and `THIRD_PARTY_NOTICES.md`.

## Verify

```bash
python scripts/verify_release.py
```

The release builder creates `SHA256SUMS.txt` and `FILE_MANIFEST.csv`. The verifier checks every listed byte count and SHA-256 and enforces the repository exclusion policy.

## Reproduce compact results

Install `requirements.txt`, set `PYTHONPATH=instrument_frozen/ultralytics_local`, obtain the datasets from their official sources, and verify any downloaded checkpoint against the release-asset manifest. Commands are documented in `docs/REPRODUCIBILITY.md`. Do not substitute legacy coarse/fine boundary artifacts for v3.

## Licence

This release applies AGPL-3.0-only to the software, derived tabular data, and author-trained checkpoints to remain compatible with the integrated Ultralytics source. AI-TOD-v2 and VisDrone images and annotations are excluded. See `LICENSE_SCOPE.md`. No DOI is asserted until Zenodo mints one.
