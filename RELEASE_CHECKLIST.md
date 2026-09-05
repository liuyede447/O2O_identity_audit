# Release checklist

## Completed in this local candidate

- [x] Ordinary repository assembled by explicit whitelist.
- [x] v3 evidence authority and compact cited artifacts copied with source SHA checks.
- [x] Active 300-image event table and all six-τ post-processing outputs included.
- [x] 1/1024 dense curve excluded from Git history and staged as a separate release asset.
- [x] Seed 0--3 checkpoints excluded from Git history and staged as separate release assets.
- [x] Raw datasets, annotations, third-party figure pixels and prediction dumps excluded.
- [x] Lockbox no-verdict record separated from post-access software QA.
- [x] `CITATION.cff` and `.zenodo.json` contain no invented DOI.
- [x] `VERSION` and `RELEASE_TAG` record the requested candidate identifiers.
- [x] File manifest and SHA-256 verification generated.

## Author and rights decisions

- [x] All authors/institution approve AGPL-3.0-only for original code and modified Ultralytics redistribution.
- [x] Checkpoint redistribution rights approved; all four checkpoints are release assets.
- [x] Third-party datasets remain excluded; only identifiers and derived statistics are released.
- [x] GitHub Release selected for the 1.411 GB curve and four 60 MB checkpoints; Zenodo binding follows publication.
- [x] Final version and tag are `3.0.0` and `v3.0.0`, newer than public `v2.0.0`.

## Required during and after public upload

- [ ] Create the final Git tag from an isolated clean clone of the public repository.
- [ ] Mint an archive DOI, if desired, and only then add the real DOI to metadata and manuscript.
- [ ] Update manuscript Data/Code Availability from “release candidate” to the actual immutable URL/tag/DOI.
- [x] Re-run the builder and verifier after final code, figure-source and evidence-document changes.
- [ ] Inspect the public Git tree and release assets once uploaded; compare every server-side download hash.
