# Data and Code Availability release record

Status date: 2026-09-06. Article-matched GitHub v3.2.0 publication is being completed; Zenodo DOI minting remains pending.

## Current manuscript wording

AI-TOD-v2 and VisDrone are third-party datasets available from their official repositories under the providers' access and licence terms. Raw third-party images and annotations are not redistributed. The article-matched manuscript code, derived data, figure source data, evidence records, complete 1/1024 dense curve, and four trained checkpoints are released in GitHub `v3.2.0` at <https://github.com/liuyede447/O2O_identity_audit/releases/tag/v3.2.0>, with asset digests recorded in the release manifest. Raw prediction dumps are not redistributed, and reproduction from source pixels requires obtaining the benchmark datasets under their providers' terms. A Zenodo DOI has not yet been minted.

## Verified release facts

- Public release: `v3.2.0`; not a draft and not a prerelease.
- Publication time and commit: recorded in the immutable GitHub release after tag creation.
- Public assets: five scientific assets (four checkpoints and the complete dense curve); sizes and SHA-256 digests are recorded in `RELEASE_ASSET_MANIFEST.json`.
- Full public-download verification: four checkpoints and the 1,411,448,003-byte dense curve all match their local SHA-256 values.
- Licence decision: AGPL-3.0-only for the released software, derived data, and checkpoint assets.
- Exclusions: raw AI-TOD-v2/VisDrone images and annotations, and raw prediction dumps.

## DOI backfill after Zenodo publication

Replace the final sentence with the minted DOI, add the DataCite-style software citation to the bibliography, update `CITATION.cff`, `.zenodo.json`, and the submission portal research-data field, then test the DOI outside the author account. Do not invent or reserve a placeholder DOI in the manuscript.
