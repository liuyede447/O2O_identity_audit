# Stress-Dependent Cardinal Assignment Boundaries — reproducibility package

Article-matched release `v3.2.0` for **Stress-Dependent Cardinal Boundaries of Native Assignment States in Aerial Tiny-Object Detection**.

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint; cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py`, then use the released figure source data, generated manuscript tables, and dense-grid post-processing outputs to regenerate figures and tables.

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>. Configure an absolute dataset path from `configs/aitod_v2.template.yaml`, verify the four checkpoint assets against `RELEASE_ASSET_MANIFEST.csv`, prepend `instrument_frozen/ultralytics_local` to `PYTHONPATH`, and run the documented replay command in `docs/REPRODUCIBILITY.md`.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, resolved `optimizer=auto` state, and complete training lineage were **not preserved**; exact from-scratch training reproduction is therefore not claimed.

## Contents

The package includes the `final_evidence_v4` authority, claim-to-evidence map, generated numerical tables, figure source CSV/JSON and image-free Figure 2–6 PDFs/SVGs, manuscript/Supplement sources, dense-grid event/post-processing data, support map, checkpoint provenance, numerical QA, and reproduction commands. Figure 1 and Figure S1 contain third-party benchmark pixels and are represented by audit source records only; their image-bearing renders are not redistributed. The complete 1/1024 curve and four protocol-matched checkpoints are attached as GitHub release assets, not committed to Git history.

## Lockbox status

The historical Lockbox-v1 record is retained as governance and analyzer QA only: one permitted invocation terminated before any estimate or verdict, was not rerun, and post-access results remain exploratory with formal status `null`. Lockbox-v2 was abandoned during provenance QA before sample authorization/outcome access and is not a scientific release artifact.

## License and data

Original code, derived audit data, and author-trained checkpoints are released under AGPL-3.0-only. AI-TOD-v2 and VisDrone images/annotations are not redistributed; obtain them from the official providers above and comply with their terms. No Zenodo DOI is asserted in this release; the DOI must be minted from this exact `v3.2.0` release.
