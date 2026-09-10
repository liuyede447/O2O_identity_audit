# Stress-Dependent Assignment Stability — reproducibility package

Article-matched release `v3.3.0` for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection**. This release supersedes the immutable `v3.2.3` predecessor. No Zenodo DOI is asserted yet; it is minted from this exact GitHub release and then backfilled (see `RELEASE_NOTES.md`).

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint; cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py` (expected: `PASS: ... files verified`). Then regenerate figures and tables from the released source data. `python figures/src/plot_fig2_lowfi.py` regenerates Figure 2 end-to-end from this repository alone and is the one to try first. Figures 3, 4, 6, and S1 additionally need the complete `1/1024` dense-grid asset (four GitHub release parts `.part01`-`.part04`, joined with `scripts/join_dense_grid_asset.py`); Figure 5 additionally needs the `o2m_positive_count_scale_controls_v5` analysis run, which is not in this snapshot. Those `plot_fig*.py` scripts fail with an explicit `FileNotFoundError` naming the missing input until that data is in place.

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>, configure an absolute dataset path from `instrument_frozen/configs/aitod_v2.template.yaml`, and verify the four checkpoint assets against `RELEASE_ASSET_MANIFEST.csv`. The read-only replay driver is `instrument_frozen/scripts/assignment_stability_audit_v2.py` (needs a GPU); run it with `--checkpoint <asset> --expected-sha256 <from RELEASE_ASSET_MANIFEST.csv> --data <your configured yaml> --output-dir <dir>`. Its outputs feed the downstream `scripts/analyze_*.py` analyses. Full end-to-end equivalence to the frozen per-object/per-direction intermediates should be confirmed in your environment, since it depends on the GPU/runtime stack.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, resolved `optimizer=auto` state, and complete training lineage were **not preserved**; exact from-scratch training reproduction is therefore not claimed.

## Contents

The package includes the `final_evidence_v4` authority, claim-to-evidence map, generated numerical tables (including the corrected dense common-trajectory sensitivity in `data/support_sensitivity_20260909/`), figure source CSV/JSON and image-free Figure 2–6 PDFs/SVGs, dense-grid event/post-processing data, support map, checkpoint provenance, numerical QA, the read-only replay driver and downstream analysis scripts, and reproduction commands. Figure 1 and Figure S1 contain third-party benchmark pixels and are represented by audit source records only; their image-bearing renders are not redistributed. The complete 1/1024 curve and four protocol-matched checkpoints are attached as GitHub release assets, not committed to Git history. The dense-grid gzip is uploaded as four ordered parts (`.part01`-`.part04`); concatenate them in order, then verify the manifest SHA-256 (the repository includes `scripts/join_dense_grid_asset.py`).

## Lockbox status

The historical Lockbox-v1 record is retained as governance and analyzer QA only: one permitted invocation terminated before any estimate or verdict, was not rerun, and post-access results remain exploratory with formal status `null`. Lockbox-v2 was abandoned during provenance QA before sample authorization/outcome access and is not a scientific release artifact.

## License and data

Original code, derived audit data, and author-trained checkpoints are released under AGPL-3.0-only. AI-TOD-v2 and VisDrone images/annotations are not redistributed; obtain them from the official providers above and comply with their terms.

This release intentionally excludes main.tex, supplementary.tex, main.pdf, supplementary.pdf, author submission files, and image-bearing Figure 1/Figure S1 renders.
