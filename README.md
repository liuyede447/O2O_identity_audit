# Stress-Dependent Assignment Stability — reproducibility package

Release `v3.4.0` for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection**, matching the revised article. It keeps the `v3.3.3` package unchanged and adds `revision/`, which reproduces the results added in the first revision; `revision/README.md` maps each revised table and figure to its command. Zenodo archives this release. Cite the version DOI shown on its Zenodo record. The all-versions concept DOI is [10.5281/zenodo.22478276](https://doi.org/10.5281/zenodo.22478276); `RELEASE_NOTES.md` lists what changed.

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated, and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint. The cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication analyses.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py` (expected: `PASS: ... files verified`), then regenerate figures and tables from the released source data. Two figures regenerate end to end from this repository alone:

- `python figures/src/plot_fig2_lowfi.py` regenerates Figure 2.
- `python figures/src/rebuild_fig3_corrected.py` re-derives Figure 3 from the base-defined correction outputs in `data/post_submission_correction_20260908/`. Its `README.md` explains how to recompute those outputs from the frozen rows.

Run `verify_release.py` before you regenerate figures. `plot_fig2_lowfi.py` rewrites the tracked `figures/draft/Fig2.{pdf,svg}` and `figures/qa/Fig2.validation.json`. The PDF and SVG embed a creation timestamp, so their hashes change, although the PNG is identical. `rebuild_fig3_corrected.py` reproduces its tracked outputs byte for byte.

The corrected support-sensitivity tables (Supplementary Tables S37 and S38) re-run with `scripts/run_support_sensitivity.py`; see `data/support_sensitivity_20260909/inputs/README.md`.

Figures 4, 6, and S1 also need the complete `1/1024` dense-grid asset. It ships as four GitHub release parts, `.part01`–`.part04`, which you join with `scripts/join_dense_grid_asset.py`. The `o2m_positive_count_scale_controls_v5` records that Figure 5 needs are in `revision/figures/analysis_root/`; `revision/figures/render_fig5_s1_revision.py` draws the article's Figure 5 from them. Until the dense-grid asset is in place, the other `plot_fig*.py` scripts stop with an explicit `FileNotFoundError` that names the missing input.

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>. Configure an absolute dataset path from `instrument_frozen/configs/aitod_v2.template.yaml`, and verify the checkpoint assets against `RELEASE_ASSET_MANIFEST.csv` (four epoch-280 discovery checkpoints and, for the revision analyses, four final-epoch checkpoints). The read-only replay driver is `instrument_frozen/scripts/assignment_stability_audit_v2.py` and needs a GPU. Run it with `--checkpoint <asset> --expected-sha256 <from RELEASE_ASSET_MANIFEST.csv> --data <your configured yaml> --output-dir <dir>`. Its outputs feed the downstream `scripts/analyze_*.py` analyses. End-to-end equivalence to the frozen per-object and per-direction intermediates depends on the GPU and runtime stack, so confirm it in your environment.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, the resolved `optimizer=auto` state, and the complete training lineage were **not preserved**, so exact from-scratch training reproduction is not claimed.

## Contents

The package includes:

- the `final_evidence_v4` authority and the claim-to-evidence map;
- generated numerical tables, including the corrected dense common-trajectory sensitivity in `data/support_sensitivity_20260909/`;
- the post-submission base-defined correction in `data/post_submission_correction_20260908/` (inputs, unmodified scripts, and outputs);
- figure source CSV/JSON files and image-free PDFs/SVGs for Figures 2–6;
- dense-grid event and post-processing data, the support map, checkpoint provenance, and numerical QA;
- the read-only replay driver, the downstream analysis scripts, and reproduction commands.

Pre-correction entries that remain in the frozen generation snapshot are listed in `manuscript/generated_final_v2/SUPERSEDED_VALUES.md`.

Figure 1 and Figure S1 contain third-party benchmark pixels. Only their audit source records are included; the image-bearing renders are not redistributed.

The complete 1/1024 curve, the four epoch-280 checkpoints and the four final-epoch checkpoints are attached as GitHub release assets and are not committed to Git history. The dense-grid gzip ships as four ordered parts (`.part01`–`.part04`). Concatenate them in order with `scripts/join_dense_grid_asset.py`, then check the result against the SHA-256 in the manifest.

## Line endings and checksums

`.gitattributes` turns off line-ending conversion. A clone on any platform, the GitHub source archive, and the Zenodo zip therefore all contain the committed bytes. `SHA256SUMS.txt`, `FILE_MANIFEST.csv`, and `ARTICLE_MATCHED_RELEASE_SHA256.txt` are computed on those bytes, so `sha256sum -c SHA256SUMS.txt` checks the whole tree. The frozen `evidence/final_evidence_v4` files were written on Windows with CRLF line endings and are shipped unchanged. Their own `SHA256SUMS.txt` also uses CRLF, so from inside that folder check it with `tr -d '\r' < SHA256SUMS.txt | sha256sum -c`.

## Lockbox status

The historical Lockbox-v1 record is kept as governance and analyzer QA only. Its one permitted invocation stopped before producing any estimate or verdict and was not rerun. Its post-access results remain exploratory, with formal status `null`. Lockbox-v2 was abandoned during provenance QA, before sample authorization or outcome access, and is not a scientific release artifact.

## License and data

Original code, derived audit data, and author-trained checkpoints are released under AGPL-3.0-only. AI-TOD-v2 and VisDrone images and annotations are not redistributed. Obtain them from the official providers above and comply with their terms.

This release intentionally excludes main.tex, supplementary.tex, main.pdf, supplementary.pdf, author submission files, and the image-bearing Figure 1 and Figure S1 renders.
