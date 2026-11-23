# v3.4.1

Adds a CPU reproduction of the base-margin analyses in Supplementary Section 18
and Tables S49–S50. Includes the minimum derived inputs, pinned dependencies,
the analysis code and five archived output tables with a verification command.

The checkpoints and complete dense-grid records are available as [v3.4.0 release
assets](https://github.com/liuyede447/O2O_identity_audit/releases/tag/v3.4.0).

# v3.4.0 release for the revised article

This release builds on v3.3.3 and adds `revision/`, the reproducibility bundle for the results added in the first revision (main Tables 3, 4, 5, 7 and 8, Figures 2-5, and Supplementary Tables S39-S48 and Section 17).

- `revision/rerun_revision_analyses.py` recomputes the revision estimates and their bootstrap intervals from `revision/revision_inputs.tar.gz` and checks every output against the archived copy. It needs numpy and pandas only.
- The runner option `--only s40_per_direction` reproduces the RFLA/ATSS per-direction estimates and bootstrap intervals in Results and Table S40.
- `revision/figures/` regenerates Figures 2-5, including the `o2m_positive_count_scale_controls_v5` records that v3.3.3 lacked for Figure 5.
- The four final-epoch (epoch 300) checkpoints are new release assets; their SHA-256 values are in `RELEASE_ASSET_MANIFEST.csv` and `revision/README.md`.
- `CITATION.cff` and `.zenodo.json` now carry Fang Wang's ORCID, and `.zenodo.json` declares this record a new version of `10.5281/zenodo.22744638` (v3.3.3).

Unchanged: all v3.3.3 code, evidence, derived data and release assets.

# v3.3.3 article-matched release

This release supersedes the immutable `v3.3.2` predecessor. It is the article-matched package for the corrected manuscript **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection**. The `v3.3.2` tag and its assets remain unchanged.

## What changed since v3.3.2

- **Figure 3 matches the article.** In v3.3.2, the Figure 3 files and their source table predated the post-submission correction.
  - `figures/rendered/Fig3.*` are now the files used in the corrected manuscript.
  - Every row of `figures/source_data/fig3_source.csv` is now bound to the base-defined correction outputs.
  - `figures/src/rebuild_fig3_corrected.py` regenerates the table and the figure from those outputs. On the reference stack, its PNG is byte-identical to the article figure.
  - `figures/src/plot_fig3.py` now holds the corrected layout. Its `main()` records the original `final_evidence_v3` binding and is no longer an entry point.
- **Post-submission correction added.** `data/post_submission_correction_20260908/` contains the frozen input rows, the six unmodified correction scripts, and the outputs behind the corrected eligibility-lock, eligibility-stable, first-divergence/any-stage, and pre-resolution estimates. Its `README.md` gives the re-run commands and the result of re-running them.
- **Support-sensitivity inputs added.** `data/support_sensitivity_20260909/inputs/` contains the two per-GT endpoint files. Supplementary Tables S37 and S38 now re-run from the repository alone.
- **Missing pinned source added.** `figures/FIGURE_EVIDENCE_MANIFEST.json` pins `evidence/artifacts/runs/20260901_first_divergence_cooccurrence_v1/`, but v3.3.2 did not include it.
- **Evidence authority byte-exact again.** v3.3.2 stored the `evidence/final_evidence_v4/` files with converted line endings. They are now the frozen originals (CRLF), so `FINAL_EVIDENCE_MANIFEST.json` has the SHA-256 cited in the Supplement (`00513d01…`).
- **Checksums valid on every platform.** v3.3.2's manifests were computed on a Windows (CRLF) checkout, so 496 entries failed on Linux, in the GitHub source archive, and in the Zenodo zip.
  - `.gitattributes` (`* -text`) now turns off line-ending conversion.
  - `FILE_MANIFEST.csv`, `SHA256SUMS.txt`, `ARTICLE_MATCHED_RELEASE_SHA256.txt`, and `manuscript/article_science_freeze_v1/CONSOLIDATED_MANIFEST.json` are now computed on the committed bytes.
- **Metadata corrected.** v3.3.2's `CITATION.cff`, `.zenodo.json`, and `ARTICLE_MATCHED_RELEASE_MANIFEST.json` recorded `10.5281/zenodo.22683896`, which is not this record's concept DOI.
  - The all-versions concept DOI is now `10.5281/zenodo.22478276`.
  - `.zenodo.json` declares this record a new version of `10.5281/zenodo.22684364` (v3.3.2).
- **Housekeeping.**
  - `manuscript/generated_final_v2/SUPERSEDED_VALUES.md` lists the pre-correction entries that remain in the frozen generation snapshot.
  - The figure evidence manifests use the article's wording for the RMCBD question, and their Figure S1 trace paths are fixed.
  - The stale internal `RELEASE_CHECKLIST.md` (v3.2.0 wording) was removed.
  - `scripts/verify_release.py` now ignores `__pycache__/`, so it still passes after the analysis or figure scripts have been run.
  - `README.md` and `docs/REPRODUCIBILITY.md` were updated.

Unchanged: the eight release assets (checkpoints and dense-grid parts, with the same SHA-256 values), the primary Table 1 endpoint, the Rank-Set and dense-boundary data, and all other code, evidence, and derived data.

## Assets and exclusions

Four protocol-eligible seed-0–3 checkpoints and the complete 1/1024 dense curve are attached as separate GitHub release assets, each with SHA-256 verification (see `RELEASE_ASSET_MANIFEST.csv`). The dense curve ships as four ordered gzip parts, `.part01`–`.part04`, which `scripts/join_dense_grid_asset.py` joins. The following are not redistributed: AI-TOD-v2 and VisDrone images and annotations, raw prediction dumps, and the image-bearing Figure 1 and Figure S1 renders. Original code and derived data are AGPL-3.0-only.

## DOI

Zenodo generates the corresponding version from this GitHub release. Cite the version DOI shown on that record. The all-versions concept DOI, [10.5281/zenodo.22478276](https://doi.org/10.5281/zenodo.22478276), stays the same across versions.
