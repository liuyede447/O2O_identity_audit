# v3.3.0 article-matched release

This release supersedes the immutable `v3.2.3` predecessor and is the article-matched package for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection** (post-audit corrected manuscript). The `v3.2.3` tag and its assets remain unchanged.

## What changed since v3.2.3

- **Corrected dense common-trajectory sensitivity (Supplementary Table).** `scripts/run_support_sensitivity.py` previously read index 0 of a 5,000-row bootstrap array as the point estimate; that function now prepends the identity-weight row, matching the production KM/RMCBD estimator (`scripts/analyze_assignment_boundary_geometry.py`). The corrected O2O-minus-O2M RMCBD contrasts are 0.01962 (overall), 0.01747 (8–16 px), and 0.03319 (16–32 px), replacing 0.02037 / 0.01707 / 0.03452. Corrected outputs and a bug-fix report are in `data/support_sensitivity_20260909/`. The primary dense-grid RMCBD numbers are unchanged.
- **Replay driver added.** `instrument_frozen/scripts/assignment_stability_audit_v2.py`, the read-only frozen-checkpoint assignment-state replay driver, is now included (it was omitted from earlier releases). It needs a GPU and AI-TOD-v2/VisDrone (obtained separately) to run.
- **Manuscript title** updated to "Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection".
- **Generated table refresh.** `sample_efficiency_1000.tex` merges the definitionally identical sign- and branch-order-recovery columns into one; `stress_contract_primary_full.tex` is transposed (one column per stress contract) to fit the page; `boundary_rmsr_summary.tex` uses the "assigned-set Top-1" label. No headline numbers changed.
- **Repository hygiene.** Stale figure/analysis-script path references to the pre-release private tree (`figures_final/`, `results/`, `runs/`) were corrected to this repository's `figures/` and `evidence/artifacts/{results,runs}/`; `python figures/src/plot_fig2_lowfi.py` regenerates Figure 2 end-to-end from a clean checkout. `docs/REPRODUCIBILITY.md`'s version string and its self-referential Level-2 instruction were fixed. The release manifests (`FILE_MANIFEST.csv`, `SHA256SUMS.txt`, `ARTICLE_MATCHED_RELEASE_SHA256.txt`) were regenerated from current tracked-file content.

## Assets and exclusions

Four protocol-eligible seed-0–3 checkpoints and the complete 1/1024 dense curve are attached as separate GitHub release assets with SHA-256 verification (see `RELEASE_ASSET_MANIFEST.csv`); the dense curve ships as four ordered `.part01`–`.part04` gzip parts joined with `scripts/join_dense_grid_asset.py`. AI-TOD-v2 and VisDrone images/annotations, raw prediction dumps, and image-bearing Figure 1/Figure S1 renders are not redistributed. Original code and derived data are AGPL-3.0-only.

## DOI

No Zenodo DOI is asserted in this release. Mint the version DOI from this exact GitHub `v3.3.0` release, then backfill it into `CITATION.cff`, `.zenodo.json`, `README.md`, the manuscript Data/Code Availability statement, the Supplement, and the reference list.
