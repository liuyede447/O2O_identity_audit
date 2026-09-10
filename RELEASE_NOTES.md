# v3.3.2 article-matched release

This release supersedes the immutable `v3.3.1` predecessor and is the article-matched package for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection** (post-audit corrected manuscript). The `v3.3.1` tag and its assets remain unchanged.

## What changed since v3.3.1

- **Metadata synchronization only.** Version identifiers and Zenodo/GitHub release wording were synchronized for v3.3.2. Scientific code, evidence, derived data, figure sources, dense-grid records, checkpoints, and their hashes are unchanged from v3.3.1.

## Assets and exclusions

Four protocol-eligible seed-0–3 checkpoints and the complete 1/1024 dense curve are attached as separate GitHub release assets with SHA-256 verification (see `RELEASE_ASSET_MANIFEST.csv`); the dense curve ships as four ordered `.part01`–`.part04` gzip parts joined with `scripts/join_dense_grid_asset.py`. AI-TOD-v2 and VisDrone images/annotations, raw prediction dumps, and image-bearing Figure 1/Figure S1 renders are not redistributed. Original code and derived data are AGPL-3.0-only.

## DOI

The corresponding Zenodo version is generated from this GitHub release. Use the version DOI displayed on the Zenodo record; the concept DOI remains stable across versions.
