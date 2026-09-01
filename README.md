# O2O Identity Audit

Reproducibility package for **Validated Read-Only Geometric Stress Testing of Native Assignment States in Aerial Tiny-Object Detection**.

This repository measures how frozen native O2O and O2M assignment states respond when only one focal ground-truth centre is replayed under controlled geometric shifts. It does not modify inference, retrain a detector, or claim a validated assignment intervention.

## What is included

- `manuscript/`: the current main manuscript and Supplementary Material PDFs.
- `figures/`: Figures 1–5 as vector PDF/SVG, 600 dpi PNG, source scripts, and validation records.
- `results/stress/`: fixed-pixel, equivalent-side, magnitude, dataset, detector-contract, and archived-stage summaries.
- `results/eligibility_pathway/`: eligibility-lock, eligibility-stable, first-divergence, and pre-resolution controls.
- `results/boundary/`: survival, cumulative-incidence, RMSR, rank-crossing status, and margin-construct tables.
- `results/rank_set/`: O2M Rank–Set and transition-decomposition summaries.
- `results/margin/`: nonlinear shape sensitivity and held-out incremental-value results.
- `results/locality_budget/`: spillover and 1,000-repeat sample-efficiency results.
- `data/per_object/`: curated per-object tables, selected-image IDs, and manifests for the principal fixed/equivalent-side, VisDrone, and YOLOv10 contracts.
- `evidence/`: final evidence ledger, result-to-claim mapping, exclusions, manifest, and SHA-256 inventory.
- `scripts/`: the analysis and validation entry points used for the included derived results.
- `docs/ALL_RESULTS.md`: all Goal-mode results, including positive, null, negative, excluded, and invalidated records.

## Evidence status

The final evidence authority is `evidence/FINAL_EVIDENCE_MANIFEST.json`. The corresponding manifest SHA-256 is:

```text
4f626da49dcd7ee87ffd33311e9c08c73629e2b7b4cb0b2c7e74501a7a5a1ca9
```

The prospective lockbox produced **no normative verdict**. Its only permitted analyzer invocation wrote the outcome-access receipt and then terminated before any H1–H5 estimate or decision. Corrective values in the package are explicitly post-access exploratory and do not recover confirmatory status.

## Reproduction outline

1. Create a Python environment and install `requirements.txt`.
2. Obtain AI-TOD-v2 and/or VisDrone from their official repositories; raw benchmark files are not redistributed here.
3. Verify package integrity with `python scripts/verify_release.py` after generating `SHA256SUMS.csv`, or compare against the included SHA-256 inventory.
4. Run the relevant analysis script against the frozen derived inputs described in the evidence ledger.
5. Regenerate Figures 2–5 with `python figures/make_final_result_figures.py` after adjusting its project root to the local checkout.

The internal audit used frozen checkpoint identities recorded by SHA-256. Model binaries, raw prediction dumps, benchmark images, and third-party annotations are intentionally excluded.

## Data access

- AI-TOD-v2: <https://github.com/Chasel-Tsui/AI-TOD-v2>
- VisDrone: <https://github.com/VisDrone/VisDrone-Dataset>

## Citation

Use the metadata in `CITATION.cff`. For an archival DOI, create a tagged GitHub release and archive that release with Zenodo.

## Contact

Corresponding author: Jun Li, `lijun2022@sicnu.edu.cn`.
