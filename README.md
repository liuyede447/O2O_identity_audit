# Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection

Reproducibility materials for an audit of how controlled annotation shifts affect label assignment in aerial tiny-object detectors. The audit freezes detector predictions and replays each native assignment rule. It reports candidate eligibility, positive-set membership and ranking as distinct measurements.

[GitHub release v3.4.2](https://github.com/liuyede447/O2O_identity_audit/releases/tag/v3.4.2) · [Zenodo collection](https://doi.org/10.5281/zenodo.22478276) · [Revision reproduction guide](revision/README.md)

## Reproduce results from archived inputs

Download the source archive from the release or clone the repository at the release tag. The revision calculations use Python 3.13.5 and the dependency versions pinned in `revision/requirements.txt`. They start from archived audit tables and run on a CPU without benchmark images or model weights.

```sh
python scripts/verify_release.py
python -m pip install -r revision/requirements.txt
python -B revision/rerun_revision_analyses.py --output-dir rerun_check
```

For the new margin replication and coordinate controls in Supplementary Tables S51–S55, see the [additional analysis guide](revision/additional_checks/README.md).

For the base-margin results in Supplementary Tables S49–S50, see the [CPU reproduction instructions](revision/margin_screen/README.md).

The [reproduction guide](revision/README.md) maps tables to commands, describes the 600-image validation replication and gives instructions for rebuilding Figures 2–5. It also identifies the analyses that require source datasets. Full detector replay and exact repetition of training are separate tasks. The released training records do not preserve everything needed for exact training reproduction.

## Obtain the datasets

Download images and labels from the original providers and follow their terms. Keep the resulting data and any conversion outputs in your own data directory.

| Dataset | Source and preparation |
| --- | --- |
| AI-TOD-v2 | Obtain the images and official test annotations from the [AI-TOD-v2 project](https://github.com/Chasel-Tsui/AI-TOD-v2). For the official-validation replication, use the v2 annotations in the [dataset authors' archive at revision `1591be1`](https://huggingface.co/datasets/jwwangchn/AI-TOD/commit/1591be1330a5dc99b646b3599f67437a2de246ae). The original discovery audit uses the official test split. The 600-image replication samples the separate official validation split. |
| VisDrone | Obtain the image-based object detection data (VisDrone2019-DET) from the [VisDrone project](https://github.com/VisDrone/VisDrone-Dataset). |

For the AI-TOD-v2 replay, set the dataset root and split paths in a local copy of [the dataset template](configs/aitod_v2.template.yaml). The validation reproduction guide links the converter and identifies the required annotation archive and file hashes. The dataset licence terms are separate from the licence for this repository.

## Checkpoints and full replay

Download model checkpoints and the four dense-grid archive parts from the [release assets](https://github.com/liuyede447/O2O_identity_audit/releases/tag/v3.4.0). Verify their SHA-256 values against the [asset manifest](RELEASE_ASSET_MANIFEST.csv). The dense-grid parts can be combined with [the join script](scripts/join_dense_grid_asset.py).

The [replay driver](instrument_frozen/scripts/assignment_stability_audit_v2.py) requires a compatible GPU environment. See the [reproduction guide](revision/README.md) for the pinned validation checkpoint, dataset conversion and replay commands.

## Citation and licence

Cite the version DOI associated with the release you use. Citation metadata is provided in [CITATION.cff](CITATION.cff). The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.22478276) identifies the full version series.

This repository uses [GNU AGPL-3.0-only](LICENSE). See [the licence scope](LICENSE_SCOPE.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for details. The archive provides derived audit records and figure sources. Obtain benchmark images and raw annotations from their providers.
