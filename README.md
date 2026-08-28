# Loss-active O2O identity audit

Reproducibility package for the manuscript **“Scale-Conditioned Loss-Active
One-to-One Identity Instability in Aerial Tiny-Object Detection.”**

The repository supports the paper's diagnostic claims about native one-to-one
(O2O) loss-active identity stability, active-relative margin, undefined
competition, same-model O2M/O2O specificity, cross-dataset replication, and
the tested intervention boundary.

## What is included

- read-only O2O and O2M replay/audit scripts;
- outcome-blind random-stratified image manifests;
- per-ground-truth derived audit tables for AI-TOD-v2 and VisDrone;
- same-model YOLO26 O2M/O2O comparison data;
- strict same-stride post hoc sensitivity code and summary;
- YOLOv10-S architecture-replication data;
- three 300-epoch YOLO26 seed audits;
- checkpoint-row association matrix and bootstrap summaries;
- manuscript-facing source tables and a deterministic Figure 2 rebuild;
- SHA-256 identifiers for pretrained and author-trained model artifacts.

## What is not included

- AI-TOD-v2 or VisDrone images and annotations;
- pretrained or author-trained `.pt` files;
- raw detector prediction dumps;
- internal development queues, server paths, or unrelated failed experiments;
- the qualitative source image used in Figure 4.

These exclusions avoid redistributing third-party datasets and large model
binaries. The model registry provides hashes so locally obtained artifacts can
be verified.

## Repository layout

```text
.
├── configs/                 dataset-path templates
├── data/
│   ├── primary/             AI-TOD-v2 and VisDrone random audits
│   ├── specificity/         same-model YOLO26 O2M/O2O audit
│   ├── architecture/        YOLOv10-S replication audit
│   ├── extended_training/   300-epoch seed audits
│   ├── checkpoint_matrix/   independently fitted checkpoint rows
│   └── derived/             manuscript tables and figure source data
├── figures/                 Figure 2 code and rendered outputs
├── model_registry/          model/checkpoint SHA-256 registry
├── scripts/                 audit and bootstrap code
└── docs/                    data dictionary and upload instructions
```

## Quick verification

The integrity check uses only the Python standard library:

```bash
python scripts/verify_release.py
```

To rebuild Figure 2:

```bash
python -m pip install -r requirements.txt
python figures/make_figure2.py
```

The rebuild writes SVG, PDF and 600-dpi PNG outputs under `figures/`.

## Running a new audit

1. Obtain AI-TOD-v2 or VisDrone from the official sources listed in
   [`docs/DATASET_ACCESS.md`](docs/DATASET_ACCESS.md).
2. Install a PyTorch build suitable for the local CUDA runtime, then install
   the remaining packages in `requirements.txt`.
3. Copy the relevant YAML in `configs/` and replace its `path` value.
4. Obtain the intended checkpoint and verify its SHA-256 against
   `model_registry/MODEL_ARTIFACTS_SHA256.csv`.
5. Run the random-stratified audit, followed by outcome construction and the
   weighted bootstrap model. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The one-pixel replay is a deterministic sensitivity stress. It is not a model
of annotation noise. Margin is an observational readout, not a validated
treatment variable.

To reproduce the strict same-stride sensitivity from the released tables, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). This conditional analysis
is post hoc and does not replace the prespecified primary estimand.

## Public repository URL

The recommended repository name is `o2o-identity-audit`. If uploaded to the
author account already used by this project, the stable GitHub URL will be:

<https://github.com/liuyede447/o2o-identity-audit>

Ready-to-paste manuscript wording is provided in
[`docs/MANUSCRIPT_LINK_TEXT.md`](docs/MANUSCRIPT_LINK_TEXT.md).

## Citation

GitHub will expose a **Cite this repository** button from `CITATION.cff`.
Create a tagged release (`v1.0.0`) and archive it with Zenodo before final
publication if a DOI is desired.

## Licence status

No open-source licence has been selected on behalf of the authors. Before
making the repository public, all authors/institutional owners should approve a
licence. See [`LICENSE_DECISION.md`](LICENSE_DECISION.md).
