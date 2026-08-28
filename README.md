# Loss-active O2O identity audit

Code and derived evidence for the manuscript **“Scale-Conditioned Loss-Active
One-to-One Identity Instability in Aerial Tiny-Object Detection.”**

This repository implements a read-only audit of native one-to-one (O2O)
loss-active identity stability. It includes the outcome-blind sampling
manifests, derived per-object audit tables, bootstrap summaries, model hashes,
and scripts used for the paper's AI-TOD-v2 and VisDrone analyses.

## Contents

```text
configs/          dataset-path templates
data/             frozen derived tables and sampling manifests
docs/             data dictionary, dataset access, and reproduction guide
figures/          deterministic Figure 2 rebuild and outputs
model_registry/   checkpoint SHA-256 identifiers
scripts/          replay, outcome, bootstrap, and sensitivity analyses
```

The repository does **not** redistribute AI-TOD-v2 or VisDrone images,
annotations, model weights, or raw prediction dumps. Obtain the datasets from
their official sources and verify model artifacts with the supplied hashes.

## Verify the release

The integrity check uses the Python standard library:

```bash
python scripts/verify_release.py
```

## Rebuild Figure 2

```bash
python -m pip install -r requirements.txt
python figures/make_figure2.py
```

The script writes SVG, PDF, and 600-dpi PNG outputs to `figures/`.

## Run the audit

1. Obtain AI-TOD-v2 or VisDrone from the sources in
   [`docs/DATASET_ACCESS.md`](docs/DATASET_ACCESS.md).
2. Install a PyTorch build for the local CUDA runtime, followed by the packages
   in `requirements.txt`.
3. Copy the appropriate YAML from `configs/` and set its dataset path.
4. Obtain the checkpoint and verify its SHA-256 against
   `model_registry/MODEL_ARTIFACTS_SHA256.csv`.
5. Follow [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the replay,
   outcome construction, weighted models, and strict same-stride sensitivity.

## Scientific scope

The one-pixel replay is a deterministic sensitivity stress, not a model of
annotation noise. The active-relative margin is an observational competition
coordinate, not a calibrated confidence or validated treatment variable. The
strict same-stride analysis is post hoc and does not replace the primary
estimand.

## Citation

Use the repository's **Cite this repository** menu, generated from
[`CITATION.cff`](CITATION.cff). The release version is `v1.0.0`.
