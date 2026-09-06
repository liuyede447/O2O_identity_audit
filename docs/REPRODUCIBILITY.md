# Stress-Dependent Cardinal Assignment Boundaries — reproducibility package

Article-matched release `v3.1.0` for **Stress-Dependent Cardinal Boundaries of Native Assignment States in Aerial Tiny-Object Detection**.

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint; cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py`, then use the released figure source data, generated manuscript tables, and dense-grid post-processing outputs to regenerate figures and tables.

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>. Configure an absolute dataset path from `configs/aitod_v2.template.yaml`, verify the four checkpoint assets against `RELEASE_ASSET_MANIFEST.csv`, prepend `instrument_frozen/ultralytics_local` to `PYTHONPATH`, and run the documented replay command in `docs/REPRODUCIBILITY.md`.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, resolved `optimizer=auto` state, and complete training lineage were **not preserved**; exact from-scratch training reproduction is therefore not claimed.


See the repository scripts and `requirements.txt` for the executable routes.
