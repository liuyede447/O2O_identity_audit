# Stress-Dependent Assignment Stability — reproducibility package

Article-matched release `v3.3.0` for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection**. Supersedes the immutable `v3.2.3` predecessor.

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint; cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py`, then use the released figure source data, generated manuscript tables, and dense-grid post-processing outputs to regenerate figures and tables. See the repository root `README.md` for the per-figure commands and which figures run from this snapshot alone (Figure 2) versus which need the separately released dense-grid asset (Figures 3, 4, 6, S1) or an analysis run not in this snapshot (Figure 5).

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>. Configure an absolute dataset path from `instrument_frozen/configs/aitod_v2.template.yaml` and verify the four checkpoint assets against `RELEASE_ASSET_MANIFEST.csv`. Run the read-only replay driver `instrument_frozen/scripts/assignment_stability_audit_v2.py` (GPU required):

```bash
python instrument_frozen/scripts/assignment_stability_audit_v2.py \
  --checkpoint <checkpoint asset> \
  --expected-sha256 <sha256 from RELEASE_ASSET_MANIFEST.csv> \
  --data <your configured aitod_v2 yaml> \
  --output-dir <output dir>
```

Its outputs feed the downstream `scripts/analyze_*.py` analyses. Because the run depends on the GPU/runtime stack, confirm end-to-end equivalence to the frozen per-object/per-direction intermediates in your environment.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, resolved `optimizer=auto` state, and complete training lineage were **not preserved**; exact from-scratch training reproduction is therefore not claimed.

See the repository root `README.md`, the repository scripts, and `requirements.txt` for the executable routes.
