# Stress-Dependent Assignment Stability — reproducibility package

Article-matched release `v3.3.3` for **Stress-Dependent Assignment Stability in Aerial Tiny-Object Detection**. It supersedes the immutable `v3.3.2` predecessor.

This package measures native post-training assignment states under controlled geometric stress. It is read-only: no weights are updated, and no detector-performance improvement or treatment effect is claimed. The article's primary mechanistic estimates are conditional on the audited seed-0 checkpoint. The cross-dataset, archived-stage, detector-contract, and four-seed analyses are sensitivity or replication analyses.

## Reproduction levels

**Level 1 — derived-data reproduction.** Run `python scripts/verify_release.py`. Then regenerate figures and tables from the released figure source data, generated manuscript tables, and dense-grid post-processing outputs.

The following run from this snapshot alone:

- Figure 2: `figures/src/plot_fig2_lowfi.py`
- Figure 3: `figures/src/rebuild_fig3_corrected.py`
- the base-defined correction: `data/post_submission_correction_20260908/README.md`
- Supplementary Tables S37 and S38: `data/support_sensitivity_20260909/inputs/README.md`

Figures 4, 6, and S1 need the dense-grid asset, which is released separately. Figure 5 needs an analysis run that is not in this snapshot. See the repository root `README.md` for the per-figure commands.

**Level 2 — replay reproduction.** Obtain AI-TOD-v2 from <https://github.com/Chasel-Tsui/AI-TOD-v2> and VisDrone from <https://github.com/VisDrone/VisDrone-Dataset>. Configure an absolute dataset path from `instrument_frozen/configs/aitod_v2.template.yaml`, and verify the four checkpoint assets against `RELEASE_ASSET_MANIFEST.csv`. Then run the read-only replay driver `instrument_frozen/scripts/assignment_stability_audit_v2.py` (GPU required):

```bash
python instrument_frozen/scripts/assignment_stability_audit_v2.py \
  --checkpoint <checkpoint asset> \
  --expected-sha256 <sha256 from RELEASE_ASSET_MANIFEST.csv> \
  --data <your configured aitod_v2 yaml> \
  --output-dir <output dir>
```

Its outputs feed the downstream `scripts/analyze_*.py` analyses. End-to-end equivalence to the frozen per-object and per-direction intermediates depends on the GPU and runtime stack, so confirm it in your environment.

**Level 3 — training reproduction.** The released checkpoints and training contract are provided for provenance and replay. The initial pretrained-weight hash, the resolved `optimizer=auto` state, and the complete training lineage were **not preserved**, so exact from-scratch training reproduction is not claimed.

For the executable routes, see the repository root `README.md`, the repository scripts, and `requirements.txt`.
