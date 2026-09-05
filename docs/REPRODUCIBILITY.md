# Reproducibility routes

## Integrity-only route

Run `python scripts/verify_release.py`. This requires only the Python standard library and validates every repository file listed in `SHA256SUMS.txt`.

## Derived-statistics route

The active event table is `data/dense_grid_v3/population/per_direction_boundary_events.csv`; six-τ outputs are under `data/dense_grid_v3/postprocess/`. Re-run post-processing from the separately released curve with `scripts/run_dense_grid_postprocess.py`. Use `--help` for the exact CLI and retain the released τ set `.05, .0625, .10, .125, .20, .25`.

## Source-image replay route

1. Obtain AI-TOD-v2 from its owner and configure an absolute local path.
2. Obtain the intended checkpoint release asset and verify SHA-256.
3. Prepend `instrument_frozen/ultralytics_local` to `PYTHONPATH` so the frozen source snapshot, which reports version 8.4.51, is imported.
4. Use the released 300-image manifest; do not resample.
5. Run the audit script with the configuration recorded in the v3 population manifest.

The internal environment inventory showed a source-tree/distribution metadata discrepancy (the frozen source reports 8.4.51 while the environment's installed distribution record reported 8.3.221). The frozen source tree and its hashes, not the installed metadata string, are the code authority.

Exact from-scratch training is not claimed: the initial pretrained-weight hash, the resolved `optimizer=auto` state and full lineage reconciliation were not retained. The released `best.pt` files come from runs configured for 300 epochs and record selected checkpoint epoch 280. Do not describe them as weights taken at epoch 300.
