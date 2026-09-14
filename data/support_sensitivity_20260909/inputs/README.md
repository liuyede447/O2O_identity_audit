# Endpoint inputs for the support-sensitivity re-run

`fixed_endpoint_per_gt.csv` (SHA-256 `0596d59486c5da70a577a6928f9b14fddc6c1ba115511e20486175dabada0d7a`) and `equivalent_endpoint_per_gt.csv` (`bf6d2f87ec2b0922427870218d3341fb0d33a45b8d8794591e927bc0f35993de`) are the frozen per-GT endpoint rows listed in `../MANIFEST.json` under `inputs.fixed_endpoint` and `inputs.equivalent_endpoint`. That manifest records their original absolute paths. The dense-event input is already in the repository at `data/dense_grid_v3/population/per_direction_boundary_events.csv`.

Run from the repository root:

```bash
python scripts/run_support_sensitivity.py --fixed-endpoint data/support_sensitivity_20260909/inputs/fixed_endpoint_per_gt.csv --equivalent-endpoint data/support_sensitivity_20260909/inputs/equivalent_endpoint_per_gt.csv --dense-events data/dense_grid_v3/population/per_direction_boundary_events.csv --output-dir <new directory>
```

The run reproduces `RUNNER_FREE_ENDPOINT_SENSITIVITY.csv`, `DENSE_COMMON_RISKSET_COUNTS.csv`, and `DENSE_COMMON_TRAJECTORY_RMCBD.csv` byte for byte. These are the sources of Supplementary Tables S37 and S38. The regenerated `MANIFEST.json` differs only in the input and output paths it records. `SUPPORT_SENSITIVITY_REPORT.md` is a written report, not script output.

`fixed_endpoint_per_gt.csv` is byte-identical to `data/post_submission_correction_20260908/inputs/fixed_branch_per_gt.csv`.
