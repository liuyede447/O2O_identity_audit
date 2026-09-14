# Post-submission base-defined correction (2026-09-08)

After the original submission, a reproducibility audit found that several secondary O2O identity summaries kept rows whose base O2O identity was undefined. There were 169 such ground truths, all in the 8–16 px stratum. This folder recomputes each affected summary with the base-defined gate applied. It uses only the frozen per-object and per-direction rows and runs no detector inference, replay, or training. The corrected article reports these values in Figure 3, in Supplementary Tables S18–S19, and in the related text. The primary Table 1 endpoint, the Rank-Set analyses, and the dense-boundary analysis already required a defined base identity and are unaffected (see `outputs/BASE_UNDEFINED_PROPAGATION_AUDIT.md`).

## Layout

- `inputs/` holds frozen rows copied from the original runs:
  - `eligibility_lock_per_gt.csv`: eligibility-lock counterfactual rows, one per GT.
  - `fixed_branch_per_gt.csv`: fixed-pixel native rows, one per GT. This is the same file as the fixed endpoint input in `data/support_sensitivity_20260909/inputs/`.
  - `anatomy_{fixed,equivalent}_per_direction.csv`: per-direction pathway anatomy rows. `anatomy_{fixed,equivalent}/per_direction_anatomy.csv` are byte-identical copies in the directory layout that `compute_corrected_anatomy_sensitivities.py` expects.
  - `pre_resolution_formal/{fixed,equivalent}/`: the formal pre-resolution runs, each with its own `SHA256SUMS.txt`. Their `summary.json` and `manifest.json` are the files pinned under `evidence/artifacts/runs/20260831T16*_preresolution_*_v1/artifact/`.
- `scripts/` holds the correction scripts, unmodified. Each resolves `inputs/` and `outputs/` relative to this folder.
- `outputs/` holds the frozen correction outputs used by the article. `build_closure_outputs.py` lives here because it reads and writes its own folder.

## Outputs and where the article uses them

| Output | Content (article use) |
|---|---|
| `LOCK_CORRECTED_ESTIMATES.{json,csv}`, `LOCK_CORRECTED_BOOTSTRAP.csv` | Eligibility lock on 7,265 / 7,263 GT and 298 images. Fixed-pixel O2O gap: 15.51 pp native versus 3.49 pp locked, a reduction of 12.02 [9.77, 14.05] pp (77.50%). Equivalent-area: 1.07 versus −0.17 pp. (Figure 3, Supplement) |
| `lock_effective_denominator_*`, `LOCK_EFFECTIVE_DENOMINATOR_AUDIT.md` | The effective-denominator audit that identified the 169 excluded GTs |
| `ANATOMY_BASE_DEFINED_CORRECTION/` | Eligibility-stable subset: 6,181/7,265 and 6,963/7,263 objects (Supplement) |
| `PATHWAY_BASE_DEFINED_CORRECTION/` | First-divergence and any-stage rates on 1,931 and 988 identity-exchange directions, with 5,000-replicate image-cluster CIs. Eligibility-led: 63.91% versus 40.18%. (Figure 3, Supplementary Tables S18–S19) |
| `PRE_RESOLUTION_BASE_DEFINED_CORRECTION/` | Pre-resolution control: fixed-pixel final minus pre-conflict gap +0.096 pp, with 64 directional disagreements (Figure 3) |
| `CORRECTED_SECONDARY_ESTIMATES_MASTER.csv`, `CORRECTION_OLD_VS_NEW.csv` | All corrected values in one table, and the old-to-new map |
| `BASE_DEFINED_CORRECTION_FULL_AUDIT.md`, `BASE_UNDEFINED_PROPAGATION_AUDIT.md` | Audit records |

`figures/src/rebuild_fig3_corrected.py` re-derives every row of the Figure 3 source table from these outputs.

## Re-running

The scripts write into `outputs/`, so run them in a scratch copy:

```bash
cp -r data/post_submission_correction_20260908 /tmp/corr && cd /tmp/corr
rm -r outputs/ANATOMY_BASE_DEFINED_CORRECTION   # this script will not overwrite an existing output directory
python scripts/compute_corrected_lock_estimates.py
python scripts/audit_lock_effective_denominator.py
python scripts/compute_corrected_anatomy_sensitivities.py --fixed inputs/anatomy_fixed --normalized inputs/anatomy_equivalent --output-dir outputs/ANATOMY_BASE_DEFINED_CORRECTION
MPLBACKEND=Agg python scripts/build_corrected_pathway_figure.py
python scripts/compute_corrected_pathway_ci.py
python scripts/compute_corrected_preresolution.py
cd outputs && python build_closure_outputs.py
```

Then compare the result with the shipped folder, for example with `diff -r --strip-trailing-cr`.

This check was run for the release on Windows with Python 3.13 and pandas 2.2. Every shipped output was reproduced byte for byte, with two expected exceptions:

- `PATHWAY_BASE_DEFINED_CORRECTION/first_divergence_corrected_table.tex`: the script adds an "Equivalent minus fixed" column filled with `n/a` and `descriptive`. That column was removed when the table went into the Supplement. Every value is the same.
- `BASE_DEFINED_CORRECTION_FULL_AUDIT.md`: `build_closure_outputs.py` writes a draft. The shipped file adds one sentence on the bootstrap design of the first-divergence CIs.

The re-run also writes files that are not part of the release: `CORRECTED_FIGURE3B.{pdf,png}` (a working plot) and four closure notes (`EDITOR_CORRECTION_SUMMARY.md`, `MANUSCRIPT_IMPACT_MAP.csv`, `POST_SUBMISSION_FINAL_DECISION.json`, `PRE_RESOLUTION_RECOVERY_REPORT.md`).

`build_corrected_pathway_figure.py` uses pyplot. On the build machine, pyplot's default backend selection crashed in a non-interactive shell. `MPLBACKEND=Agg` avoids the crash and does not change the CSV output.

The frozen outputs were written on Windows with CRLF line endings. On Linux or macOS the scripts write LF, so compare with line endings normalised. The anatomy `summary.json` records its input paths with the platform's path separator; the shipped file uses Windows `\`.
