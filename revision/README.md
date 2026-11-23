# Reproducing the v3.4.0 analyses

Version 3.4.0 of the revision analyses. It extends the v3.3.3 baseline
release (commit f3faf9bbc8354d127530d882ef4e561af9d17bf1, Zenodo
10.5281/zenodo.22744638). The baseline covers the results carried over from
the original submission, as listed in its README; this bundle covers the
results added in revision. Training runs are not reproduced (see "What is not
included").

## Where each revision result is reproduced

All commands run from the bundle root (the `revision/` directory of the
repository) after `pip install -r requirements.txt`.
"Derived" means no GPU, checkpoint or benchmark dataset is needed.

| Manuscript item | Command | Level |
|---|---|---|
| Table 3, Table S39 (matched O2O/O2M readouts) | `audit_stats.py` (points); `rerun_revision_analyses.py --only r2_1` (intervals) | derived |
| Table 4, Table S40 (RFLA and ATSS rule replay) | `audit_stats.py`; `rerun_revision_analyses.py --only table4` | derived |
| Table 5 (eligibility lock, seeds 0-3) | `audit_stats.py`; `rerun_revision_analyses.py --only table5` | derived |
| Table 7, Table S41 (directional boundary sets) | `audit_stats.py`; `rerun_revision_analyses.py --only table7_sets table7_cardinal table7_diagonal table7_eight table7_min8 table7_cardinal_legal table7_diagonal_legal table7_min8_legal` | derived |
| Table 8, Table S42 (fixed-K counterfactual) | `audit_stats.py`; `rerun_revision_analyses.py --only table8 table8_strata`; `fixed_k_direction_coverage_by_seed.csv` | derived |
| Tables S43, S44 (final-epoch checkpoints, Table 2 support) | `recompute_checkpoint_tables.py` | derived |
| Tables S45, S46 (official-validation replication) | commands under "Prespecified official-validation replication" | derived |
| Table S47 and Section 17 text (short-side strata) | `rerun_revision_analyses.py --only s47 s47_seeds s47_sensitivity` | derived |
| Section 17.1 (matched comparison without 8 px boxes) | `recompute_short_side_endpoint.py` | derived |
| Table S48 (stride-8 edge ties) | `boundary8_mechanism.py`, then `boundary8_ties.py` (see Section "Short side at the smallest stride") | needs the validation images and labels |
| Figures 2-4 | `python figures/render_figures.py` | derived |
| Figure 5 | `python figures/render_fig5_s1_revision.py --analysis-root figures/analysis_root --output figures/out` | derived |
| Figure 1, Figure S1 | not redistributed: they contain benchmark image pixels | - |
| Every other table and figure | the v3.3.3 release | see its README |

`rerun_revision_analyses.py` without `--only` runs every derived analysis in
the table above that it covers and checks each output against the archived copy
in `source_snapshots/` (tolerance 1e-9 relative). A full run took 33 to 39
minutes on one desktop CPU. The report is written to
`<output-dir>/rerun_report.json`.

## Rerun the revision analyses with intervals

The scripts in `source_snapshots/` are the unmodified originals and name their
inputs by paths from the original run environment. `revision_inputs.tar.gz`
holds exactly the files those scripts read (42 files, 392 MB uncompressed),
stored under the same layout with the drive letter as the first directory.
`revision_inputs_manifest.csv` lists each member with its SHA-256.
`rerun_revision_analyses.py` extracts the archive to a temporary directory,
copies the scripts, rewrites each absolute path literal to the extracted copy,
deletes the archived outputs from the copy, runs the scripts in order and
compares the new outputs with the archived ones, numerically to 1e-9 relative
error, including the bracketed interval strings in the CSVs. All verification
runs were on Windows. The rewritten paths use forward slashes and the archive
keeps the original letter case of every path, but the bundle has not been run
on Linux or macOS.

The Table 4 and Table 8 scripts load `assigner_audit.py` for its numpy/pandas
`summarise()` bootstrap. That module also imports torch for its replay code.
When torch is not installed, the runner supplies a placeholder module that
satisfies the import and raises on any attribute access, so the rerun needs
only numpy and pandas.

`requirements.txt` pins the versions of the original runs (Python 3.13.5,
numpy 2.2.6, pandas 2.2.3); with them every analysis reproduces its archived
output. In the original 16-analysis test, 15 passed with numpy 2.5.5 and pandas 3.1.1.
In the Section 17 aspect-ratio sensitivity, one validation GT whose long/short
ratio sits exactly on a band edge parses to the other side, so the 2-4 band
holds 701 instead of 700 targets and its reduction moves from 14.31 to 14.29 pp.

    python -B rerun_revision_analyses.py --output-dir rerun_check

To rerun the RFLA/ATSS per-direction sensitivity in the main-text Results and
Supplementary Table S40 on its own:

    python -B rerun_revision_analyses.py --output-dir s40_direction_check --only s40_per_direction

The runner checks the counts, estimates and 95% intervals in
`s40_per_direction/per_direction_sensitivity.json` against the archived copy.
The analysis uses 5,701 common targets, with 22,740 fixed and 22,732
size-scaled legal directions, and 5,000 image-cluster bootstrap draws
(seed 20261123). Its inputs are already in `revision_inputs.tar.gz`.

The inputs are per-object and per-direction replay records (identities,
change indicators, sampling weights and image identifiers). They contain no
images, raw annotations or detector predictions. The cardinal boundary-ray
records and three multiseed endpoint and Rank-Set runs are copies of baseline
records that the v3.3.3 Git tree does not carry in this form; the archive includes them so
the analyses run without a v3.3.3 checkout.

## Recompute the revision point estimates

Install the dependencies in requirements.txt and run python audit_stats.py from
this directory. The compact input tables preserve the relevant joined direction,
object, or ray rows and sampling weights. expected.json is extracted from the
saved experiment summaries. A successful run checks membership counts and
recomputes the weighted rates, restricted means and standardised gaps that it
reports.

- R1.2 applies the published base-defined O2O gate before recording the input rows.
- R1.3 uses the stored event or censoring distances on each evaluated ray set.
- R1.4 retains the directional scores used after the common-valid gate, with
  separate strict and relaxed shifted-count summaries.
- R2.1 joins the original endpoint and Rank–Set tables by image, GT and direction
  on the common-valid support.
- R2.2 uses the 5,701 joint base-nonempty GTs and the rule-specific change indicators.

The point-estimate check runs without GPU inference. The original replay and
bootstrap scripts and reports are retained in source_snapshots/ for provenance.
Full replay requires the v3.3.3 runtime and checkpoints, the author-obtained
benchmark datasets, and the archived third-party assigner revision where applicable.
The snapshots preserve the paths and runtime assumptions from the original runs.
They must be mapped to the corresponding inputs before a new full replay.

This bundle contains revision evidence, code snapshots and derived inputs; the
four final-epoch checkpoints listed under "Checkpoints" are release assets. Manuscript files, raw
benchmark images and annotations, and raw detector predictions are outside its
contents.

## Checkpoint and competition-support results

The checkpoint snapshot adds the paired final-epoch and epoch-280 reductions
on the discovery sample, plus the lock estimates on the Table 2 competition
support. The CSV and JSON files provide the estimates and intervals reported in
Supplementary Tables S43 and S44. Recompute S43 and S44, including intervals, with the archived derived inputs as described below. The portable point-estimate check above covers the five initial
revision analyses. The official validation split replication is documented below.


## Prespecified official-validation replication

The historical protocol, frozen hashes, selected image list, logs and formal
results for Supplementary Tables S45 and S46 are in
source_snapshots/R2_2_heldout_replication/. The original file
PREREGISTRATION_DRAFT.md keeps its frozen filename and contents. Its SHA-256 was
recorded before inference in FREEZE_RECORD.json. The public copy retains the
freeze timestamp, file hashes and original record digest. It lists filenames
without local path prefixes. For local file verification, use the paths
indexed in bundle_frozen_files.json.

To recompute estimates and intervals from the archived derived CSV inputs,
run these commands from the bundle root with the requirements installed:

    mkdir heldout_check
    tar -xzf heldout_inputs.tar.gz -C heldout_check
    python -B source_snapshots/R2_2_heldout_replication/analyze_replication.py --lock-dir heldout_check/lock --gate heldout_check/base_gate.csv --expected-selection source_snapshots/R2_2_heldout_replication/selected_images_preregistered.csv --out heldout_check/primary.json
    python -B source_snapshots/R2_2_heldout_replication/analyze_s2.py --root heldout_check --out heldout_check/s2.json

Compare the estimates, intervals, counts and decision with the JSON files in
source_snapshots/R2_2_heldout_replication/formal/. Input paths in primary.json
will differ because they identify the local copy. This check needs no GPU
or dataset.

For a detector replay, obtain the validation image and annotation archives
using the dataset-author URLs and hashes in Section 3 of the frozen plan.
The archived conversion and sample-selection scripts show how the 800 x 800
YOLO dataset and manifest were generated. The original orchestration scripts
record the invocations and requirements of the Windows/Git Bash run. Map the
historical input and output paths to your environment for a local reproduction.
Use the v3.3.3 runtime and the frozen replay script snapshots indexed in
bundle_frozen_files.json. The release asset yolo26s_aitodv2_seed0_final_epoch.pt
(see "Checkpoints") is the fixed final-epoch checkpoint, with SHA-256
46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b.
Check the dataset and sample against the recorded manifests. Obtain the
official validation annotations from the dataset authors.


## Recompute the checkpoint and competition-support tables

Run python -B recompute_checkpoint_tables.py --output-dir checkpoint_check
from the bundle root. The program extracts checkpoint_lock_inputs.tar.gz
into a temporary directory, redirects the path settings of the two archived
analysis scripts, and runs the existing bootstrap estimators. Compare the
resulting CSVs and JSON summaries with source_snapshots/R1_2_checkpoint_last.
With the pinned numpy 2.2.6 and pandas 2.2.3 the CSVs match byte for byte;
with numpy 2.5.5 and pandas 3.1.1 some values differ in the 16th significant
digit because summation order changed.
The input-provenance manifest in that directory maps the archived per-seed
roles to their original replay outputs and support flags. The same directory
records the seed-0 support-exclusion comparison. The file
fixed_k_direction_coverage_by_seed.csv records the active $K=5$ counts and
weight totals on which the range in Supplementary Table S42 is based.

## Short side at the smallest stride (post hoc)

source_snapshots/R3_eligibility_rule_strata/ holds the post hoc split of the
fixed 1 px eligibility-lock result by box short side (below 8 px, equal to
8 px, above 8 px) behind Supplementary Section 17 and Tables S47 and S48. It
uses the same estimator and bootstrap settings as the validation analysis.
seeds_boundary8.py covers discovery seeds 0-3; analyze_rule_strata.py and
sensitivity_rule_strata.py cover seed 0 and the validation sample;
boundary8_mechanism.py and boundary8_ties.py recompute the stride-8 inside test
for each legal shift on the validation sample. The JSON and CSV files are the
outputs. For seeds 0-3 the three strata recombine to the published 8-16 px
native and locked rates to within 1e-16 (seeds_boundary8.json); on the
validation sample the recombined gaps equal the published 16.13 and 3.32 pp.
`rerun_revision_analyses.py --only s47 s47_seeds s47_sensitivity` reruns the
first three scripts from the archived inputs. For four fixed-only validation
GTs without a size-scaled row, `analyze_rule_strata.py` reads the box size from
tokens 4 and 5 of the YOLO label line, and it checks that mapping on 2,000
other GTs. `val_label_sizes.csv` holds only those two tokens for each label line
the scripts read; the runner writes them into placeholder label files whose
class and position tokens are `-`.

The two boundary8 scripts rebuild each float32 box from the YOLO-format
validation labels, so they need the converted validation sample
(`prepare_validation_split.py` in `R2_2_heldout_replication/`) and the
`ultralytics_local` runtime from v3.3.3 (`instrument_frozen/ultralytics_local`).
Point `DATA` in `boundary8_mechanism.py` at the converted `data.yaml` and
`FORMAL` at the extracted `heldout_inputs.tar.gz`, then run
`boundary8_mechanism.py` followed by `boundary8_ties.py` (about two minutes on
a CPU). Their outputs, `boundary8_per_gt.csv`, `boundary8_tie_events.csv` and
the two JSON summaries, are archived beside the scripts and were reproduced
exactly in a rerun on 2026-11-22.

## Figures

`figures/render_figures.py` draws Figures 2-4 from `fig2_source.csv`,
`fig3_source.csv` and `fig4_source.csv` and checks their SHA-256 first.
`figures/render_fig5_s1_revision.py` draws Figure 5 with the baseline
generator `plot_fig5.py`; `figures/analysis_root/` holds that generator and the
records it reads (the Rank-Set summaries, the fixed-branch replay rows and the
`o2m_positive_count_scale_controls_v5` positive-count strata) under the
baseline layout. Install `requirements-figures.txt` as well.

    python figures/render_figures.py
    python figures/render_fig5_s1_revision.py --analysis-root figures/analysis_root --output figures/out

The same script then draws Figure S1, which needs the AI-TOD-v2 tile
`0000170_00401_d_0000001__160_0.png` at
`figures/analysis_root/figures_final/source_data/figs1_primary_trace/`. The
tile is not redistributed, so without it the command writes Figure 5 and stops
with a `FileNotFoundError` naming the tile. Figure 1 also contains benchmark
pixels and is not included.

`figures/reference/` holds the article's Figure 2-5 renders. In the authors'
Anaconda environment (matplotlib 3.10.0 built against FreeType 2.13.3,
Pillow 12.2.0, Arial), the scripts reproduced these PNGs pixel for pixel. The
pip wheel of matplotlib 3.10.0 bundles FreeType 2.6.1, which rasterizes text
differently; the changed text extents also shift whole panels by a few pixels,
so 0.5-7% of pixels differ.

## Checkpoints

The final-epoch (epoch 300) YOLO26s checkpoints of the four protocol-eligible
training runs are attached to the v3.4.0 GitHub release, not committed to the
repository. Download them into `assets/` and check each SHA-256 below. The four
epoch-280 checkpoints used in the discovery analyses are v3.3.3 release assets.
None of the derived reruns needs a checkpoint.

| Seed | File | SHA-256 | Used for |
|---|---|---|---|
| 0 | `yolo26s_aitodv2_seed0_final_epoch.pt` | 46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b | Tables S43, S45-S48 |
| 1 | `yolo26s_aitodv2_seed1_final_epoch.pt` | 1c689b3e6ae7dd27009fc911dc1e8d7928aa4aff9327dfc627ce770da4e12f5f | Table S43 |
| 2 | `yolo26s_aitodv2_seed2_final_epoch.pt` | b47855abfb339a682299b40dca54298c62f2d897f6a542af2134051db9a352f7 | Table S43 |
| 3 | `yolo26s_aitodv2_seed3_final_epoch.pt` | 326a5bc8ca16a390f5ba93fa8a99c4314eaa0e83648aefc866906b9b54f836ae | Table S43 |

Each hash equals the `checkpoint_sha256` recorded in the corresponding replay
manifest. `source_snapshots/R1_2_checkpoint_last/run_last.sh` gives the replay
invocation.

## Detector replay

Every derived input above came from a read-only replay of native assignment on
frozen predictions. To repeat a replay, obtain AI-TOD-v2 (Section "Dataset and
path provenance"), use the v3.3.3 runtime (`instrument_frozen/`) and the
checkpoints above, and follow the shell scripts kept with each analysis:
`run_seeds.sh` (Table 5), `run_last.sh` (S43), `run_fixed_k.sh` (Table 8),
`run_formal.sh` (S45, S46), `analyze_assignment_boundary_radius_multidir.py`
(Table 7) and `assigner_audit.py` (Table 4). The RFLA and ATSS replay executes
the original assigner files of `Chasel-Tsui/mmdet-rfla` at commit
198977b5e198005fec783e81f10f9363946d2ef8 (MIT licence), which is not
redistributed here; clone it at that commit. GPU replay matches the archived
records only up to the numerical behaviour of the GPU and runtime stack, so
compare a new replay with the archived inputs before rerunning the statistics.

## What is not included

- Benchmark images and annotations (AI-TOD-v2, VisDrone). Obtain them from the
  dataset authors.
- Raw detector predictions. The replay regenerates them from the checkpoints.
- Training. The released checkpoints come from the runs described in
  Supplementary Table S6. The initial pretrained-weight hash and the resolved
  optimizer state were not preserved, so exact retraining is not claimed.
- Figure 1 and Figure S1 renders, which contain benchmark pixels.

## Dataset and path provenance

Obtain AI-TOD-v2 through the dataset authors' archive at
https://huggingface.co/datasets/jwwangchn/AI-TOD/tree/1591be1330a5dc99b646b3599f67437a2de246ae.
The SHA-256 of annotations_v2.zip is
60b6e0d992ab87efac8411b6a90d66c3a38fbd32bd726d4234a0de217210facd.
The official image toolkit and license are at
https://github.com/jwwangchn/AI-TOD. The validation image and annotation
archives carry the dataset's CC BY-NC-SA 4.0 licence. Benchmark images and
raw annotations are obtained separately.

The analyses used these AI-TOD-v2 annotation files (28,036 images and
752,746 boxes in total):

| Split | File | Images / boxes | SHA-256 |
|---|---|---|---|
| train | aitodv2_train.json | 11,214 / 301,534 | ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e |
| val | aitod_val_v2.json | 2,804 / 75,091 | d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf |
| test | aitodv2_test.json | 14,018 / 376,121 | 9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98 |

The train file is byte-identical to aitod_train_v2.json in the archive above,
and the val file comes from the same archive. That archive has no test file,
so the test annotations cannot be matched byte for byte against it. The archived YOLO conversion
dependency is in the validation study's replay_sources directory, with its
hash in conversion_manifest.json.

The archive preserves original run paths for provenance. The runner maps those
paths to the bundled files for statistical reruns.
For full detector replay, obtain the dataset, use the provided final-epoch
checkpoint, and map the historical paths to your checkout. The formal
validation replay used Git Bash and one GPU, with approximately 70 minutes
for the eligibility-lock replay and 45 minutes for the S2 block. See
SHA256SUMS.txt and VERIFICATION.json for bundle integrity and analysis checks.

## Matched short-side endpoint comparison (post hoc)

Run `python recompute_short_side_endpoint.py` to reproduce the matched
comparison in Supplementary Section 17. The compact inputs contain sums of
weights and weighted O2O changes by anonymised image index, size group and
short-side stratum, including sampled images with no eligible targets.
The script pairs bootstrap draws across both perturbations. With
`--archive PATH_TO_BASELINE_CHECKOUT` it rebuilds the compact table from
the two original endpoint runs. The output JSON records the input hashes,
the support rule and the exact point estimates and intervals.
