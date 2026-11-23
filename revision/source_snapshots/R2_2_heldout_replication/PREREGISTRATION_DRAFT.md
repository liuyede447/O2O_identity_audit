# Pre-specified replication on the official AI-TOD-v2 validation split

Status: design approved by the authors on 2026-11-21. All rehearsal items in Section 6 passed.
`FREEZE_RECORD.json` holds the hashes of this file and every script and input, written before any
forward pass on replication images; `run_formal.sh` verifies it before starting.

## 1. Question

Holding base eligibility fixed reduces the fixed 1 px O2O scale gap (8–16 px minus 16–32 px
fragility) on the 300-image discovery sample drawn from the official AI-TOD-v2 test split. For the
model used here (seed-0 final-epoch weights), the discovery estimates on the base-O2O-defined support
are: native gap 15.46 [12.10, 18.57] pp, locked gap 3.30 [1.14, 5.27] pp, reduction 12.16
[10.09, 14.08] pp (`rehearsal_results.json`, D2). This replication asks whether the native
tiny-object disadvantage and its attenuation under eligibility locking reappear on images used
neither for training nor for checkpoint selection.

## 2. Why this split

- Training used only the official train split (11,214 images).
- Post-training mAP evaluation and the epoch-280 checkpoint selection used the official test split
  (14,018 images), which is also the source of the discovery and historical lockbox samples.
- The official validation split (2,804 images) is disjoint from both by file name (0 and 0 overlaps).
- No audit is known to have used validation images; Section 9 lists the one unresolved early run.

The manuscript will describe this as a pre-specified replication on the official AI-TOD-v2
validation split, which is disjoint from the splits used to train and select the audited model.
It will not claim that the project never touched these images.

## 3. Data

| Item | Value |
|---|---|
| Annotations | `aitod_val_v2.json` (SHA-256 `d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf`; 2,804 images, 75,091 boxes) from `annotations_v2.zip`, the Hugging Face archive released by the dataset authors (`jwwangchn/AI-TOD`, commit `1591be1`; archive SHA-256 `60b6e0d992ab87efac8411b6a90d66c3a38fbd32bd726d4234a0de217210facd`). The `aitod_train_v2.json` in the same archive is byte-identical to the local train annotations used for training (SHA-256 `ed7b37a1…`) |
| Images | Local copy from the official toolkit output, `G:\two_paper\SWD-MicroDet\third_party\AI-TOD-official\aitodtoolkit\aitod\images\val` (2,804 files, names match the annotations one to one). Converted images and labels are hashed file by file in `validation_files_sha256.txt` |
| Licence | CC BY-NC-SA 4.0 (images and annotations are not redistributed) |
| Conversion | `SWD-MicroDet/scripts/prepare_aitod_v2_yolo.py`, unchanged, called through `prepare_validation_split.py`; 75,082 boxes kept after the script's existing crowd and zero-size filter |
| Conversion check | Passed: 100 randomly chosen official test images (seed 20261121), re-converted with the same code, were byte-identical to the existing labels the audited model was evaluated on |

## 4. Model

Seed-0 final-epoch weights (`last.pt`, SHA-256
`46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b`), taken by the fixed rule
"last training epoch" and not selected on any evaluation split. On the discovery sample these weights
give the same lock result as the epoch-280 weights (`R1_2_checkpoint_last/RESULTS.md`).

## 5. Sample

- Eligible population: 2,613 of 2,804 images (`t_only` 1,374, `s_only` 12, `t_and_s` 1,227).
- 600 images, proportional allocation (315 / 3 / 282), sampled without replacement within strata,
  selection seed 20261122, drawn by `select_sample.py` from annotation sizes only:
  `selected_images_preregistered.csv`, SHA-256 `389dc7ed8fe254ed11617157da8465817b644b9fbb0536519bef9d38b96ee9ba`.
- The replay script draws its own sample from the same seed and has no list override. It is launched
  through `run_lock_checked.py`, which runs the unchanged script but intercepts the selected dataset
  indices before the data loader is built and aborts unless images and strata equal the
  preregistered list. The check therefore happens before the first forward pass on any replication
  image. The analysis step checks the written list again, including weights.
- Row structure, predicted from geometry alone (`predict_row_structure.py`, exact on discovery):
  31,664 (image, GT, stress) rows in 599 images. One image (`9999999_00346_d_0000152__600_0`) has no
  legal shift because its only in-domain box extends past the image border; four GTs have legal
  shifts only under fixed 1 px. No image is single-stress.

## 6. Procedure

1. Lock replay: unchanged `run_eligibility_lock_counterfactual.py` (SHA-256
   `cfcb4bc8bc88670013acaaf113371608b028d906170071daf216169916141c5d`), `--sample-images 600
   --selection-seed 20261122 --kappa 0.0625 --imgsz 800 --bootstrap 5000`. Its own `summary.json`
   uses the pre-correction estimator and is not the reported estimate.
2. Base gate: `base_gate.py` (same code path as the replay) gives each GT's base O2O identity and the
   Table-2 support flag.
3. Primary analysis: `analyze_replication.py`. Frozen corrected lock estimator on the
   base-O2O-defined support. Bootstrap: 5,000 replicates, seed 20260823 + 901, stratified image-cluster
   resampling. The cluster index is the sorted union of images with at least one gated row under
   either stress, and an image missing under one stress contributes zero to that stress. Selected
   images without gated rows are listed and contribute nothing. Replicates with an empty denominator
   are counted and left out of the percentile interval. An empty denominator in a point estimate is
   a structural failure and is reported as such.
4. Secondary, descriptive: the same estimator on the Table-2 competition-defined support.
5. S2 pipeline (`run_s2.sh`): `run_reviewer_killer_controls.py` (SHA-256 `a3a472f0…7dfc`,
   equivalent-area, κ = 0.0625, `--normalized-only`, frozen list) and
   `analyze_o2m_rank_set_decoupling.py` (SHA-256 `6a288811…6b27`), both on the preregistered list,
   analysed by `analyze_s2.py` with the unchanged R2.1 estimator.

Rehearsal, completed before freezing (`rehearsal_results.json`):
- D1: on the discovery outputs of the epoch-280 seed-0 weights, the analyzer reproduced the
  published corrected estimates and intervals (maximum absolute difference 8.3e-16).
- D2: discovery comparator for the replication model (Section 1).
- Synthetic tests: single-stress image, image whose GTs all fail the gate, zero-row image, all-zero
  outcomes, single-stress GT, duplicate keys (abort), missing gate row (abort), stratum disagreement
  (abort), row-order invariance, empty size cell. These are the failure classes that terminated the
  historical lockbox analyzer.
- `base_gate.py` reproduced the existing seed-0 final-epoch gate on all 8,282 discovery GTs.
- S2 pipeline end to end on the discovery images with the same weights: the equivalent-area replay
  reproduced the existing final-epoch output (7,432 rows, 0 differences), the R2.1 gate and join
  checks passed, and the discovery S2 estimate is 35.97 [33.14, 38.74] pp (6,373 objects).
- Selection pre-flight (`run_lock_checked.py`): passed on discovery, aborted as intended on a wrong
  list, and passed as a no-inference check on the replication list (600 images); each check stopped
  before data loading and inference. A full wrapped replay on the discovery images with the same
  weights produced `per_gt.csv`, `per_direction.csv` and `selected_images.csv` byte-identical to the
  unwrapped run, so the wrapper does not alter replay outputs.
- Both S2 scripts accept the preregistered list on the replication dataset (`check_s2_manifest.py`:
  the rank-set script's own `validate_selection_manifest()` and the killer-controls manifest check,
  600 images each, no model loaded).
- `freeze.py --verify`, run at every start of `run_formal.sh`, re-hashes the plan, scripts, checkpoint
  and the 1,200 selected image and label files against the per-file manifest.
- Primary analysis isolation: a secondary block that is undefined leaves the primary estimates and
  intervals unchanged (synthetic tests T11 and T12); the primary result is written before any
  secondary analysis runs.

## 7. Decision rule

Primary (fixed 1 px, absolute pp, 95% intervals as in Section 6):

- R1: native O2O scale gap, lower bound > 0;
- R2: eligibility-lock reduction (native gap minus locked gap), lower bound > 0.

The core finding is declared replicated only if R1 and R2 both hold. No percentage threshold is a
pass condition, because a small native gap inflates percentages. If R2 holds but R1 does not, the
report states that the new sample did not clearly reproduce the native disadvantage and gives both
estimates. Every outcome is reported with estimates and intervals in the manuscript and the response
letter, whatever its direction.

Secondary (estimated, no pass/fail):

- S1: fixed 1 px native O2O gap minus equivalent-area native O2O gap, on GTs gated under both
  stresses, with the same resampling counts;
- S2: under equivalent-area stress, O2M positive-set change minus O2O change among 8–16 px objects,
  object level, on the Table-1 common-valid support (the R2.1 measure).

## 8. One run

The replication is run once. A step that fails for an infrastructure reason (out of memory, disk or
path error, interrupted process) may be re-executed with byte-identical code from the saved upstream
outputs, provided no one has read the primary output; `run_formal.sh` skips completed steps, verifies
the freeze record on every start, and logs each start and re-execution in `timeline.txt`, which is
disclosed. Any change to code or inputs after the freeze makes the run a reported failure; it is not
repeated on the same images.

## 9. History audit

- The validation file names overlap with neither the 11,214 train images nor the 14,018 official test
  images.
- The seed-0 training log shows that the YOLO `val` folder read during training held 14,018 images
  (`beifen/archive_backup/server_A/logs/r1_baseline_13zone_e300_s0_batch8.log`, line 49), and every
  file in `G:\data\processed\AI-TOD-v2-YOLO\val\images` is listed in `aitodv2_test.json`.
- A review of 118 readable image lists from the original experiments, the historical lockbox and the
  revision experiments found no validation image. Spot checks of six validation file names found no
  hits in SWD-MicroDet, `archive_backup/server_A`, `server_B` or `artifacts`.
- All 224 archived `args.yaml` files with a data path point to `configs/aitod_v2.yaml` or VisDrone
  configs; none points to a trainval or validation-split dataset.
- Unresolved: the early 40-epoch SWD-MicroDet run
  (`SWD-MicroDet/runs/detect/experiments_yolo26/baseline/yolo26s_aitodv2_baseline_e40`) ran validation,
  and the route plan mentions `small_val_v2_1.0.json`, but no image list or label cache from that run
  survives. It belongs to a different model line and played no role in training, selecting or
  auditing the checkpoints in this study.

## 10. Record

`FREEZE_RECORD.json` holds the SHA-256 of this file, all scripts, the image list, the conversion
manifest and the checkpoint, written before the first forward pass on replication images. By author
decision the record is kept locally, with no third-party timestamp, so its timing is self-attested;
the manuscript and response letter will say so. The record, `timeline.txt` and all outputs will be
included in the reproducibility package.
