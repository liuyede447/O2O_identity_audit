# Prespecified replication of the assigned-relative margin screen on new images

Status: written before any forward pass on the replication images. `FREEZE_RECORD.json` holds SHA-256
hashes of this file, all scripts, the image list, the fitted coefficients, the checkpoints and the input
files. `run_formal.sh` verifies the record before extracting anything. The timestamp is local and
self-attested; there is no third-party registration.

## 1. Scope

The margin analysis of Supplement S18 (Tables S49 and S50) was run after exploratory results had been
seen. It stays exploratory, and this replication does not change that. It tests, on images outside that
sample and with coefficients fixed in advance, whether the pattern holds. The design choices below
(support, outcomes, model form, shift size) were made knowing the exploratory results. They were fixed
before the new images were run.

## 2. Images

Source: official AI-TOD-v2 validation split, 2,804 images, 2,613 eligible by the image-strata rule.
The 600 images used in the R2.2 replication are excluded because that run wrote margins for them.
2,013 eligible images remain. All of them are used (census, weight 1 per image); no image sampling.

Source scenes are inferred from file names only: the stem before `__`, or the sequence token for
VisDrone-style names. A name without `__` is its own scene. About 900 of the images are untiled
Airbus/DIOR pictures whose names say nothing about shared scenes, so scene separation is by name and
cannot be shown to be complete.

| Set | Images | Scenes | Definition |
|---|---|---|---|
| Primary | 1,460 | 1,266 | no scene in common with the 300 discovery images or the 600 R2.2 images |
| All eligible | 2,013 | 1,483 | every remaining eligible image |
| Name-disjoint from train/test | 953 | 951 | primary, and no scene in common with the official train or test split |

Untiled names are their own scene by rule, so most of the 953 images are in that set automatically and
their separation from train and test cannot be checked; it is a name-based restriction, not a stricter
test. Primary-set objects come mainly from the tiled images, because untiled images have about one focal
object each. The column for the 953 images is called `fully_disjoint` in the files.

The frozen list is `images.csv`.

## 3. Checkpoints

Four YOLO26s epoch-280 checkpoints, seeds 0–3, the ones of Tables S49 and S50. Epoch 280 was chosen on
the official test split; the validation split played no part in that choice. One early run
(40 epochs, SWD-MicroDet) is unresolved: whether it read validation images is not known. The paper
discloses it. It is a different model and did not produce the audited checkpoints.

## 4. Objects, predictors, outcomes

Objects: focal size bins 8–16 and 16–32 px, with a defined margin (assigned candidate and runner-up) and
legal shifts under both stresses. No object is dropped because its margin is negative.

M2: log area, log(1 + O2M positives), log assigned q, assigned CIoU. M3: M2 plus the assigned-relative
margin clipped to [0, 1]. Standardised weighted logistic regression, C = 1e6. Coefficients and scalers
are fitted once per seed on the discovery extract of that seed (original discovery sampling weights),
stored in `frozen_models.json`, and applied unchanged to the new images. Nothing is refitted.

Shifts: size-scaled δ = κ√(wh), κ = 0.0625 (primary) and fixed 1 px (secondary), four axis directions.
Replay is native and with eligibility locked at the base state. Outcomes, each "any direction":

- native O2O identity change (primary);
- lock-persistent: the identity also changes with eligibility locked;
- lock-removed: a native change that does not occur with eligibility locked, in at least one direction.

An object can be in both lock outcomes. Missed detection: Section 6.

## 5. Hypotheses, estimators and decision rules

Every estimate is the equal-weight mean of the four per-seed estimates on the primary images. Intervals
are 95% percentile bootstrap (5,000 replicates, seed 20261123; 1,000 for sensitivity outputs) over source scenes, crossed with
equal-probability resampling of the four seeds (seed 20261124). The scene draw is the same for both
models, all endpoints and all seeds.

Fixed sequence; a later test is confirmatory only if every earlier one is met:

| Test | Estimate | Met if |
|---|---|---|
| P1 | ROC-AUC of M3 minus M2, native change, size-scaled | lower bound > 0 |
| P2 | AUC gain for lock-persistent minus AUC gain for lock-removed, size-scaled | lower bound > 0 |
| P3 | recall of changed objects when the 20% with the lowest margin are audited, minus the same for the 20% with the highest M2 risk, native change, size-scaled | lower bound > 0 |

Discovery reference, same code and coefficients on the 300 discovery images (in-sample, full support,
unweighted pooled AUC, scene bootstrap, 5,000 replicates): P1 0.109 [0.088, 0.130], P2 0.175 [0.147, 0.204], P3 0.192 [0.145, 0.244].
These differ from the values in S18 because S18 used out-of-fold, design-weighted AUCs averaged over
folds. Attenuation on the new images is judged against these reference values
(`rehearsal_decisions.json`), and the in-sample fit makes them upper bounds.

P1 and P2 are the conditions carried over from S18. P3 asks whether ranking by margin alone is a useful
audit rule. Results in either direction are reported.

## 6. Missed detections

The extractor was checked on the discovery images of seed 0 before the freeze. Its miss flag (no
same-class end-to-end detection with confidence > 0.001 and IoU ≥ 0.5) differed from the published flag
for 163 of 6,410 objects; the cause was not established. The rule written before that check was to drop
the miss endpoint from the confirmatory sequence on any mismatch, so it is dropped. The AUC gain for the
extractor's own miss flag is reported descriptively as a different definition.

## 7. Prespecified sensitivity and descriptive analyses

Reported whatever they show: image-level bootstrap (stratified); seeds fixed rather than resampled;
per-seed results; raw margin instead of clipped; valid-order support (margin ≥ 0, models fitted on the
valid-order discovery objects), which is the support of Table S50; fixed 1 px outcomes; average
precision; recall at 10% and 30% budgets; margin ranked alone; the all-eligible and fully disjoint sets;
objects without a runner-up kept and ranked last; counts of objects without an assigned candidate or a
runner-up. No multiplicity correction is made for these outputs.

## 8. Gate and run rule

`gate.py` requires the batched extractor to reproduce, on the 300 discovery images and all four seeds,
the published base fields, the native and locked direction flags and the size-scaled fragility with no
mismatch, and the published seed-0 out-of-fold AUCs to 1e-6. Extraction prints no outcomes. The formal
analysis runs once. A rerun is allowed only for an infrastructure failure, is logged in `timeline.log`,
and changes no setting. Deviations are listed in `RESULTS.md`.

## 9. Not replicated

The continuous-trajectory analysis of Table S49 (dense shift grid, seed 0 only) and VisDrone.
