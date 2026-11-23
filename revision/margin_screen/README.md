# Base-margin diagnostics

Recomputes the weighted first-event incidences by base-margin quintile and the
image-grouped out-of-fold discrimination checks reported in Supplementary
Section 18 and Tables S49–S50. The two trajectory readouts are fixed-pair rank
crossing and the first change of the replayed O2O identity. Endpoint models use
frozen image folds and cover native identity changes, changes under eligibility
locking, native changes prevented by locking, and coverage-based misses where
available. The analysis was specified after the initial exploratory inspection.

## Run and verify

From the repository root with Python 3.13.5 and the pinned dependencies:

```sh
python -m pip install -r revision/margin_screen/requirements.txt
python -B revision/margin_screen/margin_screen.py --output-dir margin_outputs
python -B revision/margin_screen/verify_outputs.py margin_outputs
```

The full run uses 5,000 stratified image-cluster draws. Endpoint summaries also
include the crossed seed-and-image resampling intervals. The verifier checks all
five output tables against the archived values. The full run has been exercised
on Windows and can take several minutes on a CPU.

## Inputs

The compressed CSVs contain the derived scores, event labels, sampling weights and
fold identifiers required for the analysis. Image names are replaced with stable
aliases that preserve the ordering used for stratified resampling. The manifest
lists the columns, row counts and SHA-256 values for every input. No benchmark
images, raw annotations, model weights or prediction dumps are needed for this
calculation. The root README identifies the original dataset providers for users
who wish to run the detector replay.

The order-restricted endpoint analysis omits negative margins. Full-support
comparisons report raw and clipped margins. The interval labels in
`four_seed_auc_gain.csv` distinguish image-cluster resampling (`lo`, `hi`)
from crossed seed-and-image resampling (`seed_lo`, `seed_hi`).
