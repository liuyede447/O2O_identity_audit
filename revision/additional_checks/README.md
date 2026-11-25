# Additional revision analyses

This directory provides portable CPU recomputation for the validation-split
margin replication, screening by margin availability, augmentation controls,
first divergence, and the paired half-pixel check. Run these commands in
this directory using Python 3.13.5 and the pinned dependencies:

    python -m pip install -r requirements.txt
    python -B verify.py --only controls
    python -B verify.py --only margin
    python -B verify.py --only two-track

The verifier restores derived input tables under inputs/, checks SHA-256 and
compares results under out/ with archived results in expected/. The default
5,000 main margin resamples can take several minutes on a CPU. For a
point-estimate smoke check, append --margin-draws 100 to either margin command.
The reported resample count in an archive refers to the original run.

| Evidence | Calculation |
| --- | --- |
| Supplementary Section 19, Table S51 | analyze.py applies frozen discovery coefficients to the official validation inputs |
| Section 19.1 | miss_comparison.py compares coverage flags and reruns grouped out-of-fold ROC-AUC |
| Table S52 | posthoc_two_track.py recomputes screening on the assigned pool |
| Section 20.1 and Table S53 | augmentation.py and analyze_controls.py recompute the augmentation comparisons |
| Table S54 | first_divergence.py recomputes direction-weighted shares |
| Section 20.3 and Table S55 | half_pixel_final.py uses matched outcome rows and stable image ordering |

The freeze record is a local attestation. Its workstation path prefixes
have been removed, and its original SHA-256 is recorded. The optional dataset
replay and image/model prerequisites in the parent reproduction guide are
separate from these CPU checks. The archive contains derived CSV records and
model coefficients, with no image pixels, dataset label files, or weights.
