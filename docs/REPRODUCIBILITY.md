# Reproducibility guide

## Integrity-only reproduction

```bash
python scripts/verify_release.py
```

This verifies every released file against `SHA256SUMS.csv`, checks the expected
300-image primary samples, parses all JSON/CSV files, and confirms that no model
weights are bundled.

## Rebuild Figure 2 from released data

```bash
python -m pip install -r requirements.txt
python figures/make_figure2.py
```

The checkpoint-row forest plot reads
`data/checkpoint_matrix/cross_dataset_checkpoint_matrix.csv`. The same-model
rates, undefined-competition rates, and compatible-detector row are frozen in
the plotting script from the manuscript-facing evidence table.

## Refit the primary weighted models

AI-TOD-v2 example:

```bash
python scripts/weighted_random_stratified_models.py \
  --selected-images data/primary/aitodv2/selected_images.csv \
  --audit data/primary/aitodv2/per_gt.csv \
  --outcomes data/primary/aitodv2/outcomes/per_gt_outcomes.csv \
  --output-dir outputs/aitodv2_weighted \
  --replicates 5000 \
  --seed 20260823
```

Replace `aitodv2` with `visdrone` for the second dataset.

## Reproduce the post hoc strict same-stride sensitivity

```bash
python scripts/same_stride_conditional_analysis.py \
  --ai-selected data/primary/aitodv2/selected_images.csv \
  --ai-o2o data/primary/aitodv2/per_gt.csv \
  --ai-o2m data/specificity/yolo26_same_model/per_gt.csv \
  --vis-selected data/primary/visdrone/selected_images.csv \
  --vis-o2o data/primary/visdrone/per_gt.csv \
  --output-dir outputs/same_stride \
  --replicates 5000 \
  --seed 20260824
```

The strict object-level subset requires every recorded legal replay to retain
an active candidate on the base stride. Shifted-active-missing and cross-stride
ground truths are excluded. Conditioning can induce selection, so this is a
post hoc sensitivity rather than a replacement for the primary analysis.

## Run a new read-only audit

```bash
python scripts/assignment_stability_random_stratified.py \
  --checkpoint /path/to/checkpoint.pt \
  --expected-sha256 <full-checkpoint-sha256> \
  --data configs/aitod_v2.yaml \
  --output-dir outputs/aitodv2_audit \
  --sample-images 300 \
  --seed 20260823 \
  --imgsz 800 \
  --device 0
```

The audit selects images before inference and does not use margin, fragility,
detection outcomes, or false-negative labels during sampling.

## Reproducibility boundary

- Exact audit-table and bootstrap reproduction is possible from the released
  derived CSV/JSON files.
- Full detector replay requires the third-party datasets and a checkpoint whose
  hash matches the model registry.
- Training reproduction is outside this lightweight release because checkpoint
  redistribution and the complete upstream training implementation require
  separate licence review.
