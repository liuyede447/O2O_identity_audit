# Data dictionary and manuscript mapping

## Primary audit files

Each dataset directory contains:

- `selected_images.csv`: outcome-blind selected image IDs, strata, and
  inclusion probabilities;
- `sampling_manifest.json`: sample size, seed, contract, and source hashes;
- `per_gt.csv`: one row per audited ground truth with scale, native identity,
  runner-up, margin, and replay state fields;
- `outcomes/per_gt_outcomes.csv`: object-level FN@IoU50 outcomes joined by
  image and ground-truth identity;
- `weighted_models_v1/summary.json`: inverse-probability weighted estimates and
  5,000 image-cluster bootstrap intervals;
- descriptive summaries where available.

## Evidence families

| Directory | Manuscript role |
|---|---|
| `data/primary/aitodv2` | AI-TOD-v2 population-facing estimates |
| `data/primary/visdrone` | VisDrone population-facing replication |
| `data/specificity/yolo26_same_model` | paired O2M/O2O difference-in-scale-gradients |
| `data/architecture/yolov10s_aitod` | compatible-detector replication |
| `data/extended_training/seed_*` | 300-epoch seed stability checks |
| `data/checkpoint_matrix` | independently fitted checkpoint rows in Figure 2C,D |
| `data/derived/table2_specificity.csv` | manuscript Table 2 |
| `data/derived/table3_associations.csv` | manuscript Table 3 |
| `data/derived/table4_contrasts.csv` | manuscript Table 4 |
| `data/derived/same_stride_conditional_summary.json` | post hoc strict same-stride sensitivity |
| `data/derived/table5_interventions.csv` | manuscript Table 5 and Figure 3A,B |
| `data/derived/figure3_transfer_gate.csv` | Figure 3C transfer-gate points/intervals |
| `data/derived/qualitative_trace_v1.json` | Figure 4 identity/box trace metadata; no pixels |

## Core fields

Field names vary slightly across detector contracts. Important concepts are:

- `image`: stable dataset image identifier;
- `gt_index` / `gt_id`: ground-truth index within the selected image;
- `size_bin`: scale category based on object extent;
- `active_identity`: post-resolution loss-active O2O candidate;
- `runner_up`: highest valid positive alternative for the same ground truth;
- `margin`: active-relative native alignment margin;
- replay-state fields: base and legal one-input-pixel shifts;
- `sampling_weight`: inverse image-inclusion probability where recorded;
- `fragile`: any-direction identity change indicator;
- `fn_iou50`: object-level false negative at IoU 0.5 under the frozen
  prediction contract.

Missing active identity or missing valid runner-up defines an **undefined**
competition case. Undefined cases are excluded from the continuous-margin
model without imputation.
