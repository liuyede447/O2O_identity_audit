# Phase 1 low-fidelity QA

Overall status: **PASS**

| Check | Status | Detail |
|---|---|---|
| evidence manifest allows low-fidelity | PASS |  |
| final figures remain gated | PASS |  |
| all evidence sources exist | PASS | 67 |
| all evidence source hashes match | PASS |  |
| Figure 1 editable PPTX exists | PASS |  |
| Figure 1 layout stays inside slide bounds | PASS |  |
| Figure 1 minimum visible font >= 7 pt | PASS | 8.6325 |
| Figure 1 body text uses dark neutral/white only | PASS | [] |
| Figure 1 slide text is Arial only | PASS | ["Arial"] |
| Figure 1 low-fi contains no raster or historical GT overlay | PASS |  |
| Figure 2 panel-source row count | PASS | 30 |
| Figure 2 source CSV hash binding | PASS |  |
| Figure 2 source hash: results/five_experiment_upgrade_20260830/learning_dynamics/learning_dynamics_trajectory.csv | PASS |  |
| Figure 2 source hash: results/reviewer_controls_20260829/formal_s0e300_k00625/normalized_perturbation_summary.json | PASS |  |
| Figure 2 source hash: results/measurement_validation_20260831/fixed_branch_replay_v1/normalized_perturbation_summary.json | PASS |  |
| Figure 2 source hash: results/stress_contract_sensitivity_20260829/frozen_matrix/kappa_sensitivity_matrix.csv | PASS |  |
| Figure 2 source hash: results/five_experiment_upgrade_20260830/yolov10_normalized_k00625/normalized_perturbation_summary.json | PASS |  |
| Figure 2 SVG keeps editable text | PASS |  |
| Figure 2 SVG uses Arial | PASS |  |
| Figure 2 PNG is 600 dpi | PASS | {"pixels": [4147, 2751], "dpi": 599.9988} |
| Figure 2 PDF embeds editable Arial TrueType | PASS | ["ELUGPM+ArialMT                       CID TrueType      Identity-H       yes yes yes     19  0", "ELUGPM+Arial-BoldMT                  CID TrueType      Identity-H       yes yes yes     26  0"] |
| Figure 2 prohibited interpretation flags are false | PASS | {"architecture_generalisation_claim": false, "archived_stage_interpolation": false, "continuous_dose_response_claim": false, "same_mechanism_claim": false} |
| latest rendered previews visually inspected for overlap/clipping | PASS | "manual visual audit recorded 2026-09-04" |

PASS authorizes Phase 2 data-figure production only; final figures remain blocked by FigS1 checkpoint identity conflict.
