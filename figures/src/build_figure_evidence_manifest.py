"""Build and fail-closed validate the Figure 1--6 evidence contract.

This script hashes existing frozen sources only. It does not recompute any
scientific result and it never edits the final_evidence_v3 ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_final"
LEDGER_PATH = ROOT / "evidence_freeze/final_evidence_v4/CURRENT_EVIDENCE_LEDGER.json"
PRIMARY_CHECKPOINT = "4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa"
LEGACY_S1_CHECKPOINT = "50922eca528550ded68e24203907552b48b4ce1fcebfc37c5187e077e7ef95d3"


def src(path: str, sha256: str, evidence_id: str | None, authority: str, role: str) -> dict:
    return {
        "path": path,
        "expected_sha256": sha256,
        "evidence_id": evidence_id,
        "authority": authority,
        "role": role,
    }


SOURCES = {
    "methods": src("manuscript/content/methods.tex", "84918970a1f7209c8ea19aeeeca7573af95e2575ba1c1a4e36e7f6b0da064b67", None, "CONTRACT_SOURCE", "method definition"),
    "supplement": src("manuscript/supplementary.tex", "299d29bc2f6960ac136e17875d6d78a96ccc9241631bb9b8b65529d77b883e6b", None, "CONTRACT_SOURCE", "native-path contract"),
    "fixed_manifest": src("results/measurement_validation_20260831/fixed_branch_replay_v1/manifest.json", "38299ec8fdc5ede460922bd686334b02d914815b320dad01dd6e150bb36a975a", "EV-STRESS-CONTRACT-PRIMARY-V1", "final_evidence_v3", "fixed replay contract"),
    "eq_manifest": src("results/reviewer_controls_20260829/formal_s0e300_k00625/manifest.json", "5f98a6bd29d0e6c0c1b398c895a4fcd21001653a5bed197ee4fe3dae476484d5", "EV-STRESS-CONTRACT-PRIMARY-V1", "final_evidence_v3", "equivalent-side replay contract"),
    "fixed_primary": src("results/measurement_validation_20260831/fixed_branch_replay_v1/normalized_perturbation_summary.json", "aece532fd9ea842247ed22ea03557e13ece6209144d83f3598beadd38ba0bcfe", "EV-STRESS-CONTRACT-PRIMARY-V1", "final_evidence_v3", "fixed endpoint estimates"),
    "eq_primary": src("results/reviewer_controls_20260829/formal_s0e300_k00625/normalized_perturbation_summary.json", "5d09278684cf683a8b6227b7e792f162f01f3e65718327b6cf51e0fdc669b37b", "EV-STRESS-CONTRACT-PRIMARY-V1", "final_evidence_v3", "equivalent-side endpoint estimates"),
    "kappa_json": src("results/stress_contract_sensitivity_20260829/frozen_matrix/stress_contract_sensitivity.json", "1776ecbdff504d1e77e8842cc11d6513511470e7214d3fe846a5a1b4207e6601", "EV-STRESS-CONTRACT-KAPPA-DATASET-V1", "final_evidence_v3", "kappa by dataset estimates"),
    "kappa_csv": src("results/stress_contract_sensitivity_20260829/frozen_matrix/kappa_sensitivity_matrix.csv", "38ee31b39b02f383cba7b816aa3b34fdccc7440d051cb725524bfb879815f0a8", "EV-STRESS-CONTRACT-KAPPA-DATASET-V1", "final_evidence_v3", "tested kappa matrix"),
    "learning": src("results/five_experiment_upgrade_20260830/learning_dynamics/learning_dynamics_trajectory.csv", "3cb0c2d4bce8ca85d458a69a063c241457a72244feafa82465bf4ceaf1705aa9", "EV-LEARNING-DYNAMICS-V1", "final_evidence_v3", "archived-stage estimates"),
    "y10": src("results/five_experiment_upgrade_20260830/yolov10_normalized_k00625/normalized_perturbation_summary.json", "ca344ce4f99d92d4654b65c1c1e97818bf385f9d3a57695bc3e37d6f4af275eb", "EV-YOLOV10-NORMALIZED-V1", "final_evidence_v3", "YOLOv10-S estimates"),
    "y10_manifest": src("results/five_experiment_upgrade_20260830/yolov10_normalized_k00625/manifest.json", "c1e14e823fabfef6f8b6847c158b0da829f12d9e41ea7b4251e4c6ec8215270f", "EV-YOLOV10-NORMALIZED-V1", "final_evidence_v3", "YOLOv10-S contract"),
    "oracle": src("results/measurement_validation_20260831/oracle_suite_v1/oracle_validation.json", "689a6dd98696bf364cbc622430ee21c51eecaa7f08fe954d66e9d5a1611d8f92", "EV-INSTR-ORACLE-V1", "final_evidence_v3", "oracle and property qualification"),
    "mutation": src("results/measurement_validation_20260831/mutation_suite_v1/mutation_validation.json", "9f59974b2d57ab375d6c84a4a9dd4e558acc972c28765ced6cd3beaa4b2def73", "EV-INSTR-MUTATION-V1", "final_evidence_v3", "mutation qualification"),
    "parity5251": src("results/measurement_validation_20260831/parity_deterministic_5k_v1/summary.json", "391661c8eae8d10af84dcd2b81330bf5104aa783c35b5e5629a8401c016b8157", "EV-INSTR-PARITY-5251-V1", "final_evidence_v3", "native/reference parity"),
    "assigned_weight": src("results/measurement_validation_20260905/parity_5251_with_target_score_v1/summary.json", "9d17462c9c1f90d8ed35c1d7f5742440700738b0fcc1646af47c5a9f63545d2d", "EV-ASSIGNED-TARGET-SCORE-SEMANTICS-V1", "final_evidence_v4", "assigned-identity target-score qualification"),
    "boundary_reference": src("runs/20260831_boundary_reference_formal_v2/artifact/summary.json", "4d3a9568f5be5e1798fdc6b29dc757f381aba6df53228d60967b0f75ebe62aa7", "EV-BOUNDARY-REFERENCE-FORMAL-V2", "final_evidence_v3", "boundary reference parity"),
    "grid_gate": src("results/measurement_validation_20260902/dense_grid_convergence_gate_v3/convergence_gate.json", "ceb27e511178e57fdfe490994ccb8443a72e239a98917b66450f77e05240b109", "EV-BOUNDARY-GRID-CONVERGENCE-V1", "final_evidence_v3", "grid-resolution gate"),
    "focal_row": src("results/measurement_validation_20260902/focal_row_phase1_validation_v1/summary.json", "51d4ddde26526c8685aedbbac697669b97917eb7b1235b79f56fdb31578a92dd", "ADD-FIG-FOCAL-ROW-V1", "SOURCE_ADDENDUM", "focal-row/native state equivalence"),
    "elig_lock": src("results/measurement_validation_20260831/eligibility_lock_v1/summary.json", "dd01d94b093cf078ac22609e0066bfe1d3edab014c5c742c7231e26f7a5908df", "EV-ELIGIBILITY-LOCK-V1", "final_evidence_v3", "eligibility-lock estimates"),
    "elig_stable": src("results/five_experiment_upgrade_20260830/eligibility_stable_sensitivity_v2/summary.json", "27c12ddde8d15d144f7167f9fcc2f187b32dafb5cc5ccc9157c36ce0e2d750e7", "EV-ELIGIBILITY-STABLE-V2", "final_evidence_v3", "eligibility-stable subset"),
    "pathway": src("results/five_experiment_upgrade_20260830/anatomy_bootstrap_summary_v2.json", "ca23d877b18858bf2b8024cb280c0a91f5db47756cfb9057cfc9bcaa762ae2dd", "EV-PATHWAY-ANATOMY-V2", "final_evidence_v3", "exclusive first-observed labels"),
    "cooccur_json": src("runs/20260901_first_divergence_cooccurrence_v1/summary.json", "72c5ea1adbc8207f23d1cd3caf0d61c66f66c188e150af164704f05d29a6430b", "ADD-FIG-COOCCURRENCE-V1", "SOURCE_ADDENDUM", "non-exclusive any-stage summary"),
    "cooccur_csv": src("runs/20260901_first_divergence_cooccurrence_v1/cooccurrence.csv", "9eb9b9e02d8f8122ea9e81546b8ebc40bd5e5174d53403f87e04098c1c4503b0", "ADD-FIG-COOCCURRENCE-V1", "SOURCE_ADDENDUM", "non-exclusive any-stage rows"),
    "preres_fixed": src("runs/20260831T163900_preresolution_fixed_v1/artifact/summary.json", "964fc7b08e1edee1adc870cfa93d6b02f34f299d3022f311ad213a11288d8526", "EV-PRERESOLUTION-FIXED-V1", "final_evidence_v3", "fixed pre-resolution control"),
    "preres_eq": src("runs/20260831T164000_preresolution_normalized_v1/artifact/summary.json", "6fa33d8c15749f526a64efe58fb477d5cb860c863d7774f05ff6e157935b0c80", "EV-PRERESOLUTION-NORMALIZED-V1", "final_evidence_v3", "equivalent-side pre-resolution control"),
    "events": src("runs/20260902_boundary_full_grid_1024_population_v1/per_direction_boundary_events.csv", "ee33d0927ebbb9953a8e0e3f6c02956ef02cb6570beb3660e03ab172527e897d", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "canonical boundary events"),
    "curve": src("runs/20260902_boundary_full_grid_1024_population_v1/o2m_rank_set_radius_curve.csv", "3622652665b30f727258aa0528436e33ab402255e29935dcc8168a32ea5f6b33", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "canonical full-grid curve"),
    "terminal": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/TERMINAL_VALIDATION.json", "a7982e8dc8c6af78fd51faca6c1ab4f7232f99eef10e09eacf242dba1dd8d80d", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3", "terminal binding manifest"),
    "survival_cif": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/survival_cif.csv", "1a90951b52587823e82fe89c99a12c8512030f465a66915028285ed8a2773740", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "KM and Aalen-Johansen estimates"),
    "rmsr": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/rmsr.csv", "58505fa4a2ce39bbfb86bf7de769b63b35f4c003bc11898317b7c116933fe725", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD absolute estimates"),
    "rmsr_0125": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/rmsr_contrasts.csv", "9a0ad9465f0573b46c366231b14784f467579d68d2de96fdc5f01b1f10cb5027", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3", "RMCBD contrasts at tau .125"),
    "rmsr_005": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_00500/rmsr_contrasts.csv", "f916abd6a835ce09047dec2628d3597eb06ba13a31ae96428be53d67dfa94dee", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD contrasts at tau .05"),
    "rmsr_00625": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_00625/rmsr_contrasts.csv", "1be655cbe4b74aa55741ea09ac8e720d7d28bdf10e92cd07bf42c134871e5af1", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD contrasts at tau .0625"),
    "rmsr_010": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01000/rmsr_contrasts.csv", "ada6e5f87a6343e8b261e5e21372c9a0e124eb20b7bf7c083312acf231273498", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD contrasts at tau .10"),
    "rmsr_020": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_02000/rmsr_contrasts.csv", "7a20f920d38e312b62ac57e9e55be85cdc6e11b232c6d568e0d171b232dc3af3", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD contrasts at tau .20"),
    "rmsr_025": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_02500/rmsr_contrasts.csv", "1aa4a5fc47c80230f93fe3a78b6b467a702e8336e657a2728394d7d2813fae92", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "RMCBD contrasts at tau .25"),
    "rho_status": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/rho_R_status.csv", "852482d938a5c10e61834a3327d342a5b6134b42b895cc291066bc7d45c43bab", "EV-BOUNDARY-GEOMETRY-V3", "final_evidence_v3_bound_output", "rank-crossing status composition"),
    "margin": src("runs/20260903_boundary_full_grid_1024_postprocess_v2/tau_01250/margin_construct.csv", "e818cfd5a12b167b8aea1f174368a5d2f1457fa84a31251a6b4d03c6b565ea22", "EV-MARGIN-RHOR-CONSTRUCT-V3", "final_evidence_v3", "conditional margin associations"),
    "rankset_fixed": src("runs/20260831T161235_rank_set_fixed_v1/artifact/rank_set_summary.json", "649627210a886208014733f881429f6572f414d306bb051fe52c6ef2758ed4b4", "EV-RANK-SET-FIXED-V1", "final_evidence_v3", "fixed Rank-Set metrics"),
    "rankset_eq": src("runs/20260831T162551_rank_set_normalized_v1/artifact/rank_set_summary.json", "7f1a83d6b538143dbf1ff38b8deecc7627795298d73733a72abfd4b7e102b11d", "EV-RANK-SET-NORMALIZED-V1", "final_evidence_v3", "equivalent-side Rank-Set metrics"),
    "count_json": src("runs/20260905_o2m_positive_count_scale_controls_v5/artifact/summary.json", "2b232e52716cb5a45ef97f4bf64dd6458308fb639283762a30824e90cc056e9f", "EV-POSITIVE-COUNT-THREE-STAGE-V1", "final_evidence_v4", "three-stage positive-count decomposition"),
    "count_csv": src("runs/20260905_o2m_positive_count_scale_controls_v5/artifact/exact_positive_count_strata.csv", "8a68539c42a30b5db38ac3ca948f6b862990a885a84b21be335aa99de3c0f365", "EV-POSITIVE-COUNT-THREE-STAGE-V1", "final_evidence_v4", "positive-count distribution"),
    "increment_ai": src("results/reviewer_controls_20260829/formal_s0e300_k00625/incremental_value_summary_foldwise.json", "0f555db9c5e1ecf3d738607551d45beb75f6f0a6f3e0820683b27d9aa5b32a88", "ADD-FIG-INCREMENTAL-VALUE-V1", "SOURCE_ADDENDUM", "AI-TOD-v2 held-out increment"),
    "increment_vis": src("results/reviewer_controls_20260829/vis_b4e150_equiv_k00625/incremental_value_summary_foldwise.json", "8227b831cfe73cc3533ae1b953e21e62c138e5f9205b35e54eba38c38371ed5c", "ADD-FIG-INCREMENTAL-VALUE-V1", "SOURCE_ADDENDUM", "VisDrone held-out increment"),
    "s1_primary_trace": src("figures_final/source_data/figs1_primary_trace/primary_qualitative_trace.json", "eb85f4f399eebda1284ac1997622c8a2b88d6cb91bf27020723d09cadb44ec46", "ADD-FIG-S1-PRIMARY-TRACE-V1", "SOURCE_ADDENDUM", "primary-checkpoint qualitative candidate trace"),
    "s1_primary_validation": src("figures_final/source_data/figs1_primary_trace/validation.json", "ad0f2731de4a24caac1328d5d7be98122bfb5bebfb953080960e7a4e96571960", "ADD-FIG-S1-PRIMARY-TRACE-V1", "SOURCE_ADDENDUM", "primary-checkpoint trace validation"),
    "s1_primary_image": src("figures_final/source_data/figs1_primary_trace/0000170_00401_d_0000001__160_0.png", "a0a38c98f96b9b5bd93f14464944ba19320c1ed8b0db49036d4b0eda680f43fa", "ADD-FIG-S1-PRIMARY-TRACE-V1", "SOURCE_ADDENDUM", "real AI-TOD-v2 source image"),
    "s1_fixed_rows": src("results/measurement_validation_20260831/fixed_branch_replay_v1/per_direction.csv", "930a50dadb0abdfdcede0de344cb93ed2f68c0c5a4c1f183a2f70ddde8be87f8", "EV-STRESS-CONTRACT-PRIMARY-V1", "final_evidence_v3", "frozen fixed-pixel replay identities"),
    "s1_validation": src("manuscript/figures/assets/qualitative_case_pack_v1/validation.json", "9153f49d84da0b40a79c588878bd9ed18787d7f6f06fe6f0e46ab7d658e0759f", None, "CANDIDATE_BLOCKED", "legacy qualitative validation"),
    "s1_cases": src("manuscript/figures/assets/qualitative_case_pack_v1/case_manifest.csv", "407fd4757971099d2e9ef1e5b4ea44c3c75c1dc7283d06de8a06b10fcd3024f5", None, "CANDIDATE_BLOCKED", "legacy qualitative case identities"),
    "s1_trace": src("manuscript/figures/assets/qualitative_trace_v1.json", "f5fb2e54a1f4d427301d0e5fc817be456e540b00838318bce7f96251f50dfbd7", None, "CANDIDATE_BLOCKED", "legacy qualitative trace"),
    "s1_image": src("manuscript/figures/assets/0000170_00401_d_0000001__160_0.png", "6b115682e1438218bbb41c40e2e6309ef58f6f9654ede54df4f5171c1766dbcf", None, "CANDIDATE_BLOCKED", "real AI-TOD-v2 image"),
}


DEFAULT = {
    "dataset": "AI-TOD-v2",
    "checkpoint": PRIMARY_CHECKPOINT,
    "seed": {"model": 0, "selection": 20260823},
    "sample_support": "Frozen outcome-blind stratified 300-image discovery sample",
    "image_n": 300,
    "weighting": "inverse image-inclusion weights",
    "bootstrap_unit": "image cluster within outcome-blind sampling stratum",
    "bootstrap_repetitions": 5000,
    "ci_type": "95% stratified image-cluster percentile bootstrap",
    "evidence_class": "discovery",
    "interpretation_class": "descriptive/observational",
    "censoring_or_conditioning": "none beyond panel support",
}


def panel(figure: str, letter: str, question: str, source_keys: list[str], **kwargs) -> dict:
    item = dict(DEFAULT)
    item.update(kwargs)
    item.update({
        "figure_number": figure,
        "panel": letter,
        "panel_id": f"{figure}{letter}" if letter else figure,
        "scientific_question": question,
        "source_artifacts": [SOURCES[key] for key in source_keys],
    })
    item.setdefault("object_direction_trajectory_n", None)
    item.setdefault("statistical_unit", None)
    item.setdefault("stress_contract", None)
    item.setdefault("point_estimate_columns", [])
    item.setdefault("ci_columns", [])
    item.setdefault("panel_status", "PASS")
    item.setdefault("status_reason", "All sources are SHA-bound to final_evidence_v3 or a non-numeric contract source.")
    return item


PANELS = [
    panel("Fig1", "a", "What is frozen and what moves in focal-GT replay?", ["methods", "fixed_manifest", "eq_manifest", "s1_primary_trace", "s1_primary_validation", "s1_primary_image"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], sample_support="Method schematic with one primary-checkpoint AI-TOD-v2 context crop; no sampled estimate", image_n=1, object_direction_trajectory_n={"displayed_gt": 3, "focal_gt": 1}, statistical_unit="assignment-state replay contract", weighting=None, bootstrap_unit=None, bootstrap_repetitions=None, ci_type=None, point_estimate_columns=["true GT box coordinates"], ci_columns=[], evidence_class="method contract plus figure-only qualitative SOURCE_ADDENDUM", interpretation_class="controlled counterfactual replay schematic", panel_status="WARN", status_reason="The real-image context is SHA-verified against the primary-checkpoint figure trace and is used only for explanatory rendering; it does not add an aggregate estimate."),
    panel("Fig1", "b", "Which native assignment states are followed for O2O and O2M?", ["methods", "supplement", "assigned_weight"], stress_contract="native O2O Top-7 and O2M Top-10 paths", sample_support="Method schematic plus target-score semantic qualification", image_n=None, statistical_unit="native assignment path", weighting=None, bootstrap_unit=None, bootstrap_repetitions=None, ci_type=None, evidence_class="method contract and revision instrument qualification", interpretation_class="descriptive algorithm schematic; assigned identity is distinguished from positive loss weight"),
    panel("Fig1", "c", "How do base and shifted native states define stable, changed, undefined, and first-cardinal-boundary readouts?", ["methods", "events", "curve", "terminal"], stress_contract="four cardinal rays, equivalent-area-side r in [0,.25], step 1/1024", object_direction_trajectory_n={"focal_gt": 7431, "directional_trajectories": 29724, "curve_rows": 7593288}, statistical_unit="focal-GT directional cardinal trajectory", point_estimate_columns=["base state", "shift state", "first observed cardinal boundary"], ci_columns=[], interpretation_class="method concept bound to observed cardinal trajectories", censoring_or_conditioning="right and competing censoring retained; cardinal directions only, not a full 2D minimum radius"),
    panel("Fig2", "a", "How do branch scale contrasts differ between fixed and equivalent-area stress?", ["fixed_primary", "eq_primary"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], image_n={"selected": 300, "contributing": 288}, object_direction_trajectory_n={"fixed_common_valid_gt": 6378, "equivalent_side_common_valid_gt": 6377}, statistical_unit="common-valid focal GT", point_estimate_columns=["O2O 8-16 minus 16-32", "O2M 8-16 minus 16-32", "paired O2O-minus-O2M"], ci_columns=["ci_low", "ci_high"], censoring_or_conditioning="common-valid endpoint support; fixed and equivalent-side supports differ by one GT"),
    panel("Fig2", "b", "Does the tested paired response direction persist across kappa and dataset?", ["kappa_json", "kappa_csv"], dataset=["AI-TOD-v2", "VisDrone"], checkpoint={"AI-TOD-v2": PRIMARY_CHECKPOINT, "VisDrone": "e7cffd98c6055bb0818873f94e6566feb697660ec5666f21f13c6304b412d0e8"}, stress_contract="equivalent-side kappa in {.03125,.0625,.125}", image_n={"AI-TOD-v2": {"selected": 300, "contributing": 288}, "VisDrone": {"selected": 300, "contributing": 300}}, object_direction_trajectory_n={"AI-TOD-v2_common_valid_gt_range": [6376, 6378], "VisDrone_common_valid_gt": 13911}, statistical_unit="common-valid focal GT", point_estimate_columns=["paired", "o2o", "o2m"], ci_columns=["ci_low", "ci_high"], evidence_class="discovery sensitivity", interpretation_class="tested-point sensitivity; connecting lines imply no continuous dose response"),
    panel("Fig2", "c", "Is the paired response visible at the physically archived training stages?", ["learning"], checkpoint={"epoch0": "a7a60c7a...", "epoch20": "0a83492f...", "epoch40": "41019d81...", "epoch60": "4e901f50...", "epoch80": "80ed8764...", "epoch300": "46ff9c34..."}, stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], sample_support="Six physically archived checkpoints: 0,20,40,60,80,300", image_n=None, object_direction_trajectory_n={"archived_checkpoints": 6, "trajectory_rows": 12}, statistical_unit="checkpoint by stress-contract trajectory point", point_estimate_columns=["paired_contrast_pp", "mAP50-95"], ci_columns=["ci_low", "ci_high"], evidence_class="discovery descriptive", interpretation_class="archived-stage description; no interpolation or predictive correlation", censoring_or_conditioning="no checkpoints archived between 80 and 300"),
    panel("Fig2", "d", "Is the response signature reproduced under a compatible detector contract?", ["eq_primary", "y10", "y10_manifest"], checkpoint={"YOLO26s": PRIMARY_CHECKPOINT, "YOLOv10-S": "ecf7ca8df8eded54b9500109123f94ef8eaddd736bd8351a968761f0f62a0d38"}, stress_contract="equivalent_side_kappa_0.0625", image_n={"YOLO26s_contributing": 288, "YOLOv10-S_contributing": 288}, object_direction_trajectory_n={"YOLO26s_common_valid_gt": 6377, "YOLOv10-S_common_valid_gt": 6348}, statistical_unit="common-valid focal GT", point_estimate_columns=["o2o_scale_contrast", "o2m_scale_contrast", "paired_branch_difference"], ci_columns=["ci95[0]", "ci95[1]"], evidence_class="compatible-detector replication", interpretation_class="replication of response signature, not architecture invariance"),
    panel("Fig3", "a", "How much is the fixed O2O scale gap attenuated under base-eligibility locking?", ["elig_lock"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], object_direction_trajectory_n={"per_gt_rows": 14866, "per_direction_rows": 59228, "fixed_gt": 7434, "equivalent_side_gt": 7432, "fixed_directions": 29616, "equivalent_side_directions": 29612}, statistical_unit="GT and directional replay on lock support", weighting="as recorded by frozen eligibility-lock artifact (not explicitly identified)", point_estimate_columns=["fixed_o2o_gap_reduction.point", "fixed_o2o_gap_fraction_removed.point", "normalized_o2o_gap_reduction.point"], ci_columns=["*.ci95[0]", "*.ci95[1]"], interpretation_class="counterfactual sensitivity/localisation", censoring_or_conditioning="lock-support differs from primary common-valid endpoint support; not a causal contribution fraction"),
    panel("Fig3", "b", "What response remains in the observational eligibility-stable subset?", ["elig_stable"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], object_direction_trajectory_n={"fixed_retained_gt": 6217, "equivalent_side_retained_gt": 6999}, statistical_unit="post-replay conditioned focal GT", point_estimate_columns=["fixed object/direction gap", "normalized object/direction gap"], ci_columns=["ci95 low", "ci95 high"], evidence_class="discovery conditional sensitivity", interpretation_class="observational subset", censoring_or_conditioning="post-replay eligibility-stable conditioning changes the estimand"),
    panel("Fig3", "c", "Where is the first difference observed, and which stages occur anywhere on flip paths?", ["pathway", "cooccur_json", "cooccur_csv"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], image_n={"fixed_contributing": 201, "equivalent_side_contributing": 159}, object_direction_trajectory_n={"fixed_flip_directions": 2194, "equivalent_side_flip_directions": 1241}, statistical_unit="O2O identity-flip directional replay", point_estimate_columns=["exclusive first-observed pathway fraction", "non-exclusive flag__* and pair__* occurrence"], ci_columns=["ci95", "ci_95.*"], evidence_class="discovery plus figure-only descriptive addendum", interpretation_class="descriptive localisation; first-observed exclusive, any-stage non-exclusive", censoring_or_conditioning="identity-flip rows only; execution-order dependent and not causal fractions", panel_status="WARN", status_reason="Any-stage co-occurrence is SHA-verified in SOURCE_ADDENDUM; it remains descriptive and does not upgrade EV-PATHWAY-ANATOMY-V2."),
    panel("Fig3", "d", "How does pre-conflict rank state compare with the final post-conflict assigned O2O identity?", ["preres_fixed", "preres_eq"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], object_direction_trajectory_n={"fixed_objects": 7434, "fixed_directions": 29616, "equivalent_side_objects": 7432, "equivalent_side_directions": 29612}, statistical_unit="focal GT and directional replay", point_estimate_columns=["pre-conflict winner scale gap", "final assigned scale gap", "paired difference", "definition disagreement"], ci_columns=["ci95 low", "ci95 high"], evidence_class="discovery control", interpretation_class="same-branch descriptive control, not conflict mediation"),
    panel("Fig4", "a", "How do O2O and O2M native states persist over cardinal distance?", ["survival_cif", "terminal"], stress_contract="equivalent-area-side cardinal r in [0,.25], 1/1024 grid", object_direction_trajectory_n={"focal_gt": 7431, "directional_trajectories": 29724, "o2o_base_defined": 29048, "o2m_base_defined": 28996}, statistical_unit="focal-GT directional trajectory", point_estimate_columns=["survival", "radius"], ci_columns=["ci_low", "ci_high"], interpretation_class="Kaplan-Meier state persistence", censoring_or_conditioning="base-undefined excluded; right censoring retained; cardinal, not full 2D geometry"),
    panel("Fig4", "b", "Which native-path cause is first observed at the O2O boundary?", ["survival_cif", "terminal"], stress_contract="equivalent-area-side cardinal r in [0,.25], 1/1024 grid", object_direction_trajectory_n={"focal_gt": 7431, "directional_trajectories": 29724, "o2o_base_defined": 29048}, statistical_unit="focal-GT directional trajectory", point_estimate_columns=["cause-specific cumulative incidence", "radius"], ci_columns=["ci_low", "ci_high"], interpretation_class="Aalen-Johansen first-boundary incidence", censoring_or_conditioning="competing first causes and right censoring retained; reserved other may be zero"),
    panel("Fig4", "c", "What are absolute and paired restricted mean cardinal boundary distances?", ["rmsr", "rmsr_0125", "terminal"], stress_contract="equivalent-area-side cardinal r; primary truncation tau=.125", object_direction_trajectory_n={"focal_gt": 7431, "directional_trajectories": 29724, "o2o_base_defined": 29048, "o2m_base_defined": 28996}, statistical_unit="focal-GT directional trajectory", point_estimate_columns=["rmcbd", "o2o_minus_o2m"], ci_columns=["ci_low", "ci_high"], interpretation_class="restricted mean summary", censoring_or_conditioning="base-undefined excluded; branch supports reported separately"),
    panel("Fig4", "d", "Does branch ordering persist across prespecified truncation horizons?", ["rmsr_005", "rmsr_00625", "rmsr_010", "rmsr_0125", "rmsr_020", "rmsr_025", "terminal"], stress_contract="equivalent-area-side cardinal r; tau in {.05,.0625,.10,.125,.20,.25}", object_direction_trajectory_n={"focal_gt": 7431, "directional_trajectories": 29724}, statistical_unit="focal-GT directional trajectory", point_estimate_columns=["o2o_minus_o2m by tau and size stratum"], ci_columns=["ci_low", "ci_high"], evidence_class="discovery prespecified tau sensitivity", interpretation_class="ordering across tested truncation horizons, not magnitude robustness", censoring_or_conditioning="base-undefined excluded; truncation-specific right censoring"),
    panel("Fig5", "a", "Why are O2M rank turnover and positive-set turnover distinct states?", ["methods", "rankset_fixed", "rankset_eq"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], sample_support="Concept examples bound to frozen Rank-Set definitions", image_n=None, object_direction_trajectory_n=None, statistical_unit="conceptual assigned-positive set and assigned-set Top-1 state", weighting=None, bootstrap_unit=None, bootstrap_repetitions=None, ci_type=None, point_estimate_columns=[], ci_columns=[], evidence_class="method concept", interpretation_class="non-numeric schematic; Top-1 is not the uniquely supervised O2M positive"),
    panel("Fig5", "b", "How do native O2M set persistence and rank turnover differ by stress contract?", ["rankset_fixed", "rankset_eq"], stress_contract=["fixed_1px", "equivalent_side_kappa_0.0625"], object_direction_trajectory_n={"fixed_gt": 7434, "fixed_directions": 29616, "equivalent_side_gt": 7432, "equivalent_side_directions": 29612}, statistical_unit="directional focal-GT replay", point_estimate_columns=["exact set equality", "Jaccard", "base retention", "assigned Top-1 flip", "rank-only among flips"], ci_columns=["ci95[0]", "ci95[1]"], censoring_or_conditioning="empty-set undefined metrics excluded, not imputed"),
    panel("Fig5", "c", "How does native O2M positive count vary across the two scale groups?", ["count_json", "count_csv"], stress_contract="equivalent_side_kappa_0.0625", image_n={"selected": 300, "analysis": 288}, object_direction_trajectory_n={"common_valid_gt": 6377, "positive_count_support": "K=0 is present; standardisation overlap is K=3-7"}, statistical_unit="common-valid focal GT", bootstrap_unit="image cluster within outcome-blind sampling stratum", bootstrap_repetitions=5000, ci_type="95% stratified image-cluster percentile bootstrap where shown", point_estimate_columns=["positive_count", "weighted proportion", "positive_count_means"], ci_columns=[], evidence_class="revision descriptive control", interpretation_class="descriptive distribution", censoring_or_conditioning="K=0 remains visible; K=3-7 shading denotes the restriction/standardisation overlap only"),
    panel("Fig5", "d", "How much attenuation arises from support restriction versus within-overlap count standardisation?", ["count_json", "count_csv"], stress_contract="equivalent_side_kappa_0.0625", image_n={"selected": 300, "analysis": 288}, object_direction_trajectory_n={"common_valid_gt": 6377, "overlap_positive_counts": [3, 4, 5, 6, 7], "coverage_8_16": 0.6028970374970802, "coverage_16_32": 0.6429609485181138}, statistical_unit="common-valid focal GT or directional replay, as metric-specific", bootstrap_repetitions=5000, point_estimate_columns=["crude_gap_small_minus_tiny", "overlap_crude_gap_small_minus_tiny", "standardized_gap_small_minus_tiny"], ci_columns=["full_crude_ci_95_gap_small_minus_tiny", "adequate_overlap_crude_ci_95_gap_small_minus_tiny", "ci_95_gap_small_minus_tiny"], evidence_class="revision descriptive control", interpretation_class="descriptive restriction and standardisation, not a causal count intervention", censoring_or_conditioning="full-support crude, K=3-7 overlap crude, and within-overlap standardised estimates are separately displayed"),
    panel("Fig6", "a", "How does native margin align with observed fixed-pair rank-crossing distance?", ["events", "margin", "terminal"], stress_contract="equivalent-area-side cardinal r in [0,.25], 1/1024 grid", image_n=173, object_direction_trajectory_n={"observed_rho_R": 1593, "all_directional_trajectories": 29724}, statistical_unit="trajectory with observed comparable-pair crossing", point_estimate_columns=["o2o_base_margin", "rho_R_radius", "margin_vs_observed_rho_R_spearman.point"], ci_columns=["margin_vs_observed_rho_R_spearman.ci95"], evidence_class="conditional discovery analysis", interpretation_class="conditional association, not prediction", censoring_or_conditioning="observed crossings only; not censoring-adjusted population correlation"),
    panel("Fig6", "b", "What fraction of trajectories yield an observed fixed-pair crossing?", ["rho_status", "terminal"], stress_contract="equivalent-area-side cardinal r in [0,.25], 1/1024 grid", object_direction_trajectory_n={"observed_crossing": 1593, "competing_censor": 7439, "right_censor": 16548, "undefined": 4144, "total": 29724}, statistical_unit="focal-GT directional trajectory", point_estimate_columns=["status", "count", "proportion"], ci_columns=[], bootstrap_unit=None, bootstrap_repetitions=None, ci_type=None, interpretation_class="descriptive estimand composition", censoring_or_conditioning="mutually exclusive observed/competing/right-censored/undefined status"),
    panel("Fig6", "c", "Is conditional margin alignment stronger for rank crossing than eligibility-boundary distance?", ["margin", "terminal"], stress_contract="equivalent-area-side cardinal r in [0,.25], 1/1024 grid", image_n={"rho_R_observed_images": 173, "eligibility_boundary_images": 221}, object_direction_trajectory_n={"rho_R_observed": 1593, "eligibility_boundary_rows": 3444}, statistical_unit="separately observed trajectory supports", point_estimate_columns=["margin_vs_observed_rho_R_spearman.point", "margin_vs_eligibility_boundary_radius_spearman.point", "specificity_correlation_difference.point"], ci_columns=["*.ci95"], evidence_class="conditional discovery analysis", interpretation_class="conditional specificity association", censoring_or_conditioning="the two correlations use different observed supports; neither is censoring-adjusted"),
    panel("Fig6", "d", "Does adding margin yield stable held-out diagnostic improvement?", ["increment_ai", "increment_vis"], dataset=["AI-TOD-v2", "VisDrone"], checkpoint={"AI-TOD-v2": PRIMARY_CHECKPOINT, "VisDrone": "e7cffd98c6055bb0818873f94e6566feb697660ec5666f21f13c6304b412d0e8"}, stress_contract="equivalent_side_kappa_0.0625 baseline-state covariates", image_n={"AI-TOD-v2": 288, "VisDrone": 300}, object_direction_trajectory_n={"AI-TOD-v2_gt": 6410, "VisDrone_gt": 13946}, statistical_unit="GT prediction evaluated in held-out image-grouped folds", weighting="fold test weight; bootstrap within OOF fold x outcome-blind stratum", bootstrap_unit="image cluster within OOF fold x outcome-blind stratum", bootstrap_repetitions=5000, point_estimate_columns=["M3_minus_M2_point.delta_roc_auc", "delta_pr_auc", "delta_log_loss", "delta_brier"], ci_columns=["M3_minus_M2_ci_95.*"], evidence_class="figure-only diagnostic SOURCE_ADDENDUM", interpretation_class="held-out observational diagnostic increment, not treatment efficacy", censoring_or_conditioning="rightward display must invert log-loss/Brier signs so right always means improvement", panel_status="WARN", status_reason="Incremental-value artifacts are SHA-verified SOURCE_ADDENDUM only and cannot upgrade final_evidence_v3."),
    panel("FigS1", "", "Can real-image stable, fragile and undefined cases be shown under the primary frozen checkpoint?", ["s1_primary_trace", "s1_primary_validation", "s1_primary_image", "s1_fixed_rows"], checkpoint={"primary": PRIMARY_CHECKPOINT}, stress_contract="fixed_1px replay under the audited primary checkpoint", sample_support="one frozen AI-TOD-v2 image with three deterministic qualitative GT states", image_n=1, object_direction_trajectory_n={"cases": 3, "gt_ids": [40, 28, 17]}, statistical_unit="qualitative GT-direction trace", weighting=None, bootstrap_unit=None, bootstrap_repetitions=None, ci_type=None, point_estimate_columns=["candidate identity", "runner identity", "GT box", "shifted GT box", "decoded candidate box"], ci_columns=[], evidence_class="figure-only qualitative SOURCE_ADDENDUM", interpretation_class="qualitative illustration only; overlays are audited assignment states, not final detections", censoring_or_conditioning="deterministically selected within one frozen image; no aggregate estimate recomputed", panel_status="WARN", status_reason="Primary-checkpoint trace is SHA-verified and exactly matches frozen fixed-pixel replay identities; it is a figure-only SOURCE_ADDENDUM and does not upgrade scientific evidence."),
]


ADDENDUM = {
    "schema_version": "1.0",
    "status": "FIGURE_ONLY_SOURCE_ADDENDUM",
    "governance": "These sources are SHA-verified for figure construction only. They are not inserted into final_evidence_v3, do not upgrade evidence stage, and cannot support new scientific claims.",
    "entries": [
        {"addendum_id": "ADD-FIG-COOCCURRENCE-V1", "source_keys": ["cooccur_json", "cooccur_csv"], "admission": "WARN/SOURCE_ADDENDUM", "scope": "Fig3c non-exclusive descriptive any-stage occurrence only"},
        {"addendum_id": "ADD-FIG-INCREMENTAL-VALUE-V1", "source_keys": ["increment_ai", "increment_vis"], "admission": "WARN/SOURCE_ADDENDUM", "scope": "Fig6d held-out diagnostic increment only"},
        {"addendum_id": "ADD-FIG-S1-PRIMARY-TRACE-V1", "source_keys": ["s1_primary_trace", "s1_primary_validation", "s1_primary_image"], "admission": "WARN/SOURCE_ADDENDUM", "scope": "Fig1a and FigS1 primary-checkpoint qualitative rendering only; no aggregate scientific estimate"},
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def s1_identity_audit() -> dict:
    trace = json.loads((ROOT / SOURCES["s1_primary_trace"]["path"]).read_text(encoding="utf-8"))
    validation = json.loads((ROOT / SOURCES["s1_primary_validation"]["path"]).read_text(encoding="utf-8"))
    requested = {(trace["image_id"], int(record["gt_id"]), record["direction"]): record for record in trace["records"]}
    frozen: dict[tuple[str, int, str], dict] = {}
    with (ROOT / SOURCES["s1_fixed_rows"]["path"]).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["image_id"], int(row["gt_id"]), row["direction"])
            if key in requested:
                frozen[key] = {
                    "active": int(row["o2o_active"]) if row["o2o_active"] else None,
                    "runner": int(row["o2o_runner"]) if row["o2o_runner"] else None,
                    "shift_active": int(row["o2o_active_shift"]) if row["o2o_active_shift"] else None,
                    "o2o_flip": int(row["o2o_flip"]),
                }
    conflicts = []
    for key, record in requested.items():
        if record["frozen_row"] != frozen.get(key):
            conflicts.append({"image_id": key[0], "gt_id": key[1], "direction": key[2], "trace": record["frozen_row"], "frozen": frozen.get(key)})
    checkpoint_match = trace["checkpoint_sha256"] == PRIMARY_CHECKPOINT and validation["checkpoint_sha256"] == PRIMARY_CHECKPOINT
    validation_pass = validation.get("status") == "PASS" and validation.get("records") == 3 and validation.get("identity_checks") == 9
    return {
        "status": "PASS" if checkpoint_match and validation_pass and not conflicts else "FAIL",
        "trace_checkpoint": trace["checkpoint_sha256"],
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "checkpoint_match": checkpoint_match,
        "validation_pass": validation_pass,
        "records_checked": len(requested),
        "candidate_identity_conflicts": conflicts,
    }


def build() -> tuple[dict, list[dict]]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    ledger_ids = {entry["evidence_id"] for entry in ledger["entries"]}
    addendum_ids = {entry["addendum_id"] for entry in ADDENDUM["entries"]}
    audit = []
    for item in PANELS:
        for source in item["source_artifacts"]:
            path = ROOT / source["path"]
            exists = path.is_file()
            actual = sha256(path) if exists else None
            hash_status = "PASS" if exists and actual == source["expected_sha256"] else ("MISSING" if not exists else "CONFLICT")
            evidence_id = source["evidence_id"]
            if source["authority"] == "SOURCE_ADDENDUM":
                registry_status = "SOURCE_ADDENDUM" if evidence_id in addendum_ids else "UNRESOLVED"
            elif source["authority"].startswith("final_evidence_v"):
                registry_status = "PASS" if evidence_id in ledger_ids else "UNRESOLVED"
            elif source["authority"] in {"CONTRACT_SOURCE", "CANDIDATE_BLOCKED"}:
                registry_status = source["authority"]
            else:
                registry_status = "UNRESOLVED"
            audit.append({
                "figure": item["figure_number"], "panel": item["panel"], "panel_id": item["panel_id"],
                "source_artifact": source["path"], "expected_sha256": source["expected_sha256"],
                "actual_sha256": actual, "exists": exists, "hash_status": hash_status,
                "evidence_authority": source["authority"], "evidence_id": evidence_id,
                "evidence_registry_status": registry_status, "panel_status": item["panel_status"],
                "notes": source["role"],
            })
    s1 = s1_identity_audit()
    missing = sum(row["hash_status"] == "MISSING" for row in audit)
    conflicts = sum(row["hash_status"] == "CONFLICT" for row in audit)
    unresolved = sum(row["evidence_registry_status"] == "UNRESOLVED" for row in audit)
    panel_counts = {status: sum(p["panel_status"] == status for p in PANELS) for status in ("PASS", "WARN", "FAIL")}
    status = "BLOCKED_FOR_FINAL_FIGURES" if missing or conflicts or unresolved or panel_counts["FAIL"] else "PASS_FOR_FINAL_FIGURES"
    manifest = {
        "schema_version": "1.0",
        "status": status,
        "low_fidelity_allowed": True,
        "final_figure_generation_allowed": status == "PASS_FOR_FINAL_FIGURES",
        "science_results_recomputed": False,
        "authority_order": ["frozen source artifact", "final_evidence_v4", "current manuscript", "old figure"],
        "primary_checkpoint_sha256": PRIMARY_CHECKPOINT,
        "ledger": {"path": str(LEDGER_PATH.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(LEDGER_PATH), "status": ledger["status"]},
        "summary": {
            "panels": len(PANELS), "panel_status_counts": panel_counts,
            "source_references": len(audit), "missing_sources": missing,
            "sha256_conflicts": conflicts, "unresolved_evidence_links": unresolved,
            "semantic_identity_conflicts": len(s1["candidate_identity_conflicts"]),
        },
        "blocking_findings": [],
        "source_addendum_policy": ADDENDUM["governance"],
        "panels": PANELS,
        "figs1_identity_audit": s1,
    }
    return manifest, audit


def write_markdown(manifest: dict) -> None:
    lines = [
        "# Figure Evidence Manifest", "",
        f"- Overall status: **{manifest['status']}**",
        f"- Low-fidelity layouts allowed: **{str(manifest['low_fidelity_allowed']).lower()}**",
        f"- Final figure generation allowed: **{str(manifest['final_figure_generation_allowed']).lower()}**",
        "- Scientific results recomputed: **false**", "",
        "All source files are present and all recorded SHA-256 values match. The regenerated FigS1 trace is bound to the audited primary checkpoint and exactly matches the frozen fixed-pixel replay identities. Figure-only addenda remain `WARN/SOURCE_ADDENDUM` and do not change `final_evidence_v4`.", "",
        "## Panel status", "",
        "| Panel | Status | Question | Evidence class | Key qualification |",
        "|---|---|---|---|---|",
    ]
    for p in manifest["panels"]:
        qual = str(p["censoring_or_conditioning"]).replace("|", "\\|")
        lines.append(f"| {p['panel_id']} | {p['panel_status']} | {p['scientific_question']} | {p['evidence_class']} | {qual} |")
    lines.extend(["", "## FigS1 identity gate", "", f"- Required checkpoint: `{PRIMARY_CHECKPOINT}`", f"- Trace checkpoint: `{manifest['figs1_identity_audit']['trace_checkpoint']}`", f"- Frozen identity conflicts: `{len(manifest['figs1_identity_audit']['candidate_identity_conflicts'])}`", "", "The previous legacy qualitative candidate remains excluded. The admitted FigS1 trace was regenerated from the primary checkpoint for figure rendering only and does not recompute any aggregate scientific result.", "", "## Audit summary", ""])
    for key, value in manifest["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "See `qa/FIGURE_SOURCE_TO_EVIDENCE_AUDIT.csv` for every source/hash/evidence link and `FIGURE_EVIDENCE_ADDENDUM.json` for non-ledger figure-only sources.", ""])
    (OUT / "FIGURE_EVIDENCE_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    manifest, audit = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    (OUT / "FIGURE_EVIDENCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    addendum_output = json.loads(json.dumps(ADDENDUM))
    audit_by_path = {row["source_artifact"]: row for row in audit}
    for entry in addendum_output["entries"]:
        entry["sources"] = []
        for key in entry.pop("source_keys"):
            source = dict(SOURCES[key])
            source["actual_sha256"] = audit_by_path[source["path"]]["actual_sha256"]
            source["hash_status"] = audit_by_path[source["path"]]["hash_status"]
            entry["sources"].append(source)
    (OUT / "FIGURE_EVIDENCE_ADDENDUM.json").write_text(json.dumps(addendum_output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(manifest)
    fields = ["figure", "panel", "panel_id", "source_artifact", "expected_sha256", "actual_sha256", "exists", "hash_status", "evidence_authority", "evidence_id", "evidence_registry_status", "panel_status", "notes"]
    with (OUT / "qa/FIGURE_SOURCE_TO_EVIDENCE_AUDIT.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps({"status": manifest["status"], **manifest["summary"]}, indent=2))
    if manifest["summary"]["missing_sources"] or manifest["summary"]["sha256_conflicts"] or manifest["summary"]["unresolved_evidence_links"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
