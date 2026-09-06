"""Build an isolated manuscript copy with the approval figures inserted.

The authoritative manuscript tree is read only. This staging build exists to
check figure numbering, floats, and page rendering before author approval.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manuscript"
FIG = ROOT / "figures_final"
BUILD_ROOT = FIG / "approval_build_20260904_v1"
TARGET = BUILD_ROOT / "manuscript"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    if TARGET.exists():
        raise RuntimeError(f"staging target already exists; refusing to overwrite: {TARGET}")
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE,
        TARGET,
        ignore=shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.fls", "*.fdb_latexmk", "*.synctex.gz", "*.pdf", "tmp"),
    )
    approval_dir = TARGET / "figures/approval"
    approval_dir.mkdir(parents=True, exist_ok=True)
    for figure in ("Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "FigS1"):
        shutil.copy2(FIG / f"draft/{figure}.pdf", approval_dir / f"{figure}.pdf")

    methods_path = TARGET / "content/methods.tex"
    methods = methods_path.read_text(encoding="utf-8")
    old = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{final_v1/Figure_1.pdf}
\\caption{Moving only the focal GT centre can cross native assignment-state boundaries while every model output remains fixed. (A) A registered AI-TOD-v2 crop and magnified inset show the exact base GT box and its one-pixel left shift; the example changes the post-conflict assigned O2O identity from candidate 4230 to 12757. (B) Candidate horizontal position is the recorded anchor location and vertical height is the native alignment score $q$, so the perspective view encodes measured quantities rather than decorative depth. (C) O2O and O2M retain their native eligibility, selection, conflict, and final-state semantics. (D) Four cardinal rays end in an observed event, right censoring, an undefined state, or a change followed by return. The audit does not estimate a two-dimensional minimum boundary. The example is explanatory and is not a detector-performance claim.}
\\label{fig:measurement_overview}
\\end{figure*}"""
    new = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig1.pdf}
\\caption{Read-only assignment-state measurement. (A) Controlled focal-ground-truth (GT) replay on a frozen AI-TOD-v2 image. The displayed rectangles are true annotations for GTs 17, 28, and 40; GT 28 is focal. Only the focal-GT centre is displaced, whereas image pixels, features, model parameters, candidate lattice, box dimensions, and non-focal GTs remain fixed. (B) Native O2O and O2M assignment paths. O2M assigned-set Top-1 is a rank-state summary, not a uniquely supervised positive. (C) Grid-resolved probing along four cardinal rays, including first boundary, return, right censoring, and representative first-boundary causes. The readout is not a full two-dimensional minimum radius. (D) Principal audit readouts and mutually exclusive per-direction state meanings after software-instrument qualification.}
\\label{fig:measurement_overview}
\\end{figure*}"""
    methods = replace_once(methods, old, new, "Figure 1 block")
    methods_path.write_text(methods, encoding="utf-8")

    results_path = TARGET / "content/results.tex"
    results = results_path.read_text(encoding="utf-8")
    old = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{final_v1/Figure_2.pdf}
\\caption{The sign of the O2O-minus-O2M branch contrast persisted across the audited stress magnitudes, datasets, archived stages, and detector contracts, while its branch-wise decomposition differed across stress contracts. (A) AI-TOD-v2 O2O, O2M, and paired scale contrasts under fixed one-pixel and equivalent-area-side replay. (B) Paired contrasts across the three prespecified equivalent-area-side magnitudes on AI-TOD-v2 and VisDrone. (C) Paired contrasts at the six physically archived training stages; points are not connected, and the 80--300 interval contains no archived intermediate checkpoint. (D) Equivalent-area-side replication under compatible YOLO26s and YOLOv10-S detector contracts. Points are inverse-inclusion-weighted estimates; whiskers are 95\\% intervals from 5,000 stratified image-cluster bootstrap replicates.}
\\label{fig:stress_response}
\\end{figure*}"""
    new = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig2.pdf}
\\caption{Stress-contract-dependent scale response. (A) O2O, O2M, and paired 8--16-minus-16--32 px contrasts under fixed one-pixel and equivalent-area-side stress. Points are estimates and bars are 95\\% stratified image-cluster bootstrap intervals. (B) Paired O2O-minus-O2M contrast at the three prespecified equivalent-area-side $\\kappa$ values in AI-TOD-v2 and VisDrone; segments join tested values only. (C) Paired contrasts at the six physically archived stages. The archived \\texttt{last.pt} state is displayed as Final (source archive label 300) and is distinct from the retained post-training-selected epoch-280 artifacts named \\texttt{best.pt}; no intermediate checkpoints were archived between 80 and Final. (D) Compatible native-contract replication with YOLO26s and YOLOv10-S; different native assignment semantics preclude a same-mechanism or broad architecture-generalisation claim.}
\\label{fig:stress_response}
\\end{figure*}"""
    results = replace_once(results, old, new, "Figure 2 block")

    old = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{final_v1/Figure_3.pdf}
\\caption{Eligibility locking removed most of the fixed-pixel O2O contrast, while within-set changes occurred more often anywhere along an exchange path than the exclusive first labels suggested. (A) Native and eligibility-locked O2O contrasts on the same lock-analysis support. The 17.10-to-3.52 pp reduction was 79.4\\% [70.5\\%, 91.9\\%] of the 17.10 pp denominator, which differs from the primary 14.37 pp common-valid estimate. (B) Object- and direction-level contrasts conditional on eligibility stability. (C) Exclusive first-divergence composition and nonexclusive occurrence of each stage among 2,194 fixed and 1,241 equivalent-area-side directional identity exchanges; eligibility and within-set changes co-occurred on 44.63\\% and 31.59\\% of the respective paths. (D) Pre-conflict Top-7 winner versus final post-conflict assigned contrasts. Whiskers are 95\\% stratified image-cluster bootstrap intervals. First divergence localises the earliest observed difference and is not a causal contribution.}
\\label{fig:pathway_localisation}
\\end{figure*}"""
    new = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig3.pdf}
\\caption{Eligibility and native-path localisation. (A) Native versus eligibility-locked O2O scale contrasts on the eligibility-lock support, which differs from the primary endpoint support. (B) Eligibility-stable observational subset; open markers emphasize post-replay conditioning and the estimand change. (C) Exclusive first-observed pathway composition (left) and non-exclusive any-stage occurrence (right). (D) Pre-resolution same-branch control; filled markers denote fixed one-pixel stress and open markers equivalent-area-side stress. First-observed localisation depends on execution order; the pathway summaries are descriptive and are not causal contribution fractions or mediation effects.}
\\label{fig:pathway_localisation}
\\end{figure*}"""
    results = replace_once(results, old, new, "Figure 3 block")

    boundary_anchor = """\\end{table}

% Evidence: EV-RANK-SET-FIXED-V1, EV-RANK-SET-NORMALIZED-V1"""
    boundary_figure = """\\end{table}

Figure~\\ref{fig:boundary_geometry} summarises dense-grid native-state persistence, first-boundary incidence, restricted mean cardinal boundary distance, and the prespecified truncation-horizon sensitivity.

\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig4.pdf}
\\caption{Grid-resolved cardinal boundary geometry. (A) Weighted Kaplan--Meier summaries of remaining free of a first observed state departure under the recorded administrative and geometric censoring rules, for O2O identity and O2M assigned-set Top-1 rank state in the 8--16 and 16--32 px groups. (B) Aalen--Johansen cause-specific cumulative incidence for the first O2O boundary. (C) Absolute branch-specific restricted mean cardinal boundary distance (RMCBD) and the paired O2O-minus-O2M contrast at $\\tau=0.125$. (D) Paired contrast across the six prespecified truncation horizons; segments connect tested horizons only and assess ordering rather than magnitude invariance. Base-undefined trajectories are excluded, right censoring is retained, and distances refer to four cardinal rays rather than a full two-dimensional radius. KM/AJ/RMCBD are descriptive summaries under the recorded censoring rules.}
\\label{fig:boundary_geometry}
\\end{figure*}

% Evidence: EV-RANK-SET-FIXED-V1, EV-RANK-SET-NORMALIZED-V1"""
    results = replace_once(results, boundary_anchor, boundary_figure, "Figure 4 insertion")

    rank_anchor = """\\end{table*}

% Evidence: EV-SAME-STRIDE-V1"""
    rank_figure = """\\end{table*}

Figure~\\ref{fig:rank_set_geometry} separates O2M assigned-set rank turnover from positive-set persistence and makes the positive-count conditioning support explicit.

\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig5.pdf}
\\caption{O2M Rank--Set decomposition. (A) Distinction among rank-only turnover, simultaneous rank-and-set turnover, and set-only turnover with stable assigned-set Top-1. (B) Native O2M Rank--Set contrasts under fixed and equivalent-area-side stress. (C) Weighted native positive-count distributions; $K=0$ remains visible and shading marks only the $K=3$--7 overlap used for standardisation. (D) Full-support crude and overlap-support count-standardised contrasts. The two estimates have different supports, and standardisation is descriptive rather than a causal positive-count intervention.}
\\label{fig:rank_set_geometry}
\\end{figure*}

% Evidence: EV-SAME-STRIDE-V1"""
    results = replace_once(results, rank_anchor, rank_figure, "Figure 5 insertion")

    old_combined = """Figure~\\ref{fig:boundary_geometry} complements the endpoint contrasts with full-domain $1/1024$ cardinal-direction identity survival, cause-specific incidence, restricted mean cardinal boundary distance, and O2M Rank--Set geometry.

\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{final_v1/Figure_4.pdf}
\\caption{On the exhaustive $1/1024$ grid, post-conflict assigned O2O identities persisted farther than O2M assigned-positive Top-1 identities, with a larger RMCBD separation for 16--32 px objects. (A) Weighted Kaplan--Meier identity-survival curves for O2O and O2M states, with branch and size encoded independently. (B) Aalen--Johansen cumulative incidence of the instrumented O2O first-divergence causes. (C) Restricted mean cardinal boundary distance through $\\tau=0.125$ and the paired O2O-minus-O2M contrasts for 8--16 and 16--32 px objects. (D) Tiny-minus-small contrasts in O2M positive-set Jaccard similarity, base-set retention, assigned-set Top-1 flips, and the rank-only fraction among Top-1 flips under fixed and equivalent-area-side replay. Shading and whiskers are 95\\% stratified image-cluster bootstrap intervals.}
\\label{fig:boundary_geometry}
\\end{figure*}

"""
    results = replace_once(results, old_combined, "", "legacy combined Figure 4 removal")

    old = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{final_v1/Figure_5.pdf}
\\caption{Native margin tracked observed rank-crossing distance but supplied little stable held-out diagnostic improvement beyond native score and geometry. (A) Spearman associations with observed fixed-pair rank-crossing distance and the eligibility-boundary specificity control, shown for all, 8--16 px, and 16--32 px trajectories. (B) Weighted status composition of the rank-crossing estimand; competing censoring, right censoring, and undefined states prevent a population-wide uncensored interpretation. (C) Held-out metric differences between restricted-cubic-spline and linear margin specifications. (D) Out-of-fold increment from adding margin after area, O2M coverage, native active score, and active CIoU for the coverage-based-miss outcome. Whiskers are 95\\% stratified image-cluster bootstrap intervals. The association is conditional on observed crossings and does not establish population-level construct validity or treatment efficacy.}
\\label{fig:margin_boundary}
\\end{figure*}"""
    new = """\\begin{figure*}[t]
\\centering
\\includegraphics[width=\\textwidth]{approval/Fig6.pdf}
\\caption{Margin and observed rank-crossing geometry. (A) Assigned-relative margin versus observed fixed-pair rank-crossing distance among 1,593 trajectories with an observed crossing (conditional Spearman $\\rho=0.628$); no regression curve is fitted. (B) Weighted composition of observed crossing, competing censoring, right censoring, and undefined states. (C) Conditional Spearman associations with observed rank crossing and eligibility-boundary distance on their separate observed supports. (D) Held-out diagnostic change after adding margin in AI-TOD-v2 and VisDrone; log-loss and Brier signs are oriented so rightward values indicate improvement. These are conditional association and diagnostic-increment analyses, not a guaranteed predictor or intervention effect.}
\\label{fig:margin_boundary}
\\end{figure*}"""
    results = replace_once(results, old, new, "Figure 6 block")
    results_path.write_text(results, encoding="utf-8")

    supp_path = TARGET / "supplementary.tex"
    supp = supp_path.read_text(encoding="utf-8")
    old = """\\begin{figure}[ht]
\\centering
\\includegraphics[width=0.98\\textwidth]{fig4_qualitative_replay.pdf}
\\caption{Frozen qualitative scene-to-crop correspondence. Colored focal-GT boxes are audit annotations, not detector outputs. Crop identities, assigned O2O candidates, runners, and labels were deterministically derived from the frozen replay records.}
\\label{fig:supp_qualitative}
\\end{figure}"""
    new = """\\begin{figure}[ht]
\\centering
\\includegraphics[width=0.98\\textwidth]{approval/FigS1.pdf}
\\caption{Audited primary-checkpoint assignment-state examples. Each row shows context, base state, and one-pixel shift state for stable, fragile, and post-shift undefined cases. Green solid boxes mark focal GTs, red dashed boxes shifted GTs, blue solid boxes assigned O2O candidates, and purple dashed boxes runner candidates. Image pixels are unchanged and only the focal-GT centre moves. These overlays are audited assignment states, not final detections; row-specific zoom is used only for legibility.}
\\label{fig:supp_qualitative}
\\end{figure}"""
    supp = replace_once(supp, old, new, "Figure S1 block")
    supp_path.write_text(supp, encoding="utf-8")

    summary = {
        "status": "STAGED_NOT_AUTHORITATIVE",
        "source_manuscript": str(SOURCE),
        "target_manuscript": str(TARGET),
        "source_manuscript_modified": False,
        "main_figures": 6,
        "supplementary_figures_replaced": 1,
        "approval_figure_paths": [f"figures/approval/Fig{i}.pdf" for i in range(1, 7)] + ["figures/approval/FigS1.pdf"],
    }
    (BUILD_ROOT / "STAGING_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
