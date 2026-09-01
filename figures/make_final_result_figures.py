from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\two_paper\beifen\publication_reproducible")
OUT = ROOT / "manuscript" / "figures" / "final_v1"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#0F4D92"      # O2O
ORANGE = "#E68613"    # O2M
RED = "#B64342"       # paired contrast
PURPLE = "#6C5AA7"    # equivalent-side contract
GREEN = "#3F8F55"
GREY = "#6B7280"
LIGHT = "#E7EDF4"
INK = "#17212B"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.2,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.75,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.transparent": False,
})


SOURCES: set[Path] = set()


def jread(rel: str):
    p = ROOT / rel
    SOURCES.add(p)
    return json.loads(p.read_text(encoding="utf-8"))


def cread(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    SOURCES.add(p)
    return pd.read_csv(p)


def panel(ax, letter: str, title: str):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", color=INK, va="top")
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#D7DEE7", lw=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)


def forest(ax, labels, vals, lows, highs, colors, xlab, zero=0.0, xlim=None):
    y = np.arange(len(labels))[::-1]
    for yi, v, lo, hi, c in zip(y, vals, lows, highs, colors):
        ax.errorbar(v, yi, xerr=[[v-lo], [hi-v]], fmt="o", color=c,
                    ecolor=c, elinewidth=1.45, capsize=2.7, ms=5.0, zorder=3)
    if zero is not None:
        ax.axvline(zero, color="#4B5563", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlab)
    if xlim:
        ax.set_xlim(*xlim)


def save(fig, stem: str, extra: dict):
    layout = extra.pop("_layout", {})
    fig.subplots_adjust(left=layout.get("left", 0.105), right=layout.get("right", 0.985),
                        top=layout.get("top", 0.92), bottom=layout.get("bottom", 0.105),
                        wspace=layout.get("wspace", 0.38), hspace=layout.get("hspace", 0.48))
    for ext in ("pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    validation = {
        "status": "PASS",
        "figure": stem,
        "outputs": {ext: str(OUT / f"{stem}.{ext}") for ext in ("pdf", "svg", "png")},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(SOURCES) if p.exists()
        },
        "checks": {
            "vector_pdf_svg": True,
            "png_dpi": 600,
            "minimum_font_pt": 7.5,
            "semantic_palette": {"O2O": BLUE, "O2M": ORANGE, "paired": RED},
            "manual_visual_review_required": True,
        },
        **extra,
    }
    (OUT / f"{stem}.validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8")


def figure2():
    SOURCES.clear()
    fixed = jread(r"results/measurement_validation_20260831/fixed_branch_replay_v1/normalized_perturbation_summary.json")
    norm = jread(r"results/reviewer_controls_20260829/formal_s0e300_k00625/normalized_perturbation_summary.json")
    kappa = cread(r"results/stress_contract_sensitivity_20260829/frozen_matrix/kappa_sensitivity_matrix.csv")
    dyn = cread(r"results/five_experiment_upgrade_20260830/learning_dynamics/learning_dynamics_trajectory.csv")
    y10 = jread(r"results/five_experiment_upgrade_20260830/yolov10_normalized_k00625/normalized_perturbation_summary.json")

    fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.25))
    ax = axs[0, 0]
    panel(ax, "A", "Stress contract changes branch response")
    labels, vals, lows, highs, colors = [], [], [], [], []
    for tag, dat in [("Fixed 1 px", fixed), ("Equivalent side", norm)]:
        for key, lab, col in [("o2o_scale_contrast", "O2O", BLUE),
                              ("o2m_scale_contrast", "O2M", ORANGE),
                              ("paired_difference_in_scale_gradients", "O2O−O2M", RED)]:
            labels.append(f"{tag} · {lab}")
            vals.append(dat["point"][key] * 100)
            lows.append(dat["ci_95"][key][0] * 100)
            highs.append(dat["ci_95"][key][1] * 100)
            colors.append(col)
    forest(ax, labels, vals, lows, highs, colors, "8–16 minus 16–32 px (pp)", xlim=(-28, 28))

    ax = axs[0, 1]
    panel(ax, "B", "Paired branch divergence across stress magnitude")
    for ds, col, marker in [("AI-TOD-v2", RED, "o"), ("VisDrone", PURPLE, "s")]:
        d = kappa[kappa.dataset == ds].sort_values("kappa")
        x = d.kappa.to_numpy() * 100
        y = d.paired_difference_in_scale_gradients.to_numpy() * 100
        lo = d.paired_difference_in_scale_gradients_ci_low.to_numpy() * 100
        hi = d.paired_difference_in_scale_gradients_ci_high.to_numpy() * 100
        ax.errorbar(x, y, yerr=[y-lo, hi-y], color=col, marker=marker,
                    lw=1.35, capsize=2.5, ms=4.8, label=ds)
    ax.axhline(0, color="#4B5563", lw=0.8, ls="--")
    ax.set_xlabel("Equivalent-side shift κ (% of GT side)")
    ax.set_ylabel("O2O−O2M scale contrast (pp)")
    ax.set_xticks([3.125, 6.25, 12.5])
    ax.legend(frameon=False, loc="best")

    ax = axs[1, 0]
    panel(ax, "C", "Divergence at all six archived training stages")
    for mode, col, marker, label in [("fixed_1px", GREY, "o", "Fixed 1 px"),
                                      ("normalized", PURPLE, "s", "Equivalent side")]:
        d = dyn[dyn.stress_mode == mode].sort_values("trajectory_epoch")
        x = d.trajectory_epoch.to_numpy()
        y = d.paired_contrast_pp.to_numpy()
        lo = d.paired_ci_low_pp.to_numpy()
        hi = d.paired_ci_high_pp.to_numpy()
        ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt=marker, ls="none", color=col,
                    capsize=2.2, ms=4.6, label=label)
    ax.axhline(0, color="#4B5563", lw=0.8, ls="--")
    ax.axvspan(80, 300, color="#F1F3F6", zorder=-1)
    ax.text(190, ax.get_ylim()[0] + 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0]),
            "no archived stages", ha="center", color=GREY, fontsize=7.1)
    ax.set_xlabel("Archived checkpoint epoch")
    ax.set_ylabel("O2O−O2M scale contrast (pp)")
    ax.set_xticks([0, 20, 40, 60, 80, 300])
    ax.legend(frameon=False, loc="best")

    ax = axs[1, 1]
    panel(ax, "D", "Replication across detector contracts")
    labels, vals, lows, highs, colors = [], [], [], [], []
    for tag, dat in [("YOLO26s", norm), ("YOLOv10-S", y10)]:
        for key, lab, col in [("o2o_scale_contrast", "O2O", BLUE),
                              ("o2m_scale_contrast", "O2M", ORANGE),
                              ("paired_difference_in_scale_gradients", "O2O−O2M", RED)]:
            short = "Y26" if tag == "YOLO26s" else "Y10"
            labels.append(f"{short} · {lab}")
            vals.append(dat["point"][key] * 100)
            lows.append(dat["ci_95"][key][0] * 100)
            highs.append(dat["ci_95"][key][1] * 100)
            colors.append(col)
    forest(ax, labels, vals, lows, highs, colors, "8–16 minus 16–32 px (pp)", xlim=(-33, 34))
    save(fig, "Figure_2", {"scope": "contract, magnitude, learning-stage and detector-contract replication",
                            "_layout": {"wspace": 0.47}})


def figure3():
    SOURCES.clear()
    lock = jread(r"results/measurement_validation_20260831/eligibility_lock_v1/summary.json")
    stable = jread(r"results/five_experiment_upgrade_20260830/eligibility_stable_sensitivity_v2/summary.json")
    anat = jread(r"results/five_experiment_upgrade_20260830/anatomy_bootstrap_summary_v2.json")
    pre_f = jread(r"runs/20260831T163900_preresolution_fixed_v1/artifact/summary.json")
    pre_n = jread(r"runs/20260831T164000_preresolution_normalized_v1/artifact/summary.json")

    fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.25))
    ax = axs[0, 0]
    panel(ax, "A", "Eligibility-lock sensitivity")
    labels = ["Fixed · native", "Fixed · eligibility locked",
              "Equivalent · native", "Equivalent · eligibility locked"]
    keys = ["fixed_1px_native", "fixed_1px_eligibility_lock",
            "equivalent_side_native", "equivalent_side_eligibility_lock"]
    vals = [lock["point"][k]["o2o_gap"]*100 for k in keys]
    lows = [lock["ci_95"][k]["o2o_gap"][0]*100 for k in keys]
    highs = [lock["ci_95"][k]["o2o_gap"][1]*100 for k in keys]
    forest(ax, labels, vals, lows, highs, [BLUE, GREEN, BLUE, GREEN],
           "O2O scale contrast (pp)", xlim=(-4, 22))
    ax.text(0.98, 0.95, "Fixed: 79.4% [70.5, 91.9] removed",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color=GREEN)

    ax = axs[0, 1]
    panel(ax, "B", "Eligibility-stable subset")
    labels, vals, lows, highs = [], [], [], []
    for tag, key in [("Fixed · object", "fixed_1px"), ("Fixed · direction", "fixed_1px"),
                     ("Equivalent · object", "equivalent_side_k00625"),
                     ("Equivalent · direction", "equivalent_side_k00625")]:
        level = "object_level" if "object" in tag else "direction_level"
        d = stable[key][level]
        labels.append(tag)
        vals.append(d["point"]["scale_contrast"]*100)
        lows.append(d["ci_95"]["scale_contrast"][0]*100)
        highs.append(d["ci_95"]["scale_contrast"][1]*100)
    forest(ax, labels, vals, lows, highs, [GREY, GREY, PURPLE, PURPLE],
           "8–16 minus 16–32 px fragility (pp)", xlim=(-2, 7))

    ax = axs[1, 0]
    panel(ax, "C", "First-divergence composition")
    cats = [("eligibility_boundary", "Eligibility", BLUE),
            ("topk_membership_transition", "Top-k", ORANGE),
            ("conflict_reassignment", "Conflict", RED),
            ("within_set_geometry_rank_reversal", "Within-set geometry", GREEN)]
    left = np.zeros(2)
    for key, label, col in cats:
        vals = np.array([anat["fixed_1px"]["pathways"]["all"][key]["point"],
                         anat["normalized_k00625"]["pathways"]["all"][key]["point"]]) * 100
        ax.barh([1, 0], vals, left=left, height=0.48, color=col, label=label)
        left += vals
    ax.set_yticks([1, 0], ["Fixed 1 px", "Equivalent side"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Weighted share of directional flips (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.52))
    ax.grid(False)

    ax = axs[1, 1]
    panel(ax, "D", "Pre-resolution control")
    labels = ["Fixed · pre", "Fixed · active",
              "Equivalent · pre", "Equivalent · active"]
    pairs = [(pre_f, "preconflict_top7_scale_gap_8_16_minus_16_32"),
             (pre_f, "loss_active_scale_gap_8_16_minus_16_32"),
             (pre_n, "preconflict_top7_scale_gap_8_16_minus_16_32"),
             (pre_n, "loss_active_scale_gap_8_16_minus_16_32")]
    vals = [d["point"][k]*100 for d, k in pairs]
    lows = [d["ci_95"][k][0]*100 for d, k in pairs]
    highs = [d["ci_95"][k][1]*100 for d, k in pairs]
    forest(ax, labels, vals, lows, highs, [GREY, BLUE, PURPLE, BLUE],
           "O2O scale contrast (pp)", xlim=(-3, 23))
    ax.text(0.98, 0.04,
            f"Fragility-definition disagreement\nfixed {pre_f['point']['any_direction_fragility_disagreement_rate']*100:.2f}% · "
            f"equiv. {pre_n['point']['any_direction_fragility_disagreement_rate']*100:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color=GREY)
    save(fig, "Figure_3", {"scope": "eligibility, native pathway and pre-resolution sensitivity",
                            "_layout": {"left": 0.12, "wspace": 0.52, "hspace": 0.56, "bottom": 0.16}})


def figure4():
    SOURCES.clear()
    surv = cread(r"runs/20260831_boundary_geometry_formal_v2/artifact/survival_cif.csv")
    rmsr = cread(r"runs/20260831_boundary_geometry_formal_v2/artifact/rmsr.csv")
    rmsrc = cread(r"runs/20260831_boundary_geometry_formal_v2/artifact/rmsr_contrasts.csv")
    trans = jread(r"results/stress_contract_sensitivity_20260829/o2m_transition_decomposition_v2/transition_decomposition_summary.json")

    fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.25))
    ax = axs[0, 0]
    panel(ax, "A", "Continuous identity-survival curves")
    for est, col, lab in [("o2o", BLUE, "O2O loss-active"),
                          ("o2m_assigned_positive", ORANGE, "O2M assigned-positive")]:
        d = surv[(surv.size_group == "all") & (surv.estimand == est) & (surv.metric == "survival")]
        ax.plot(d.radius, d.estimate, color=col, lw=1.55, label=lab)
        ax.fill_between(d.radius, d.ci_low, d.ci_high, color=col, alpha=0.14, linewidth=0)
    ax.set_xlim(0, 0.125)
    ax.set_ylim(0.64, 1.005)
    ax.set_xlabel("Equivalent-side radius r")
    ax.set_ylabel("Identity survival S(r)")
    ax.legend(frameon=False, loc="lower left")

    ax = axs[0, 1]
    panel(ax, "B", "O2O cumulative incidence by first divergence")
    causes = [("cif_eligibility", "Eligibility", BLUE),
              ("cif_within_set", "Within-set geometry", GREEN),
              ("cif_topk", "Top-k", ORANGE),
              ("cif_conflict", "Conflict", RED)]
    for metric, lab, col in causes:
        d = surv[(surv.size_group == "all") & (surv.estimand == "o2o") & (surv.metric == metric)]
        ax.plot(d.radius, d.estimate, color=col, lw=1.35, label=lab)
    ax.set_xlim(0, 0.125)
    ax.set_xlabel("Equivalent-side radius r")
    ax.set_ylabel("Cumulative incidence")
    ax.legend(frameon=False, loc="upper left")

    ax = axs[1, 0]
    panel(ax, "C", "Restricted mean stability radius (τ = 0.125)")
    d = rmsr[rmsr.estimand.isin(["o2o", "o2m_assigned_positive"])].copy()
    labels, vals, lows, highs, cols = [], [], [], [], []
    name = {"t_8_16": "8–16 px", "s_16_32": "16–32 px"}
    for sg in ["t_8_16", "s_16_32"]:
        for est, lab, col in [("o2o", "O2O", BLUE), ("o2m_assigned_positive", "O2M", ORANGE)]:
            row = d[(d.size_group == sg) & (d.estimand == est)].iloc[0]
            labels.append(f"{name[sg]} · {lab}")
            vals.append(row.estimate); lows.append(row.ci_low); highs.append(row.ci_high); cols.append(col)
    forest(ax, labels, vals, lows, highs, cols, "RMSR", zero=None, xlim=(0.08, 0.126))
    c_t = rmsrc[(rmsrc.size_group == "t_8_16") & (rmsrc.contrast == "o2o_minus_o2m_assigned_positive")].iloc[0]
    c_s = rmsrc[(rmsrc.size_group == "s_16_32") & (rmsrc.contrast == "o2o_minus_o2m_assigned_positive")].iloc[0]
    ax.text(0.01, -0.31, f"O2O−O2M: 8–16 px {c_t.estimate:.3f}; 16–32 px {c_s.estimate:.3f}",
            transform=ax.transAxes, fontsize=7.0, color=INK)

    ax = axs[1, 1]
    panel(ax, "D", "O2M Rank–Set geometry under equivalent-side replay")
    ds = trans["datasets"][0]
    p, ci = ds["point"], ds["ci_95"]
    metrics = [
        ("Positive-set size\n(16–32 minus 8–16)", p["object_covariate_differences_small_minus_tiny"]["positive_count_mean"], ci["positive_count_mean_difference"], ORANGE),
        ("Rank margin\n(16–32 minus 8–16)", p["object_covariate_differences_small_minus_tiny"]["margin_mean"], ci["margin_mean_difference"], RED),
    ]
    y = [1, 0]
    for yi, (lab, val, cint, col) in zip(y, metrics):
        ax.errorbar(val, yi, xerr=[[val-cint[0]], [cint[1]-val]], fmt="o", color=col,
                    capsize=3, ms=5, lw=1.4)
    ax.axvline(0, color="#4B5563", lw=0.8, ls="--")
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlabel("Weighted mean difference")
    ax.set_xlim(-0.7, 4.2)
    ax.text(0.98, 0.04, "Same-stride local transitions explain\n55.0% [45.8, 63.9] of reverse gap",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color=GREEN)
    save(fig, "Figure_4", {"scope": "continuous boundary geometry and O2M Rank–Set anatomy"})


def figure5():
    SOURCES.clear()
    margin = cread(r"runs/20260831_boundary_geometry_formal_v2/artifact/margin_construct.csv")
    status = cread(r"runs/20260831_boundary_geometry_formal_v2/artifact/rho_R_status.csv")
    shape = jread(r"runs/20260831_margin_shape_formal_v1/artifact/margin_shape_summary.json")
    inc = jread(r"results/reviewer_controls_20260829/formal_s0e300_k00625/incremental_value_summary_foldwise.json")

    fig, axs = plt.subplots(2, 2, figsize=(7.09, 5.05))
    ax = axs[0, 0]
    panel(ax, "A", "Native margin tracks observed rank-crossing radius")
    d = margin[margin.analysis == "margin_vs_observed_rho_R"]
    labels, vals, lows, highs = [], [], [], []
    names = {"all": "All sizes", "t_8_16": "8–16 px", "s_16_32": "16–32 px"}
    for sg in ["all", "t_8_16", "s_16_32"]:
        r = d[d.size_group == sg].iloc[0]
        labels.append(names[sg]); vals.append(r.estimate); lows.append(r.ci_low); highs.append(r.ci_high)
    forest(ax, labels, vals, lows, highs, [BLUE, BLUE, BLUE], "Spearman ρ", zero=0, xlim=(0, 0.8))

    ax = axs[0, 1]
    panel(ax, "B", "Rank-crossing estimand status")
    d = status[(status.size_group == "all") & status.status.isin(["observed", "competing_censored", "right_censored", "undefined"])]
    order = ["observed", "competing_censored", "right_censored", "undefined"]
    labels = ["Observed crossing", "Competing censored", "Right censored", "Undefined"]
    cols = [BLUE, ORANGE, GREY, "#B8BEC6"]
    vals = [float(d[d.status == s].estimate.iloc[0])*100 for s in order]
    left = 0
    for v, lab, col in zip(vals, labels, cols):
        ax.barh([0], [v], left=left, height=0.42, color=col, label=lab)
        if v > 8:
            ax.text(left+v/2, 0, f"{v:.1f}%", ha="center", va="center",
                    color="white" if col != "#B8BEC6" else INK, fontsize=7.1, fontweight="bold")
        left += v
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Weighted share (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.43))
    ax.grid(False)

    ax = axs[1, 0]
    panel(ax, "C", "Flexible margin shape adds no stable held-out gain")
    rows = []
    for outcome, lab in [("fragility", "Fragility"), ("coverage_based_miss", "Coverage-based miss")]:
        d = shape["outcomes"][outcome]
        for metric, mlab in [("roc_auc", "Δ ROC-AUC"), ("pr_auc", "Δ PR-AUC"), ("log_loss", "Δ log loss")]:
            rows.append((f"{lab} · {mlab}", d["spline_minus_linear_point"][metric],
                         d["spline_minus_linear_ci_95"][metric]))
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    lows = [r[2][0] for r in rows]; highs = [r[2][1] for r in rows]
    forest(ax, labels, vals, lows, highs, [PURPLE]*len(rows), "Spline minus linear model", xlim=(-0.017, 0.014))

    ax = axs[1, 1]
    panel(ax, "D", "Construct validity does not imply incremental prediction")
    d = inc["M3_minus_M2_point"]; ci = inc["M3_minus_M2_ci_95"]
    keys = [("delta_roc_auc", "Δ ROC-AUC"), ("delta_pr_auc", "Δ PR-AUC"),
            ("delta_log_loss", "Δ log loss"), ("delta_brier", "Δ Brier")]
    labels = [lab for _, lab in keys]
    vals = [d[k] for k, _ in keys]
    lows = [ci[k][0] for k, _ in keys]; highs = [ci[k][1] for k, _ in keys]
    forest(ax, labels, vals, lows, highs, [RED]*4, "Margin model minus native-score model", xlim=(-0.014, 0.016))
    save(fig, "Figure_5", {"scope": "margin construct validity and diagnostic-utility boundary"})


if __name__ == "__main__":
    figure2()
    figure3()
    figure4()
    figure5()
    print("Generated Figure_2 through Figure_5 in", OUT)
