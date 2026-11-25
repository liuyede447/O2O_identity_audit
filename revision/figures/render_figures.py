"""Render the R2.3 artwork from frozen, SHA-verified panel data."""
from pathlib import Path
import argparse
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HASHES = {
    2: "380ef2f6688462598dcf4a1d010fd73d0fb0f407777a1d0a4e66f6b765a89ddb",
    3: "d350e0200b8e15b1b55dc5454d5001341a129e8438c9fde3899990573d0a685e",
    4: "1a359e2b9a9802b24d320b42456a634c03166cce9eb8c71708ca6234d10053d7",
}
BRANCH = {"O2O": "#1F4E79", "O2M": "#C86718", "Paired": "#805399"}
MARKER = {"O2O": "o", "O2M": "s", "Paired": "D"}
CAUSE = {
    "Eligibility": "#008463", "Within-set rank": "#237FAB",
    "Conflict": "#42474B", "Top-k": "#B77400", "Other": "#777B7E",
}
STRESS = ["Fixed 1 px", "Equivalent-area-side"]
SHORT_STRESS = ["Fixed 1 px", "Eq. area"]
SIZES = ["All", "8–16 px", "16–32 px"]
GAP = "8–16 minus 16–32 px (pp)"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"],
    "font.size": 8.6, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelsize": 8.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "legend.frameon": False, "legend.fontsize": 8.6,
    "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": 0.45,
})


def marker_key(label, color, marker="o", opened=False):
    return Line2D([], [], marker=marker, ls="none", color=color, markersize=4.8,
                  markerfacecolor="white" if opened else color, label=label)


def forest(ax, x, y, row, color, marker="o", opened=False, vertical=False):
    err = [[row.estimate - row.ci_low], [row.ci_high - row.estimate]]
    ax.errorbar(x, y, yerr=err if vertical else None, xerr=None if vertical else err,
                color=color, fmt=marker, markersize=4.7, elinewidth=0.8,
                markerfacecolor="white" if opened else color, capsize=2, zorder=3)


def style(ax, title, *, horizontal=False, zero=False):
    ax.set_title(title, loc="left", pad=7)
    axis = "x" if horizontal else "y"
    ax.grid(axis=axis, color="#E3E6E8", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.65, pad=3)
    if zero:
        rule = ax.axvline if horizontal else ax.axhline
        rule(0, color="#767E84", linewidth=0.75, zorder=2)


def groups(ax, labels, top_pad=0.5):
    ax.set_yticks(np.arange(len(labels))[::-1], labels)
    ax.set_ylim(-0.5, len(labels) - 1 + top_pad)


def canvas(height, **kw):
    fig, axes = plt.subplots(2, 2, figsize=(178 / 25.4, height / 25.4))
    fig.subplots_adjust(**kw)
    return fig, axes.flat


def figure_2(data):
    fig, (a, b, c, d) = canvas(125, left=.100, right=.973, bottom=.104,
                              top=.842, wspace=.34, hspace=.67)
    for j, condition in enumerate(STRESS):
        for i, branch in enumerate(BRANCH):
            row = data.loc[data.panel.eq("a") & data.x.eq(condition) & data.series.eq(branch)].iloc[0]
            forest(a, row.estimate, 2 - i + (.13 if j == 0 else -.13), row,
                   BRANCH[branch], MARKER[branch], opened=bool(j))
    groups(a, ["O2O", "O2M", "Paired"])
    a.set(xlim=(-27, 35), xlabel=GAP)
    style(a, "a  Branch contrasts", horizontal=True, zero=True)
    fig.legend(handles=[marker_key(label, "#464A50", opened=bool(i))
                        for i, label in enumerate(SHORT_STRESS)],
               loc="upper center", bbox_to_anchor=(.51, .987), ncol=2,
               title="Stress in (a) and (c)", title_fontsize=8.6)

    for i, dataset in enumerate(["AI-TOD-v2", "VisDrone"]):
        for row in data.loc[data.panel.eq("b") & data.series.eq(dataset)].itertuples():
            forest(b, float(row.x), row.estimate, row, BRANCH["Paired"], "os"[i],
                   opened=bool(i), vertical=True)
    b.set(xlim=(.019, .138), ylim=(10, 43), xlabel="Equivalent-area factor κ",
          ylabel="Paired scale gap (pp)")
    b.set_xticks([.03125, .0625, .125], ["0.03125", "0.0625", "0.125"])
    b.legend(handles=[marker_key(name, BRANCH["Paired"], "os"[i], bool(i))
                      for i, name in enumerate(["AI-TOD-v2", "VisDrone"])],
             loc="center", bbox_to_anchor=(.64, .53))
    style(b, "b  Perturbation magnitude")

    epochs = [0, 20, 40, 60, 80, 300]
    for i, condition in enumerate(STRESS):
        for row in data.loc[data.panel.eq("c") & data.series.eq(condition)].itertuples():
            x = epochs.index(int(float(row.x))) + (-.10 if i == 0 else .10)
            forest(c, x, row.estimate, row, BRANCH["Paired"], opened=bool(i), vertical=True)
    c.set(xlim=(-.45, 5.45), ylim=(4, 30), xlabel="Archived checkpoint (epoch)",
          ylabel="Paired scale gap (pp)")
    c.set_xticks(range(6), [str(e) for e in epochs])
    style(c, "c  Archived checkpoints")

    detector_names = ["YOLO26s", "YOLOv10-S"]
    for j, detector in enumerate(detector_names):
        for i, branch in enumerate(BRANCH):
            row = data.loc[data.panel.eq("d") & data.x.eq(detector) & data.series.eq(branch)].iloc[0]
            forest(d, row.estimate, 2 - i + (.13 if j == 0 else -.13), row,
                   BRANCH[branch], "o^"[j], opened=bool(j))
    groups(d, ["O2O", "O2M", "Paired"], top_pad=1.10)
    d.set(xlim=(-33, 43), xlabel=GAP)
    d.legend(handles=[marker_key(name, "#464A50", "o^"[j], bool(j))
                      for j, name in enumerate(detector_names)], loc="upper right",
             ncol=2, columnspacing=.9, borderpad=.25, handletextpad=.5,
             frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    style(d, "d  Detector comparison", horizontal=True)
    d.vlines(0, -.5, 2.34, color="#767E84", linewidth=.75, zorder=2)
    return fig


def figure_3(data):
    fig, (a, b, c, d) = canvas(151, left=.167, right=.974, bottom=.145,
                              top=.935, wspace=.58, hspace=.66)
    locks = data.loc[data.kind.eq("lock_figure_cohorts")]
    lock_order = [
        "1 px: whole (S0)", "1 px: half (S0)", "1 px: val 600",
        "1 px: val + trans.", "1 px: train + Mosaic", "Eq.: whole (S0)", "Eq.: half (S0)",
    ]
    cohorts_a = []
    for label in lock_order:
        pair = locks.loc[locks.contract.eq(label)].set_index("label")
        cohorts_a.append((label, pair.loc["Native"], pair.loc["Eligibility locked"], label.startswith("Eq.")))
    n_a = len(cohorts_a)
    for idx, (_, r_nat, r_loc, opened) in enumerate(cohorts_a):
        y = (n_a - 1) - idx
        a.plot([r_nat.estimate, r_loc.estimate], [y, y], color="#D3D6D8", linewidth=1.2)
        forest(a, r_nat.estimate, y, r_nat, BRANCH["O2O"], "o", opened=opened)
        forest(a, r_loc.estimate, y, r_loc, CAUSE["Eligibility"], "s", opened=opened)
    groups(a, [x[0] for x in cohorts_a], top_pad=1.05)
    a.set(xlim=(-4, 22), xlabel="O2O scale gap (pp)")
    a.legend(handles=[marker_key("Native", BRANCH["O2O"]),
                      marker_key("Locked", CAUSE["Eligibility"], "s")], loc="upper right", ncol=2,
             borderpad=.2, handletextpad=.4, columnspacing=.9)
    style(a, "a  Eligibility locking", horizontal=True, zero=True)

    composition = data.loc[data.kind.eq("first_divergence_figure_cohorts")]
    order = ["Eligibility", "Within-set rank", "Conflict", "Top-k"]
    hatches = {"Within-set rank": "///", "Conflict": "...", "Top-k": "xx"}
    bars_b = [
        (label, rows.set_index("label").estimate.to_dict())
        for label, rows in composition.groupby("contract", sort=False)
    ]
    n_b = len(bars_b)
    for idx, (_, vals) in enumerate(bars_b):
        y = (n_b - 1) - idx
        assert np.isclose(sum(vals.values()), 100.0, atol=0.02)
        left = 0.0
        for label in order:
            value = vals[label]
            b.barh(y, value, left=left, height=.52, color=CAUSE[label],
                   edgecolor="white", linewidth=0, hatch=hatches.get(label))
            left += value
    groups(b, [x[0] for x in bars_b], top_pad=0.65)
    b.set(xlim=(0, 100), xlabel="First observed stage (%)")
    b.set_xticks([0, 25, 50, 75, 100])
    style(b, "b  Exclusive first divergence", horizontal=True)
    fig.legend(handles=[Patch(facecolor=CAUSE[label], edgecolor="white", linewidth=0,
                              hatch=hatches.get(label), label=label) for label in order], loc="lower center", ncol=4, bbox_to_anchor=(.54, .008))

    labels = ["Eligibility", "Within-set rank", "Eligibility ∩ Within-set"]
    for i, contract in enumerate(STRESS):
        for j, label in enumerate(labels):
            row = data.loc[data.kind.eq("nonexclusive_any_stage") &
                           data.contract.eq(contract) & data.label.eq(label)].iloc[0]
            forest(c, row.estimate, 2-j + (.13 if i == 0 else -.13), row,
                   CAUSE.get(label, "#33383D"), "os"[i], opened=bool(i))
    groups(c, ["Eligibility", "Within-set rank", "Both stages"], top_pad=.92)
    c.set(xlim=(0, 103), xlabel="Any-stage occurrence (%)")
    c.set_xticks([0, 25, 50, 75, 100])
    c.legend(handles=[marker_key(label, "#41464A", "os"[i], bool(i))
                      for i, label in enumerate(SHORT_STRESS)], loc="upper left", ncol=2)
    style(c, "c  Stage co-occurrence", horizontal=True)

    prepost = ["Pre-conflict Top-7 winner gap", "Final assigned gap"]
    for i, contract in enumerate(STRESS):
        pair = data.loc[data.kind.eq("pre_resolution_control") &
                        data.contract.eq(contract)].set_index("label")
        for j, label in enumerate(prepost):
            row = pair.loc[label]
            forest(d, row.estimate, 1-i + (.12 if j == 0 else -.12), row,
                   BRANCH["O2O"], "so"[j], opened=j == 0)
    groups(d, SHORT_STRESS, top_pad=.81)
    d.set(xlim=(-4, 22), xlabel="O2O scale gap (pp)")
    d.legend(handles=[marker_key("Pre-conflict", BRANCH["O2O"], "s", True),
                      marker_key("Final", BRANCH["O2O"])], loc="upper right", ncol=2)
    style(d, "d  Pre-resolution control", horizontal=True, zero=True)
    return fig


def step_band(ax, rows, color, linestyle="-"):
    rows = rows.sort_values("radius")
    ax.step(rows.radius, rows.estimate, where="post", color=color, linestyle=linestyle, linewidth=1.25)
    ax.fill_between(rows.radius, rows.ci_low, rows.ci_high, step="post",
                    color=color, alpha=.13, linewidth=0)


def radius_axis(ax):
    ax.set_xlim(-.002, .128)
    ax.set_xticks([0, .0625, .125], ["0", "0.0625", "0.125"])


def figure_4(data):
    fig = plt.figure(figsize=(178 / 25.4, 131 / 25.4))
    grid = fig.add_gridspec(2, 2, left=.098, right=.978, bottom=.109, top=.842,
                           wspace=.43, hspace=.69)
    survival = grid[0, 0].subgridspec(1, 2, wspace=.31)
    handles = [
        Line2D([], [], color=BRANCH["O2O"], marker="o", label="O2O assigned identity"),
        Line2D([], [], color=BRANCH["O2M"], marker="s", ls="--", label="O2M assigned-set Top-1"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.54, .991))
    for i, size in enumerate(["t_8_16", "s_16_32"]):
        ax = fig.add_subplot(survival[0, i])
        for branch, estimand, ls in [("O2O", "o2o", "-"), ("O2M", "o2m_assigned_positive", "--")]:
            step_band(ax, data.loc[data.panel.eq("a") & data.size_group.eq(size) &
                                   data.estimand.eq(estimand)], BRANCH[branch], ls)
        radius_axis(ax)
        ax.set(ylim=(.47, 1.015), xlabel="Distance r", yticks=[.5, .75, 1])
        style(ax, SIZES[i+1])
        if i == 0:
            ax.annotate("a  First-departure survival", (0, 1.28), xycoords="axes fraction",
                        fontweight="bold", fontsize=9.5, annotation_clip=False)
            ax.set_ylabel("Survival S(r)")
        else:
            ax.tick_params(axis="y", labelleft=False)

    b = fig.add_subplot(grid[0, 1])
    causes = [("Eligibility", "Eligibility", "-"), ("Within-set", "Within-set rank", "--"),
              ("Conflict", "Conflict", "-."), ("Top-k", "Top-k", ":"), ("Other", "Other", "--")]
    for label, color_key, ls in causes:
        step_band(b, data.loc[data.panel.eq("b") & data.series.eq(label)], CAUSE[color_key], ls)
    radius_axis(b)
    b.set(ylim=(0, .087), xlabel="Distance r", ylabel="Cumulative incidence",
          yticks=[0, .02, .04, .06, .08])
    b.legend(handles=[Line2D([], [], color=CAUSE[key], ls=ls, label=key)
                      for label, key, ls in causes], ncol=2, loc="upper left",
             columnspacing=.8, handlelength=1.4)
    style(b, "b  First O2O boundary cause")

    c_grid = grid[1, 0].subgridspec(1, 2, width_ratios=(1.2, 1), wspace=.25)
    absolute = fig.add_subplot(c_grid[0, 0])
    delta = fig.add_subplot(c_grid[0, 1])
    c = data.loc[data.panel.eq("c")]
    for i, label in enumerate(SIZES):
        rows = c.loc[c.size_label.eq(label)]
        for j, branch in enumerate(["O2O", "O2M"]):
            row = rows.loc[rows.series.eq(branch)].iloc[0]
            forest(absolute, row.estimate, 2-i + (.12 if j == 0 else -.12), row,
                   BRANCH[branch], MARKER[branch])
        row = rows.loc[rows.kind.eq("rmcbd_paired")].iloc[0]
        forest(delta, row.estimate, 2-i, row, BRANCH["Paired"], "D")
    groups(absolute, SIZES)
    groups(delta, SIZES)
    delta.tick_params(axis="y", labelleft=False, length=0)
    absolute.annotate("c  Restricted mean distance", (0, 1.28), xycoords="axes fraction",
                      fontweight="bold", fontsize=9.5, annotation_clip=False)
    absolute.set(xlim=(.079, .126), xticks=[.08, .10, .12], xlabel="Absolute RMCBD")
    delta.set(xlim=(0, .042), xticks=[0, .02, .04], xlabel="O2O − O2M")
    style(absolute, "Absolute", horizontal=True)
    style(delta, "Difference", horizontal=True)

    d = fig.add_subplot(grid[1, 1])
    for i, label in enumerate(SIZES):
        rows = data.loc[data.panel.eq("d") & data.series.eq(label)].sort_values("tau")
        errors = [rows.estimate-rows.ci_low, rows.ci_high-rows.estimate]
        d.errorbar(rows.tau, rows.estimate, yerr=errors, color=BRANCH["Paired"],
                   marker="Dos"[i], markerfacecolor="white" if i == 1 else BRANCH["Paired"],
                   linestyle=["-", "--", ":"][i], linewidth=1.10, markersize=4.7,
                   elinewidth=.75, capsize=2, label=label)
    d.axvline(.125, color="#8D949C", linewidth=.7, linestyle="--")
    d.set(xlim=(.039, .259), ylim=(-.006, .109), xlabel="Truncation horizon τ",
          ylabel="O2O − O2M RMCBD", yticks=[0, .025, .05, .075, .10])
    d.set_xticks([.05, .10, .15, .20, .25], ["0.05", "0.10", "0.15", "0.20", "0.25"])
    d.legend(loc="upper left", handlelength=1.55)
    style(d, "d  Truncation sensitivity")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", type=int, nargs="*", metavar="N")
    selected = set(parser.parse_args().figures or [2, 3, 4])
    if selected - {2, 3, 4}:
        parser.error("figure numbers must be 2, 3, or 4")
    for number, draw in [(2, figure_2), (3, figure_3), (4, figure_4)]:
        if number not in selected:
            continue
        source = HERE / f"fig{number}_source.csv"
        if hashlib.sha256(source.read_bytes()).hexdigest() != HASHES[number]:
            raise ValueError(f"Source SHA-256 mismatch for {source.name}")
        fig = draw(pd.read_csv(source))
        for extension in ["pdf", "svg", "png"]:
            fig.savefig(HERE / f"Figure_{number}.{extension}", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()

