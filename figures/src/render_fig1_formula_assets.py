from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(r"E:\two_paper\beifen\publication_reproducible\figures_final")
OUT = ROOT / "source_data"


def render_formula(filename: str, lines: list[str], width: float, height: float) -> None:
    mpl.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "svg.fonttype": "path",
            "text.color": "#111111",
        }
    )
    fig = plt.figure(figsize=(width, height), dpi=300, facecolor="none")
    if len(lines) == 1:
        ys = [0.50]
    else:
        ys = [0.71, 0.27]
    for formula, y in zip(lines, ys, strict=True):
        fig.text(
            0.02,
            y,
            formula,
            fontsize=10.2,
            ha="left",
            va="center",
            color="#111111",
        )
    # Tight vector bounds are important here: PowerPoint otherwise scales the
    # formula together with a large invisible canvas and makes the glyphs tiny.
    fig.savefig(OUT / filename, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
render_formula("fig1_fixed_formula.svg", [r"$\delta=1\,\mathrm{px}$"], 1.05, 0.28)
render_formula(
    "fig1_equivalent_area_formula.svg",
    [r"$\delta=\kappa s_g$", r"$s_g=\sqrt{w_g h_g}$"],
    1.15,
    0.52,
)
render_formula("fig1_rho_card_formula.svg", [r"$\rho_{\mathrm{card}}$"], 0.72, 0.28)
