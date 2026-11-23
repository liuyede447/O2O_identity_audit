"""Re-layout Figures 5 and S1 using their original evidence-bound generators."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as DrawPath
from matplotlib.text import Text
import numpy as np
from PIL import Image

def coverage(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def export(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=600, facecolor="white")
    plt.close(fig)
    png = stem.with_suffix(".png")
    with Image.open(png) as image:
        rgb = image.convert("RGB")
    rgb.save(png, dpi=(600, 600))

def set_legend(ax, labels, **kw):
    handles, _ = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    return ax.legend(handles, labels, frameon=False, handletextpad=.42,
                     columnspacing=.72, borderaxespad=0, **kw)

def apply_fig5_layout(fig):
    a, b, c, d = fig.axes
    case_centres = [[.17 + .31 * case + .067 * value for value in range(3)]
                    for case in range(3)]
    for index, patch in enumerate(a.patches):
        group, step = divmod(index, 6)
        patch.center = (case_centres[group][step % 3], patch.center[1])
        patch.radius = .027
    label_counts = {"Base": 0, "Shift": 0}
    member_index = 0
    headings = {"Rank-only": (0, "Rank only"), "Rank + set": (1, "Rank + set"),
                "Set-only; Top-1 stable": (2, "Set only")}
    for text in list(a.texts):
        label = text.get_text()
        _, y = text.get_position()
        if label in headings:
            index, replacement = headings[label]
            text.set_text(replacement)
            text.set_position((case_centres[index][1], y))
        elif label in label_counts:
            label_counts[label] += 1
            if label_counts[label] == 1:
                text.set_position((.01, y))
            else:
                text.remove()
        elif label in {"A", "B", "C", "D"}:
            case, item = divmod(member_index, 6)
            text.set_position((case_centres[case][item % 3], y))
            member_index += 1
    assert member_index == 18
    for text in list(b.texts):
        if text.get_text() in {"SET PERSISTENCE", "RANK TURNOVER"}:
            text.remove()
    b.set_title("Native rank and set changes", loc="left", pad=20)
    set_legend(b, ["Fixed 1 px", "Eq. area"], loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2)
    b.set_xlabel("16–32 minus 8–16 px (pp)")
    c.set_title("Positive-count distribution", loc="left", pad=20)
    # Match the size symbols in Figure 4(d). Move the paired stems apart
    # within each integer count so the low-proportion markers stay distinct.
    size_color = "#805399"
    for line, stems, marker, face, offset in zip(
        c.lines, c.collections, ("o", "s"), ("white", size_color), (-0.055, 0.055), strict=True
    ):
        line.set_color(size_color)
        line.set_marker(marker)
        line.set_markerfacecolor(face)
        line.set_xdata(line.get_xdata() + offset)
        stems.set_color(size_color)
        stems.set_segments([segment + (offset, 0) for segment in stems.get_segments()])
    set_legend(c, ["8–16 px", "16–32 px"], loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2)
    d.set_title("Support and count adjustment", loc="left", pad=20)
    set_legend(d, ["Full", "Overlap", "Std. overlap"], loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3)
    d.set_xlabel("16–32 minus 8–16 px (pp)")
    fig.set_size_inches(178/25.4, 135/25.4)
    fig.tight_layout(pad=.6, h_pad=.9, w_pad=.85)
    fig.canvas.draw()
    return fig

RUNNER_YELLOW = "#FFDD32"


def add_runner_stroke(mark):
    mark.set_linewidth(1.10)
    mark.set_linestyle((0, (2.2, 1.15)))
    mark.set_path_effects([
        patheffects.Stroke(linewidth=2.05, foreground="#242424"),
        patheffects.Normal(),
    ])
    return mark

def add_assigned_stroke(mark):
    mark.set_linewidth(1.0)
    mark.set_path_effects([
        patheffects.Stroke(linewidth=1.95, foreground="#FAFAFA"),
        patheffects.Normal(),
    ])
    return mark


def corner_handle(color, linestyle, width):
    return PathPatch(DrawPath([(0, 0), (0, 1), (1, 1)]), fill=False,
                     edgecolor=color, linestyle=linestyle, linewidth=width)

def corner_key(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    x, y = -xdescent + .09 * width, -ydescent + .08 * height
    w, h = .82 * width, .84 * height
    vertices = [(x,y+h*.56), (x,y+h), (x+w*.32,y+h),
                (x+w*.68,y+h), (x+w,y+h), (x+w,y+h*.56),
                (x+w,y+h*.44), (x+w,y), (x+w*.68,y),
                (x+w*.32,y), (x,y), (x,y+h*.44)]
    codes = [DrawPath.MOVETO, DrawPath.LINETO, DrawPath.LINETO] * 4
    return PathPatch(DrawPath(vertices, codes), fill=False, edgecolor=orig_handle.get_edgecolor(),
                     linestyle=orig_handle.get_linestyle(), linewidth=orig_handle.get_linewidth(),
                     path_effects=orig_handle.get_path_effects())

def audit_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip()]
    # Major tick objects outside the axis limits are not painted.
    all_ticks = {id(tick) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
                 for tick in axis.get_ticklabels()}
    items = [t for t in visible if id(t) not in all_ticks]
    clipped = [t.get_text() for t in items
               if not fig.bbox.padded(1).contains(*t.get_window_extent(renderer).p0)
               or not fig.bbox.padded(1).contains(*t.get_window_extent(renderer).p1)]
    if clipped: raise RuntimeError(f"Clipped labels: {clipped}")
    return min(t.get_fontsize() for t in visible)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.analysis_root.resolve() / "figures_final" / "src"
    sys.path.insert(0, str(baseline))
    import figure_style
    import plot_fig5
    import plot_figs1
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fonts = {"panel_label": 10.0, "panel_title": 9.0, "axis_label": 8.7,
             "legend": 8.7, "tick": 8.7, "annotation": 8.7, "minimum": 8.7}
    figure_style.FONT.update(fonts)
    plot_fig5.BRANCH = {**plot_fig5.BRANCH, "O2M": "#C86718", "Paired": "#35383C"}
    rows, meta = plot_fig5.source_rows()
    f5_validation = {}
    def save_fig5(fig, scratch_stem, *, dpi=600):
        apply_fig5_layout(fig)
        f5_validation["minimum_canvas_font_pt"] = audit_text(fig)
        export(fig, scratch_stem)
    with TemporaryDirectory(prefix="fig5_style_") as d:
        plot_fig5.FIG = Path(d)
        plot_fig5.save_figure = save_fig5
        plot_fig5.render(rows)
        for ext in ("pdf", "svg", "png"):
            (out / f"Figure_5.{ext}").write_bytes((Path(d)/"draft"/f"Fig5.{ext}").read_bytes())
    f5_validation.update({"figure":"Fig5","source_rows":len(rows),
                          "contrast_orientation":"16–32 minus 8–16 px in b and d",
                          "source_artifacts":plot_fig5.verify(),"original_plot_script_sha256":coverage(baseline/"plot_fig5.py")})

    trace, _, replay, verified = plot_figs1.load_and_verify_sources()
    for result in [plot_figs1.validate_identities(trace,replay), plot_figs1.validate_shift_coordinates(trace)]:
        if result["status"] != "PASS": raise RuntimeError(result)
    with Image.open(plot_figs1.ROOT / plot_figs1.IMAGE_REL) as source:
        image = np.array(source.convert("RGB"))
        size = source.size
    bbox = plot_figs1.detect_nonpadding_content_bbox(image)
    rows, _ = plot_figs1.build_rows(trace, verified, size, bbox)
    checks = [plot_figs1.coordinate_parity(rows,trace),
              plot_figs1.box_and_crop_validation(rows,size),
              plot_figs1.padding_crop_validation(rows,image,bbox)]
    if any(c["status"] != "PASS" for c in checks): raise RuntimeError(checks)
    fig, _ = plot_figs1.make_figure(trace, rows, image)
    # Keep the audited geometry and add a high-contrast stroke to the runner corners.
    runner_color = to_rgba(plot_figs1.RUNNER_COLOR)
    runner_corners = [
        line for ax in fig.axes for line in ax.lines
        if to_rgba(line.get_color()) == runner_color
    ]
    runner_count = sum(row["role"] == "runner_candidate" for row in rows)
    if not runner_count or len(runner_corners) != 4 * runner_count:
        raise RuntimeError("Runner-up corner count does not match the audited overlays")
    for line in runner_corners:
        line.set_color(RUNNER_YELLOW)
        add_runner_stroke(line)
    active_color = to_rgba(plot_figs1.ACTIVE_COLOR)
    assigned_corners = [
        line for ax in fig.axes for line in ax.lines
        if to_rgba(line.get_color()) == active_color
    ]
    assigned_count = sum(row["role"] == "active_candidate" for row in rows)
    if not assigned_count or len(assigned_corners) != 4 * assigned_count:
        raise RuntimeError("Assigned O2O corner count does not match the audited overlays")
    for line in assigned_corners:
        add_assigned_stroke(line)
    for t in list(fig.texts):
        if t.get_text().startswith(("Representative","Detail insets","Overlays are")):
            t.remove()
    for ax in fig.axes:
        t = ax.get_title(loc="left")
        if not t: continue
        t = t.replace("Identity-disappearance departure", "Identity disappearance").replace("assigned O2O ", "O2O ")
        ax.set_title(t.replace("undefined","none"), loc="left", pad=4)
    fig.subplots_adjust(left=.035,right=.992,top=.903,bottom=.093,hspace=.31,wspace=.08)
    fig.canvas.draw()
    for t, ax in zip(fig.texts, fig.axes[:3]):
        p = ax.get_position()
        t.set_position(((p.x0+p.x1)*.5,.955))
    fig.legends[0].remove()
    p = plot_figs1
    handles = [
       Rectangle((0,0),1,1,fill=False,edgecolor=p.FOCAL_GT_COLOR,linewidth=1.0,label="Focal GT"),
       Rectangle((0,0),1,1,fill=False,edgecolor=p.SHIFT_GT_COLOR,linewidth=1.0,linestyle="--",label="Shifted GT"),
       add_assigned_stroke(corner_handle(p.ACTIVE_COLOR,"-",1.0)), add_runner_stroke(corner_handle(RUNNER_YELLOW,"--",1.0)),
    ]
    fig.legend(handles, ["Focal GT","Shifted GT","Assigned O2O","Runner-up"],
               handler_map={PathPatch:HandlerPatch(patch_func=corner_key)},
               loc="lower center", bbox_to_anchor=(.5,.027), ncol=4,
               handlelength=1.6, columnspacing=1.1, handletextpad=.45, frameon=False)
    fig.canvas.draw()
    geom = p.geometry_audit(fig,fig.axes)
    if geom["status"] != "PASS": raise RuntimeError(geom)
    fs1_validation={"figure":"FigS1","minimum_canvas_font_pt":audit_text(fig),
                    "source_artifacts":verified,"coordinate_checks":checks,"text_geometry":geom,
                    "runner_overlay_count":runner_count,"assigned_overlay_count":assigned_count,
                    "original_plot_script_sha256":coverage(baseline/"plot_figs1.py")}
    export(fig,out/"Figure_S1")
    for name,rec in [("Figure_5",f5_validation),("Figure_S1",fs1_validation)]:
        rec["status"]="PASS"
        rec["outputs"]={e:coverage(out/f"{name}.{e}") for e in ["pdf","svg","png"]}
        (out/f"{name}.validation.json").write_text(json.dumps(rec,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
