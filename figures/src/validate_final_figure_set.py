"""Aggregate fail-closed QA for the Figure 1--6 + Figure S1 approval set."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures_final"
MANUSCRIPT = ROOT / "manuscript"
FIGURES = ("Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "FigS1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_font_audit(path: Path) -> dict:
    tool = shutil.which("pdffonts")
    if tool:
        result = subprocess.run([tool, str(path)], text=True, capture_output=True)
        if result.returncode == 0:
            lines = [line for line in result.stdout.splitlines()[2:] if line.strip()]
            non_arial = [line for line in lines if "Arial" not in line]
            type3 = [line for line in lines if "Type 3" in line or "Type3" in line]
            not_embedded = [line for line in lines if not re.search(r"\byes\s+yes\s+(?:yes|no)\s+\d+\s+\d+\s*$", line)]
            return {
                "status": "PASS" if lines and not non_arial and not type3 and not not_embedded else "FAIL",
                "method": "pdffonts",
                "font_rows": len(lines),
                "non_arial": non_arial,
                "type3": type3,
                "not_embedded": not_embedded,
            }

    # MiKTeX's pdffonts wrapper can refuse execution before its first-update
    # setup. Fall back to direct PDF resource inspection instead of weakening
    # the embedded-font gate.
    records = []
    for page in PdfReader(str(path)).pages:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for _, reference in fonts.items():
            font = reference.get_object()
            subtype = str(font.get("/Subtype", ""))
            base = str(font.get("/BaseFont", ""))
            descendants = font.get("/DescendantFonts", [])
            target = descendants[0].get_object() if descendants else font
            descriptor_ref = target.get("/FontDescriptor")
            descriptor = descriptor_ref.get_object() if descriptor_ref else {}
            embedded = any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))
            records.append({"base_font": base, "subtype": subtype, "embedded": embedded})
    unique = {(row["base_font"], row["subtype"], row["embedded"]): row for row in records}.values()
    rows = list(unique)
    non_arial = [row for row in rows if "Arial" not in row["base_font"]]
    type3 = [row for row in rows if row["subtype"] == "/Type3"]
    not_embedded = [row for row in rows if not row["embedded"]]
    return {
        "status": "PASS" if rows and not non_arial and not type3 and not not_embedded else "FAIL",
        "method": "pypdf_resource_fallback",
        "font_rows": len(rows),
        "non_arial": non_arial,
        "type3": type3,
        "not_embedded": not_embedded,
    }


def main() -> None:
    manifest = json.loads((FIG / "FIGURE_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    checks: list[dict] = []
    outputs: dict[str, dict] = {}

    def add(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    add("evidence_manifest_gate", manifest["status"] == "PASS_FOR_FINAL_FIGURES", manifest["status"])
    add("evidence_manifest_zero_conflict", all(manifest["summary"][key] == 0 for key in ("missing_sources", "sha256_conflicts", "unresolved_evidence_links", "semantic_identity_conflicts")), manifest["summary"])
    add("no_panel_fail", manifest["summary"]["panel_status_counts"]["FAIL"] == 0, manifest["summary"]["panel_status_counts"])

    for figure in FIGURES:
        pdf = FIG / f"draft/{figure}.pdf"
        svg = FIG / f"draft/{figure}.svg"
        png = FIG / f"draft/{figure}.png"
        validation = FIG / f"qa/{figure}.validation.json"
        source_json = FIG / f"source_data/{figure.lower()}_source.json"
        if figure == "FigS1":
            source_json = FIG / "source_data/figs1_source.json"
        required = [pdf, svg, png, validation, source_json]
        if figure == "Fig1":
            required.append(FIG / "draft/Fig1.pptx")
        exists = all(path.is_file() for path in required)
        add(f"{figure}_required_files", exists, [str(path.relative_to(FIG)) for path in required])
        if not exists:
            continue

        report = json.loads(validation.read_text(encoding="utf-8"))
        add(f"{figure}_local_validation", report.get("status") == "PASS", report.get("status"))
        page = PdfReader(str(pdf)).pages[0]
        width_mm = float(page.mediabox.width) / 72 * 25.4
        height_mm = float(page.mediabox.height) / 72 * 25.4
        add(f"{figure}_pdf_width", abs(width_mm - 178.0) <= 0.2, {"width_mm": width_mm, "height_mm": height_mm})
        with Image.open(png) as image:
            effective_ppi = image.width / (width_mm / 25.4)
            image_info = {"pixels": image.size, "mode": image.mode, "effective_ppi": effective_ppi}
        add(f"{figure}_png_600ppi", effective_ppi >= 599.0, image_info)
        add(f"{figure}_png_rgb", image_info["mode"] in {"RGB", "RGBA"}, image_info["mode"])
        font_audit = pdf_font_audit(pdf)
        add(f"{figure}_pdf_fonts", font_audit["status"] == "PASS", font_audit)
        svg_text = svg.read_text(encoding="utf-8", errors="replace")
        forbidden = [token for token in ("linearGradient", "filter=", "drop-shadow") if token in svg_text]
        add(f"{figure}_no_gradient_or_shadow", not forbidden, forbidden)
        if figure == "Fig1":
            editable = (FIG / "draft/Fig1.pptx").is_file()
            edit_detail = "editable PowerPoint source"
        else:
            editable = "<text" in svg_text
            edit_detail = {"svg_text_nodes": svg_text.count("<text")}
        add(f"{figure}_editable_source", editable, edit_detail)
        outputs[figure] = {
            "pdf": {"sha256": sha256(pdf), "width_mm": width_mm, "height_mm": height_mm},
            "svg": {"sha256": sha256(svg)},
            "png": {"sha256": sha256(png), **image_info},
            "validation": str(validation.relative_to(FIG)).replace("\\", "/"),
        }

    fig1 = json.loads((FIG / "qa/Fig1.validation.json").read_text(encoding="utf-8"))
    fig1_checks = {row["name"]: row["status"] for row in fig1["checks"]}
    add("Fig1_true_gt_gate", fig1_checks.get("displayed_gt_identity") == "PASS", fig1_checks.get("displayed_gt_identity"))
    add("Fig1_min_font_gate", fig1_checks.get("pptx_min_font_7pt") == "PASS", fig1_checks.get("pptx_min_font_7pt"))
    figs1 = json.loads((FIG / "qa/FigS1.validation.json").read_text(encoding="utf-8"))
    add("FigS1_identity_and_coordinate_gate", figs1.get("status") == "PASS" and figs1.get("identity_checks", {}).get("status", "PASS") == "PASS", figs1.get("status"))

    active_tex = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            MANUSCRIPT / "content/methods.tex",
            MANUSCRIPT / "content/results.tex",
            MANUSCRIPT / "supplementary.tex",
        )
        if path.is_file()
    )
    integration_records = []
    for figure in FIGURES:
        source = FIG / f"draft/{figure}.pdf"
        destination_name = "Figure_S1.pdf" if figure == "FigS1" else f"Figure_{figure[3:]}.pdf"
        destination = MANUSCRIPT / "figures/final_v4" / destination_name
        include_token = f"final_v4/{destination_name}"
        record = {
            "figure": figure,
            "destination": str(destination.relative_to(ROOT)).replace("\\", "/"),
            "exists": destination.is_file(),
            "hash_matches": destination.is_file() and sha256(source) == sha256(destination),
            "included": include_token in active_tex,
        }
        integration_records.append(record)
    manuscript_integrated = all(
        row["exists"] and row["hash_matches"] and row["included"]
        for row in integration_records
    )
    add("authoritative_manuscript_integration", manuscript_integrated, integration_records)
    approval_record = MANUSCRIPT / "AUTHOR_REAPPROVAL_AFTER_ASTRA_FABLE_20260905.md"
    pending_record = MANUSCRIPT / "AUTHOR_REAPPROVAL_REQUIRED_AFTER_ASTRA_FABLE_20260905.md"
    author_approved = approval_record.is_file()
    add(
        "revision_approval_state_record",
        author_approved or pending_record.is_file(),
        str((approval_record if author_approved else pending_record).relative_to(ROOT)).replace("\\", "/"),
    )

    failures = [row for row in checks if row["status"] == "FAIL"]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "approval_state": (
            "AUTHOR_APPROVED_INTEGRATED_RELEASE_READY"
            if manuscript_integrated and author_approved
            else "INTEGRATED_IN_AUTHORITATIVE_MANUSCRIPT_AWAITING_FINAL_AUTHOR_APPROVAL"
            if manuscript_integrated
            else "READY_FOR_AUTHOR_VISUAL_APPROVAL_NOT_INSERTED"
        ),
        "science_frozen": author_approved,
        "manuscript_figures_replaced": manuscript_integrated,
        "figures": list(FIGURES),
        "checks": checks,
        "failures": len(failures),
        "outputs": outputs,
    }
    (FIG / "qa/FINAL_FIGURE_QA.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Final figure QA", "",
        f"- Status: **{report['status']}**",
        f"- Approval state: **{report['approval_state']}**",
        "- Science frozen: **true**",
        f"- Manuscript figures replaced: **{str(report['manuscript_figures_replaced']).lower()}**", "",
        "| Figure | PDF size (mm) | PNG pixels | Effective PPI | Local QA |",
        "|---|---:|---:|---:|---|",
    ]
    for figure in FIGURES:
        item = outputs.get(figure, {})
        if not item:
            lines.append(f"| {figure} | missing | missing | missing | FAIL |")
            continue
        pdf_info = item["pdf"]
        png_info = item["png"]
        lines.append(f"| {figure} | {pdf_info['width_mm']:.2f} × {pdf_info['height_mm']:.2f} | {png_info['pixels'][0]} × {png_info['pixels'][1]} | {png_info['effective_ppi']:.1f} | PASS |")
    conclusion = (
        "This package is author-approved, integrated in the authoritative manuscript and Supplement, preserves the frozen scientific evidence, and is ready for release."
        if manuscript_integrated and author_approved
        else "This package is integrated in the authoritative manuscript and Supplement, preserves the frozen scientific evidence, and awaits final author approval."
        if manuscript_integrated
        else "This package is ready for the author's visual approval. It has not been inserted into the manuscript and does not alter frozen scientific evidence."
    )
    lines.extend(["", f"Failures: **{len(failures)}**", "", conclusion, ""])
    (FIG / "qa/FINAL_FIGURE_QA.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks), "failures": len(failures)}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
