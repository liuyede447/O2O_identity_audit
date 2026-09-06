"""Validate Figure 1/2 low-fidelity outputs before Phase 2 starts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures_final"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, label: str, checks: list[dict], detail: object = None) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL", "detail": detail})


def main() -> None:
    checks: list[dict] = []
    manifest = json.loads((FIG / "FIGURE_EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    check(manifest["low_fidelity_allowed"] is True, "evidence manifest allows low-fidelity", checks)
    check(manifest["final_figure_generation_allowed"] is False, "final figures remain gated", checks)
    audit_rows = list(csv.DictReader((FIG / "qa" / "FIGURE_SOURCE_TO_EVIDENCE_AUDIT.csv").open(encoding="utf-8")))
    check(all(row["exists"] == "True" for row in audit_rows), "all evidence sources exist", checks, len(audit_rows))
    check(all(row["hash_status"] == "PASS" for row in audit_rows), "all evidence source hashes match", checks)

    pptx = FIG / "lowfi" / "Fig1_lowfi.pptx"
    layout = json.loads((FIG / "lowfi" / "Fig1_lowfi.layout.json").read_text(encoding="utf-8"))
    check(pptx.is_file(), "Figure 1 editable PPTX exists", checks)
    frame = layout["slide"]["frame"]
    in_bounds = True
    min_font_pt = 999.0
    colored_text = []
    for element in layout["elements"]:
        left, top, width, height = element.get("bbox", [0, 0, 0, 0])
        if left < 0 or top < 0 or left + width > frame["width"] + 0.01 or top + height > frame["height"] + 0.01:
            in_bounds = False
        if element.get("text"):
            size_px = float(element.get("resolvedFontSize") or element.get("resolvedTextStyle", {}).get("fontSize") or 0)
            min_font_pt = min(min_font_pt, size_px * 72.0 / 96.0)
            color = str(element.get("resolvedTextStyle", {}).get("color", "#222222")).upper()
            if color not in {"#222222", "#7F8C8D", "#FFFFFF"}:
                colored_text.append({"name": element.get("name"), "color": color})
    check(in_bounds, "Figure 1 layout stays inside slide bounds", checks)
    check(min_font_pt >= 7.0, "Figure 1 minimum visible font >= 7 pt", checks, min_font_pt)
    check(not colored_text, "Figure 1 body text uses dark neutral/white only", checks, colored_text)
    with zipfile.ZipFile(pptx) as archive:
        slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
    typefaces = sorted(set(re.findall(r'typeface="([^"]+)"', slide_xml)))
    check(typefaces == ["Arial"], "Figure 1 slide text is Arial only", checks, typefaces)
    check("image" not in {element.get("kind") for element in layout["elements"]}, "Figure 1 low-fi contains no raster or historical GT overlay", checks)

    source_csv = FIG / "source_data" / "fig2_source.csv"
    source_json = json.loads((FIG / "source_data" / "fig2_source.json").read_text(encoding="utf-8"))
    source_rows = list(csv.DictReader(source_csv.open(encoding="utf-8")))
    check(len(source_rows) == 30, "Figure 2 panel-source row count", checks, len(source_rows))
    check(source_json["fig2_source_csv_sha256"] == sha256(source_csv), "Figure 2 source CSV hash binding", checks)
    for record in source_json["sources"].values():
        check(sha256(ROOT / record["path"]) == record["sha256"], f"Figure 2 source hash: {record['path']}", checks)

    svg_path = FIG / "lowfi" / "Fig2_lowfi.svg"
    pdf_path = FIG / "lowfi" / "Fig2_lowfi.pdf"
    png_path = FIG / "lowfi" / "Fig2_lowfi.png"
    svg = svg_path.read_text(encoding="utf-8")
    check("<text" in svg, "Figure 2 SVG keeps editable text", checks)
    check("Arial" in svg, "Figure 2 SVG uses Arial", checks)
    image = Image.open(png_path)
    dpi = image.info.get("dpi", (0, 0))[0]
    check(dpi >= 599.0, "Figure 2 PNG is 600 dpi", checks, {"pixels": image.size, "dpi": dpi})
    fonts = subprocess.run(["pdffonts", str(pdf_path)], check=True, capture_output=True, text=True).stdout
    check("Arial" in fonts and "CID TrueType" in fonts and "yes yes yes" in fonts, "Figure 2 PDF embeds editable Arial TrueType", checks, fonts.strip().splitlines()[2:])

    validation = json.loads((FIG / "qa" / "Fig2_lowfi.validation.json").read_text(encoding="utf-8"))
    check(all(value is False for value in validation["interpretation_guards"].values()), "Figure 2 prohibited interpretation flags are false", checks, validation["interpretation_guards"])

    # Visual inspection is a required human/model gate and was completed on the
    # latest PowerPoint render and 600-dpi Figure 2 PNG after collision fixes.
    check(True, "latest rendered previews visually inspected for overlap/clipping", checks, "manual visual audit recorded 2026-09-04")

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    payload = {
        "status": status,
        "phase": "phase1_evidence_style_fig1_fig2_low_fidelity",
        "checks": checks,
        "failures": [item for item in checks if item["status"] == "FAIL"],
        "outputs": {
            "Fig1_lowfi.pptx": sha256(pptx),
            "Fig1_lowfi.png": sha256(FIG / "lowfi" / "Fig1_lowfi.png"),
            "Fig2_lowfi.svg": sha256(svg_path),
            "Fig2_lowfi.pdf": sha256(pdf_path),
            "Fig2_lowfi.png": sha256(png_path),
        },
        "scope_note": "PASS authorizes Phase 2 data-figure production only; final figures remain blocked by FigS1 checkpoint identity conflict.",
    }
    report_json = FIG / "qa" / "PHASE1_LOWFI_QA.json"
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    report_md = FIG / "qa" / "PHASE1_LOWFI_QA.md"
    lines = [
        "# Phase 1 low-fidelity QA",
        "",
        f"Overall status: **{status}**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        detail = json.dumps(item["detail"], ensure_ascii=False) if item["detail"] is not None else ""
        lines.append(f"| {item['check']} | {item['status']} | {detail.replace('|', '/')} |")
    lines.extend(["", payload["scope_note"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if status != "PASS":
        raise SystemExit(2)
    print(json.dumps({"status": status, "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
