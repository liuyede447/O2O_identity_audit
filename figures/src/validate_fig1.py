"""Validate the editable Figure 1 against its frozen source and journal geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
import win32com.client


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures_final"
PPTX = FIG / "draft/Fig1.pptx"
PDF = FIG / "draft/Fig1.pdf"
PNG = FIG / "draft/Fig1.png"
SVG = FIG / "draft/Fig1.svg"
SOURCE = FIG / "source_data/fig1_source.json"
TRACE = FIG / "source_data/figs1_primary_trace/primary_qualitative_trace.json"
ORIGINAL = FIG / "source_data/figs1_primary_trace/0000170_00401_d_0000001__160_0.png"
CROP = FIG / "source_data/fig1_focal_audit_crop_500x375.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    expected_gt = {int(row["gt_id"]): row["gt_xyxy"] for row in trace["records"]}
    displayed_gt = {int(row["gt_id"]): row["xyxy"] for row in source["displayed_gt"]}

    checks: list[dict] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    check("all_outputs_exist", all(path.is_file() for path in (PPTX, PDF, PNG, SVG)), [str(path) for path in (PPTX, PDF, PNG, SVG)])
    check("primary_checkpoint", source["trace"]["checkpoint_sha256"] == trace["checkpoint_sha256"], trace["checkpoint_sha256"])
    check("displayed_gt_identity", displayed_gt == expected_gt and set(displayed_gt) == {17, 28, 40}, displayed_gt)
    check("display_crop_sha", source["image"]["sha256"] == sha256(CROP), sha256(CROP))
    check("original_image_sha", sha256(ORIGINAL) == trace["source_image_sha256"], sha256(ORIGINAL))
    with Image.open(CROP) as image:
        check("display_crop_pixels", image.size == (500, 375), image.size)
    source_ppi = source["image"]["effective_source_ppi"]
    check("embedded_crop_source_ppi", min(source_ppi) >= 300.0, source_ppi)
    with Image.open(PNG) as image:
        effective_ppi = image.width / (178.0 / 25.4)
        check("png_effective_ppi", image.width >= 4205 and effective_ppi >= 599.5, {"pixels": image.size, "effective_ppi_at_178mm": effective_ppi})

    reader = PdfReader(str(PDF))
    page = reader.pages[0]
    width_pt = float(page.mediabox.width)
    height_pt = float(page.mediabox.height)
    check("pdf_size_178mm", abs(width_pt / 72 * 25.4 - 178.0) <= 0.2, {"width_mm": width_pt / 72 * 25.4, "height_mm": height_pt / 72 * 25.4})

    app = win32com.client.DispatchEx("PowerPoint.Application")
    deck = None
    try:
        deck = app.Presentations.Open(str(PPTX), WithWindow=False)
        slide = deck.Slides(1)
        font_sizes: list[float] = []
        font_names: list[str] = []
        text_shapes = 0
        for shape in slide.Shapes:
            if shape.HasTextFrame == -1 and shape.TextFrame.HasText == -1:
                text_shapes += 1
                font_sizes.append(float(shape.TextFrame.TextRange.Font.Size))
                font_names.append(str(shape.TextFrame.TextRange.Font.Name))
        width_mm = float(deck.PageSetup.SlideWidth) / 72 * 25.4
        height_mm = float(deck.PageSetup.SlideHeight) / 72 * 25.4
        check("pptx_size_178mm", abs(width_mm - 178.0) <= 0.2, {"width_mm": width_mm, "height_mm": height_mm})
        check("pptx_min_font_7pt", min(font_sizes) >= 7.0, {"minimum_pt": min(font_sizes), "maximum_pt": max(font_sizes), "text_shapes": text_shapes})
        non_arial = sorted({name for name in font_names if name.lower() != "arial"})
        check("pptx_arial_only", not non_arial, non_arial)
    finally:
        if deck is not None:
            deck.Close()
        app.Quit()

    check("no_legacy_qualitative_asset", "qualitative_case_pack_v1" not in SOURCE.read_text(encoding="utf-8"), "legacy qualitative pack excluded")
    check("source_role_non_numeric", "no aggregate estimate" in source["role"], source["role"])
    failures = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "figure": "Fig1",
        "checks": checks,
        "failures": len(failures),
        "scientific_results_recomputed": False,
    }
    (FIG / "qa/Fig1.validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
