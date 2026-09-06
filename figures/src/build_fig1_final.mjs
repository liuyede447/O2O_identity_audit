import fs from "node:fs/promises";
import crypto from "node:crypto";
import sharp from "sharp";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "E:/two_paper/beifen/publication_reproducible";
const OUT = `${ROOT}/figures_final`;
const IMAGE = `${OUT}/source_data/fig1_focal_audit_crop_500x375.png`;
const ORIGINAL_IMAGE = `${OUT}/source_data/figs1_primary_trace/0000170_00401_d_0000001__160_0.png`;
const TRACE = `${OUT}/source_data/figs1_primary_trace/primary_qualitative_trace.json`;
const FIXED_FORMULA = `${OUT}/source_data/fig1_fixed_formula.svg`;
const EQ_FORMULA = `${OUT}/source_data/fig1_equivalent_area_formula.svg`;
const RHO_FORMULA = `${OUT}/source_data/fig1_rho_card_formula.svg`;
const INSET_IMAGE = `${OUT}/source_data/fig1_focal_gt_inset_nearest.png`;
const SCALE = 672.76 / 1280;
const FONT_SCALE = 4 / 3;
const SLIDE_HEIGHT = 680 * SCALE;

const C = {
  white: "#FFFFFF", ink: "#111111", muted: "#70797D", guide: "#D3D7DA", pale: "#F5F7F8",
  o2o: "#1F4E79", o2m: "#E67E22", focal: "#007F5F", change: "#B03A2E", neutral: "#929B9F",
};
const X = value => value * SCALE;
const P = (left, top, width, height) => ({ left: X(left), top: X(top), width: X(width), height: X(height) });

function sha256(buffer) { return crypto.createHash("sha256").update(buffer).digest("hex"); }

function shape(slide, name, geometry, left, top, width, height, fill = C.white, stroke = C.guide, strokeWidth = 0.7, dashed = false) {
  return slide.shapes.add({
    geometry, name, position: P(left, top, width, height), fill,
    line: { style: dashed ? "dashed" : "solid", fill: stroke, width: strokeWidth },
  });
}

function text(slide, name, value, left, top, width, height, options = {}) {
  const item = slide.shapes.add({ geometry: "textbox", name, position: P(left, top, width, height), fill: "none", line: { fill: "none", width: 0 } });
  item.text = value;
  item.text.style = {
    typeface: "Arial", fontSize: (options.fontSize ?? 7.0) * FONT_SCALE,
    bold: options.bold ?? false, color: options.color ?? C.ink,
    alignment: options.alignment ?? "left", anchor: options.anchor ?? 2,
    wrap: "square", insets: { left: X(1), right: X(1), top: 0, bottom: 0 },
  };
  return item;
}

function line(slide, name, x1, y1, x2, y2, color = C.ink, width = 0.7, dashed = false) {
  return slide.shapes.add({
    geometry: "line", name,
    position: { left: X(Math.min(x1, x2)), top: X(Math.min(y1, y2)), width: X(Math.abs(x2 - x1)), height: X(Math.abs(y2 - y1)), horizontalFlip: x2 < x1, verticalFlip: y2 < y1 },
    fill: "none", line: { style: dashed ? "dashed" : "solid", fill: color, width },
  });
}

function arrow(slide, name, geometry, left, top, width, height, color) {
  return shape(slide, name, geometry, left, top, width, height, color, color, 0.35);
}

function downArrow(slide, name, cx, y1, y2, color) {
  line(slide, `${name}-line`, cx, y1, cx, y2 - 7, color, 0.75);
  shape(slide, `${name}-head`, "triangle", cx - 4, y2 - 8, 8, 8, color, color, 0.3);
}

function panelHeading(slide, letter, titleValue, x, width) {
  text(slide, `${letter}-label`, letter, x, 15, 20, 19, { fontSize: 8.0, bold: true });
  text(slide, `${letter}-title`, titleValue, x + 25, 15, width - 25, 19, { fontSize: 7.6, bold: true });
}

function candidate(slide, name, label, cx, cy, stroke, fill = C.white, fontColor = C.ink, radius = 13) {
  const item = shape(slide, name, "ellipse", cx - radius, cy - radius, 2 * radius, 2 * radius, fill, stroke, 0.75);
  item.text = label;
  item.text.style = { typeface: "Arial", fontSize: 7.0 * FONT_SCALE, color: fontColor, alignment: "center", anchor: 2, wrap: "square", insets: { left: 0, right: 0, top: 0, bottom: 0 } };
  return item;
}

function stage(slide, name, label, x, y, width, stroke, fill = C.white, bold = false) {
  const item = shape(slide, name, "rect", x, y, width, 39, fill, stroke, 0.65);
  item.text = label;
  item.text.style = { typeface: "Arial", fontSize: 7.0 * FONT_SCALE, bold, color: C.ink, alignment: "center", anchor: 2, wrap: "square", insets: { left: X(3), right: X(3), top: X(2), bottom: X(2) } };
  return item;
}

function stressScalePair(slide, name, x, y, scaleProportional) {
  text(slide, `${name}-tiny-label`, "tiny", x, y, 34, 14, { fontSize: 7.0, color: C.muted, alignment: "center" });
  text(slide, `${name}-small-label`, "small", x + 48, y, 42, 14, { fontSize: 7.0, color: C.muted, alignment: "center" });
  shape(slide, `${name}-tiny-object`, "rect", x + 9, y + 23, 8, 8, C.pale, C.neutral, 0.55);
  shape(slide, `${name}-small-object`, "rect", x + 57, y + 18, 18, 18, C.pale, C.neutral, 0.55);
  arrow(slide, `${name}-tiny-shift`, "rightArrow", x + 20, y + 23, scaleProportional ? 9 : 14, 8, C.change);
  arrow(slide, `${name}-small-shift`, "rightArrow", x + 78, y + 23, scaleProportional ? 22 : 14, 8, C.change);
}

async function main() {
  const imageBytes = await fs.readFile(IMAGE);
  const originalBytes = await fs.readFile(ORIGINAL_IMAGE);
  const traceBytes = await fs.readFile(TRACE);
  const fixedFormulaBytes = await fs.readFile(FIXED_FORMULA);
  const eqFormulaBytes = await fs.readFile(EQ_FORMULA);
  const rhoFormulaBytes = await fs.readFile(RHO_FORMULA);
  const insetCrop = { left: 383, top: 316, width: 32, height: 42 };
  const insetBytes = await sharp(imageBytes)
    .extract(insetCrop)
    .resize(320, 420, { kernel: "nearest" })
    .png()
    .toBuffer();
  const trace = JSON.parse(traceBytes.toString("utf8"));
  const presentation = Presentation.create({ slideSize: { width: 672.76, height: SLIDE_HEIGHT } });
  const slide = presentation.slides.add();
  slide.background.fill = C.white;

  line(slide, "separator-a-b", 425, 20, 425, 660, C.guide, 0.5);
  line(slide, "separator-b-c", 955, 20, 955, 660, C.guide, 0.5);
  panelHeading(slide, "a", "Controlled focal-GT replay", 15, 395);
  panelHeading(slide, "b", "Native assignment-state extraction", 445, 490);
  panelHeading(slide, "c", "State comparison", 975, 290);

  // a. Dense real-image context at >300 source ppi.
  const imagePos = { left: 55, top: 65, width: 300, height: 225 };
  slide.images.add({ blob: imageBytes, contentType: "image/png", alt: "Dense AI-TOD-v2 crop with exact GT boxes", fit: "fill", position: P(imagePos.left, imagePos.top, imagePos.width, imagePos.height) });
  shape(slide, "a-image-frame", "rect", imagePos.left, imagePos.top, imagePos.width, imagePos.height, "none", C.guide, 0.45);
  const crop = { x: 120, y: 165, width: 500, height: 375 };
  const gtRecords = trace.records.map(record => ({ id: record.gt_id, box: record.gt_xyxy }));
  const mapBox = box => ({
    left: imagePos.left + ((box[0] - crop.x) / crop.width) * imagePos.width,
    top: imagePos.top + ((box[1] - crop.y) / crop.height) * imagePos.height,
    width: ((box[2] - box[0]) / crop.width) * imagePos.width,
    height: ((box[3] - box[1]) / crop.height) * imagePos.height,
  });
  for (const record of gtRecords) {
    const b = mapBox(record.box);
    const focal = record.id === 28;
    shape(slide, `a-gt-${record.id}`, "rect", b.left, b.top, b.width, b.height, "none", focal ? C.focal : C.neutral, focal ? 1.0 : 0.55);
  }
  const focalBox = mapBox(gtRecords.find(row => row.id === 28).box);

  // True-pixel magnified inset for the core one-pixel manipulation.
  // The raster crop is enlarged only by nearest-neighbour pixel replication;
  // both GT rectangles and the direction cue remain editable vectors.
  const insetPos = { left: 272, top: 73, width: 70, height: 92 };
  line(slide, "a-inset-leader-left", focalBox.left, focalBox.top, insetPos.left, insetPos.top + insetPos.height, C.guide, 0.55, true);
  line(slide, "a-inset-leader-right", focalBox.left + focalBox.width, focalBox.top, insetPos.left + insetPos.width, insetPos.top + insetPos.height, C.guide, 0.55, true);
  slide.images.add({ blob: insetBytes, contentType: "image/png", alt: "Nearest-neighbour magnified focal ground truth", fit: "fill", position: P(insetPos.left, insetPos.top, insetPos.width, insetPos.height) });
  shape(slide, "a-inset-frame", "rect", insetPos.left, insetPos.top, insetPos.width, insetPos.height, "none", C.ink, 0.65);
  const focalLocal = [390, 324, 408, 350];
  const mapInsetBox = box => ({
    left: insetPos.left + ((box[0] - insetCrop.left) / insetCrop.width) * insetPos.width,
    top: insetPos.top + ((box[1] - insetCrop.top) / insetCrop.height) * insetPos.height,
    width: ((box[2] - box[0]) / insetCrop.width) * insetPos.width,
    height: ((box[3] - box[1]) / insetCrop.height) * insetPos.height,
  });
  const baseInset = mapInsetBox(focalLocal);
  const shiftedInset = mapInsetBox([focalLocal[0], focalLocal[1] - 1, focalLocal[2], focalLocal[3] - 1]);
  shape(slide, "a-inset-base-gt", "rect", baseInset.left, baseInset.top, baseInset.width, baseInset.height, "none", C.focal, 1.35);
  shape(slide, "a-inset-shifted-gt", "rect", shiftedInset.left, shiftedInset.top, shiftedInset.width, shiftedInset.height, "none", C.change, 1.10, true);
  arrow(slide, "a-inset-up-arrow", "upArrow", 278, 43, 8, 19, C.change);
  text(slide, "a-inset-one-pixel", "1 px", 290, 42, 42, 20, { fontSize: 7.0, bold: true });

  // A single restrained frame organizes the lower explanatory band. The
  // header rule separates the GT key, and thin dividers distinguish replay,
  // fixed displacement, and scale-proportional displacement.
  shape(slide, "a-lower-group", "roundRect", 45, 296, 365, 214, "#FAFBFC", C.guide, 0.55);
  line(slide, "a-lower-header-rule", 58, 330, 397, 330, C.guide, 0.45);
  line(slide, "a-replay-contract-divider", 180, 336, 180, 498, C.guide, 0.45);
  line(slide, "a-contract-divider", 291, 360, 291, 498, C.guide, 0.45);
  shape(slide, "a-focal-key", "rect", 72, 308, 12, 12, "none", C.focal, 0.9);
  text(slide, "a-focal-label", "Focal GT", 90, 303, 78, 21, { fontSize: 7.0 });
  shape(slide, "a-other-key", "rect", 220, 308, 12, 12, "none", C.neutral, 0.6);
  text(slide, "a-other-label", "Non-focal GT", 238, 303, 155, 21, { fontSize: 7.0 });

  // Replay cross and stress contracts occupy a separate lower band. No leader
  // crosses this band: the crop annotation and replay schematic are separated.
  shape(slide, "a-centre", "ellipse", 114, 376, 14, 14, C.focal, C.focal, 0.4);
  arrow(slide, "a-left", "leftArrow", 79, 379, 32, 8, C.change);
  arrow(slide, "a-right", "rightArrow", 131, 379, 32, 8, C.change);
  arrow(slide, "a-up", "upArrow", 117, 352, 8, 22, C.change);
  arrow(slide, "a-down", "downArrow", 117, 393, 8, 22, C.change);
  text(slide, "a-left-label", "L", 62, 373, 18, 18, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "a-right-label", "R", 164, 373, 18, 18, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "a-up-label", "U", 112, 333, 18, 18, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "a-down-label", "D", 112, 417, 18, 18, { fontSize: 7.0, bold: true, alignment: "center" });

  text(slide, "a-contract-title", "Stress contracts", 186, 335, 216, 19, { fontSize: 7.2, bold: true });
  text(slide, "a-fixed", "Fixed", 186, 362, 100, 21, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "a-equivalent", "Eq.-area", 296, 362, 106, 21, { fontSize: 7.0, bold: true, alignment: "center" });
  stressScalePair(slide, "a-fixed-scale", 190, 391, false);
  stressScalePair(slide, "a-eq-scale", 298, 391, true);
  slide.images.add({ blob: fixedFormulaBytes, contentType: "image/svg+xml", alt: "Fixed displacement formula delta equals one pixel", fit: "contain", position: P(198, 456, 77, 25) });
  slide.images.add({ blob: eqFormulaBytes, contentType: "image/svg+xml", alt: "Equivalent-area-side displacement formula", fit: "contain", position: P(303, 448, 94, 55) });
  text(slide, "a-only-focal", "Only the focal GT centre moves.", 48, 524, 320, 21, { fontSize: 7.0, bold: true });
  text(slide, "a-frozen", "Model outputs and non-focal GTs remain frozen.", 48, 553, 342, 42, { fontSize: 7.0 });

  // b. Two vertical native lanes with branch-specific frozen predictions.
  shape(slide, "b-frozen-box", "rect", 555, 55, 290, 58, C.pale, C.neutral, 0.55);
  text(slide, "b-frozen-label", "Branch-specific outputs frozen\nsame focal-GT replay", 572, 68, 256, 32, { fontSize: 7.0, bold: true, alignment: "center", verticalAlignment: "middle" });
  line(slide, "b-source-down", 700, 113, 700, 130, C.neutral, 0.65);
  line(slide, "b-source-split", 570, 130, 830, 130, C.neutral, 0.65);
  downArrow(slide, "b-o2o-entry", 570, 130, 150, C.o2o);
  downArrow(slide, "b-o2m-entry", 830, 130, 150, C.o2m);

  const lane = [470, 730];
  const laneWidth = 200;
  text(slide, "b-o2o-heading", "O2O branch", lane[0], 148, laneWidth, 20, { fontSize: 7.4, bold: true, alignment: "center" });
  text(slide, "b-o2o-sub", "single-assigned state", lane[0], 169, laneWidth, 18, { fontSize: 7.0, alignment: "center" });
  text(slide, "b-o2m-heading", "O2M branch", lane[1], 148, laneWidth, 20, { fontSize: 7.4, bold: true, alignment: "center" });
  text(slide, "b-o2m-sub", "positive-set state", lane[1], 169, laneWidth, 18, { fontSize: 7.0, alignment: "center" });

  const o2oYs = [214, 284, 354, 424];
  const o2mYs = [194, 248, 302, 356, 410];
  const o2oLabels = ["Eligible candidates", "Top-7", "Conflict resolution", "Assigned O2O ID"];
  const o2mLabels = ["Eligible candidates", "Top-10", "Conflict resolution", "Positive set\n{A, B, C}", "Assigned-set Top-1"];
  for (let i = 0; i < o2oYs.length; i += 1) {
    stage(slide, `b-o2o-stage-${i}`, o2oLabels[i], lane[0] + 10, o2oYs[i], 180, C.o2o, i === o2oYs.length - 1 ? "#EEF3F7" : C.white, i === o2oYs.length - 1);
    if (i < o2oYs.length - 1) downArrow(slide, `b-o2o-arrow-${i}`, lane[0] + laneWidth / 2, o2oYs[i] + 39, o2oYs[i + 1], C.o2o);
  }
  for (let i = 0; i < o2mYs.length; i += 1) {
    stage(slide, `b-o2m-stage-${i}`, o2mLabels[i], lane[1] + 10, o2mYs[i], 180, C.o2m, i === o2mYs.length - 1 ? "#FFF3E8" : C.white, i === o2mYs.length - 1);
    if (i < o2mYs.length - 1) downArrow(slide, `b-o2m-arrow-${i}`, lane[1] + laneWidth / 2, o2mYs[i] + 39, o2mYs[i + 1], C.o2m);
  }
  text(slide, "b-o2o-example", "A   B   C   …", lane[0] + 12, 476, 176, 20, { fontSize: 7.0, color: C.muted, alignment: "center" });
  text(slide, "b-o2m-example", "A   B   C   …", lane[1] + 12, 476, 176, 20, { fontSize: 7.0, color: C.muted, alignment: "center" });
  candidate(slide, "b-o2o-example-a", "A", lane[0] + 55, 522, C.o2o, C.o2o, C.white, 12);
  candidate(slide, "b-o2m-example-a", "A", lane[1] + 55, 522, C.o2m, C.o2m, C.white, 12);
  text(slide, "b-o2o-example-label", "assigned", lane[0] + 76, 512, 75, 20, { fontSize: 7.0 });
  text(slide, "b-o2m-example-label", "Top-1 readout", lane[1] + 76, 512, 92, 20, { fontSize: 7.0 });

  // c. State comparison table.
  const baseX = 1018, shiftX = 1104, stateX = 1170;
  text(slide, "c-base-head", "Base", baseX - 25, 69, 50, 19, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "c-shift-head", "Shift", shiftX - 27, 69, 54, 19, { fontSize: 7.0, bold: true, alignment: "center" });
  text(slide, "c-state-head", "Outcome", stateX, 69, 90, 19, { fontSize: 7.0, bold: true });
  line(slide, "c-header-rule", 986, 98, 1256, 98, C.guide, 0.5);
  const rowY = [137, 211, 285];
  candidate(slide, "c-base-stable", "A", baseX, rowY[0], C.o2o);
  candidate(slide, "c-shift-stable", "A", shiftX, rowY[0], C.o2o);
  text(slide, "c-stable", "Stable", stateX, rowY[0] - 10, 90, 20, { fontSize: 7.0, bold: true });
  candidate(slide, "c-base-changed", "A", baseX, rowY[1], C.o2o);
  candidate(slide, "c-shift-changed", "B", shiftX, rowY[1], C.change, C.change, C.white);
  text(slide, "c-changed", "Identity change", stateX, rowY[1] - 10, 95, 20, { fontSize: 7.0, bold: true });
  candidate(slide, "c-base-undefined", "A", baseX, rowY[2], C.o2o);
  text(slide, "c-shift-undefined", "—", shiftX - 15, rowY[2] - 12, 30, 24, { fontSize: 8.0, color: C.neutral, alignment: "center" });
  text(slide, "c-undefined", "Identity disappearance", stateX - 4, rowY[2] - 10, 105, 20, { fontSize: 6.5, bold: true });
  line(slide, "c-row-rule-1", 986, 174, 1256, 174, C.guide, 0.45);
  line(slide, "c-row-rule-2", 986, 248, 1256, 248, C.guide, 0.45);
  line(slide, "c-section-rule", 986, 332, 1256, 332, C.guide, 0.5);
  // d. Cardinal boundary readout, visually separated from the state table.
  text(slide, "d-label", "d", 975, 347, 20, 19, { fontSize: 8.0, bold: true });
  text(slide, "d-title", "Cardinal boundary readout", 1000, 347, 255, 19, { fontSize: 7.6, bold: true });
  line(slide, "d-boundary-ray", 1014, 436, 1236, 436, C.neutral, 0.75);
  candidate(slide, "d-boundary-a", "A", 1038, 436, C.o2o, C.o2o, C.white, 12);
  line(slide, "d-boundary-mark", 1133, 418, 1133, 454, C.change, 1.0);
  candidate(slide, "d-boundary-b", "B", 1212, 436, C.change, C.change, C.white, 12);
  slide.images.add({ blob: rhoFormulaBytes, contentType: "image/svg+xml", alt: "Cardinal boundary distance rho card", fit: "contain", position: P(1107, 457, 52, 24) });
  text(slide, "d-ray-note", "one cardinal ray", 1052, 493, 165, 18, { fontSize: 7.0, color: C.muted, alignment: "center" });
  text(slide, "d-guard", "Cardinal directions only", 990, 552, 260, 20, { fontSize: 7.0, color: C.muted, alignment: "center" });

  slide.speakerNotes.textFrame.setText(
    "[Figure 1 evidence]\n" +
    `- Display crop: ${IMAGE}; SHA-256 ${sha256(imageBytes)}; source extent [120,165,620,540].\n` +
    `- Original source image: ${ORIGINAL_IMAGE}; SHA-256 ${sha256(originalBytes)}.\n` +
    `- Primary trace: ${TRACE}; SHA-256 ${sha256(traceBytes)}; checkpoint ${trace.checkpoint_sha256}.\n` +
    "- Displayed GT rectangles map directly from GT 17, 28 and 40; GT 28 is focal. IDs are omitted from the panel.\n" +
    "- Only the focal GT centre moves. Model outputs, box extent and non-focal GTs remain frozen.\n" +
    "- O2M assigned-set Top-1 is a rank readout, not a uniquely supervised positive. The boundary inset represents one cardinal ray, not the full 2-D plane."
  );

  await fs.mkdir(`${OUT}/draft`, { recursive: true });
  await fs.mkdir(`${OUT}/source_data`, { recursive: true });
  await fs.mkdir(`${OUT}/qa`, { recursive: true });
  await fs.writeFile(INSET_IMAGE, insetBytes);
  const preview = await presentation.export({ slide, format: "png", scale: 3 });
  await fs.writeFile(`${OUT}/draft/Fig1_preview.png`, new Uint8Array(await preview.arrayBuffer()));
  await fs.writeFile(`${OUT}/qa/Fig1.layout.json`, await (await slide.export({ format: "layout" })).text());
  const inspect = await presentation.inspect({ kind: "slide,textbox,shape", maxChars: 120000 });
  await fs.writeFile(`${OUT}/qa/Fig1.inspect.ndjson`, inspect.ndjson);
  await (await PresentationFile.exportPptx(presentation)).save(`${OUT}/draft/Fig1_candidate_v7.pptx`);

  const imageWidthMm = imagePos.width / 1280 * 178;
  const imageHeightMm = imagePos.height / 1280 * 178;
  const source = {
    status: "SOURCE_BOUND",
    role: "three-column measurement schematic with a true-GT aerial crop; no aggregate estimate",
    image: {
      path: IMAGE.replace(`${ROOT}/`, ""), sha256: sha256(imageBytes), original_path: ORIGINAL_IMAGE.replace(`${ROOT}/`, ""), original_sha256: sha256(originalBytes),
      displayed_source_extent: [120, 165, 620, 540], source_pixels: [500, 375], displayed_size_mm: [imageWidthMm, imageHeightMm],
      effective_source_ppi: [500 / (imageWidthMm / 25.4), 375 / (imageHeightMm / 25.4)],
    },
    magnified_inset: {
      path: INSET_IMAGE.replace(`${ROOT}/`, ""), sha256: sha256(insetBytes),
      source_crop_xywh: [insetCrop.left, insetCrop.top, insetCrop.width, insetCrop.height],
      source_pixels: [insetCrop.width, insetCrop.height], replicated_pixels: [320, 420],
      enlargement: "nearest-neighbour pixel replication; no interpolation or enhancement",
      base_gt_xyxy_in_display_crop: focalLocal, shifted_gt_xyxy_in_display_crop: [390, 323, 408, 349],
      displayed_shift: "up by one input pixel",
    },
    trace: { path: TRACE.replace(`${ROOT}/`, ""), sha256: sha256(traceBytes), checkpoint_sha256: trace.checkpoint_sha256 },
    displayed_gt: gtRecords.map(row => ({ gt_id: row.id, xyxy: row.box, focal: row.id === 28 })),
    constraints: ["three-column four-panel narrative", "no overall title", "no backbone", "no dashboard", "black Arial text", "true GT coordinates", "source crop above 300 ppi", "nearest-neighbour focal inset", "one-pixel shift shown explicitly", "fixed versus scale-proportional stress visualised", "four cardinal directions", "no full 2-D radius", "O2M Top-1 is a rank readout"],
  };
  await fs.writeFile(`${OUT}/source_data/fig1_source.json`, JSON.stringify(source, null, 2) + "\n");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
