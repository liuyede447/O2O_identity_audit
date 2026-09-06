import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = process.argv[2];
if (!OUT_DIR) throw new Error("usage: node build_fig1_lowfi.mjs <output-dir>");

const C = {
  white: "#FFFFFF",
  ink: "#222222",
  muted: "#7F8C8D",
  guide: "#D9DEE3",
  panel: "#F8FAFB",
  o2o: "#1F4E79",
  o2m: "#E67E22",
  paired: "#8E5AA7",
  eligibility: "#009E73",
  within: "#56B4E9",
  topk: "#E69F00",
  conflict: "#A569BD",
  other: "#95A5A6",
};

const presentation = Presentation.create({ slideSize: { width: 1280, height: 676 } });
const slide = presentation.slides.add();
slide.background.fill = C.white;

function shape(name, geometry, left, top, width, height, fill = C.white, stroke = C.guide, strokeWidth = 1) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: stroke, width: strokeWidth },
  });
}

function text(name, value, left, top, width, height, options = {}) {
  const item = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  item.text = value;
  item.text.style = {
    typeface: "Arial",
    fontSize: options.fontSize ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.alignment ?? "left",
    anchor: options.anchor ?? 2,
    wrap: "square",
    autoFit: "shrinkText",
    insets: options.insets ?? { left: 3, right: 3, top: 2, bottom: 2 },
  };
  return item;
}

function pill(name, value, left, top, width, height, stroke, fill = C.white, fontSize = 17) {
  const item = shape(name, "roundRect", left, top, width, height, fill, stroke, 1.3);
  item.text = value;
  item.text.style = {
    typeface: "Arial",
    fontSize,
    bold: false,
    color: C.ink,
    alignment: "center",
    anchor: 2,
    wrap: "square",
    autoFit: "shrinkText",
    insets: { left: 5, right: 5, top: 2, bottom: 2 },
  };
  return item;
}

function arrow(name, left, top, width, height, color = C.muted) {
  return shape(name, "rightArrow", left, top, width, height, color, color, 0.8);
}

function line(name, left, top, width, height, color = C.muted, dashed = false, strokeWidth = 1.4) {
  const normalizedLeft = width < 0 ? left + width : left;
  const normalizedTop = height < 0 ? top + height : top;
  return slide.shapes.add({
    geometry: "line",
    name,
    position: {
      left: normalizedLeft,
      top: normalizedTop,
      width: Math.abs(width),
      height: Math.abs(height),
    },
    fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width: strokeWidth },
  });
}

const panels = {
  a: { left: 20, top: 18, width: 610, height: 310 },
  b: { left: 650, top: 18, width: 610, height: 310 },
  c: { left: 20, top: 348, width: 610, height: 310 },
  d: { left: 650, top: 348, width: 610, height: 310 },
};

// Panel surfaces and all connectors/arrows are created before visible nodes.
for (const [label, p] of Object.entries(panels)) {
  shape(`panel-${label}`, "rect", p.left, p.top, p.width, p.height, C.white, C.guide, 1);
}

// a: displacement arrows and contract flow.
arrow("a-base-to-replay", 250, 161, 58, 13, C.muted);
line("a-left-ray", 350, 180, -38, 0, C.eligibility, true, 1.4);
line("a-right-ray", 382, 180, 38, 0, C.eligibility, true, 1.4);
line("a-up-ray", 366, 164, 0, -36, C.eligibility, true, 1.4);
line("a-down-ray", 366, 196, 0, 36, C.eligibility, true, 1.4);

// b: lane arrows behind nodes.
for (const y of [126, 234]) {
  for (const x of [810, 910, 1010, 1112]) arrow(`lane-${y}-${x}`, x, y, 13, 10, C.guide);
}

// c: cardinal rays and state evolution.
line("c-ray-right", 220, 510, 300, 0, C.ink, false, 1.7);
line("c-ray-left", 220, 510, -95, 0, C.other, false, 1.2);
line("c-ray-up", 220, 510, 0, -100, C.other, false, 1.2);
line("c-ray-down", 220, 510, 0, 92, C.other, false, 1.2);
line("c-scan-horizon", 520, 495, 0, 30, C.muted, true, 1.0);

// d: a visual reading order from qualification to readouts.
arrow("d-qualification-to-readouts", 914, 501, 54, 12, C.guide);

// Panel headings.
for (const [label, titleValue] of Object.entries({
  a: "Controlled focal-GT replay",
  b: "Native assignment paths",
  c: "Grid-resolved cardinal boundary",
  d: "Qualification and readouts",
})) {
  const p = panels[label];
  text(`${label}-label`, label, p.left + 12, p.top + 8, 26, 26, { fontSize: 21, bold: true });
  text(`${label}-title`, titleValue, p.left + 42, p.top + 8, p.width - 58, 27, { fontSize: 20, bold: true });
  line(`${label}-title-rule`, p.left + 12, p.top + 40, p.width - 24, 0, C.guide, false, 1.0);
}

// Panel a nodes.
const imagePlane = shape("a-frozen-image-plane", "rect", 52, 91, 170, 174, C.panel, C.guide, 1.1);
text("a-plane-label", "Frozen image + candidate lattice", 64, 96, 146, 34, { fontSize: 17, bold: true, alignment: "center" });
for (let row = 0; row < 3; row += 1) {
  for (let col = 0; col < 4; col += 1) {
    const dot = shape(`a-grid-${row}-${col}`, "ellipse", 78 + col * 34, 148 + row * 34, 8, 8, C.muted, C.muted, 0.5);
    dot.text = "";
  }
}
shape("a-nonfocal-gt", "rect", 80, 161, 42, 34, "none", C.other, 1.0);
shape("a-focal-base", "rect", 139, 181, 44, 38, "none", C.eligibility, 2.0);
text("a-base-equation", "g = (x, y, w, h)", 220, 103, 124, 32, { fontSize: 18, bold: true, alignment: "center" });
shape("a-base-box", "rect", 270, 154, 62, 52, "none", C.eligibility, 2.1);
text("a-replay-equation", "gᵣ = (x + Δx, y + Δy, w, h)", 302, 73, 210, 34, { fontSize: 17, bold: true, alignment: "center" });
shape("a-replay-centre", "ellipse", 355, 169, 22, 22, C.white, C.eligibility, 1.6);
for (const [name, value, x, y] of [
  ["left", "L", 286, 166], ["right", "R", 421, 166], ["up", "U", 355, 110], ["down", "D", 355, 231],
]) text(`a-${name}`, value, x, y, 24, 24, { fontSize: 16, bold: true, alignment: "center" });
text("a-contracts", "Fixed: δ = 1 input pixel\nEquivalent-area-side:\nδ = κ√(wh)\nAxis-normalized: sensitivity only", 452, 111, 150, 114, { fontSize: 16 });
text("a-frozen-list", "Frozen: pixels · feature tensors · parameters · decoded boxes · lattice · non-focal GTs", 48, 275, 544, 24, { fontSize: 16, alignment: "center", color: C.muted });
text("a-only-focal", "Only the focal GT centre moves.", 175, 301, 300, 22, { fontSize: 17, bold: true, alignment: "center" });

// Panel b nodes.
text("b-o2o-lane", "O2O", 666, 94, 60, 28, { fontSize: 19, bold: true, color: C.ink, alignment: "center" });
text("b-o2m-lane", "O2M", 666, 202, 60, 28, { fontSize: 19, bold: true, color: C.ink, alignment: "center" });
const stageXs = [720, 820, 920, 1020, 1125];
const topStages = ["Eligibility", "q ranking", "Native\nTop-7", "Multi-GT\nconflict", "Assigned O2O\nidentity"];
const bottomStages = ["Eligibility", "q ranking", "Native\nTop-10", "Assigned-\npositive set", "Assigned-set\nTop-1 rank"];
for (let i = 0; i < stageXs.length; i += 1) {
  const width = i === 4 ? 110 : (i === 3 ? 90 : 82);
  pill(`b-o2o-${i}`, topStages[i], stageXs[i], 101, width, 58, C.o2o, i === 4 ? "#EAF1F7" : C.white, 16);
  pill(`b-o2m-${i}`, bottomStages[i], stageXs[i], 209, width, 58, C.o2m, i >= 3 ? "#FFF3E8" : C.white, 16);
}
text("b-candidates-top", "A  B  C  D", 734, 166, 145, 22, { fontSize: 16, color: C.muted, alignment: "center" });
text("b-candidates-bottom", "A  B  C  D", 734, 274, 145, 22, { fontSize: 16, color: C.muted, alignment: "center" });
text("b-o2m-note", "Rank-state summary ≠ uniquely supervised O2M positive", 918, 274, 316, 24, { fontSize: 16, bold: true, alignment: "center" });

// Panel c nodes.
shape("c-base", "ellipse", 194, 484, 52, 52, C.o2o, C.o2o, 1.0);
text("c-base-text", "A", 203, 491, 34, 34, { fontSize: 19, bold: true, color: C.white, alignment: "center" });
shape("c-boundary", "diamond", 342, 493, 34, 34, C.white, C.paired, 2.0);
text("c-boundary-label", "first observed\nboundary", 315, 450, 90, 38, { fontSize: 16, bold: true, alignment: "center" });
shape("c-state-b", "ellipse", 402, 484, 52, 52, C.o2m, C.o2m, 1.0);
text("c-state-b-text", "B", 411, 491, 34, 34, { fontSize: 19, bold: true, color: C.white, alignment: "center" });
shape("c-return-a", "ellipse", 468, 484, 52, 52, C.white, C.o2o, 2.0);
text("c-return-a-text", "A", 477, 491, 34, 34, { fontSize: 19, bold: true, color: C.ink, alignment: "center" });
text("c-stable", "stable segment", 251, 519, 94, 24, { fontSize: 16, alignment: "center" });
text("c-return", "optional return", 424, 546, 108, 24, { fontSize: 16, alignment: "center" });
text("c-right-censor", "no event → right-censored", 242, 397, 210, 28, { fontSize: 17, bold: true, alignment: "center" });
text("c-directions", "L", 97, 498, 26, 24, { fontSize: 16, bold: true, alignment: "center" });
text("c-up", "U", 208, 380, 26, 24, { fontSize: 16, bold: true, alignment: "center" });
text("c-down", "D", 208, 610, 26, 24, { fontSize: 16, bold: true, alignment: "center" });
text("c-horizon", "scan horizon", 514, 526, 86, 22, { fontSize: 16, color: C.muted });
text("c-footer", "Cardinal rays only · not a 2-D stability radius", 148, 627, 360, 24, { fontSize: 17, bold: true, alignment: "center" });

// Panel d nodes.
text("d-qual-title", "Instrument qualification", 675, 397, 238, 26, { fontSize: 18, bold: true });
const quals = [
  ["6/6", "Analytical oracles", C.eligibility],
  ["7/7", "Property invariants", C.within],
  ["6/6", "Mutations detected", C.topk],
  ["0", "Native/reference mismatch", C.conflict],
  ["0", "Focal-row/native mismatch", C.paired],
  ["PASS", "Grid-resolution gate", C.other],
];
for (let i = 0; i < quals.length; i += 1) {
  const y = 430 + i * 31;
  shape(`d-qual-dot-${i}`, "ellipse", 680, y + 5, 13, 13, quals[i][2], quals[i][2], 0.5);
  text(`d-qual-count-${i}`, quals[i][0], 700, y, 52, 24, { fontSize: 16, bold: true, alignment: "center" });
  text(`d-qual-label-${i}`, quals[i][1], 755, y, 162, 24, { fontSize: 16 });
}
text("d-read-title", "Core readouts", 978, 397, 230, 26, { fontSize: 18, bold: true });
const readouts = [
  ["Endpoint fragility", C.o2o],
  ["First-observed pathway", C.eligibility],
  ["Cardinal boundary distance", C.paired],
  ["O2M Rank-Set persistence", C.o2m],
];
for (let i = 0; i < readouts.length; i += 1) {
  const y = 438 + i * 39;
  shape(`d-readout-${i}`, "rect", 980, y, 16, 16, readouts[i][1], readouts[i][1], 0.5);
  text(`d-readout-label-${i}`, readouts[i][0], 1004, y - 4, 218, 26, { fontSize: 17 });
}
text("d-margin", "Secondary: conditional active-runner rank separation", 978, 591, 242, 32, { fontSize: 15, color: C.muted, alignment: "center" });
text("d-footer", "Read-only offline diagnostic · not used during training or inference", 704, 626, 520, 24, { fontSize: 17, bold: true, alignment: "center" });

slide.speakerNotes.textFrame.setText(
  "[Sources]\n" +
  "- manuscript/content/methods.tex (current frozen measurement definitions)\n" +
  "- final_evidence_v3: EV-INSTR-ORACLE-V1, EV-INSTR-MUTATION-V1, EV-INSTR-PARITY-5251-V1, EV-BOUNDARY-GRID-CONVERGENCE-V1\n" +
  "- results/measurement_validation_20260902/focal_row_phase1_validation_v1/summary.json\n" +
  "Low-fidelity schematic only. No real image, candidate ID, or scientific estimate is shown."
);

await fs.mkdir(OUT_DIR, { recursive: true });
const preview = await presentation.export({ slide, format: "png", scale: 2 });
await fs.writeFile(path.join(OUT_DIR, "Fig1_lowfi.png"), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(path.join(OUT_DIR, "Fig1_lowfi.layout.json"), await layout.text());
const inspect = await presentation.inspect({ kind: "slide,textbox,shape", maxChars: 60000 });
await fs.writeFile(path.join(OUT_DIR, "Fig1_lowfi.inspect.ndjson"), inspect.ndjson);
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(path.join(OUT_DIR, "Fig1_lowfi.pptx"));
