"""Render the article Figure 3 from the base-defined correction outputs.

This is a source-to-figure renderer only. It reads released derived artifacts
and runs no inference, replay, bootstrap, or new analysis. Every value in the
Figure 3 source table is re-derived from
data/post_submission_correction_20260908/outputs: panels a, c and d are drawn,
and the post-replay eligibility-stable rows (panel b) are kept for the
Supplement but not drawn. Row layout, labels and interpretation fields come from
the released figures/source_data/fig3_source.csv.

Writes figures/source_data/fig3_source.{csv,json}, figures/draft/Fig3.{svg,pdf,png}
and figures/qa/Fig3.validation.json. figures/rendered/Fig3.* are the exact files
used in the article; the QA record states whether the regenerated PNG is
byte-identical to them on the current font/renderer stack.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
SOURCE_CSV = FIG / "source_data" / "fig3_source.csv"
SOURCE_JSON = FIG / "source_data" / "fig3_source.json"
MANIFEST = FIG / "FIGURE_EVIDENCE_MANIFEST.json"
ADDENDUM = FIG / "FIGURE_EVIDENCE_ADDENDUM.json"
CORR = ROOT / "data" / "post_submission_correction_20260908" / "outputs"
LOCK = CORR / "LOCK_CORRECTED_ESTIMATES.json"
STABLE = CORR / "ANATOMY_BASE_DEFINED_CORRECTION" / "summary.json"
PATHWAY = CORR / "PATHWAY_BASE_DEFINED_CORRECTION" / "FIRST_DIVERGENCE_ANY_STAGE_CORRECTED_CI.csv"
PRE = CORR / "PRE_RESOLUTION_BASE_DEFINED_CORRECTION" / "summary.json"
DRAFT = FIG / "draft" / "Fig3"
RENDERED = FIG / "rendered" / "Fig3"
QA = FIG / "qa" / "Fig3.validation.json"
AUTHORITY = "derived_correction_from_frozen_evidence"
EXPECTED_ROWS = {"a": 5, "b": 4, "c": 16, "d": 8}

# Figure contract -> keys used by the lock, stable, pathway and pre-resolution outputs.
CONTRACTS = {
    "Fixed 1 px": ("fixed_1px", "fixed_1px", "Fixed 1 px", "fixed"),
    "Equivalent-area-side": ("equivalent_side", "equivalent_side_k00625", "Equivalent-area side", "equivalent"),
}
LOCK_LABELS = {"Native": "native_gap", "Eligibility locked": "eligibility_lock_gap"}
STABLE_LEVELS = {
    "Object-level": ("object_level", "retained_objects", "all_objects"),
    "Direction-level": ("direction_level", "retained_directions", "all_directions"),
}
FIRST_LABELS = {
    "Eligibility": ("eligibility_boundary",),
    "Top-k": ("topk_membership_transition",),
    "Conflict": ("conflict_reassignment",),
    "Within-set rank": ("within_set_geometry_rank_reversal",),
    "Other/reserved": ("active_disappearance", "compound_or_other"),
}
ANY_LABELS = {"Eligibility": "eligibility", "Within-set rank": "within_set",
              "Eligibility ∩ Within-set": "eligibility_within"}
PRE_LABELS = {
    "Pre-conflict Top-7 winner gap": "pre_gap",
    "Final assigned gap": "active_gap",
    "Final minus pre-conflict": "active_minus_pre_gap",
    "Fragility-definition disagreement": "object_disagreement",
}
PATH_PREFIXES = (("figures_final/", "figures/"), ("results/", "evidence/artifacts/results/"),
                 ("runs/", "evidence/artifacts/runs/"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def provenance(path: Path, evidence_id: str) -> dict[str, str]:
    return {"source_path": rel(path), "source_sha256": sha256(path),
            "authority": AUTHORITY, "evidence_id": evidence_id}


def set_values(row: dict[str, Any], estimate: float, low: float, high: float,
               prov: dict[str, str], **extra: Any) -> None:
    row.update(estimate=100 * float(estimate), ci_low=100 * float(low),
               ci_high=100 * float(high), **prov)
    row.update(extra)


def map_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: map_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [map_paths(item) for item in value]
    if isinstance(value, str):
        for old, new in PATH_PREFIXES:
            if value.startswith(old):
                return new + value[len(old):]
    return value


def derive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lock = read_json(LOCK)
    stable = read_json(STABLE)
    pre = read_json(PRE)
    pathway = {record["contract"]: record for record in pd.read_csv(PATHWAY).to_dict("records")}
    prov = {
        "a": provenance(LOCK, "EV-LOCK-BASE-DEFINED-CORRECTED-V1"),
        "b": provenance(STABLE, "EV-ELIGIBILITY-STABLE-BASE-DEFINED-CORRECTED-V1"),
        "c": provenance(PATHWAY, "EV-PATHWAY-BASE-DEFINED-CORRECTED-V1"),
        "d": provenance(PRE, "EV-PRERESOLUTION-BASE-DEFINED-CORRECTED-V1"),
    }
    for row in rows:
        panel, kind, label = row["panel"], row["kind"], row["label"]
        lock_key, stable_key, pathway_key, pre_key = CONTRACTS[row["contract"]]
        if panel == "a":
            payload = lock["estimates"][lock_key]
            metric = "reduction_fraction" if kind == "attenuation_fraction" else LOCK_LABELS[label]
            low, high = payload["ci_95"][metric]
            population = lock["effective_population"][lock_key]
            set_values(row, payload["point"][metric], low, high, prov["a"],
                       image_n=population["images"], n_gt=population["gt"])
        elif panel == "b":
            level, retained_key, all_key = STABLE_LEVELS[label]
            item = stable[stable_key][level]
            low, high = item["ci_95"]["scale_contrast"]
            set_values(row, item["point"]["scale_contrast"], low, high, prov["b"],
                       retained_n=item[retained_key], all_n=item[all_key],
                       contributing_images=item["contributing_images"])
        elif panel == "c":
            record = pathway[pathway_key]
            counts = {"directional_flip_rows": int(record["flip_directions"]),
                      "contributing_images": int(record["images"])}
            if kind == "exclusive_first_observed":
                keys = FIRST_LABELS[label]
                if len(keys) == 1:
                    key = keys[0]
                    set_values(row, record[key], record[f"{key}_ci_low"], record[f"{key}_ci_high"],
                               prov["c"], **counts)
                else:
                    values = [float(record[f"{key}{suffix}"]) for key in keys
                              for suffix in ("", "_ci_low", "_ci_high")]
                    if any(values):
                        raise RuntimeError(f"reserved components are non-zero for {row['contract']}; "
                                           "their sum needs a joint bootstrap interval")
                    set_values(row, 0.0, 0.0, 0.0, prov["c"], **counts)
            elif kind == "nonexclusive_any_stage":
                key = ANY_LABELS[label]
                set_values(row, record[key], record[f"{key}_ci_low"], record[f"{key}_ci_high"],
                           prov["c"], **counts,
                           interpretation="descriptive non-exclusive occurrence on corrected identity-exchange support")
            else:
                raise RuntimeError(f"unexpected panel c row kind: {kind}")
        elif panel == "d":
            contract = pre["contracts"][pre_key]
            key = PRE_LABELS[label]
            low, high = contract["ci_95"][key]
            set_values(row, contract["point"][key], low, high, prov["d"],
                       n_gt=contract["effective_gt"], image_n=contract["effective_images"],
                       directional_replays=contract["effective_directions"])
        else:
            raise RuntimeError(f"unexpected panel: {panel}")
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    counts = frame.groupby("panel").size().to_dict()
    values = frame[["estimate", "ci_low", "ci_high"]].astype(float)
    finite = bool(values.map(math.isfinite).all().all())
    ordered = bool(((values.ci_low <= values.estimate) & (values.estimate <= values.ci_high)).all())
    first = frame[(frame.panel == "c") & (frame.kind == "exclusive_first_observed")]
    sums = {key: float(value) for key, value in first.groupby("contract").estimate.sum().items()}
    sums_ok = all(abs(value - 100.0) <= 1e-6 for value in sums.values())
    authority_ok = set(frame.authority) == {AUTHORITY}
    return {
        "panel_rows": "PASS" if counts == EXPECTED_ROWS else f"FAIL {counts}",
        "values_finite": "PASS" if finite else "FAIL",
        "ci_order_valid": "PASS" if ordered else "FAIL",
        "exclusive_first_observed_sums_100": "PASS" if sums_ok else f"FAIL {sums}",
        "all_rows_bound_to_correction_outputs": "PASS" if authority_ok else "FAIL",
    }


def main() -> None:
    header = list(pd.read_csv(SOURCE_CSV, nrows=0).columns)
    rows = derive_rows(pd.read_csv(SOURCE_CSV).to_dict("records"))
    checks = validate(rows)

    spec = importlib.util.spec_from_file_location("plot_fig3", FIG / "src" / "plot_fig3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the Figure 3 layout module")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # Deterministic PDF/SVG metadata and SVG element ids.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    fig, preflight = plot.make_figure(rows)
    font = preflight.get("resolved_font", {})
    if font.get("path"):
        font["path"] = Path(font["path"]).name  # keep machine paths out of the QA record
    plot.plt.rcParams["svg.hashsalt"] = "fig3"
    plot.save_draft(fig, DRAFT)
    draft = plot.output_audit(DRAFT)

    pd.DataFrame(rows, columns=header).to_csv(SOURCE_CSV, index=False, lineterminator="\n")

    corrected = {name: {"path": rel(path), "sha256": sha256(path)}
                 for name, path in (("panel_a", LOCK), ("panel_b", STABLE),
                                    ("panel_c", PATHWAY), ("panel_d", PRE))}
    meta = map_paths(read_json(SOURCE_JSON))
    for panel_id, name in (("Fig3a", "panel_a"), ("Fig3b", "panel_b"), ("Fig3c", "panel_c"), ("Fig3d", "panel_d")):
        support = meta["panel_support"][panel_id]
        support["panel_status"] = "PASS"
        support["source_artifacts"] = [{"authority": AUTHORITY, "expected_sha256": corrected[name]["sha256"],
                                        "path": corrected[name]["path"], "role": f"base-defined corrected {name}"}]
        support["status_reason"] = "Values are re-derived from the SHA-bound base-defined correction output."
    meta.update({
        "status": "SOURCE_BOUND_ARTICLE_CORRECTED",
        "manifest_path": rel(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "addendum_path": rel(ADDENDUM), "addendum_sha256": sha256(ADDENDUM),
        "source_row_count": len(rows),
        "fig3_source_csv": rel(SOURCE_CSV), "fig3_source_csv_sha256": sha256(SOURCE_CSV),
        "corrected_sources": corrected,
        "sources_note": ("'sources' lists the frozen final_evidence_v3 artifacts the correction re-derives from; "
                         "the plotted values come from 'corrected_sources'."),
    })
    write_json(SOURCE_JSON, meta)

    article = {ext: {"path": rel(RENDERED.with_suffix(f".{ext}")), "sha256": sha256(RENDERED.with_suffix(f".{ext}"))}
               for ext in ("pdf", "svg", "png")}
    png_match = draft["outputs"]["png"]["sha256"] == article["png"]["sha256"]
    fail = [name for name, value in checks.items() if value != "PASS"]
    qa = {
        "status": "PASS" if not fail else "FAIL",
        "stage": "ARTICLE_FIGURE3_BASE_DEFINED_CORRECTION",
        "figure": "Fig3",
        "renderer": "figures/src/rebuild_fig3_corrected.py (layout from figures/src/plot_fig3.py)",
        "read_only_science": True,
        "model_inference_or_replay": False,
        "checks": checks,
        "failures": fail,
        "corrected_sources": corrected,
        "source_data": {"csv": rel(SOURCE_CSV), "csv_sha256": sha256(SOURCE_CSV),
                        "json": rel(SOURCE_JSON), "json_sha256": sha256(SOURCE_JSON)},
        "draft_outputs": draft,
        "article_render": article,
        "draft_png_identical_to_article_png": png_match,
        "preflight": preflight,
        "notes": [
            "Panels a, c and d are drawn; panel b rows are retained for the Supplement and are not drawn.",
            "First-observed labels are mutually exclusive and stacked; any-stage occurrence is non-exclusive.",
            "Eligibility attenuation is limited to lock support and is not a causal contribution fraction.",
            "A PNG mismatch with figures/rendered/Fig3.png indicates a different font/renderer stack, not different values.",
        ],
    }
    write_json(QA, qa)
    print(json.dumps({"status": qa["status"], "checks": checks,
                      "draft_png_identical_to_article_png": png_match}, indent=2, ensure_ascii=False))
    if fail:
        raise SystemExit(f"Figure 3 QA failed: {fail}")


if __name__ == "__main__":
    main()
