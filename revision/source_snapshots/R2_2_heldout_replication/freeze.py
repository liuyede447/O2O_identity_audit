"""Write (default) or verify (--verify) FREEZE_RECORD.json: SHA-256 of the plan, scripts and inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PR = Path(r"E:\two_paper\beifen\publication_reproducible")
FILES = [
    HERE / "PREREGISTRATION_DRAFT.md",
    HERE / "selected_images_preregistered.csv",
    HERE / "conversion_manifest.json",
    HERE / "validation_files_sha256.txt",
    HERE / "rehearsal_results.json",
    HERE / "prepare_validation_split.py",
    HERE / "select_sample.py",
    HERE / "predict_row_structure.py",
    HERE / "run_lock_checked.py",
    HERE / "base_gate.py",
    HERE / "analyze_replication.py",
    HERE / "rehearsal.py",
    HERE / "run_s2.sh",
    HERE / "analyze_s2.py",
    HERE / "run_formal.sh",
    HERE / "freeze.py",
    HERE / "check_s2_manifest.py",
    HERE.parent / "R2_1_comparable_measure" / "comparable_measure.py",
    PR / "scripts" / "run_eligibility_lock_counterfactual.py",
    PR / "scripts" / "run_reviewer_killer_controls.py",
    PR / "scripts" / "analyze_o2m_rank_set_decoupling.py",
    PR / "scripts" / "margin_o2o_replay.py",
    Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO\data.yaml"),
    Path(r"G:\revision_20261120\heldout_replication\extracted\annotations\aitod_val_v2.json"),
    Path(r"E:\two_paper\beifen\archive_backup\artifacts\analysis_runs\detect\experiments\baseline_13zone\yolo26s_aitodv2_baseline_13zone_e300_s0\weights\last.pt"),
]
RECORD = HERE / "FREEZE_RECORD.json"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


DATA_ROOT = Path(r"G:\revision_20261120\heldout_replication\AI-TOD-v2-officialval-YOLO")


def verify_selected_data() -> int:
    """Re-hash the selected images and labels against validation_files_sha256.txt."""
    import csv
    manifest = {}
    for line in (HERE / "validation_files_sha256.txt").read_text(encoding="utf-8").splitlines():
        h, rel = line.split("  ", 1)
        manifest[rel] = h
    with (HERE / "selected_images_preregistered.csv").open(encoding="utf-8") as f:
        ids = [row["image_id"] for row in csv.DictReader(f)]
    bad, n = [], 0
    for image_id in ids:
        for rel in (f"val/images/{image_id}.png", f"val/labels/{image_id}.txt"):
            n += 1
            if rel not in manifest or digest(DATA_ROOT / rel) != manifest[rel]:
                bad.append(rel)
    if bad:
        raise SystemExit(f"DATA VERIFY FAILED: {bad[:10]}")
    return n


def main() -> None:
    if "--verify" in sys.argv:
        rec = json.loads(RECORD.read_text(encoding="utf-8"))
        bad = [f for f, h in rec["sha256"].items() if digest(Path(f)) != h]
        if bad:
            raise SystemExit(f"FREEZE VERIFY FAILED: {bad}")
        n = verify_selected_data()
        print(f"freeze verified: {len(rec['sha256'])} files unchanged since {rec['frozen_at_utc']}; {n} selected image/label files match the per-file manifest")
        return
    verify_selected_data()
    if RECORD.exists():
        raise SystemExit("FREEZE_RECORD.json exists; refusing to overwrite")
    rec = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": "Written before any forward pass on replication images.",
           "sha256": {str(p): digest(p) for p in FILES}}
    RECORD.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"frozen {len(FILES)} files at {rec['frozen_at_utc']}; record sha256 {digest(RECORD)}")


if __name__ == "__main__":
    main()
