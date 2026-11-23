#!/usr/bin/env bash
# Formal run of the pre-specified replication (PREREGISTRATION_DRAFT.md, Sections 6 and 8).
# Verifies FREEZE_RECORD.json on every start. Completed steps (output present) are skipped, so a step
# that failed for an infrastructure reason can be re-executed with identical code; each start and
# re-execution is logged in timeline.txt. A partial step output is moved aside, never reused.
# Order: pre-flight-checked lock replay -> base gate -> primary analysis (written first) -> S2.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=/g/revision_20261120/heldout_replication/formal
CK="E:/two_paper/beifen/archive_backup/artifacts/analysis_runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s0/weights/last.pt"
SHA=46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b
DATA="G:/revision_20261120/heldout_replication/AI-TOD-v2-officialval-YOLO/data.yaml"
SEL="$HERE/selected_images_preregistered.csv"
cd "$HERE"
python -I freeze.py --verify
mkdir -p "$OUT"
log() { echo "$1 $(date -Is)" | tee -a "$OUT/timeline.txt"; }
aside() { [ -e "$1" ] && mv "$1" "$1.failed_$(date +%Y%m%dT%H%M%S)" && log "moved partial output aside: $1"; return 0; }
log "start (freeze verified)"

if [ -f "$OUT/lock/manifest.json" ] && grep -q '"status": "complete"' "$OUT/lock/manifest.json"; then
  log "skip lock replay (complete)"
else
  aside "$OUT/lock"; log "lock replay begin"
  python -I run_lock_checked.py --expected-selection "$SEL" -- --checkpoint "$CK" --expected-sha256 $SHA --data "$DATA" \
    --output-dir "$OUT/lock" --sample-images 600 --selection-seed 20261122 --bootstrap 5000 --imgsz 800 \
    --device 0 --workers 0 --kappa 0.0625 > "$OUT/lock.log" 2>&1
  log "lock replay done"
fi

if [ -f "$OUT/base_gate.csv" ]; then
  log "skip base gate (present)"
else
  log "base gate begin"
  python -I base_gate.py --checkpoint "$CK" --expected-sha256 $SHA --data "$DATA" --selected-images "$SEL" \
    --out "$OUT/base_gate.csv" > "$OUT/base_gate.log" 2>&1
  log "base gate done"
fi

if [ -f "$OUT/primary.json" ]; then
  log "skip primary analysis (present)"
else
  log "primary analysis begin"
  python -I analyze_replication.py --lock-dir "$OUT/lock" --gate "$OUT/base_gate.csv" --expected-selection "$SEL" \
    --out "$OUT/primary.json" > "$OUT/primary.txt" 2>&1
  log "primary analysis done"
fi

if [ -f "$OUT/s2.json" ]; then
  log "skip S2 (present)"
else
  aside "$OUT/normalized"; aside "$OUT/rank_set"; log "S2 begin"
  if bash run_s2.sh "$CK" $SHA "$DATA" "$SEL" "$OUT" && python -I analyze_s2.py --root "$OUT" --out "$OUT/s2.json" > "$OUT/s2.txt" 2>&1; then
    log "S2 done"
  else
    log "S2 FAILED"
  fi
fi
log "end"
