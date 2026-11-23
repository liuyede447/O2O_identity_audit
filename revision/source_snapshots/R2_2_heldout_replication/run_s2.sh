#!/usr/bin/env bash
# S2 pipeline: equivalent-area normalized replay + O2M rank-set decoupling on a frozen selection.
# Usage: run_s2.sh <checkpoint> <sha256> <data.yaml> <selected_images.csv> <out_root>
set -eu
CK=$1; SHA=$2; DATA=$3; SEL=$4; OUT=$5
PR=/e/two_paper/beifen/publication_reproducible
cd "$PR"
python -I scripts/run_reviewer_killer_controls.py --checkpoint "$CK" --expected-sha256 "$SHA" --data "$DATA" \
  --selected-images "$SEL" --output-dir "$OUT/normalized" --stress-mode normalized --normalization equivalent_side \
  --normalized-only --kappa 0.0625 --detector-contract yolo26 --checkpoint-role "pre-specified replication checkpoint" --bootstrap 5000 \
  --device 0 --workers 0 > "$OUT/normalized.log" 2>&1
python -I scripts/analyze_o2m_rank_set_decoupling.py --checkpoint "$CK" --expected-sha256 "$SHA" --data "$DATA" \
  --selected-images "$SEL" --output-dir "$OUT/rank_set" --stress equivalent_side --kappa 0.0625 \
  --bootstrap 5000 --bootstrap-seed 20260831 --device 0 --workers 0 > "$OUT/rank_set.log" 2>&1
echo "S2 done $(date -Is)"
