#!/usr/bin/env bash
# R1.4: fixed-K O2M replay (K = 1, 3, 5), equivalent-area kappa=0.0625, YOLO26s seeds 0-3.
# Script is unchanged since the aborted 2026-09-01 formal run (sha256 51a501bf...).
set -u
PR=/e/two_paper/beifen/publication_reproducible
CK=$PR/release_candidates/O2O_assignment_state_audit_v3.0.0/release_assets
SEL=$PR/results/five_experiment_upgrade_20260830/anatomy_fixed_1px_v2/selected_images.csv
OUT=/g/revision_20261120/fixed_k
mkdir -p "$OUT"
declare -A SHA=(
  [0]=4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa
  [1]=d9432d90e9b7c024456a4bca94da8e5e805b03beeedb6158052792e3e755b549
  [2]=d83f86f570393d9e573d9c93900322bc4e8fca687a40233a108e97832d8e15f6
  [3]=28fee07d38f6914a4e0d74b574f2f4318314f42e02b1160de495b2082af60ff2
)
cd "$PR"
for s in 0 1 2 3; do
  echo "=== seed $s start $(date -Is)"
  python -I scripts/analyze_o2m_fixed_k_replay.py \
    --checkpoint "$CK/yolo26s_aitodv2_seed${s}_best_epoch280.pt" \
    --expected-sha256 "${SHA[$s]}" \
    --data "$PR/configs/aitod_v2_local_g.yaml" \
    --selected-images "$SEL" \
    --output-dir "$OUT/seed$s" \
    --fixed-k 1 3 5 --kappa 0.0625 --imgsz 800 --device 0 --workers 0 \
    --bootstrap 5000 --bootstrap-seed 20260901 \
    > "$OUT/seed${s}.log" 2>&1
  echo "=== seed $s exit $? $(date -Is)"
done
