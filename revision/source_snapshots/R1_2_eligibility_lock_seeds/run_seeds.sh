#!/usr/bin/env bash
# R1.2: eligibility-lock counterfactual on YOLO26s seeds 1-3 (seed 0 already exists).
# Same script, same data config, same selection seed and kappa as the seed-0 run
# (results/measurement_validation_20260831/eligibility_lock_v1/manifest.json).
set -u
PR=/e/two_paper/beifen/publication_reproducible
CK=$PR/release_candidates/O2O_assignment_state_audit_v3.0.0/release_assets
OUT=/g/revision_20261120/eligibility_lock
mkdir -p "$OUT"
declare -A SHA=(
  [1]=d9432d90e9b7c024456a4bca94da8e5e805b03beeedb6158052792e3e755b549
  [2]=d83f86f570393d9e573d9c93900322bc4e8fca687a40233a108e97832d8e15f6
  [3]=28fee07d38f6914a4e0d74b574f2f4318314f42e02b1160de495b2082af60ff2
)
cd "$PR"
for s in 1 2 3; do
  echo "=== seed $s start $(date -Is)"
  python -I scripts/run_eligibility_lock_counterfactual.py \
    --checkpoint "$CK/yolo26s_aitodv2_seed${s}_best_epoch280.pt" \
    --expected-sha256 "${SHA[$s]}" \
    --data "$PR/configs/aitod_v2_local_g.yaml" \
    --output-dir "$OUT/seed$s" \
    --sample-images 300 --selection-seed 20260823 --bootstrap 5000 \
    --imgsz 800 --device 0 --workers 0 --kappa 0.0625 \
    > "$OUT/seed${s}.log" 2>&1
  echo "=== seed $s exit $? $(date -Is)"
done
