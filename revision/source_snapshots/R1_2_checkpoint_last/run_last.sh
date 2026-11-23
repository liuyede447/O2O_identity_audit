#!/usr/bin/env bash
# Checkpoint sensitivity: eligibility-lock replay on the final-epoch (last.pt) weights of seeds 0-3.
# Same script, data config, 300-image selection seed, kappa and bootstrap as the epoch-280 runs
# (../R1_2_eligibility_lock_seeds/run_seeds.sh). Pairing to epoch 280 is verified by training
# name, seed, training-argument hash and save time; see PROVENANCE.md.
set -u
PR=/e/two_paper/beifen/publication_reproducible
B=/e/two_paper/beifen/archive_backup
OUT=/g/revision_20261120/checkpoint_last
mkdir -p "$OUT"
declare -A CKPT=(
  [0]=$B/artifacts/analysis_runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s0/weights/last.pt
  [1]=$B/server_A/runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s1/weights/last.pt
  [2]=$B/server_A/runs/detect/experiments/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s2/weights/last.pt
  [3]=$B/artifacts/consolidated_training_20260801/active_main/server_B/baseline_13zone/yolo26s_aitodv2_baseline_13zone_e300_s3/weights/last.pt
)
declare -A SHA=(
  [0]=46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b
  [1]=1c689b3e6ae7dd27009fc911dc1e8d7928aa4aff9327dfc627ce770da4e12f5f
  [2]=b47855abfb339a682299b40dca54298c62f2d897f6a542af2134051db9a352f7
  [3]=326a5bc8ca16a390f5ba93fa8a99c4314eaa0e83648aefc866906b9b54f836ae
)
cd "$PR"
for s in 0 1 2 3; do
  echo "=== seed $s start $(date -Is)"
  python -I scripts/run_eligibility_lock_counterfactual.py \
    --checkpoint "${CKPT[$s]}" \
    --expected-sha256 "${SHA[$s]}" \
    --data "$PR/configs/aitod_v2_local_g.yaml" \
    --output-dir "$OUT/seed$s" \
    --sample-images 300 --selection-seed 20260823 --bootstrap 5000 \
    --imgsz 800 --device 0 --workers 0 --kappa 0.0625 \
    > "$OUT/seed${s}.log" 2>&1
  echo "=== seed $s exit $? $(date -Is)"
done
