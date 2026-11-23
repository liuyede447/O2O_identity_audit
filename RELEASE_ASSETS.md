# Separate release assets

These files are deliberately outside `github_repository/`.

| File | Bytes | SHA-256 | Role | Status |
|---|---:|---|---|---|
| `yolo26s_aitodv2_seed0_best_epoch280.pt` | 60,231,446 | `4b57787f7351c77dfe6c85e64b30206245207722b7ed86cae0c741b1f06828fa` | primary seed-0 discovery checkpoint | unchanged from public v3.0.0; article-matched v3.1.0 release |
| `yolo26s_aitodv2_seed1_best_epoch280.pt` | 60,231,446 | `d9432d90e9b7c024456a4bca94da8e5e805b03beeedb6158052792e3e755b549` | protocol-matched additional seed checkpoint | unchanged from public v3.0.0; article-matched v3.1.0 release |
| `yolo26s_aitodv2_seed2_best_epoch280.pt` | 60,231,446 | `d83f86f570393d9e573d9c93900322bc4e8fca687a40233a108e97832d8e15f6` | multiseed sensitivity checkpoint | unchanged from public v3.0.0; article-matched v3.1.0 release |
| `yolo26s_aitodv2_seed3_best_epoch280.pt` | 60,231,446 | `28fee07d38f6914a4e0d74b574f2f4318314f42e02b1160de495b2082af60ff2` | multiseed sensitivity checkpoint | unchanged from public v3.0.0; article-matched v3.1.0 release |
| `boundary_full_grid_1024_o2m_rank_set_radius_curve.csv` | 1,411,448,003 | `3622652665b30f727258aa0528436e33ab402255e29935dcc8168a32ea5f6b33` | complete 1/1024 rank-set curve; 7,593,288 rows | unchanged from public v3.0.0; article-matched v3.1.0 release |
| `yolo26s_aitodv2_seed0_final_epoch.pt` | 20,396,990 | `46ff9c348e4eda88d439b776717d2dcfb3c386ea8833dc681e77ea1aa01d349b` | seed-0 final-epoch checkpoint; Tables S43, S45-S48 | new in v3.4.0 |
| `yolo26s_aitodv2_seed1_final_epoch.pt` | 20,396,990 | `1c689b3e6ae7dd27009fc911dc1e8d7928aa4aff9327dfc627ce770da4e12f5f` | seed-1 final-epoch checkpoint; Table S43 | new in v3.4.0 |
| `yolo26s_aitodv2_seed2_final_epoch.pt` | 20,396,990 | `b47855abfb339a682299b40dca54298c62f2d897f6a542af2134051db9a352f7` | seed-2 final-epoch checkpoint; Table S43 | new in v3.4.0 |
| `yolo26s_aitodv2_seed3_final_epoch.pt` | 20,396,990 | `326a5bc8ca16a390f5ba93fa8a99c4314eaa0e83648aefc866906b9b54f836ae` | seed-3 final-epoch checkpoint; Table S43 | new in v3.4.0 |
