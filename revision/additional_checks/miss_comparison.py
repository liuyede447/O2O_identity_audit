"""Recompute the miss-definition comparison from saved continuous IoU and OOF input rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from core import M2, fit, predict
from verify import compare

here = Path(__file__).resolve().parent
data = pd.read_csv(here / "inputs" / "margin" / "miss_comparison.csv", dtype={"image_id": str})
expected = json.loads((here / "expected" / "exp1_miss_contract_reconciliation.json").read_text(encoding="utf-8"))
pairs = {
    "mismatch_ext_miss_vs_recomputed_800_fp32": ("ext_miss", "miss_800_fp32"),
    "mismatch_oof_fn_vs_ext_or_800_fp32": ("fn_at_iou50", "miss_800_fp32"),
    "mismatch_oof_fn_vs_832_fp16_pad16": ("fn_at_iou50", "miss_832_fp16"),
    "mismatch_oof_fn_vs_832_fp32_pad16": ("fn_at_iou50", "miss_832_fp32"),
    "mismatch_800_fp32_vs_832_fp16_pad16": ("miss_800_fp32", "miss_832_fp16"),
    "mismatch_832_fp16_vs_832_fp32": ("miss_832_fp16", "miss_832_fp32"),
}
compare(len(data), expected["n_overlap_6410"], "overlap")
for key, (a, b) in pairs.items():
    compare(int(data[a].ne(data[b]).sum()), expected[key], key)
y = data.miss_800_fp32.to_numpy(int)
w = data.sampling_weight.to_numpy(float)
data["miss"] = y
data["margin_clip"] = data["o2o_margin"].clip(0., 1.)
groups = data.image_id.str.split("__").str[0]
weights = np.zeros((len(data), 2))
for train, test in GroupKFold(n_splits=5).split(data, y, groups):
    for index, features in enumerate((M2, M2 + ["margin_clip"])):
        weights[test, index] = predict(data.iloc[test], fit(data.iloc[train], "miss", features, w[train]))
for index, model in enumerate(("M2", "M3clip")):
    compare(float(roc_auc_score(y, weights[:, index], sample_weight=w)), expected[f"overlap6410_unified_auc_{model}"], model)
