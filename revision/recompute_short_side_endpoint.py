
"""Reproduce the post hoc matched-endpoint short-side sensitivity in Section 17."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs/short_side_endpoint_by_image.csv.gz"
OUTPUT = ROOT / "short_side_endpoint_results.json"
TAGS = ["t_8_16", "s_16_32"]
KEYS = ["image_id", "gt_id"]
DIRECTION_KEYS = [*KEYS, "direction"]
RUNS = (
    "results/measurement_validation_20260831/fixed_branch_replay_v1",
    "results/reviewer_controls_20260829/formal_s0e300_k00625",
)
BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 20260823


def archive_inputs(baseline):
    run = [baseline / p for p in RUNS]
    csvs = [p / name for p in run for name in ("per_gt.csv", "per_direction.csv")]
    roster_path = run[0] / "selected_images.csv"
    csvs.append(roster_path)
    provenance = {p.relative_to(baseline).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in csvs}
    columns = KEYS + ["width", "height", "size_bin", "sampling_weight", "common_valid_margin"]
    per_gt = [pd.read_csv(p / "per_gt.csv", usecols=columns) for p in run]
    valid = [x.loc[x.common_valid_margin.eq(1) & x.size_bin.isin(TAGS)] for x in per_gt]
    matched = valid[0].merge(valid[1][KEYS + ["sampling_weight", "width", "height"]], 
                             on=KEYS, validate="one_to_one", suffixes=("", "_scaled"))
    for col in ("sampling_weight", "width", "height"):
        assert np.allclose(matched[col], matched[col + "_scaled"], rtol=0, atol=1e-9), col
    direction = [pd.read_csv(p / "per_direction.csv", usecols=DIRECTION_KEYS + ["o2o_flip"]) for p in run]
    same = direction[0].merge(direction[1], on=DIRECTION_KEYS, validate="one_to_one",
                              suffixes=("_fixed", "_scaled"))
    same = same.merge(matched[KEYS], on=KEYS, validate="many_to_one")
    endpoints = same.groupby(KEYS, as_index=False).agg(
        fixed=("o2o_flip_fixed", "max"), scaled=("o2o_flip_scaled", "max"),
        n_directions=("direction", "size"))
    joined = matched.merge(endpoints, on=KEYS, validate="one_to_one")
    short_side = joined[["width", "height"]].min(axis=1)
    joined["subset"] = np.where((short_side - 8).abs().le(0.001), "boundary_8", "other")
    joined["fixed_weight"] = joined.fixed * joined.sampling_weight
    joined["scaled_weight"] = joined.scaled * joined.sampling_weight
    joined["n_gt"] = 1
    roster = pd.read_csv(roster_path)[["image_id", "stratum"]].reset_index(drop=True)
    assert roster.image_id.is_unique
    roster["image_index"] = np.arange(len(roster))
    summary = joined.merge(roster, on="image_id", validate="many_to_one").groupby(
        ["image_index", "size_bin", "subset"], as_index=False
    )[["n_gt", "n_directions", "sampling_weight", "fixed_weight", "scaled_weight"]].sum()
    grid = pd.MultiIndex.from_product([roster.image_index, TAGS, ["boundary_8", "other"]],
                                      names=["image_index", "size_bin", "subset"])
    compact = summary.set_index(list(grid.names)).reindex(grid, fill_value=0).reset_index()
    compact = compact.merge(roster.drop(columns="image_id"), on="image_index", validate="many_to_one")
    compact.to_csv(INPUT,index=False, compression={"method":"gzip","mtime":0})
    return provenance


def compute(compact):
    roster = compact.drop_duplicates("image_index")[["image_index", "stratum"]]
    assert roster.image_index.is_unique
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for stratum in roster.stratum.unique():
        indices = np.flatnonzero(roster.stratum.eq(stratum).to_numpy())
        draws.append(rng.choice(indices, size=(BOOTSTRAPS, len(indices)), replace=True))
    results = {}
    for label, data in (("matched_full", compact), ("excluding_8px_short_side", compact[compact.subset.eq("other")])):
        fields = ["sampling_weight", "fixed_weight", "scaled_weight"]
        by_image = data.groupby(["image_index", "size_bin"])[fields].sum().unstack(fill_value=0)
        by_image = by_image.reindex(roster.image_index, fill_value=0)
        cube = np.stack([by_image.xs(tag, axis=1, level=1)[fields].to_numpy() for tag in TAGS], axis=1)
        boot = sum(cube[ix].sum(axis=1) for ix in draws)
        def gaps(sums):
            rates = sums[..., 1:] / sums[..., :1]
            estimates = 100 * (rates[..., 0, :] - rates[..., 1, :])
            return np.concatenate((estimates, (estimates[..., :1] - estimates[..., 1:])), axis=-1)
        point = gaps(cube.sum(axis=0))
        lo,hi = np.percentile(gaps(boot), [2.5, 97.5], axis=0)
        results[label] = {"objects":int(data.n_gt.sum()), "directions":int(data.n_directions.sum()),
                          "contributing_images":int(data.loc[data.n_gt.gt(0),"image_index"].nunique()),
                          "estimates_pp":{name:{"point":float(p),"ci_95":[float(a),float(b)]}
                          for name,p,a,b in zip(("fixed_1px", "size_scaled", "fixed_minus_scaled"),point,lo,hi)}}
    return {"design":"Post hoc seed-0 discovery sample. Contract-specific Table 2 GTs intersected, then legal directions matched within each GT. An object changes if any matched direction changes its O2O identity.",
            "short_side_exclusion":{"centre_px":8,"tolerance_px":0.001,"applies_to_both_size_groups":True},
            "bootstrap":{"seed":BOOTSTRAP_SEED,"replicates":BOOTSTRAPS,"unit":"image within original sampling stratum"},
            "results":results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,help="Create compact inputs again from the original analysis archive")
    args=parser.parse_args()
    if args.archive:
        provenance=archive_inputs(args.archive)
        result=compute(pd.read_csv(INPUT))
        result["upstream_sha256"]=provenance
        OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    else:
        result=compute(pd.read_csv(INPUT))
        expected=json.loads(OUTPUT.read_text("utf8"))
        for label,row in result["results"].items():
            saved=expected["results"][label]
            for n in ("objects","directions","contributing_images"):
                assert row[n]==saved[n], (label,n)
            for metric,calculated in row["estimates_pp"].items():
                recorded=saved["estimates_pp"][metric]
                assert np.isclose(calculated["point"],recorded["point"],rtol=0,atol=1e-10)
                assert np.allclose(calculated["ci_95"],recorded["ci_95"],rtol=0,atol=1e-10)
    print(json.dumps(result["results"],indent=2))


if __name__=="__main__":
    main()

