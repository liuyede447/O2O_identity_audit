"""Recompute the revision point estimates from the compact audit inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SIZES = ("t_8_16", "s_16_32")

def read(name):
    return pd.read_csv(ROOT / f"inputs/{name}.csv.gz")

def rates(rows, col):
    return [float(np.average(x[col], weights=x.sampling_weight))
            for key in SIZES for x in [rows.loc[rows.size_bin.eq(key)]]]

def gap(rows, col):
    left, right = rates(rows, col)
    return left - right

def restricted_mean(rows, branch, tau=.125):
    present = rows[rows[branch+"_base"].notna()]
    times = present[branch+"_radius"].where(present[branch+"_event"].eq(1),present[branch+"_censor"])
    weight = present.sampling_weight.to_numpy()
    times = times.to_numpy()
    event = present[branch+"_event"].eq(1).to_numpy()
    area = 0.0
    survival = 1.0
    prev = 0.0
    for time in sorted(set(times[event & (times<=tau+1e-12)])):
        area += (time-prev)*survival
        risk = weight[times>=time-1e-12].sum()
        deaths = weight[event & np.isclose(times,time,atol=1e-12,rtol=0)].sum()
        survival *= 1-deaths/risk
        prev = time
    return float(area+(tau-prev)*survival)

def compute():
    out = dict()
    lock = read("eligibility_lock")
    out["R1.2"] = {}
    for k, rows in lock.groupby(["seed","stress"],sort=True):
        a,b = (gap(rows,x) for x in ("native_o2o_fragile","eligibility_lock_o2o_fragile"))
        out["R1.2"][f"{k[0]}:{k[1]}"] = [a,b,a-b]
    events = read("boundary")
    out["R1.3"] = {str(ds):[restricted_mean(x,"o2o"),restricted_mean(x,"o2m")]
                   for ds,x in events.groupby("direction_set")}
    fk=read("fixed_k")
    out["R1.4"] = {}
    for seed,direction in fk.groupby("seed"):
        for k in (3,5):
            col=f"k{k}_set_turnover"
            raw=direction.loc[direction[f"k{k}_eligible"].eq(1)&direction.base_native_assigned_count.gt(k)]
            for name,rows in (("binding",raw.loc[raw.shift_native_assigned_count.gt(k)]),("base_only",raw)):
                rows=rows.assign(bucket=rows.base_native_assigned_count.clip(upper=k+5))
                a=[]
                for _, b in rows.groupby("bucket"):
                    if set(b.size_bin)==set(SIZES):
                        a.append((gap(b,col),float(b.loc[b.size_bin.eq(SIZES[0]),"sampling_weight"].sum())))
                v,w=zip(*a)
                out["R1.4"][f"{seed}:{k}:{name}"]=[gap(rows,col),float(np.average(v,weights=w))]
    comparable=read("comparable")
    out["R2.1"] = {}
    for seed,direction in comparable.groupby("seed"):
        measures=["o2o_change","o2m_top1_paper","o2m_assigned_top1","o2m_set_change"]
        obj=direction.groupby(["image_id","gt_id"],as_index=False).agg(
           **{k:(k,"max") for k in measures},size_bin=("size_bin","first"),sampling_weight=("sampling_weight","first"))
        out["R2.1"][str(seed)]=[q for v in measures for q in [*rates(obj,v),gap(obj,v)]]
    indep=read("independent")
    cols=["yolo_o2o_change","yolo_o2m_top1_change","rfla_set_change","rfla_top1_change","atss_set_change","atss_top1_change"]
    out["R2.2"]={str(k):[len(x),*[gap(x,y) for y in cols]] for k,x in indep.groupby("stress")}
    return out

def main():
    reference=json.loads((ROOT/"expected.json").read_text(encoding="utf-8"))
    values=compute()
    assert set(values)==set(reference)
    max_error=0.0
    for family, entries in reference.items():
        assert set(values[family])==set(entries),family
        for k, expected in entries.items():
            observed=np.asarray(values[family][k], dtype=float)
            np.testing.assert_allclose(observed,expected,rtol=0,atol=2e-10,err_msg=f"{family}/{k}")
            max_error=max(max_error,float(np.max(np.abs(observed-np.asarray(expected)))))
    print(json.dumps({"status":"point_estimates_match","max_abs_error":max_error,"replicated_rows":{k:len(v) for k,v in reference.items()}},indent=2))

if __name__=="__main__": main()

