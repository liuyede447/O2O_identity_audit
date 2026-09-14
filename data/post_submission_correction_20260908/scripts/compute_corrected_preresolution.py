"""Exact base-defined reaggregation of copied formal pre-resolution artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'inputs'/'pre_resolution_formal'; OUT=ROOT/'outputs'/'PRE_RESOLUTION_BASE_DEFINED_CORRECTION'
SIZES=('t_8_16','s_16_32'); REPS=5000; SEED=20260831

def rate(df,col,size=None):
    x=df if size is None else df[df.size_bin.eq(size)]
    return float((x.sampling_weight*x[col].astype(float)).sum()/x.sampling_weight.sum())
def stats(gt, direction):
    r={}
    for label,col in [('pre','pre_fragile'),('active','active_fragile')]:
        for name,size in [('tiny',SIZES[0]),('small',SIZES[1])]: r[f'{label}_{name}']=rate(gt,col,size)
        r[f'{label}_gap']=r[f'{label}_tiny']-r[f'{label}_small']
    r['active_minus_pre_gap']=r['active_gap']-r['pre_gap']
    for name,col in [('object_disagreement','fragility_disagreement'),('object_conflict','any_conflict_involved'),('object_disagreement_with_conflict','disagreement_with_conflict')]:r[name]=rate(gt,col)
    for name,col in [('directional_disagreement','flip_disagreement'),('directional_disagreement_with_conflict','flip_disagreement_with_conflict'),('directional_conflict','conflict_involved')]:r[name]=rate(direction,col)
    r['directional_disagreement_count']=int(direction.flip_disagreement.sum());r['object_disagreement_count']=int(gt.fragility_disagreement.sum())
    r['directional_disagreement_conflict_count']=int(direction.flip_disagreement_with_conflict.sum())
    return r
def boot(gt,direction):
    info=gt[['image_id','stratum']].drop_duplicates(); groups={s:g.image_id.to_numpy() for s,g in info.groupby('stratum')}; gi={k:v.index.to_numpy() for k,v in gt.groupby('image_id')}; di={k:v.index.to_numpy() for k,v in direction.groupby('image_id')}; rng=np.random.default_rng(SEED); rows=[]
    for rep in range(REPS):
        images=np.concatenate([rng.choice(v,len(v),replace=True) for v in groups.values()]); gidx=np.concatenate([gi[x] for x in images]); didx=np.concatenate([di[x] for x in images]); row={'replicate':rep};row.update(stats(gt.loc[gidx],direction.loc[didx]));rows.append(row)
    return pd.DataFrame(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True); allout={'protocol':'PRE_RESOLUTION_BASE_DEFINED_CORRECTION_V1','read_only':True,'base_gate':'loss_active_base must be defined','contracts':{}}
    master=[]
    for contract in ['fixed','equivalent']:
        base=IN/contract; gt=pd.read_csv(base/'per_gt.csv'); direction=pd.read_csv(base/'per_direction.csv'); valid_keys=direction.loc[direction.loss_active_base.notna(),['image_id','gt_id']].drop_duplicates()
        gt=gt.merge(valid_keys,on=['image_id','gt_id'],how='inner'); direction=direction.merge(valid_keys,on=['image_id','gt_id'],how='inner'); point=stats(gt,direction); b=boot(gt,direction); b.to_csv(OUT/f'{contract}_bootstrap.csv',index=False); ci={k:[float(x) for x in np.percentile(b[k],[2.5,97.5])] for k in point if k not in {'directional_disagreement_count','object_disagreement_count','directional_disagreement_conflict_count'}}
        allout['contracts'][contract]={'effective_gt':len(gt),'effective_images':gt.image_id.nunique(),'effective_directions':len(direction),'excluded_base_undefined_gt':int(pd.read_csv(base/'per_gt.csv').shape[0]-len(gt)),'point':point,'ci_95':ci}
        master += [{'analysis':'pre_resolution','contract':contract,'estimand':k,'point':v,'ci_low':ci.get(k,[None,None])[0],'ci_high':ci.get(k,[None,None])[1]} for k,v in point.items()]
    pd.DataFrame(master).to_csv(OUT/'PRE_RESOLUTION_CORRECTED_ESTIMATES.csv',index=False);(OUT/'summary.json').write_text(json.dumps(allout,indent=2)+'\n')
if __name__=='__main__':main()
