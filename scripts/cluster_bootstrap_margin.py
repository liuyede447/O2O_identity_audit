"""Image-cluster bootstrap for margin→FN association in an outcome table."""
from __future__ import annotations
import argparse,csv,json,math,random
from collections import defaultdict
from pathlib import Path
import numpy as np
def sig(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def coef(rows):
 y=np.array([float(r['fn_at_iou50']) for r in rows]);m=np.array([float(r['o2o_margin_0']) for r in rows]);a=np.array([math.log(max(float(r['area']),1)) for r in rows]);n=np.array([float(r['o2m_positive_count']) for r in rows]);X=np.column_stack([np.ones(len(y)),m,a,n]);b=np.zeros(4)
 for _ in range(80):
  p=sig(X@b);w=np.clip(p*(1-p),1e-7,None);s=np.linalg.solve((X.T*w)@X,X.T@(y-p));b+=s
  if max(abs(s))<1e-8:break
 return float(b[1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--replicates',type=int,default=500);p.add_argument('--seed',type=int,default=20260821);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 rows=[r for r in csv.DictReader(a.input.open(newline='',encoding='utf-8')) if r['size_bin'] in ('t_8_16','s_16_32') and r['o2o_margin_0'] not in ('','None')];groups=defaultdict(list)
 for r in rows:groups[r['image_id']].append(r)
 rng=random.Random(a.seed);keys=list(groups);vals=[]
 for _ in range(a.replicates):
  sample=[]
  for k in (rng.choice(keys) for _ in keys):sample.extend(groups[k])
  try:vals.append(math.exp(coef(sample)))
  except Exception:pass
 vals.sort();a.output_dir.mkdir(parents=True)
 (a.output_dir/'summary.json').write_text(json.dumps({'status':'complete','images':len(keys),'gt':len(rows),'replicates_ok':len(vals),'margin_or_median':vals[len(vals)//2],'ci_2_5':vals[int(.025*len(vals))],'ci_97_5':vals[int(.975*len(vals))-1]},indent=2),encoding='utf-8')
if __name__=='__main__':main()
