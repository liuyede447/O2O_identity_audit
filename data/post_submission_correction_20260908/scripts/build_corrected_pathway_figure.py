"""Create base-defined first-divergence / any-stage evidence and Figure 3B."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; IN=ROOT/'inputs'; OUT=ROOT/'outputs'/'PATHWAY_BASE_DEFINED_CORRECTION'; OUT.mkdir(parents=True,exist_ok=True)
paths={'Fixed 1 px':IN/'anatomy_fixed_per_direction.csv','Equivalent-area':IN/'anatomy_equivalent_per_direction.csv'}
labels=['eligibility_boundary','topk_membership_transition','conflict_reassignment','within_set_geometry_rank_reversal','active_disappearance','compound_or_other']
rows=[]
for contract,path in paths.items():
 d=pd.read_csv(path); d=d[d.active_base.notna()].copy(); flips=d[d.o2o_flip.eq(1)].copy(); w=flips.sampling_weight
 for label in labels: rows.append({'contract':contract,'family':'first','metric':label,'share':float(w[flips.first_divergence.eq(label)].sum()/w.sum()),'directions':len(flips),'images':flips.image_id.nunique()})
 for label,col in [('eligibility_any','eligibility_pair_changed'),('within_set_any','pair_rank_reversal'),('eligibility_and_within','both')]:
  mask=(d.eligibility_pair_changed.eq(1)&d.pair_rank_reversal.eq(1)) if col=='both' else d[col].eq(1)
  rows.append({'contract':contract,'family':'any','metric':label,'share':float((d.sampling_weight*mask).sum()/d.sampling_weight.sum()),'directions':len(d),'images':d.image_id.nunique()})
table=pd.DataFrame(rows);table.to_csv(OUT/'FIRST_DIVERGENCE_ANY_STAGE_CORRECTED.csv',index=False)
first=table[table.family.eq('first')]; fig,ax=plt.subplots(figsize=(8.4,3.7)); x=range(len(labels)); width=.36
for offset,contract,color in [(-width/2,'Fixed 1 px','#4477AA'),(width/2,'Equivalent-area','#CC6677')]:
 values=first[first.contract.eq(contract)].set_index('metric').loc[labels,'share']*100;ax.bar([i+offset for i in x],values,width,label=contract,color=color)
ax.set_xticks(list(x),['Eligibility','Top-k','Conflict','Within-set','Disappearance','Other'],rotation=24,ha='right');ax.set_ylabel('First-label share (%)');ax.set_ylim(0,max(first.share*100)*1.22);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'CORRECTED_FIGURE3B.pdf');fig.savefig(OUT/'CORRECTED_FIGURE3B.png',dpi=300)
