from pathlib import Path
import csv,json,hashlib,gzip
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parent
rates=[1,2,3,5];budgets=[1,2,3,5]
expected={1:[2,4,5,5],2:[302,330,333,335],3:[302,491,493,499],5:[302,604,663,672]}
tasks=[108,1720,1724,1725,284,363,475,609,833,888]
names={108:'Contextual abuse',1720:'Toxicity',1724:'Insult',1725:'Severe toxicity',284:'IMDb',363:'SST-2',475:'Yelp',609:'Potential offense',833:'Poem sentiment',888:'Reviews'}
data={};checks={};mapping=None
for rate in rates:
 file=root/'data'/f'task_scores_{rate}pct.csv.gz'
 with gzip.open(file,'rt',encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 r.sort(key=lambda x:int(x['scored_index']))
 assert [int(x['scored_index']) for x in r]==list(range(30220))
 pairs=[(int(x['original_index']),x['task_name']) for x in r]
 assert len(set(x[0] for x in pairs))==30220
 if mapping is None:mapping=pairs
 else:assert pairs==mapping
 p=np.array([int(x['poison']) for x in r]);s=np.array([float(x['influence_score']) for x in r])
 assert int(p.sum())==int(30220*rate/100) and np.isfinite(s).all()
 ids=np.array([int(x['task_name'].split('_')[0][4:]) for x in r])
 ref=np.array([int(x.get('is_reference',0)) for x in r],dtype=bool) if rate!=2 else None
 if ref is not None:assert ref.sum()==10 and np.all(p[ref]==1)
 ranked=np.argsort(s,kind='stable');cuts={};audit=[]
 for budget,tp_expected in zip(budgets,expected[rate]):
  k=int(len(r)*budget/100);selected=ranked[:k];tp=int(p[selected].sum());assert tp==tp_expected
  cut=float(s[selected[-1]]);cuts[budget]=cut
  audit.append({'removal':budget,'removed':k,'tp':tp,'fp':k-tp,'cutoff':cut,'tie_spans_boundary':bool((s<cut).sum()<k<(s<=cut).sum())})
 data[rate]=(s,p,ids,ref,cuts)
 checks[rate]={'rows':len(r),'poisoned':int(p.sum()),'sha256':hashlib.sha256(gzip.decompress(file.read_bytes())).hexdigest(),'reference_membership_available':ref is not None,'removal_checks':audit}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
styles={1:('#444444',':'),2:('#777777','--'),3:('#555555','-.'),5:('#999999','-')}
allmetrics=[]
for rate,(scores,poison,ids,refs,cuts) in data.items():
 fig,axs=plt.subplots(5,2,figsize=(10.2,12.4),sharex=True,sharey=True)
 for ax,task in zip(axs.flat,tasks):
  m=ids==task;x=scores[m];y=poison[m];nc=int((y==0).sum());np_=int(y.sum())
  auc=float((rankdata(-x)[y==1].sum()-np_*(np_+1)/2)/(np_*nc)) if np_ and nc else None
  allmetrics.append({'poisoning_percent_nominal':rate,'scope':'all_rows','task':task,'clean':nc,'poisoned':np_,'auroc_low':auc})
  if refs is not None:
   keep=m&~refs;xx=scores[keep];yy=poison[keep];nn=int(yy.sum());cc=int((yy==0).sum())
   aa=float((rankdata(-xx)[yy==1].sum()-nn*(nn+1)/2)/(nn*cc)) if nn and cc else None
   allmetrics.append({'poisoning_percent_nominal':rate,'scope':'without_references_descriptive','task':task,'clean':cc,'poisoned':nn,'auroc_low':aa})
  for flag,color,ls in [(0,'#0072B2','-'),(1,'#D55E00','--')]:
   z=np.sort(x[y==flag])
   if len(z):ax.step(np.r_[z[0],z],np.r_[0,np.arange(1,len(z)+1)/len(z)],where='post',color=color,ls=ls,lw=1.5,zorder=3)
  for b,(col,ls) in styles.items():ax.axvline(cuts[b],color=col,ls=ls,lw=.8,alpha=.85,zorder=1)
  ax.set_title(f'task{task}  {names[task]}',loc='left',pad=7)
  ax.text(.97,.06,f'Clean n={nc:,}; poisoned n={np_:,}\nAUROC='+('NA' if auc is None else f'{auc:.3f}'),transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':2})
  ax.set_xlim(-96000,10000);ax.set_ylim(-.02,1.03);ax.grid(alpha=.2,lw=.5);ax.spines[['top','right']].set_visible(False)
  ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
 handles=[Line2D([0],[0],color='#0072B2',label='Clean',lw=1.5),Line2D([0],[0],color='#D55E00',ls='--',label='Poisoned',lw=1.5)]
 handles +=[Line2D([0],[0],color=col,ls=ls,lw=1,label=f'Remove {b}%') for b,(col,ls) in styles.items()]
 fig.suptitle(f'Nominal poisoning {rate}%  |  all scored rows',y=.997,fontsize=12)
 fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.978),fontsize=9)
 fig.supxlabel('Influence score (lower scores selected first)',y=.015);fig.supylabel('Empirical cumulative proportion within group',x=.015)
 fig.subplots_adjust(left=.085,right=.985,top=.91,bottom=.065,hspace=.43,wspace=.16)
 fig.savefig(root/f'task_score_ecdf_{rate}pct.pdf',bbox_inches='tight');fig.savefig(root/f'task_score_ecdf_{rate}pct.png',dpi=220,bbox_inches='tight');plt.close(fig)
with (root/'task_metrics_all_rates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allmetrics[0]));w.writeheader();w.writerows(allmetrics)
(root/'audit.json').write_text(json.dumps(checks,indent=2))
print('Verified all 120,880 rows, common task mapping, and all 16 detection outcomes.')
for rate in rates:
 print('POISONING',rate,[(m['task'],None if m['auroc_low'] is None else round(m['auroc_low'],3)) for m in allmetrics if m['poisoning_percent_nominal']==rate and m['scope']=='all_rows'])
