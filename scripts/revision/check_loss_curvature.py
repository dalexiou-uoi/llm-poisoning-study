"""Paired final-token alignment and reference-covariance sensitivity check.
No training. The full reference covariance is NOT full EK-FAC or a training Hessian.
"""
import argparse,csv,hashlib,json,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def stable_low(x,k):return np.argsort(x,kind='stable')[:k]
def detection(flags,selected):
 tp=int(flags[selected].sum());fp=len(selected)-tp;p=int(flags.sum());clean=len(flags)-p
 return dict(removed=len(selected),tp=tp,fp=fp,precision=tp/len(selected),recall=tp/p,fdr=fp/len(selected),fpr=fp/clean)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path('/home/user1/PycharmProjects/Llama3/.venv/lib'))
 ap.add_argument('--historical-csv',type=Path,default=Path('/home/user1/reviewer_figures/task_scores.csv'))
 ap.add_argument('--pilot',type=int,default=64)
 ap.add_argument('--full',action='store_true')
 args=ap.parse_args()
 import torch,transformers,peft
 from datasets import load_from_disk
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(42);np.random.seed(42)
 root=args.root;dataset_path=root/'train_data_step4.1_poisoned_task'
 ds=load_from_disk(str(dataset_path));n=len(ds)
 flags=np.array(ds['poison'],dtype=np.int64)
 assert n==30220 and flags.sum()==604
 with args.historical_csv.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 rows.sort(key=lambda r:int(r['scored_index']))
 assert len(rows)==n and [int(r['scored_index']) for r in rows]==list(range(n))
 assert np.array_equal(flags,np.array([int(r['poison']) for r in rows]))
 old=np.array([float(r['influence_score']) for r in rows],dtype=np.float64)
 assert np.isfinite(old).all() and flags[stable_low(old,604)].sum()==330
 refs=np.flatnonzero(flags)[:10].tolist()
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 out=root/'evaluation_runs'/('loss_covariance_'+('full_' if args.full else 'pilot_')+stamp)
 out.mkdir(parents=True,exist_ok=False)
 protocol=dict(status='running',rows=n,references=refs,seed=42,max_length=256,damping=1.0,batch_size=1,
  dataset=str(dataset_path),adapter=str(root/'outputs/llama3_lora_poisoned_2pct_01'),
  historical_csv=str(args.historical_csv),historical_csv_sha256=hashlib.sha256(args.historical_csv.read_bytes()).hexdigest(),
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  loss_old='final nonpadding logits against token at same position',
  loss_corrected='previous-position logits against final nonpadding token; NOT mean sequence loss',
  comparison='same adapter, references, candidate gradients, damping; diagonal vs full centered reference-gradient covariance',
  scope='reference-covariance sensitivity only; not full EK-FAC or training Hessian; no recovery ASR/DSR',
  versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
  gpu=torch.cuda.get_device_name(0),full=args.full)
 def save_protocol(): (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 save_protocol();print('OUTPUT:',out,flush=True)
 try:
  started=time.monotonic()
  tok=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',use_fast=True)
  if tok.pad_token is None:tok.pad_token=tok.eos_token
  tok.padding_side='right'
  base=AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct',quantization_config=BitsAndBytesConfig(load_in_8bit=True),device_map='auto')
  model=PeftModel.from_pretrained(base,protocol['adapter'])
  for name,param in model.named_parameters():param.requires_grad_('lora_' in name.lower())
  model.eval();params=[p for p in model.parameters() if p.requires_grad]
  device=model.get_input_embeddings().weight.device
  protocol['trainable_parameters']=sum(p.numel() for p in params)
  protocol['parameter_names']=[name for name,p in model.named_parameters() if p.requires_grad]
  def gradient(i,shifted=True):
   model.zero_grad(set_to_none=True)
   enc=tok(ds[int(i)]['text'],truncation=True,padding='max_length',max_length=256,return_tensors='pt').to(device)
   pos=int(enc['attention_mask'][0].sum().item())-1
   assert pos>=1
   logits=model(input_ids=enc['input_ids'],attention_mask=enc['attention_mask']).logits
   loss=torch.nn.functional.cross_entropy(logits[:,pos-1 if shifted else pos,:],enc['input_ids'][:,pos])
   gs=torch.autograd.grad(loss,params,allow_unused=True)
   return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)]).float().detach()
  def diagonal(gradients):
   v=gradients.mean(0);h=(gradients.square().mean(0)-v.square()).clamp_min(0)+1
   return v,v/h
  print('Computing historical references...',flush=True)
  G0=torch.stack([gradient(i,False) for i in refs]);_,old_direction=diagonal(G0);del G0
  print('Computing shifted references...',flush=True)
  G=torch.stack([gradient(i,True) for i in refs]);v,diag_direction=diagonal(G)
  # Woodbury: (I + U.T U)^-1 v, U=(G-mean(G))/sqrt(m).
  # Only a 10x10 solve is needed; the D x D matrix is never allocated.
  U=(G.double()-v.double())/(len(refs)**.5);del G
  gram=U@U.T
  rhs=U@v.double();small=torch.eye(len(refs),dtype=torch.float64,device=device)+gram
  coefficients=torch.linalg.solve(small,rhs)
  cov64=v.double()-U.T@coefficients
  residual=float(torch.linalg.vector_norm(cov64+U.T@(U@cov64)-v.double())/torch.linalg.vector_norm(v.double()).clamp_min(1e-30))
  if not np.isfinite(residual) or residual>1e-5:raise RuntimeError(f'Covariance solve residual too large: {residual}')
  cov_direction=cov64.float();del cov64,U,gram,small,rhs,coefficients
  directions={'shifted_diagonal':diag_direction,'shifted_first_order':v,'shifted_reference_covariance':cov_direction}
  indices=np.sort(np.random.default_rng(42).choice(n,size=min(args.pilot,n),replace=False))
  old_check=[];pilot={name:[] for name in directions}
  print('Pilot: historical alignment...',flush=True)
  for j,i in enumerate(indices):
   g=gradient(i,False);old_check.append(-torch.dot(g,old_direction).item())
   if (j+1)%16==0:print(f'Historical check: {j+1}/{len(indices)}',flush=True)
  del old_direction
  torch.cuda.synchronize();pilot_started=time.monotonic()
  for j,i in enumerate(indices):
   g=gradient(i,True)
   for name,d in directions.items():pilot[name].append(-torch.dot(g,d).item())
   if (j+1)%16==0:print(f'Shifted check: {j+1}/{len(indices)}',flush=True)
  torch.cuda.synchronize();seconds_per_example=(time.monotonic()-pilot_started)/len(indices)
  agreement={'pearson':float(np.corrcoef(old[indices],old_check)[0,1]),'max_absolute_error':float(np.max(np.abs(old[indices]-old_check)))}
  report=dict(pilot_examples=len(indices),historical_agreement=agreement,covariance_relative_residual=residual,
   measured_seconds_per_example=seconds_per_example,estimated_full_scoring_hours=seconds_per_example*n/3600,
   estimate_excludes_model_loading_references_and_interruptions=True,
   paired_correlations={name:float(np.corrcoef(pilot['shifted_diagonal'],a)[0,1]) for name,a in pilot.items()})
  if not all(np.isfinite(a).all() for a in pilot.values()):raise RuntimeError('Nonfinite pilot scores')
  np.savez(out/'pilot_scores.npz',indices=indices,historical_saved=old[indices],historical_recomputed=np.array(old_check),**{k:np.array(a) for k,a in pilot.items()})
  (out/'pilot_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
  if args.full:
   # Full run is explicit; pilot estimates alone do not launch it.
   if agreement['pearson']<.999:raise RuntimeError('Historical agreement needs review before full scoring')
   arrays={name:np.full(n,np.nan,dtype=np.float32) for name in directions}
   begin=time.monotonic()
   with (out/'scores_partial.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['index','poison','historical_saved']+list(directions))
    for i in range(n):
     g=gradient(i,True)
     for name,d in directions.items():arrays[name][i]=-torch.dot(g,d).item()
     if not all(np.isfinite(a[i]) for a in arrays.values()):raise RuntimeError(f'Nonfinite score {i}')
     w.writerow([i,int(flags[i]),float(old[i])]+[float(a[i]) for a in arrays.values()])
     if (i+1)%200==0 or i+1==n:
      f.flush();np.savez(out/'scores_progress.npz',completed=i+1,poison=flags,**arrays)
      print(f'{i+1}/{n}; ETA {(time.monotonic()-begin)/(i+1)*(n-i-1)/60:.1f} min',flush=True)
   np.savez(out/'scores.npz',poison=flags,reference_indices=np.array(refs),historical_saved=old,**arrays)
   result=[]
   for name,a in {'historical_saved':old,**arrays}.items():
    for budget in [1,2,3,5]:result.append(dict(method=name,removal_percent=budget,**detection(flags,stable_low(a,int(n*budget/100)))))
   with (out/'comparison.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
   for r in result:print(r,flush=True)
   overlaps={}
   for name,a in arrays.items():
    A=set(stable_low(old,604));B=set(stable_low(a,604));overlaps[name]=dict(common=len(A&B),jaccard=len(A&B)/len(A|B))
   (out/'overlap_2pct.json').write_text(json.dumps(overlaps,indent=2)+'\n')
  protocol.update(status='completed',elapsed_seconds=time.monotonic()-started,pilot=report);save_protocol()
  print('COMPLETED:',out,flush=True)
 except BaseException as e:
  protocol.update(status='failed',error=f'{type(e).__name__}: {e}');save_protocol();raise
if __name__=='__main__':main()
