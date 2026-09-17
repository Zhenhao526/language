"""Independent replay audit for receiver representation runs."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from . import design,runner,policy

def sha(path): return runner.sha(path)
def finite(x):
 if isinstance(x,dict): return all(finite(v) for v in x.values())
 if isinstance(x,list): return all(finite(v) for v in x)
 if isinstance(x,(float,int,np.number)): return math.isfinite(float(x))
 return True
def load_params(path):
 with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]).copy() for k in z.files if k!="update"}
def init_params(parent,seed,rep):
 p=policy.clone(parent)
 if rep=="history": p["worker_logits"][design.TARGET_WORKER]=runner.action_policy.make_policy(seed+190000,design.FORM)["worker_logits"][0]
 else: p["worker_logits_local"]=policy.make_slot_local(seed+190000)
 return p
def audit(prepared,execution,out):
 runner.verify(prepared); execution=Path(execution); progress=json.loads((execution/"progress.json").read_text()); files=sorted(execution.glob("seed_*/result.json")); design.require(progress["completed"]==progress["total"]==len(files),"run accounting mismatch")
 seen=set(); logs=checkpoints=pair_rows=0; max_error=0.0; rows=[]
 for path in files:
  r=json.loads(path.read_text()); key=(int(r["seed"]),r["condition"]); design.require(key not in seen,"duplicate run"); seen.add(key); rep,ch=design.parse_condition(r["condition"]); seed=int(r["seed"]); parent_path=runner.parent_path(json.loads((Path(prepared)/"prepared.json").read_text())["parent_root"],seed); design.require(r["parent_checkpoint_sha256"]==sha(parent_path),"parent hash mismatch"); parent=runner.load_checkpoint(parent_path); p=init_params(parent,seed,rep); design.require(r["initial_parameter_sha256"]==policy.combined_parameter_hash(p),"initial parameter mismatch")
  log_path=path.parent/"training.jsonl"; lines=[json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]; design.require(len(lines)==r["updates"],"log length mismatch"); logs+=len(lines)
  for row in lines:
   u=int(row["update"]); ep=design.episode_stream(seed,design.BATCH_SIZE,evaluation=False,update=u,support=design.SUPPORT,heldout=design.heldout_goal(seed)); idx=runner.holdout_design.base.goal_index(ep["goal"])
   design.require(not np.any(idx==design.heldout_goal(seed)),"heldout goal leaked")
   tr=runner.rollout(p,ep,rep,ch,sample=True); g=runner.gradient(p,ep,tr,rep,design.entropy_coefficient(u)); norm=float(np.sqrt(sum(float((v*v).sum()) for v in g.values()))); scale=min(1.0,5.0/max(norm,1e-12)); max_error=max(max_error,abs(float(tr["team_return"][tr["active"]].mean())-float(row["return_mean"])),abs(norm-float(row["gradient_norm"])),abs(scale-float(row["gradient_clip_scale"]))); 
   for k in ("world_sha256","goal_sha256","partner_sha256","message_uniform_sha256","action_uniform_sha256"): design.require(row[k]==design.array_sha(ep["site_type"] if k=="world_sha256" else ep["goal"] if k=="goal_sha256" else ep["partner_id"] if k=="partner_sha256" else ep["message_uniforms"] if k=="message_uniform_sha256" else ep["action_uniforms"]),"stream mismatch")
   keyp="worker_logits_local" if rep=="slot_local" else "worker_logits"; p[keyp][0 if rep=="slot_local" else design.TARGET_WORKER]-=design.LEARNING_RATE*scale*g[keyp][0 if rep=="slot_local" else design.TARGET_WORKER]
   design.require(row["parameter_sha256"]==policy.combined_parameter_hash(p),"parameter mismatch")
   if u in r["checkpoints"]:
    cp=path.parent/f"checkpoint_{u:04d}.npz"; design.require(cp.is_file() and row.get("checkpoint_sha256")==sha(cp),"checkpoint receipt mismatch"); got=load_params(cp)
    for k in got: max_error=max(max_error,float(np.max(np.abs(got[k]-p[k]))))
    checkpoints+=1
  design.require(r["final_parameter_sha256"]==policy.combined_parameter_hash(p),"final parameter mismatch"); fresh=runner.evaluate(p,seed,design.heldout_goal(seed),rep,ch); 
  for kind in ("all","seen","heldout"):
   for mode in ("natural","permuted"):
    max_error=max(max_error,abs(float(fresh[kind][mode]["team_return_mean"])-float(r["final"][kind][mode]["team_return_mean"])))
  rows.append({"seed":seed,"condition":r["condition"],"representation":rep,"channel":ch,"heldout_goal":design.heldout_goal(seed),"heldout_natural":r["final"]["heldout"]["natural"]["team_return_mean"]});
 # Paired stream audit for each representation.  Smoke audits may contain
 # a strict subset of the frozen seed grid, so derive the audited keys from
 # the execution itself rather than assuming all 32 seeds are present.
 by={(x["seed"],x["condition"]):x for x in (json.loads(p.read_text()) for p in files)}
 audited_seeds=sorted({s for s,_ in by})
 for s in audited_seeds:
  for rep in design.REPRESENTATIONS:
   live=by[(s,f"{rep}_live")]["trajectory"]; silent=by[(s,f"{rep}_silent")]["trajectory"]; design.require(len(live)==len(silent),"pair length mismatch")
   for a,b in zip(live,silent):
    for k in ("world_sha256","goal_sha256","partner_sha256","message_uniform_sha256","action_uniform_sha256"): design.require(a[k]==b[k],"paired stream mismatch")
    pair_rows+=1
 design.require(len(seen)==progress["total"] and max_error<1e-12,"audit failure")
 report={"schema":"slot_local_composition_audit_v1","status":"passed","runs":len(files),"training_log_rows":logs,"final_checkpoints":checkpoints,"max_abs_replay_error":max_error,"paired_channel_trajectory_rows":pair_rows,"rows":rows}; Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({k:report[k] for k in ("schema","status","runs","training_log_rows","final_checkpoints","max_abs_replay_error","paired_channel_trajectory_rows")},ensure_ascii=False))
if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--prepared",required=True); ap.add_argument("--execution",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); audit(a.prepared,a.execution,a.out)
