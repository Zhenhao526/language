"""Train and audit a receiver-state representation intervention."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, time
from pathlib import Path
import numpy as np
from . import design, environment, policy
from research_program.action_dependent_signaling_study import design as action_design
from research_program.action_dependent_signaling_study import policy as action_policy
from research_program.heldout_composition_study import design as holdout_design

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()
def finite(x):
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,(float,int,np.number)): return bool(np.isfinite(float(x)))
    return True

def source_hashes():
    rels=("__init__.py","design.py","policy.py","environment.py","runner.py","aggregate.py","audit.py","plan.md","tests/test_game.py")
    return {str((HERE/x).relative_to(ROOT)):sha(HERE/x) for x in rels}
def dependency_hashes():
    rels=("research_program/action_dependent_signaling_study/design.py","research_program/action_dependent_signaling_study/policy.py","research_program/heldout_composition_study/design.py")
    return {x:sha(ROOT/x) for x in rels}
def parent_path(parent_root,seed): return Path(parent_root)/f"seed_{seed}_{design.PARENT_CONDITION}"/f"checkpoint_{design.CHECKPOINTS[-1]:04d}.npz"
def parent_hashes(parent_root):
    out={}
    for seed in design.SEEDS:
        p=parent_path(parent_root,seed); design.require(p.is_file(),f"missing parent checkpoint {p}"); out[str(seed)]=sha(p)
    return out
def parent_composable_from_analysis(path):
    payload=json.loads(Path(path).read_text()); rows=[r for r in payload["rows"] if r.get("role")=="worker" and r.get("channel")=="live"]; out={}
    for r in rows:
        s=int(r["seed"]); v=bool(r["parent_composable"])
        if s in out and out[s]!=v: raise ValueError(f"inconsistent parent status {s}")
        out[s]=v
    design.require(set(out)==set(design.SEEDS),"parent analysis does not cover all seeds"); return out

def prepare(out,parent_root,parent_analysis):
    out=Path(out).resolve(); design.require(not out.exists(),"refuse overwrite"); parent_root=Path(parent_root).resolve(); ph=parent_hashes(parent_root); status=parent_composable_from_analysis(parent_analysis); cfg=design.prepare(parent_root,ph,status); cfg.update({"runs":len(design.SEEDS)*len(design.CONDITIONS),"evaluation_episodes_per_goal":1024})
    src=source_hashes(); deps=dependency_hashes(); out.mkdir(parents=True)
    for rel in src:
        q=out/"source_snapshot"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    for rel in deps:
        q=out/"dependency_snapshot"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    shutil.copy2(parent_analysis,out/"parent_analysis.json")
    (out/"prepared.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
    plan={"schema":"slot_local_composition_study_v1","prepared_sha256":sha(out/"prepared.json"),"parent_analysis_sha256":sha(out/"parent_analysis.json"),"sources":src,"dependencies":deps,"runtime":{"python":platform.python_version(),"numpy":np.__version__},"config":cfg}
    (out/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
    (out/"freeze.json").write_text(json.dumps({"plan_sha256":sha(out/"plan.json"),"prepared_sha256":sha(out/"prepared.json"),"parent_analysis_sha256":sha(out/"parent_analysis.json")},indent=2)+"\n")
    verify(out); return {"status":"prepared","out":str(out),"plan_sha256":sha(out/"plan.json"),"prepared_sha256":sha(out/"prepared.json")}

def verify(out):
    out=Path(out); plan=json.loads((out/"plan.json").read_text()); cfg=json.loads((out/"prepared.json").read_text()); fr=json.loads((out/"freeze.json").read_text())
    design.require(sha(out/"plan.json")==fr["plan_sha256"] and sha(out/"prepared.json")==fr["prepared_sha256"]==plan["prepared_sha256"],"freeze hash mismatch")
    design.require(sha(out/"parent_analysis.json")==fr["parent_analysis_sha256"]==plan["parent_analysis_sha256"],"parent analysis changed")
    design.require(plan["config"]==cfg and plan["sources"]==source_hashes() and plan["dependencies"]==dependency_hashes(),"source or config changed")
    for rel,digest in plan["sources"].items(): design.require(sha(out/"source_snapshot"/rel)==digest,"source snapshot changed "+rel)
    for rel,digest in plan["dependencies"].items(): design.require(sha(out/"dependency_snapshot"/rel)==digest,"dependency snapshot changed "+rel)
    design.require(parent_hashes(cfg["parent_root"])==cfg["parent_checkpoint_sha256"],"parent checkpoint changed"); return plan,cfg

def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]).copy() for k in ("sender_logits_hidden","sender_logits_visible","worker_logits")}
def future_returns(rewards): return np.flip(np.cumsum(np.flip(rewards,axis=1),axis=1),axis=1)
def center(values,keys):
    values=np.asarray(values,dtype=np.float64); keys=np.asarray(keys); out=np.zeros_like(values)
    for key in np.unique(keys):
        idx=np.flatnonzero(keys==key)
        if len(idx)>1: out[idx]=values[idx]-(values[idx].sum()-values[idx])/(len(idx)-1)
    return out
def entropy_grad(prob):
    lp=np.log(np.maximum(prob,1e-300)); ent=-(prob*lp).sum(axis=-1,keepdims=True); return -prob*(lp+ent)

def rollout(p,episode,representation,channel,*,sample,message_mode="natural"):
    return environment.rollout(p,episode,design.FORM,design.PARTNER_MODE,design.VISIBILITY,design.TASK,channel,design.PROTOCOL,representation=representation,message_mode=message_mode,sample=sample,partner_filter=design.TARGET_WORKER)

def gradient(p,episode,tr,representation,beta):
    g={k:np.zeros_like(v) for k,v in p.items()}; idx=np.flatnonzero(tr["active"])
    if len(idx)==0: return g
    future=future_returns(tr["rewards"]); worker=np.zeros(len(idx),dtype=np.int64)
    for t in range(design.ACTION_START,design.HORIZON):
        ap=tr["action_probs"][t][idx] if "action_probs" in tr else tr["action_probs"][t][idx]
        state=tr["policy_state"][t][idx]; local=tr["local"][t][idx]; inv=tr["inventory"][t][idx]
        keys=state*1000+t*100+local*10+inv; adv=center(future[idx,t],keys); one=np.zeros_like(ap); one[np.arange(len(idx)),tr["actions"][idx,t]]=1.0
        d=-(adv[:,None]*(one-ap))/len(idx)-beta*entropy_grad(ap)/len(idx)
        if representation=="slot_local": np.add.at(g["worker_logits_local"],(worker,state,np.full(len(idx),t),local,inv),d)
        else: np.add.at(g["worker_logits"],(worker,state,np.full(len(idx),t),local,inv),d)
    return g

def metrics(p,episode,representation,channel,message_mode="natural"):
    tr=rollout(p,episode,representation,channel,sample=False,message_mode=message_mode); mask=tr["active"]; r=tr["team_return"][mask]
    return {"episodes":int(mask.sum()),"team_return_mean":float(r.mean()) if len(r) else None,"team_return_sd":float(r.std()) if len(r) else None,"positive_episode_rate":float((r>0).mean()) if len(r) else None}
def evaluate(p,seed,heldout,representation,channel):
    out={}
    for kind in ("all","seen","heldout"):
        ep=design.balanced_eval_stream(seed+design.EVAL_SEED_OFFSET,4096,kind,heldout)
        out[kind]={"natural":metrics(p,ep,representation,channel,"natural"),"permuted":metrics(p,ep,representation,channel,"permuted")}
    return out

def train_one(seed,condition,execution,parent_checkpoint,updates=None):
    representation,channel=design.parse_condition(condition); updates=design.UPDATES if updates is None else int(updates); run=Path(execution)/f"seed_{seed}_{condition}"; run.mkdir(parents=True,exist_ok=False)
    parent=load_checkpoint(parent_checkpoint); params=policy.clone(parent)
    if representation=="history": params["worker_logits"][design.TARGET_WORKER]=action_policy.make_policy(seed+190000,design.FORM)["worker_logits"][0]
    else: params["worker_logits_local"]=policy.make_slot_local(seed+190000)
    initial=policy.combined_parameter_hash(params); checkpoints=sorted(set([u for u in design.CHECKPOINTS if u<=updates]+[updates])); rows=[]; start=time.perf_counter()
    def save(path,update):
        arrays={"update":np.array(update,dtype=np.int64),**params}; np.savez_compressed(path,**arrays); return sha(path)
    if 0 in checkpoints: save(run/"checkpoint_0000.npz",0)
    for update in range(1,updates+1):
        heldout=design.heldout_goal(seed); ep=design.episode_stream(seed,design.BATCH_SIZE,evaluation=False,update=update,support=design.SUPPORT,heldout=heldout); tr=rollout(params,ep,representation,channel,sample=True); beta=design.entropy_coefficient(update); g=gradient(params,ep,tr,representation,beta); norm=float(np.sqrt(sum(float((v*v).sum()) for v in g.values()))); scale=min(1.0,5.0/max(norm,1e-12))
        key="worker_logits_local" if representation=="slot_local" else "worker_logits"; params[key][0 if representation=="slot_local" else design.TARGET_WORKER]-=design.LEARNING_RATE*scale*g[key][0 if representation=="slot_local" else design.TARGET_WORKER]
        row={"update":update,"seed":seed,"condition":condition,"representation":representation,"channel":channel,"heldout_goal":heldout,"world_sha256":design.array_sha(ep["site_type"]),"goal_sha256":design.array_sha(ep["goal"]),"partner_sha256":design.array_sha(ep["partner_id"]),"message_uniform_sha256":design.array_sha(ep["message_uniforms"]),"action_uniform_sha256":design.array_sha(ep["action_uniforms"]),"return_mean":float(tr["team_return"][tr["active"]].mean()),"active_count":int(tr["active"].sum()),"gradient_norm":norm,"gradient_clip_scale":scale,"parameter_sha256":policy.combined_parameter_hash(params),"elapsed_seconds":time.perf_counter()-start}
        if update in checkpoints: row["checkpoint_sha256"]=save(run/f"checkpoint_{update:04d}.npz",update)
        rows.append(row)
    log=run/"training.jsonl"; log.write_bytes(b"".join(json_bytes(x) for x in rows)); result={"seed":seed,"condition":condition,"representation":representation,"channel":channel,"heldout_goal":design.heldout_goal(seed),"parent_checkpoint":str(parent_checkpoint),"parent_checkpoint_sha256":sha(parent_checkpoint),"initial_parameter_sha256":initial,"updates":updates,"final":evaluate(params,seed,design.heldout_goal(seed),representation,channel),"checkpoints":checkpoints,"trajectory":rows,"training_log_sha256":sha(log),"final_parameter_sha256":policy.combined_parameter_hash(params)}; (run/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); return result

def run_grid(prepared,out,updates=None,seeds=None,conditions=None):
    _,cfg=verify(prepared); out=Path(out); execution=out/"execution"; execution.mkdir(parents=True); seeds=tuple(design.SEEDS if seeds is None else seeds); conditions=tuple(design.CONDITIONS if conditions is None else conditions); design.require(set(seeds)<=set(design.SEEDS) and set(conditions)<=set(design.CONDITIONS),"invalid subset"); results=[]
    for seed in seeds:
        for condition in conditions:
            results.append(train_one(seed,condition,execution,parent_path(cfg["parent_root"],seed),updates)); (execution/"progress.json").write_text(json.dumps({"completed":len(results),"total":len(seeds)*len(conditions),"seeds":list(seeds),"conditions":list(conditions)},indent=2))
    (execution/"results.json").write_text(json.dumps({"results":results},ensure_ascii=False,indent=2)+"\n"); return results

if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); a=sub.add_parser("prepare"); a.add_argument("--out",required=True); a.add_argument("--parent-root",required=True); a.add_argument("--parent-analysis",required=True); a=sub.add_parser("execute"); a.add_argument("--out",required=True); a.add_argument("--prepared",required=True); a.add_argument("--updates",type=int,default=None); a.add_argument("--seeds"); a.add_argument("--conditions"); args=ap.parse_args()
    if args.cmd=="prepare": print(json.dumps(prepare(args.out,args.parent_root,args.parent_analysis),ensure_ascii=False))
    else:
        seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x); conditions=None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x); print(json.dumps({"status":"completed","runs":len(run_grid(args.prepared,args.out,args.updates,seeds,conditions))},ensure_ascii=False))
