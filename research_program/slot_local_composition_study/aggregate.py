"""Aggregate representation-intervention runs."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
T95={31:2.039513}
def ci(v):
 v=[float(x) for x in v]
 if not v:return [None,None]
 if len(v)==1:return [v[0],v[0]]
 m=sum(v)/len(v); h=T95.get(len(v)-1,1.96)*statistics.stdev(v)/math.sqrt(len(v)); return [m-h,m+h]
def load(path):
 out={}
 for r in json.loads(Path(path).read_text())["results"]:
  k=(int(r["seed"]),r["condition"])
  if k in out: raise ValueError(k)
  out[k]=r
 return out
def value(r,kind="heldout",mode="natural"): return float(r["final"][kind][mode]["team_return_mean"])
def aggregate(by):
 seeds=sorted({s for s,_ in by}); rows=[]
 for (s,c),r in sorted(by.items()):
  rep,ch=c.rsplit("_",1)
  for kind in ("all","seen","heldout"):
   for mode in ("natural","permuted"): rows.append({"seed":s,"condition":c,"representation":rep,"channel":ch,"goal_kind":kind,"mode":mode,"return":value(r,kind,mode)})
 effects={}
 for rep in ("history","slot_local"):
  for kind in ("all","seen","heldout"):
   vals=[value(by[(s,f"{rep}_live")],kind)-value(by[(s,f"{rep}_silent")],kind) for s in seeds]
   effects[f"live_minus_silent|{rep}|{kind}"]={"values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
  vals=[value(by[(s,f"{rep}_live")],"heldout") for s in seeds]
  effects[f"zero_shot|{rep}"]={"values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals),"passes":int(sum(x>=.60 for x in vals)),"n":len(vals)}
 for ch in ("live","silent"):
  vals=[value(by[(s,f"slot_local_{ch}")],"heldout")-value(by[(s,f"history_{ch}")],"heldout") for s in seeds]
  effects[f"slot_local_minus_history|{ch}|heldout"]={"values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
 vals=[
  (value(by[(s,"slot_local_live")],"heldout")-
   value(by[(s,"slot_local_silent")],"heldout")) -
  (value(by[(s,"history_live")],"heldout")-
   value(by[(s,"history_silent")],"heldout"))
  for s in seeds
 ]
 effects["representation_interaction|heldout"]={"values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
 return {"schema":"slot_local_composition_aggregate_v1","runs":len(by),"rows":rows,"effects":effects}
def write_md(path,data):
 lines=["# Slot-local receiver representation aggregation","",f"- runs: {data['runs']}","- split: heldout goal combination","","| representation | live heldout | silent heldout | live−silent | zero-shot passes |","|---|---:|---:|---:|---:|"]
 for rep in ("history","slot_local"):
  live=data["effects"][f"zero_shot|{rep}"]; ls=data["effects"][f"live_minus_silent|{rep}|heldout"]
  lines.append(f"| `{rep}` | {live['mean']:.3f} | {float(np.mean([x['return'] for x in data['rows'] if x['representation']==rep and x['channel']=='silent' and x['goal_kind']=='heldout' and x['mode']=='natural'])):.3f} | {ls['mean']:+.3f} [{ls['ci95_t'][0]:+.3f},{ls['ci95_t'][1]:+.3f}] | {live['passes']}/{live['n']} |")
 lines += ["","| contrast | mean | 95% CI |","|---|---:|---:|"]
 for key in ("slot_local_minus_history|live|heldout","representation_interaction|heldout"):
  x=data["effects"][key]; lines.append(f"| `{key}` | {x['mean']:+.3f} | [{x['ci95_t'][0]:+.3f},{x['ci95_t'][1]:+.3f}] |")
 lines += ["","The slot-local arm is an architectural representation intervention. It tests whether staged action timing is sufficient to induce factor-wise receiver states; it does not supply a semantic dictionary."]
 Path(path).write_text("\n".join(lines)+"\n",encoding="utf8")
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); ap.add_argument("--out",required=True); ap.add_argument("--markdown",required=True); a=ap.parse_args(); d=aggregate(load(a.results)); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n"); write_md(a.markdown,d); print(json.dumps({"status":"written","runs":d["runs"]},ensure_ascii=False))
if __name__=="__main__": main()
