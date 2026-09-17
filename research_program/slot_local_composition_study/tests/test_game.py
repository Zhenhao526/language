from __future__ import annotations
import numpy as np
from research_program.action_dependent_signaling_study import policy as ap
from research_program.slot_local_composition_study import design,environment,policy,runner

def main():
 p=ap.make_policy(99001,design.FORM); ep=design.episode_stream(99001,64,update=1,support=design.SUPPORT,heldout=design.heldout_goal(99001)); p2=policy.clone(p); p2["worker_logits_local"]=policy.make_slot_local(199001)
 for rep in design.REPRESENTATIONS:
  tr=runner.rollout(p2,ep,rep,"live",sample=False); assert tr["actions"].shape==(64,design.HORIZON); assert tr["policy_state"][2].shape==(64,)
  g=runner.gradient(p2,ep,tr,rep,design.entropy_coefficient(1)); assert all(np.isfinite(x).all() for x in g.values())
 held=design.heldout_goal(99001); ep2=design.episode_stream(99001,512,update=1,support="leave_one_out",heldout=held); assert not np.any(ap.design.goal_index(ep2["goal"])==held)
 print("slot_local_composition_study tests passed")
if __name__=="__main__": main()
