"""Frozen design for a receiver-state representation intervention."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.heldout_composition_study import design as base
from research_program.action_dependent_signaling_study import design as action

HORIZON=base.HORIZON; ACTION_START=base.ACTION_START; SUBTASKS=base.SUBTASKS; STEPS_PER_SUBTASK=base.STEPS_PER_SUBTASK
FORM=base.FORM; PARTNER_MODE=base.PARTNER_MODE; VISIBILITY=base.VISIBILITY; TASK=base.TASK; PROTOCOL=base.PROTOCOL
WORKERS=base.WORKERS; CAPACITY=base.CAPACITY; MESSAGE_SLOTS=base.MESSAGE_SLOTS; NULL_MESSAGE=base.NULL_MESSAGE; ACTION_COUNT=base.ACTION_COUNT
UPDATES=3000; BATCH_SIZE=512; LEARNING_RATE=base.LEARNING_RATE; ENTROPY_INITIAL=base.ENTROPY_INITIAL; ENTROPY_ZERO_AFTER=base.ENTROPY_ZERO_AFTER
SEEDS=base.SEEDS; SUPPORT="leave_one_out"; REPRESENTATIONS=("history","slot_local"); CHANNELS=("live","silent")
CONDITIONS=tuple(f"{representation}_{channel}" for representation in REPRESENTATIONS for channel in CHANNELS)
CHECKPOINTS=base.CHECKPOINTS; TARGET_WORKER=base.TARGET_WORKER; PARENT_CONDITION=base.PARENT_CONDITION; EVAL_SEED_OFFSET=base.EVAL_SEED_OFFSET

def require(ok,msg):
    if not ok: raise ValueError(msg)
def parse_condition(condition):
    parts=condition.rsplit("_",1); require(len(parts)==2 and parts[0] in REPRESENTATIONS and parts[1] in CHANNELS,f"bad condition {condition}"); return parts[0],parts[1]
def entropy_coefficient(update): return base.entropy_coefficient(update)
def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def heldout_goal(seed): return base.heldout_goal(seed)
def episode_stream(seed,count,*,evaluation=False,update=0,support=SUPPORT,heldout=None): return base.episode_stream(seed,count,evaluation=evaluation,update=update,support=support,heldout=heldout)
def balanced_eval_stream(seed,count,goal_kind,heldout): return base.balanced_eval_stream(seed,count,goal_kind,heldout)
def prepare(parent_root,parent_hashes,parent_composable):
    return {"schema":"slot_local_composition_study_v1","horizon":HORIZON,"action_start":ACTION_START,"subtasks":SUBTASKS,"steps_per_subtask":STEPS_PER_SUBTASK,"form":FORM,"partner_mode":PARTNER_MODE,"visibility":VISIBILITY,"task":TASK,"protocol":PROTOCOL,"workers":WORKERS,"capacity":CAPACITY,"target_worker":TARGET_WORKER,"seeds":list(SEEDS),"support":SUPPORT,"representations":list(REPRESENTATIONS),"channels":list(CHANNELS),"conditions":list(CONDITIONS),"updates":UPDATES,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"entropy_initial":ENTROPY_INITIAL,"entropy_zero_after":ENTROPY_ZERO_AFTER,"checkpoints":list(CHECKPOINTS),"parent_condition":PARENT_CONDITION,"parent_root":str(parent_root),"parent_checkpoint_sha256":parent_hashes,"parent_composable":{str(k):bool(v) for k,v in parent_composable.items()},"heldout_assignment":{str(seed):heldout_goal(seed) for seed in SEEDS},"pairing":"history and slot_local, live and silent share worlds, goals, partners, messages and actions","intervention":"history uses complete staged message memory; slot_local reads slot 0 for subtask 0 and slot 1 for subtask 1","controls":["same parent sender","same replacement seed","same leave-one-goal-out support","from-scratch silent"],"zero_shot_rule":"heldout natural team return >= 0.60","no_teacher_or_language_prior":True}
