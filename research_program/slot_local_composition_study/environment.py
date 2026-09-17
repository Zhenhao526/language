"""Staged dual-token environment with history or slot-local receiver readout."""
from __future__ import annotations
import numpy as np
from research_program.action_dependent_signaling_study import policy
from research_program.action_dependent_signaling_study import design as action

def require(ok,msg):
    if not ok: raise ValueError(msg)
def _permute_sequences(sequences,partner_id):
    out=sequences.copy()
    for worker in range(action.WORKERS):
        idx=np.flatnonzero(partner_id==worker)
        if len(idx)>1: out[idx]=sequences[np.roll(idx,1)]
    return out
def _schedule(delivered,protocol,form):
    n,length=delivered.shape; messages=np.full((n,action.HORIZON),action.NULL_MESSAGE,dtype=np.int64)
    for slot,t in enumerate(action.message_arrival_times(protocol,form)): messages[:,t]=delivered[:,slot]
    return messages

def rollout(p,ep,form,partner_mode,partner_visibility,task,channel,protocol,*,representation="history",message_mode="natural",sample=True,partner_filter=None):
    require(representation in ("history","slot_local"),"bad representation"); require(channel in action.CHANNELS and message_mode in ("natural","closed","permuted"),"bad channel/mode")
    n=len(ep["goal"]); active=np.ones(n,dtype=bool) if partner_filter is None else (ep["partner_id"]==int(partner_filter)); k=action.alphabet_size(form); length=action.message_length(form)
    remain=ep["capacity"].astype(np.int16).copy(); actions=np.zeros((n,action.HORIZON),dtype=np.int64); rewards=np.zeros((n,action.HORIZON),dtype=np.float64)
    msg_probs=[None]*length; action_probs=[None]*action.HORIZON; states=[None]*action.HORIZON; policy_states=[None]*action.HORIZON; locals_=[None]*action.HORIZON; inventories=[None]*action.HORIZON
    goal_idx=action.goal_index(ep["goal"]); selected=np.zeros((n,length),dtype=np.int64); sender=p["sender_logits_hidden"]
    for slot in range(length):
        sp=policy.softmax(sender[goal_idx,slot]); msg_probs[slot]=sp
        selected[:,slot]=policy.sample(sp,ep["message_uniforms"][:,slot]) if sample else sp.argmax(axis=-1)
    delivered=selected.copy()
    if channel=="silent" or message_mode=="closed": delivered=np.full((n,length),action.NULL_MESSAGE,dtype=np.int64)
    elif message_mode=="permuted": delivered=_permute_sequences(selected,ep["partner_id"])
    delivered=np.where(active[:,None],delivered,action.NULL_MESSAGE)
    messages=_schedule(delivered,protocol,form); worker_idx=ep["partner_id"].astype(np.int64); memory=np.zeros(n,dtype=np.int64); slot_memory=np.full((n,length),action.NULL_MESSAGE,dtype=np.int64); received=np.zeros(n,dtype=np.int8)
    arrival=action.message_arrival_times(protocol,form)
    for t in range(action.HORIZON):
        incoming=messages[:,t-1] if t>0 else np.full(n,action.NULL_MESSAGE,dtype=np.int64); seen=incoming!=action.NULL_MESSAGE
        if np.any(seen):
            incoming_slot=arrival.index(t-1) if (t-1) in arrival else None
            first=seen&(received==0); later=seen&~first
            memory[first]=incoming[first]+1
            if np.any(later): memory[later]=1+(memory[later]-1)*k+incoming[later]
            received[seen]+=1
            if incoming_slot is not None: slot_memory[seen,incoming_slot]=incoming[seen]
        stage=min(max((t-action.ACTION_START)//action.STEPS_PER_SUBTASK,0),action.SUBTASKS-1)
        local=ep["site_type"][:,stage,0].astype(np.int64); inv=remain[:,stage,:].sum(axis=1).astype(np.int64)
        if representation=="slot_local":
            ps=np.where((t>=action.ACTION_START)&(worker_idx==0),slot_memory[:,stage]+1,0).astype(np.int64)
            # Null is -1; +1 maps it to zero, token 0/1 to one/two.
            logits=p["worker_logits_local"][0,ps,t,local,inv]
        else:
            ps=memory.copy(); logits=p["worker_logits"][worker_idx,ps,t,local,inv]
        ap=policy.softmax(logits); action_probs[t]=ap; policy_states[t]=ps; states[t]=memory.copy(); locals_[t]=local.copy(); inventories[t]=inv.copy()
        act=policy.sample(ap,ep["action_uniforms"][:,t]) if sample else ap.argmax(axis=-1); act=np.where((t<action.ACTION_START)|(~active),0,act).astype(np.int64); actions[:,t]=act
        if t>=action.ACTION_START:
            for i in np.flatnonzero(active):
                a=int(act[i]);
                if a==0: continue
                site=a-1; avail=int(remain[i,stage,site])
                if avail>0: remain[i,stage,site]-=1
                target=int(ep["target_bits"][i,stage]); rewards[i,t]=action.CORRECT_REWARD if avail>0 and int(ep["site_type"][i,stage,site])==target else action.WRONG_REWARD
    return {"messages":messages,"selected_messages":selected,"actions":actions,"rewards":rewards,"team_return":rewards.mean(axis=1),"active":active,"message_probs":msg_probs,"action_probs":action_probs,"state":states,"policy_state":policy_states,"local":locals_,"inventory":inventories,"final_inventory":remain,"episodes":ep,"message_mode":message_mode,"representation":representation}

def oracle_team_return(ep): return np.full(len(ep["goal"]),action.SUBTASKS*action.STEPS_PER_SUBTASK*action.CORRECT_REWARD/action.HORIZON,dtype=np.float64)
