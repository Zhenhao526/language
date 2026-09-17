"""Policy helpers for the representation intervention."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.action_dependent_signaling_study import policy as base
from research_program.action_dependent_signaling_study import design

def make_slot_local(seed):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),719,design.alphabet_size("dual2"),design.message_length("dual2"),7]))
    return rng.normal(0.0,0.02,size=(1,design.alphabet_size("dual2")+1,design.HORIZON,2,2*design.CAPACITY+1,design.ACTION_COUNT))
def clone(p): return {k:np.asarray(v).copy() for k,v in p.items()}
def combined_parameter_hash(p):
    h=hashlib.sha256()
    for key in ("sender_logits_hidden","sender_logits_visible","worker_logits","worker_logits_local"):
        if key in p: h.update(key.encode()); h.update(np.asarray(p[key],dtype=np.float64).tobytes())
    return h.hexdigest()
softmax=base.softmax; sample=base.sample
