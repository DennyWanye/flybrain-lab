"""Focused training semantics; no learned-model readiness claims."""
from types import SimpleNamespace
import math
import numpy as np
import torch
from flydrone.policy import ActorCritic
from flydrone.tellosim.training.stability_campaign import label_for,training_case,fit
from flydrone.tellosim.training.runtime import distribution
from flydrone.tellosim.training.robust_env import PROFILES

def test_phase_bootstrap_is_labeled_stop_and_heading_uses_target_error():
    obs=np.zeros(26);obs[12:15]=1;obs[0]=.1;obs[6:8]=[1,0]
    env=SimpleNamespace(phase='heading',previous_action=None,observation=obs,mask=np.ones(9,bool))
    assert label_for(env)==0
    env.previous_action=0
    assert label_for(env)==8
    env.observation[6:8]=[-1,0]
    assert label_for(env)==7
    env.observation[6:8]=[math.sin(.1),math.cos(.1)]
    assert label_for(env)==0
    env.phase='navigation'
    assert label_for(env)==1

def test_training_cases_balance_disturbance_and_keep_joint_instruction():
    cases=[training_case(151100000+i*500,i) for i in range(16)]
    assert all(c['curriculum']=='J1' and 'target_yaw_rad' in c for c in cases)
    assert all(c['start'][2]==c['goal'][2]==1. for c in cases)
    assert all(sum(c['disturbance_profile']==p for c in cases)==4 for p in PROFILES)
    assert all(np.linalg.norm(np.array(c['goal'])[:2]-c['start'][:2])<=.55 for c in cases[:8])

def test_dense_fit_updates_readout_without_requiring_physics_or_raw_observations():
    torch.set_num_threads(2);torch.manual_seed(9)
    model=ActorCritic(128,9);opt=torch.optim.AdamW(list(model.body.parameters())+list(model.actor.parameters()),lr=.01)
    critic=[p.detach().clone() for p in model.critic.parameters()]
    rows=[]
    for label in [0,1,2]:
        for k in range(8):
            x=np.zeros(128,np.float32);x[label]=1
            mask=np.zeros(9,bool);mask[:3]=True;rows.append((x,mask,label,.25 if k%2 else 1.))
    metrics=fit(model,opt,rows,20)
    assert metrics['label_accuracy']==1.
    assert all(torch.equal(a,b) for a,b in zip(critic,model.critic.parameters()))
    with torch.no_grad():pi,_=distribution(model,torch.tensor(np.stack([r[0] for r in rows])),np.stack([r[1] for r in rows]))
    assert torch.all(pi.probs[:,3:]==0)
