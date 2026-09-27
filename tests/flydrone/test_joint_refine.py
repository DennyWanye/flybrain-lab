"""Focused teacher, training and lineage checks; fixtures are not flight evidence."""
import json
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,pool
from flydrone.tellosim.training.joint_refine import settling_label,training_case,fit_settling,refined_bundle
from flydrone.tellosim.training.contracts import action_mask
import torch

def obs(distance):
    v=np.zeros(26,np.float32);v[0]=distance/6;v[12:15]=1;return v

def test_stop_labels_have_margin_and_invalid_input_never_moves():
    v=obs(.179);m=action_mask(v,'C1');assert settling_label(v,m)==0
    v=obs(.21);assert settling_label(v,m)==1
    v=obs(-.21);assert settling_label(v,m)==2
    assert settling_label(v,m,first=True)==0
    v[12]=0;assert settling_label(v,m)==0

def test_training_curriculum_uses_real_start_goal_without_teleport():
    for lane in range(8):
        c=training_case(91100000+lane*1000,lane)
        d=np.linalg.norm(np.array(c['goal'])[:2]-np.array(c['start'])[:2])
        assert (.06<=d<=.55) if lane%4<2 else (.6<=d<=2.5)
        assert c['curriculum']=='C1' and c['start'][2]==c['goal'][2]==1
        assert max(abs(x) for x in c['goal'][:2])<3

def test_supervised_fit_updates_readout_without_advancing_brain(project):
    p=pool(project,1);before={k:v.clone() for k,v in p.model.state_dict().items()};clock=p.brain.advance_count
    rng=np.random.default_rng(42);rows=[(rng.normal(size=128).astype(np.float32),np.array([1,1,1,1,1,0,0,1,1],bool),i%5,2. if i%5==0 else 1.) for i in range(20)]
    opt=torch.optim.Adam(list(p.model.body.parameters())+list(p.model.actor.parameters()),lr=.001)
    metrics=fit_settling(p,opt,rows,epochs=2)
    assert metrics['gradient_steps']==2 and p.brain.advance_count==clock
    assert any(not torch.equal(before[k],v) for k,v in p.model.state_dict().items())
