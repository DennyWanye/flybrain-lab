import copy
import numpy as np
import pytest
from flydrone.tellosim.training.c0_campaign import expert,training_case,make_splits,acceptance,Budget,BudgetExceeded,wilson
from flydrone.tellosim.training.contracts import digest


def test_expert_four_directions_and_success_boundary():
    x=np.zeros(26);x[12]=1;mask=np.array([True]*5+[False]*4)
    for error,action in [([.3,0],1),([-.3,0],2),([0,.3],3),([0,-.3],4),([.11,.11],0),([.16,.16],1)]:
        x[:2]=np.array(error)/6;assert expert(x,mask)==action
    assert expert(x,mask,first=True)==0
    x[12]=0;assert expert(x,mask)==0


def test_c0_cases_ranges_and_train_validation_separation(tmp_path):
    splits=make_splits(tmp_path);assert splits==make_splits(tmp_path)
    held={c['seed'] for key in ('validation','sealed_test') for c in splits[key]}
    assert len(held)==400
    for c in splits['validation']+splits['sealed_test']:
        assert .6<=np.linalg.norm(np.array(c['goal'])-c['start'])<=2.5
        assert np.max(np.abs(c['start'][:2]))<=2 and np.max(np.abs(c['goal'][:2]))<=2
    for seed in (11,22,33):
        for stage in range(4):
            cases=[training_case(3000000+seed*10000+stage*1000+i,stage==0) for i in range(192)]
            assert not {c['seed'] for c in cases}&held


def result(successes=280,collisions=0):
    return {'mode':'argmax','case_hash':'fixed','results':[{'case_id':f'c{i}','success':i<successes,
        'reason':'success' if i<successes else ('collision' if i<successes+collisions else 'task_deadline')} for i in range(300)]}


def test_acceptance_rejects_incomplete_best_seed_and_mismatched_cases():
    models={s:result() for s in (11,22,33)};random=result(20);random['mode']='random'
    assert acceptance(models,random,'fixed')['passed']
    bad=copy.deepcopy(models);bad[22]=result(269)
    assert not acceptance(bad,random,'fixed')['passed']
    bad=copy.deepcopy(models);bad[33]=result(280,4)
    assert not acceptance(bad,random,'fixed')['passed']
    assert not acceptance({11:result()},random,'fixed')['passed']
    bad=copy.deepcopy(models);bad[33]['results'].pop()
    assert not acceptance(bad,random,'fixed')['passed']
    bad=copy.deepcopy(models);bad[33]['results'].reverse()
    assert not acceptance(bad,random,'fixed')['passed']
    strong_random=result(230);strong_random['mode']='random'
    assert not acceptance(models,strong_random,'fixed')['passed']
    bad=copy.deepcopy(models);bad[11]['mode']='rule'
    assert not acceptance(bad,random,'fixed')['passed']


def test_budget_counts_executed_options_and_wilson_bounds():
    b=Budget(options=2);b.take();b.take()
    with pytest.raises(BudgetExceeded):b.take()
    assert b.used==2
    lo,hi=wilson(90,100);assert .82<lo<.84 and .94<hi<.95
