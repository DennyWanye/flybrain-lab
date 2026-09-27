"""Corrective demonstrations on independent boundary/recovery training cases.

The deployed actor continues to consume reservoir features only. Training-time
teacher labels and deliberate pauses are never used in evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .runtime import ReservoirAgent,load_checkpoint,save_checkpoint
from .bootstrap import teacher,fit
from .critic_warmup import warmup
from .env import TrainingEnv,curriculum_case
from .contracts import digest
from ..visual import atomic_json


def boundary_case(seed,case_id=None,boundary=True):
    case=curriculum_case(seed,case_id,'C0-near-x')
    if boundary:
        rng=np.random.default_rng(seed+71)
        direction=np.sign(case['goal'][0]-case['start'][0])
        case['goal'][0]=case['start'][0]+float(direction*rng.uniform(.205,.265))
    case['cohort']='boundary' if boundary else 'near'
    return case


def validation_cases(recovery=False):
    cases=[curriculum_case(910000+i,f'validation-{i:03d}','C0-near-x') for i in range(16)]
    for case in cases:case['cohort']='previous16'
    cases += [boundary_case(1410000+i,f'fresh-near-{i:03d}',False) for i in range(8)]
    cases += [boundary_case(1420000+i,f'fresh-boundary-{i:03d}',True) for i in range(8)]
    if recovery:
        fresh=[boundary_case(1430000+i,f'recovery-check-{i:03d}',True) for i in range(8)]
        for case in fresh:case['cohort']='recovery_holdout'
        cases+=fresh
    return cases


def evaluate(root,agent,cases,prefix,label):
    results=[]
    for mode in ('argmax','sampled'):
        for index,case in enumerate(cases):
            torch.manual_seed(960000+index)
            record=mode=='argmax' and index in (5,24)
            env=TrainingEnv(root,case,agent,record=record,
                run_id=f'{prefix}-{label}-{index}' if record else None,policy_source=f'{label}_sdk9')
            try:
                while not env.terminated:
                    action,_,_,policy=agent.decision(env.features,env.mask,mode=='argmax')
                    env.step(action,policy)
                results.append({'case_id':case['case_id'],'seed':case['seed'],'cohort':case['cohort'],
                    'mode':mode,'success':env.reason=='success','reason':env.reason,'return':env.raw_return,
                    'duration_s':(env.session.tick-env.start_tick)/120,
                    'distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
                    'run_id':env.session.run_id if record else None})
            finally:env.close()
            if (index+1)%8==0:print(json.dumps({'phase':'evaluation_progress','label':label,'mode':mode,'cases':index+1,'target':len(cases)}),flush=True)
        print(json.dumps({'phase':'evaluation','label':label,'mode':mode,'completed':len(results)}),flush=True)
    fixed=[r for r in results if r['mode']=='argmax']
    return {'label':label,'episodes':len(fixed),'successes':sum(r['success'] for r in fixed),
        'mean_return':float(np.mean([r['return'] for r in fixed])),'results':fixed,'all_modes':results,
        'checkpoint_sha256':agent.checkpoint_sha256}


def collect(root,agent,cases,rows,metadata,execute_teacher,recovery=False):
    outcomes=[];injections=0;explorations=0
    for index,case in enumerate(cases):
        env=TrainingEnv(root,case,agent)
        exploration=np.random.default_rng(case['seed']+900)
        previous=None
        try:
            while not env.terminated:
                label=teacher(env)
                rows.append((env.features.copy(),env.mask.copy(),label))
                metadata.append({'case_seed':case['seed'],'cohort':case['cohort'],
                    'measured_goal_error_m':float(env.observation[0]*6),'step':env.steps})
                if execute_teacher:action=label;policy=None
                else:action,_,_,policy=agent.decision(env.features,env.mask,index%2==0)
                # Training exploration only: pauses expose different neural histories.
                if index%2==0 and env.steps in (1,3):
                    action=0;policy=None;injections+=1
                if recovery:
                    if execute_teacher and env.steps in (2,3) and previous in (1,2):
                        action=previous;policy=None;explorations+=1
                    elif not execute_teacher and index%3==0 and 1<=env.steps<=6:
                        action=int(exploration.choice(np.flatnonzero(env.mask)))
                        policy=None;explorations+=1
                env.step(action,policy);previous=action
            outcomes.append({'seed':case['seed'],'reason':env.reason,'steps':env.steps})
        finally:env.close()
        if (index+1)%8==0:print(json.dumps({'phase':'collect','episodes':index+1,'target':len(cases),'examples':len(rows)}),flush=True)
    return {'episodes':outcomes,'pause_intervention_requests':injections,'recovery_intervention_requests':explorations,
        'intervention_note':'Recovery requests take precedence over a pause requested at the same step.'}


def compare(before,after):
    rows=[]
    for mode in ('argmax','sampled'):
        for cohort in sorted({r['cohort'] for r in before['all_modes']}):
            left=[r for r in before['all_modes'] if r['mode']==mode and r['cohort']==cohort]
            right=[r for r in after['all_modes'] if r['mode']==mode and r['cohort']==cohort]
            assert [r['case_id'] for r in left]==[r['case_id'] for r in right]
            rows.append({'mode':mode,'cohort':cohort,'episodes':len(left),
                'before':sum(r['success'] for r in left),'after':sum(r['success'] for r in right),
                'regressed_cases':[a['case_id'] for a,b in zip(left,right) if a['success'] and not b['success']]})
    return rows


def run(root,graph,out,seed,source,recovery=False):
    started=time.monotonic();root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4)
    agent=ReservoirAgent(graph,'cuda',seed,profile='balanced_rate_v3')
    load_checkpoint(source,agent,root);source_sha=agent.checkpoint_sha256
    validation=validation_cases(recovery)
    stages_cases=[]
    for stage,count in enumerate((64,96,96) if recovery else (32,48,48)):
        start=(1300000 if recovery else 1200000)+seed*1000+stage*100
        stages_cases.append([boundary_case(start+i,boundary=i%2==0) for i in range(count)])
    splits={'validation':validation,'training_stages':stages_cases,
        'critic_seeds':list(range(1600000+seed*100,1600000+seed*100+16)),
        'sealed_test_opened':False,'known_validation_case_used_in_training':False}
    training_seeds={c['seed'] for stage in stages_cases for c in stage}
    assert not training_seeds & {c['seed'] for c in validation}
    atomic_json(out/'cases.json',splits)
    protocol={'seed':seed,'source':str(source),'source_sha256':source_sha,
        'recovery_exploration':recovery,'stage_episodes':[len(cases) for cases in stages_cases],'fit_epochs_per_stage':200 if recovery else 100,'learning_rate':.0003,
        'method':'Corrective demonstrations + two DAgger rounds with training-only '+('overshoot/recovery' if recovery else 'pause')+' exploration; zero new PPO updates',
        'rehearsal':'all original supervised examples; source graph/encoder/normalizer unchanged',
        'selection':'fixed final checkpoint only; evaluation not used for epochs or checkpoint selection',
        'evaluation':f'same {len(validation)} scenarios in argmax and matched sampled modes; exploratory, not sealed acceptance',
        'publish_gate':'Known validation-005 succeeds in both modes and no previous success regresses in any cohort'}
    atomic_json(out/'protocol.json',protocol)
    save_checkpoint(out/'initial.pt',agent,None,root,0,0,{'method':agent.training_method})
    original_parameters=[p.detach().clone() for p in agent.model.parameters()]
    torch.manual_seed(seed+1700000)
    rehearsal=root/f'runs/tellosim-sdk9/nearx-bootstrap-s{seed}-20260926/demonstrations.npz'
    with np.load(rehearsal) as data:
        rows=[(x.copy(),m.copy(),int(y)) for x,m,y in zip(data['features'],data['masks'],data['teacher_actions'])]
    metadata=[{'source':'original_rehearsal'} for _ in rows]
    agent.training_method=protocol['method'];agent.value_trained=False
    optimizer=torch.optim.Adam(list(agent.model.body.parameters())+list(agent.model.actor.parameters()),lr=.0003)
    stages=[];updates=0
    for stage,cases in enumerate(stages_cases):
        collection=collect(root,agent,cases,rows,metadata,stage==0,recovery)
        metrics=fit(agent,optimizer,rows,epochs=protocol['fit_epochs_per_stage']);updates+=metrics['gradient_steps']
        stages.append({'stage':stage,**collection,**metrics})
        atomic_json(out/'progress.json',{'stages':stages})
        print(json.dumps({'phase':'fit','seed':seed,'stage':stage,**metrics}),flush=True)
    features=np.stack([r[0] for r in rows]);masks=np.stack([r[1] for r in rows]);labels=np.array([r[2] for r in rows])
    np.savez_compressed(out/'demonstrations.npz',features=features,masks=masks,teacher_actions=labels)
    atomic_json(out/'data-provenance.json',{'rows':metadata,'rehearsal_sha256':hashlib.sha256(rehearsal.read_bytes()).hexdigest()})
    # Shared hidden features changed; the old value head is invalid until refitted.
    critic=warmup(root,agent,out,splits['critic_seeds'],'C0-near-x')
    extra={'method':agent.training_method,'seed':seed,'curriculum':'C0-near-x','case_hash':digest(splits),
        'source_sha256':source_sha,'demonstrations_sha256':hashlib.sha256((out/'demonstrations.npz').read_bytes()).hexdigest()}
    save_checkpoint(out/'checkpoint.pt',agent,optimizer,root,updates,len(rows),extra)
    expected=[agent.decision(x,m,True)[3] for x,m in zip(features[:32],masks[:32])]
    load_checkpoint(out/'checkpoint.pt',agent,root)
    assert expected==[agent.decision(x,m,True)[3] for x,m in zip(features[:32],masks[:32])]
    delta=float(torch.sqrt(sum((a-p.detach()).square().sum() for a,p in zip(original_parameters,agent.model.parameters()))))
    load_checkpoint(source,agent,root)
    before=evaluate(root,agent,validation,out.name,'before')
    atomic_json(out/'evaluation-before.json',before)
    load_checkpoint(out/'checkpoint.pt',agent,root)
    after=evaluate(root,agent,validation,out.name,'boundary-repair')
    atomic_json(out/'evaluation-after.json',after)
    comparison=compare(before,after)
    known=[r for r in after['all_modes'] if r['case_id']=='validation-005']
    repaired=all(r['success'] for r in known)
    retained=all(not r['regressed_cases'] for r in comparison)
    summary={'schema':'sdk9.boundary_repair/1.0','status':'completed','seed':seed,'profile':agent.profile,
        'curriculum':'C0-near-x','feature_source':'reservoir','repair_method':protocol['method'],
        'parameter_delta_l2':delta,'training_method':protocol['method'],'ppo_updates':0,'options':len(rows),'updates':updates,
        'stages':stages,'critic_warmup':critic,'source_sha256':source_sha,'checkpoint_sha256':agent.checkpoint_sha256,
        'checkpoint_roundtrip_exact':True,'graph_sha256':agent.brain.graph_sha256,'mapping_sha256':agent.brain.mapping_sha256,
        'case_hash':digest(splits),'baselines':[before],'trained':after,'comparison':comparison,
        'KNOWN_FAILURE_REPAIRED':repaired,'NO_CASE_REGRESSIONS':retained,'MODEL_READY_FOR_NEXT_STAGE':False,
        'TS1_C0_TASK_LEARNED':False,'elapsed_s':time.monotonic()-started,'artifact_directory':str(out.relative_to(root)),
        'scope':'Corrective supervised learning on independent training cases; no runtime expert, no sealed testing or real flight.'}
    atomic_json(out/'summary.json',summary)
    print(json.dumps({'seed':seed,'repaired':repaired,'retained':retained,'comparison':comparison}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=23);parser.add_argument('--tag',default='20260926');parser.add_argument('--recovery',action='store_true')
    args=parser.parse_args();root=Path('.').resolve()
    source=root/f'runs/tellosim-sdk9/nearx-warmstart-s{args.seed}-20260926/checkpoint.pt'
    run(root,args.graph,root/f'runs/tellosim-sdk9/boundary-repair-s{args.seed}-{args.tag}',args.seed,source,args.recovery)

if __name__=='__main__':main()
