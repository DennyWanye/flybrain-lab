"""V1R fixed-budget independent altitude learning and sealed evaluation."""
import argparse,json,signal,time,math
from pathlib import Path
import numpy as np
import torch
from .altitude_refined import (AltitudeEnv,ALTITUDE_SPEC,DISTURBANCE_SPEC,PROFILES,altitude_case,with_profile,teacher,save_altitude,load_altitude,AltitudeEncoder34,ENCODER_SPEC)
from .parallel import ReservoirPool
from .checkpoint import configure_exact_execution
from .stability_campaign import fit,bundle_spec as retained_bundle,evaluate as retained_evaluate
from .heading_campaign import RuleDiagnostic
from .heading import wrap_angle
from .joint import sha
from .joint_campaign import frozen_json
from .contracts import digest
from .c0_campaign import summarize,acceptance
from ..visual import atomic_json
SEEDS=(11,22,33);STAGES=(768,1024,1024);BATCH=16
METHOD='V1R: independent altitude readout from random initialization; 768 teacher + 2x1024 DAgger options, decision and settled-state labels; smooth multiscale altitude encoding; frozen MaleCNS; no PPO'
def directory(root):return Path(root)/'reports/ts1_altitude_refined'
def run_directory(root,seed,smoke=False):return Path(root)/f'runs/tellosim-sdk9/altitude-refined-{"smoke-" if smoke else ""}s{seed}'
def training_case(seed,lane):
    case=altitude_case(seed,direction=1 if lane//4%2==0 else -1)
    rng=np.random.default_rng(seed+37)
    sign=1 if lane//4%2==0 else -1
    # Half the episodes cover the entire SDK half-step grid, not old failed cases.
    if lane//8==0:
        centre=float(rng.choice([.1,.3,.5,.7]))
        delta=float(np.clip(centre+rng.uniform(-.04,.04),.04,.75))
        case['goal'][2]=1+sign*delta
    return with_profile(case,tuple(PROFILES)[lane%4])
def pool_for(root,seed,batch):
    torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=ReservoirPool(Path(root)/'data/male-v1.npz','cuda',seed=seed,profile='balanced_rate_v3',batch=batch,physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
    pool.brain.encoder=AltitudeEncoder34()
    return pool
def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(exist_ok=True)
    cases={name:[with_profile(altitude_case(base+i,f'v1r-{name}-{i:03d}',direction=1 if i//4%2==0 else -1),tuple(PROFILES)[i%4]) for i in range(n)] for name,base,n in [('validation',213000000,100),('sealed_test',214000000,300)]}
    cases['instruction_pairs']=[]
    for pair,delta in enumerate([.4,.6]):
        for sign in [-1,1]:
            c=with_profile(altitude_case(215000000+pair,f'v1r-pair-{pair}-{sign}'),'combined');c['goal'][2]=1+sign*delta;cases['instruction_pairs'].append(c)
    cases['boundary']=[]
    for i,delta in enumerate([-.705,-.700,-.695,-.505,-.500,-.495,.495,.500,.505,.695,.700,.705]):
        c=with_profile(altitude_case(216000000+i,f'v1r-boundary-{i}'),'clean');c['goal'][2]=1+delta;cases['boundary'].append(c)
    cases['zero_cases']=[]
    for i in range(12):
        c=with_profile(altitude_case(217000000+i,f'v1r-zero-{i}'),tuple(PROFILES)[i%4]);c['goal'][2]=[.55,.65,1.35,1.45][i%4];cases['zero_cases'].append(c)
    legacy=json.loads((root/'reports/ts1_stability/cases.json').read_text())['validation'][:12]
    protocol=dict(seeds=list(SEEDS),stages=list(STAGES),batch=BATCH,epochs=80,learning_rate=.0003,weight_decay=.001,method=METHOD,
        task_spec=ALTITUDE_SPEC,disturbance=DISTURBANCE_SPEC,training_seed_formula='220000000 + model_seed*100000 + stage*10000 + lane*500 + episode; smoke base230000000',
        encoder=ENCODER_SPEC,labels='decisions weight1; settled samples weight.25; abs(error) .06-.14m x4 symmetric for STOP and movement; exclude unfinished movement; no STOP class bonus',
        development='218000000..218000063; smoke 230000000+formula; disjoint from all final splits',
        selection='fixed final checkpoints, all seeds frozen before sealed evaluation; no heldout tuning',
        zero_control='12 preregistered interior targets away from safety-mask stopping heights; identical zero-vector input, no target direction rule',
        gate=dict(each_seed_success_min=.9,clean_profile_success_min=.9,other_profile_success_min=.85,collision_or_bounds_max=.01,
            advantage_over_uniform_min=.2,instruction_pairs_required=4,zero_feature_success_max=0,boundary_success_min=11,rule_validation_success_min=99,legacy_retention='12 historical validation cases per seed, exact result equality; frozen J2R sources/weights'),
        split_hashes={k:digest(v) for k,v in cases.items()},legacy_cases=legacy,legacy_bundles={str(s):retained_bundle(root,s) for s in SEEDS})
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol);return cases

def train(root,seed,smoke=False):
    root=Path(root);prepare(root);out=run_directory(root,seed,smoke)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();pool=pool_for(root,seed,BATCH);pool.training_method=METHOD;pool.value_trained=False
    initial={k:v.detach().clone() for k,v in pool.model.state_dict().items()};save_altitude(out/'initial.pt',pool,None,root,0,dict(seed=seed,method='untrained altitude readout'))
    opt=torch.optim.AdamW(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0003,weight_decay=.001)
    rows=[];provenance=[];stages=[];options=0;envs=[];sources=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)['contract']['task_sources']
    def annotate(env,x,kind):
        label=teacher(env.observation,env.mask,env.previous_action is None);error=abs(float(env.observation[2])*3);weight=(1. if kind=='decision' else .25)*(4. if .06<=error<=.14 and env.previous_action is not None else 1.)
        rows.append((x.copy(),env.mask.copy(),label,weight));provenance.append(dict(seed=env.case['seed'],stage=stage,step=env.steps,sample_id=env.agent.sample_id,profile=env.disturbance_profile,kind=kind,label=label,weight=weight));return label
    def stop(signum,frame):raise KeyboardInterrupt('SIGTERM')
    old_signal=signal.signal(signal.SIGTERM,stop)
    try:
        for stage,count in enumerate((64,64) if smoke else STAGES):
            envs=[None]*BATCH;episodes=[0]*BATCH;done=[0]*BATCH;gens={};outcomes=[];ticks=0
            while any(n<count//BATCH for n in done):
                if time.monotonic()-started>3600:raise TimeoutError('altitude training budget')
                pending={}
                for i in range(BATCH):
                    if done[i]>=count//BATCH:continue
                    if envs[i] is None:
                        if episodes[i]>=500:raise RuntimeError('training seed range exhausted')
                        sample=(230000000 if smoke else 220000000)+seed*100000+stage*10000+i*500+episodes[i];episodes[i]+=1
                        envs[i]=AltitudeEnv(root,training_case(sample,i),pool.lanes[i])
                    env=envs[i]
                    if i not in gens:
                        label=annotate(env,env.features,'decision')
                        if stage==0:a,policy=label,None
                        else:a,_,_,policy=pool.lanes[i].decision(env.features,env.mask,deterministic=i%2==0)
                        gens[i]=env.step_iter(a,policy)
                    try:next(gens[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                    except StopIteration:
                        del gens[i];done[i]+=1;options+=1
                        if env.terminated:outcomes.append(env.reason);env.close();envs[i]=None
                if pending:
                    features=pool.observe_lanes(pending);ticks+=len(pending)
                    for i in pending:
                        env=envs[i];env.features=features[i].copy()
                        if env.agent.sample_id%2==0 and env.previous_duration>=.5 and (env.previous_action==0 or env.device.observed_operation()['client']!='sent'):annotate(env,env.features,'settled')
            for env in envs:
                if env:env.close('stage_complete')
            envs=[];advances=pool.brain.advance_count;metrics=fit(pool.model,opt,rows,2 if smoke else 80);assert pool.brain.advance_count==advances
            stages.append(dict(stage=stage,options=sum(done),base_ticks=ticks,outcomes=outcomes,**metrics))
            save_altitude(out/f'stage-{stage}.pt',pool,opt,root,options,dict(seed=seed,method=METHOD))
            atomic_json(out/'progress.json',dict(options=options,stages=stages,elapsed_s=time.monotonic()-started));print(json.dumps(dict(seed=seed,options=options,**metrics)),flush=True)
        checkpoint=out/'checkpoint.pt';save_altitude(checkpoint,pool,opt,root,options,dict(seed=seed,method=METHOD))
        expected=[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]];load_altitude(checkpoint,pool,root);assert expected==[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]]
        delta=float(torch.sqrt(sum((initial[k]-v).square().sum() for k,v in pool.model.state_dict().items())));assert delta>0
        assert all(sha(root/p)==h for p,h in sources.items())
        atomic_json(out/'training.json',dict(status='completed',smoke_only=smoke,seed=seed,options=options,new_examples=len(rows),stages=stages,
            method=METHOD,roundtrip_exact=True,parameter_delta_l2=delta,checkpoint_sha256=sha(checkpoint),task_sources=sources,elapsed_s=time.monotonic()-started))
    except BaseException as exc:atomic_json(out/'failure.json',dict(error=repr(exc),options=options));raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        signal.signal(signal.SIGTERM,old_signal);atomic_json(out/'provenance.json',provenance)
        if rows:np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]),labels=np.array([r[2] for r in rows]),weights=np.array([r[3] for r in rows]))

def result_row(env,case,actions,run_id):
    w=env.session.world;g=np.array(case['goal'])
    return {'return':env.raw_return, **dict(case_id=case['case_id'],seed=case['seed'],reason=env.reason,success=env.reason=='success',steps=env.steps,
        duration_s=(env.session.tick-env.start_tick)/120,distance_m=float(np.linalg.norm(w.position-g)),horizontal_distance_m=float(np.linalg.norm(w.position[:2]-g[:2])),
        height_error_m=abs(float(w.position[2]-g[2])),vertical_speed_mps=abs(float(w.velocity[2])),horizontal_speed_mps=float(np.linalg.norm(w.velocity[:2])),
        yaw_drift_deg=math.degrees(abs(wrap_angle(w.yaw_rad-case['initial_yaw_rad']))),angular_speed_rad_s=abs(float(w.data.qvel[5])),
        target_height_m=case['goal'][2],final_height_m=float(w.position[2]),stable_hold_s=env.hold,actions=actions,run_id=run_id,
        physics_ticks=env.session.tick-env.start_tick,brain_tick=env.agent.brain_tick,disturbance=env.disturbance_evidence())}

def grouped(rows):
    return {p:dict(episodes=len(part),successes=sum(r['success'] for r in part),success_rate=sum(r['success'] for r in part)/len(part)) for p in PROFILES for part in [[r for r in rows if r['disturbance']['profile']==p]] if part}

def baseline(root,cases,mode,out):
    started=time.monotonic();agent=RuleDiagnostic();rows=[]
    for case in cases:
        env=AltitudeEnv(root,case,agent);actions=[];rng=np.random.default_rng(case['seed']+991)
        try:
            while not env.terminated:
                a=teacher(env.observation,env.mask,env.steps==0) if mode=='rule' else int(rng.choice(np.flatnonzero(env.mask)));actions.append(a);env.step(a)
            rows.append(result_row(env,case,actions,None))
        finally:env.close()
        if len(rows)%10==0:atomic_json(out.with_suffix('.progress.json'),summarize(rows,'altitude-refined-'+mode));print(json.dumps(dict(baseline=mode,cases=len(rows),successes=sum(r['success'] for r in rows))),flush=True)
    result={**summarize(rows,'altitude-refined-'+mode),'mode':mode,'neural_model':False,'case_hash':digest(cases),'profiles':grouped(rows),'elapsed_s':time.monotonic()-started};atomic_json(out,result);return result

def freeze(root):
    cases=prepare(root);out=directory(root);models=[]
    for seed in SEEDS:
        folder=run_directory(root,seed);t=json.loads((folder/'training.json').read_text());assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
        assert all(sha(Path(root)/p)==h for p,h in t['task_sources'].items()) and sha(folder/'checkpoint.pt')==t['checkpoint_sha256']
        models.append(dict(seed=seed,path=str((folder/'checkpoint.pt').relative_to(root)),sha256=sha(folder/'checkpoint.pt')))
    frozen_json(out/'frozen-checkpoints.json',dict(models=models,case_hash=digest(cases['sealed_test'])))

def evaluate(root,checkpoint,cases,out,label,record_indices=(0,1),batch=64,zero_features=False):
    root=Path(root).resolve();out=Path(out)
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();pool=pool_for(root,int(torch.load(checkpoint,map_location='cpu',weights_only=False)['extra']['seed']),batch);load_altitude(checkpoint,pool,root)
    envs=[None]*batch;generators={};indices={};actions={};results={};cursor=0
    try:
        while len(results)<len(cases):
            if time.monotonic()-started>3600:raise TimeoutError('altitude evaluation budget')
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;actions[index]=[]
                    envs[i]=AltitudeEnv(root,cases[index],pool.lanes[i],record=index in record_indices,run_id=f'{label}-{index}' if index in record_indices else None,policy_source='altitude_argmax_malecns' if not zero_features else 'altitude_ZERO_FEATURE_ABLATION')
                env=envs[i]
                if i not in generators:
                    x=np.zeros_like(env.features) if zero_features else env.features
                    a,_,_,policy=pool.lanes[i].decision(x,env.mask,True);policy.update(skill='altitude',checkpoint_sha256=pool.checkpoint_sha256);actions[indices[i]].append(a);generators[i]=env.step_iter(a,policy)
                try:next(generators[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                except StopIteration:
                    del generators[i]
                    if env.terminated:
                        index=indices[i];results[index]=result_row(env,env.case,actions[index],env.session.run_id if env.live else None);env.close();envs[i]=None
                        if len(results)%10==0:atomic_json(out.with_suffix('.progress.json'),summarize([results[k] for k in sorted(results)],label));print(json.dumps({'label':label,'cases':len(results),'successes':sum(x['success'] for x in results.values())}),flush=True)
            if pending:
                features=pool.observe_lanes(pending)
                for i in pending:envs[i].features=features[i].copy();envs[i].publish()
    finally:
        for env in envs:
            if env:env.close()
    result={**summarize([results[i] for i in range(len(cases))],label),'profiles':grouped([results[i] for i in range(len(cases))]),'mode':'argmax' if not zero_features else 'zero_features_ablation','case_hash':digest(cases),'checkpoint_sha256':sha(checkpoint),'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result


def evaluate_seed(root,seed):
    root=Path(root);out=directory(root);cases=prepare(root);lock=json.loads((out/'frozen-checkpoints.json').read_text());entry=next(r for r in lock['models'] if r['seed']==seed);checkpoint=root/entry['path'];assert sha(checkpoint)==entry['sha256']
    for split in ['validation','sealed_test','instruction_pairs','boundary','zero_cases']:
        target=out/f's{seed}-{split}.json'
        if not target.exists():evaluate(root,checkpoint,cases[split],target,f'altitude-refined-s{seed}-{split}',record_indices=(0,1,2,3,7) if split=='sealed_test' else (),batch=min(64,len(cases[split])),zero_features=split=='zero_cases')
    protocol=json.loads((out/'protocol.json').read_text());bundle=retained_bundle(root,seed);assert bundle==protocol['legacy_bundles'][str(seed)]
    target=out/f's{seed}-joint-retention.json'
    if not target.exists():retained_evaluate(root,seed,protocol['legacy_cases'],target,f'altitude-refined-joint-retention-s{seed}',batch=64)
    historical=json.loads((root/f'reports/ts1_stability/s{seed}-validation.json').read_text())['results'][:12]
    current=json.loads(target.read_text())['results'];atomic_json(out/f's{seed}-retention-check.json',dict(exact=current==historical,cases=12,legacy_bundle_unchanged=retained_bundle(root,seed)==bundle))
    assert sha(checkpoint)==entry['sha256']

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root)
    def load(name):return json.loads((out/name).read_text())
    models={s:load(f's{s}-sealed_test.json') for s in SEEDS};random=load('random-sealed.json');gate=acceptance(models,random,digest(cases['sealed_test']));rows=[]
    if load('rule-validation.json')['successes']<99:gate['reasons'].append('rule feasibility below99/100')
    lock=load('frozen-checkpoints.json')
    for seed,result in models.items():
        entry=next(x for x in lock['models'] if x['seed']==seed);assert sha(root/entry['path'])==entry['sha256']==result['checkpoint_sha256']
        pairs=load(f's{seed}-instruction_pairs.json');zero=load(f's{seed}-zero_cases.json');boundary=load(f's{seed}-boundary.json');retention=load(f's{seed}-retention-check.json')
        if pairs['successes']!=4:gate['reasons'].append(f'{seed}: height instruction pairs failed')
        if zero['successes']!=0:gate['reasons'].append(f'{seed}: zero feature control succeeded')
        if boundary['successes']<11:gate['reasons'].append(f'{seed}: boundary below11/12')
        if not retention['exact'] or not retention['legacy_bundle_unchanged']:gate['reasons'].append(f'{seed}: legacy joint retention mismatch')
        for profile,m in result['profiles'].items():
            threshold=.9 if profile=='clean' else .85
            if m['success_rate']+1e-10<threshold:gate['reasons'].append(f'{seed}: {profile} success below{threshold:.0%}')
        actions=[a for r in result['results'] for a in r['actions']]
        if 5 not in actions or 6 not in actions:gate['reasons'].append(f'{seed}: missing vertical direction')
        for r in result['results']:
            if any(a not in [0,5,6] for a in r['actions']):gate['reasons'].append(f'{seed}: invalid altitude action')
            if r['brain_tick']!=4*(1+r['physics_ticks']//12):gate['reasons'].append(f'{seed}: clock mismatch')
            if r['disturbance']['profile'] in ['force','combined'] and r['duration_s']>=8 and r['disturbance']['absolute_vertical_impulse_ns']<=0:gate['reasons'].append(f'{seed}: missing vertical force')
            if r['success'] and not (r['height_error_m']<=.1 and r['vertical_speed_mps']<=.08 and r['horizontal_distance_m']<=.2 and r['horizontal_speed_mps']<=.08 and r['yaw_drift_deg']<=16 and r['angular_speed_rad_s']<=.08 and r['stable_hold_s']>=2-1e-8 and r['duration_s']>=8-1e-8):gate['reasons'].append(f'{seed}: invalid stable altitude success')
        rows.append(dict(seed=seed,validation=f"{load(f's{seed}-validation.json')['successes']}/100",sealed=f"{result['successes']}/300",success_rate=result['success_rate'],profiles=result['profiles'],collision_or_bounds=result['collision_or_bounds'],instruction_pairs_successes=pairs['successes'],zero_features_successes=zero['successes'],boundary_successes=boundary['successes'],legacy_exact=retention['exact'],run_id=result['results'][3]['run_id'],down_run_id=result['results'][7]['run_id'],sample_outcome=result['results'][3]['reason'],down_sample_outcome=result['results'][7]['reason']))
    gate['passed']=not gate['reasons'];passed=gate['passed']
    summary=dict(stage='V1R',status='evaluated',ALTITUDE_TASK_LEARNED=passed,MODEL_READY_FOR_NEXT_STAGE=passed,JOINT_3D_TASK_VERIFIED=False,FULL_TS1_READY=False,REAL_FLIGHT_READY=False,
        models=rows,acceptance=gate,new_training_actions=len(SEEDS)*sum(STAGES),random_successes=random['successes'],rule_successes=load('rule-validation.json')['successes'],
        readiness_scope='independent altitude control from1m to.25-1.75m, fixedXY, four declared synthetic disturbance profiles; not integrated3D navigation',
        note='V1R：独立高度读出，冻结MaleCNS和原J2R技能。20厘米上下动作，高度误差≤10厘米、垂直速度≤0.08米/秒并稳定2秒；每场至少观察8秒。固定第4场上升、第8场下降为回放入口。尚未完成三维位置与朝向联合训练。')
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','smoke','freeze','evaluate','baselines','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command in ['train','smoke']:train(root,a.seed,a.command=='smoke')
    elif a.command=='freeze':freeze(root)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    elif a.command=='baselines':
        cases=prepare(root)
        for split,mode in [('validation','rule'),('sealed_test','random')]:
            target=out/('rule-validation.json' if mode=='rule' else 'random-sealed.json')
            if not target.exists():baseline(root,cases[split],mode,target)
    else:finalize(root)
if __name__=='__main__':main()
