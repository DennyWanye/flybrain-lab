"""C2 preregistered continuous 3D curriculum, fixed budget, frozen held-out evaluation."""
import argparse,json,signal,time,math
from pathlib import Path
import numpy as np
import torch
from .spatial_settled import SpatialEnv,SpatialPool,SpatialDiagnostic,spatial_case,SPATIAL_SPEC,SKILLS,SOURCE_FILES,save_spatial,load_spatial,sha
from .altitude_refined import PROFILES,DISTURBANCE_SPEC,with_profile,teacher as altitude_teacher
from .heading import teacher as heading_teacher,wrap_angle
from .joint_refine import settling_label,training_case as navigation_case
from .stability_campaign import fit
from .checkpoint import configure_exact_execution
from .joint_campaign import frozen_json
from .runtime import distribution
from .contracts import digest
from .c0_campaign import summarize,acceptance
from ..visual import atomic_json
SEEDS=(11,22,33);STAGES=();BATCH=16
METHOD='C2S: exact frozen C2Q checkpoint transfer, no optimization; trailing measured phase-state estimator; raw policy input and final physical gates unchanged'
def directory(root):return Path(root)/'reports/ts1_spatial_settled'
def run_directory(root,seed,smoke=False):return Path(root)/f'runs/tellosim-sdk9/spatial-settled-{"smoke-" if smoke else ""}s{seed}'
def source_hashes(root):return {p:sha(Path(root)/p) for p in SOURCE_FILES}
def label_for(env):
    function={'altitude':altitude_teacher,'navigation':settling_label,'heading':heading_teacher}[env.phase]
    return function(env.observation,env.mask,env.previous_action is None)
def training_case(seed,lane):
    case=spatial_case(seed);nav=navigation_case(seed,lane//4)
    case['start']=nav['start'];case['goal'][:2]=nav['goal'][:2];case['initial_yaw_rad']=nav['initial_yaw_rad']
    if lane//4<2:
        rng=np.random.default_rng(seed+801)
        sign=1 if lane//4==0 else -1
        case['goal'][2]=1+sign*float(rng.choice([.4,.6]))
        case['target_yaw_rad']=wrap_angle(case['initial_yaw_rad']+sign*math.pi/2)
    return with_profile(case,tuple(PROFILES)[lane%4])
def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(exist_ok=True)
    assert json.loads((root/'reports/ts1_altitude_refined/summary.json').read_text())['MODEL_READY_FOR_NEXT_STAGE']
    cases={name:[with_profile(spatial_case(base+i,f'c2-{name}-{i:03d}'),tuple(PROFILES)[i%4]) for i in range(n)]
        for name,base,n in [('validation',473000000,100),('sealed_test',474000000,300)]}
    cases['instruction_pairs']=[]
    for kind in ['height','yaw']:
        for pair in range(2):
            for sign in [-1,1]:
                c=with_profile(spatial_case(475000000+(0 if kind=='height' else 2)+pair,f'c2-{kind}-{pair}-{sign}'),'combined')
                if kind=='height':c['goal'][2]=1+sign*(.4+.2*pair)
                else:c['target_yaw_rad']=wrap_angle(c['initial_yaw_rad']+sign*math.pi/2)
                cases['instruction_pairs'].append(c)
    cases['boundary']=[]
    for i,d in enumerate([-.705,-.7,-.695,-.505,-.5,-.495,.495,.5,.505,.695,.7,.705]):
        c=with_profile(spatial_case(476000000+i,f'c2-boundary-{i}'),'clean');c['goal'][2]=1+d;cases['boundary'].append(c)
    cases['zero_cases']=[with_profile(spatial_case(477000000+i,f'c2-zero-{i}'),tuple(PROFILES)[i%4]) for i in range(12)]
    protocol=dict(stage='C2',seeds=list(SEEDS),stages=list(STAGES),batch=BATCH,epochs=0,learning_rate=0.,weight_decay=0.,
        method=METHOD,task_spec=SPATIAL_SPEC,disturbance=DISTURBANCE_SPEC,initial_gate=sha(root/'reports/ts1_altitude_refined/summary.json'),
        training_seed_formula='no new training; formally trained frozen C2Q seeds11/22/33 transferred without parameter changes',development='470000000..470000063',
        rehearsal='none; no optimization or data fitting',
        labels='real10Hz states at5Hz; navigation/heading STOP x2; altitude settled-only, abs(error).06-.14m symmetric x4',
        selection='all3 final C2Q weights fixed before this software repair and all new cases; no seed/checkpoint selection or heldout tuning',
        gate=dict(each_seed_success_min=.9,clean_success_min=.9,other_profile_min=.85,collision_or_bounds_max=.01,advantage_over_random_min=.2,
            instruction_successes=8,boundary_successes_min=11,zero_successes_max=0,rule_successes_min=99,
            physical_continuity=True,vertical_speed_tolerance=.08,non1m_navigation_both_sides=True),
        split_hashes={k:digest(v) for k,v in cases.items()})
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol);return cases

def transfer(root):
    root=Path(root);prepare(root);models=[];torch.set_num_threads(4);configure_exact_execution('cuda')
    for seed in SEEDS:
        folder=run_directory(root,seed)
        if folder.exists():raise FileExistsError(folder)
        source=root/f'runs/tellosim-sdk9/spatial-orientation-s{seed}/checkpoint.pt'
        trained=json.loads(source.with_name('training.json').read_text())
        assert trained['status']=='completed' and trained['options']==5632 and not trained['smoke_only']
        pool=SpatialPool(root,seed,batch=1);before=torch.load(source,map_location='cpu',weights_only=False)
        folder.mkdir(parents=True);target=folder/'checkpoint.pt';save_spatial(target,pool,None,root,0,{'origin':'formally trained C2Q; parameter-identical transfer; no optimization'})
        after=load_spatial(target,pool,root)
        assert all(torch.equal(value,after['models'][skill][key]) for skill in SKILLS for key,value in before['models'][skill].items())
        row={'status':'transferred','new_training_actions':0,'source_training_actions':5632,'source_checkpoint':str(source.relative_to(root)),'source_sha256':sha(source),'checkpoint_sha256':sha(target),'parameter_identical':True,'source_hashes':source_hashes(root),'roundtrip_exact':True,'smoke_only':False}
        atomic_json(folder/'MODEL_ORIGIN.json',row);models.append(dict(seed=seed,path=str(target.relative_to(root)),sha256=sha(target),source_sha256=sha(source)))
        del pool
        import gc
        gc.collect();torch.cuda.empty_cache()
    frozen_json(directory(root)/'frozen-checkpoints.json',{'models':models,'case_hash':digest(prepare(root)['sealed_test']),'new_training_actions':0,'source_training_actions':16896})
    return models

def result_row(env,actions,skills,run_id):
    w=env.session.world;goal=np.array(env.case['goal']);heights=env.navigation_heights
    return dict(case_id=env.case['case_id'],seed=env.case['seed'],success=env.reason=='success',reason=env.reason,
        **{'return':env.raw_return},duration_s=(env.session.tick-env.start_tick)/120,steps=env.steps,
        distance_m=float(np.linalg.norm(w.position-goal)),horizontal_distance_m=float(np.linalg.norm(w.position[:2]-goal[:2])),
        height_error_m=abs(float(w.position[2]-goal[2])),vertical_speed_mps=abs(float(w.velocity[2])),
        horizontal_speed_mps=float(np.linalg.norm(w.velocity[:2])),heading_error_deg=math.degrees(abs(wrap_angle(env.target_yaw-w.yaw_rad))),
        angular_speed_rad_s=abs(float(w.data.qvel[5])),stable_hold_s=env.hold,target_height_m=goal[2],final_height_m=float(w.position[2]),
        actions=actions,action_skills=skills,phase_events=env.phase_events,final_phase=env.phase,run_id=run_id,
        physics_ticks=env.session.tick-env.start_tick,brain_tick=env.agent.brain_tick,disturbance=env.disturbance_evidence(),
        navigation_height_range_m=[min(heights),max(heights)] if heights else None)
def grouped(rows):
    return {p:dict(episodes=len(part),successes=sum(x['success'] for x in part),success_rate=sum(x['success'] for x in part)/len(part))
        for p in PROFILES for part in [[r for r in rows if r['disturbance']['profile']==p]] if part}
def baseline(root,cases,mode,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    agent=SpatialDiagnostic();rows=[]
    for case in cases:
        env=SpatialEnv(root,case,agent);actions=[];skills=[];rng=np.random.default_rng(case['seed']+991)
        try:
            while not env.terminated:
                a=label_for(env) if mode=='rule' else int(rng.choice(np.flatnonzero(env.mask)))
                actions.append(a);skills.append(env.phase);env.step(a)
            rows.append(result_row(env,actions,skills,None))
        finally:env.close()
        if len(rows)%10==0:
            atomic_json(out.with_suffix('.progress.json'),summarize(rows,mode));print(json.dumps(dict(baseline=mode,cases=len(rows),successes=sum(r['success'] for r in rows))),flush=True)
    result={**summarize(rows,mode),'profiles':grouped(rows),'case_hash':digest(cases),'neural_model':False,'mode':mode};atomic_json(out,result);return result

def evaluate(root,checkpoint,cases,out,label,batch=64,zero=False,record_indices=()):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    seed=torch.load(checkpoint,map_location='cpu',weights_only=False)['seed'];pool=SpatialPool(root,seed,batch=batch);load_spatial(checkpoint,pool,root)
    envs=[None]*batch;generators={};indices={};actions={};skills={};results={};cursor=0
    try:
        while len(results)<len(cases):
            if time.monotonic()-started>5400:raise TimeoutError('C2 evaluation budget')
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;actions[index]=[];skills[index]=[]
                    envs[i]=SpatialEnv(root,cases[index],pool.lanes[i],record=index in record_indices,run_id=f'{label}-{index}' if index in record_indices else None,
                        policy_source='spatial_ZERO_FEATURE_ABLATION' if zero else 'spatial_argmax_malecns')
                env=envs[i]
                if i not in generators:
                    x=np.zeros_like(env.features) if zero else env.features
                    a,_,_,policy=env.agent.decision(x,env.mask,True);actions[indices[i]].append(a);skills[indices[i]].append(env.phase);generators[i]=env.step_iter(a,policy)
                try:next(generators[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                except StopIteration:
                    del generators[i]
                    if env.terminated:
                        index=indices[i];results[index]=result_row(env,actions[index],skills[index],env.session.run_id if env.live else None);env.close();envs[i]=None
                        if len(results)%10==0:
                            atomic_json(out.with_suffix('.progress.json'),summarize([results[k] for k in sorted(results)],label));print(json.dumps(dict(label=label,cases=len(results),successes=sum(r['success'] for r in results.values()))),flush=True)
            if pending:
                x=pool.observe_lanes(pending)
                for i in pending:envs[i].features=x[i].copy();envs[i].publish()
    finally:
        for env in envs:
            if env:env.close()
    rows=[results[i] for i in range(len(cases))]
    result={**summarize(rows,label),'profiles':grouped(rows),'case_hash':digest(cases),'checkpoint_sha256':sha(checkpoint),'mode':'zero' if zero else 'argmax','elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result

def evaluate_seed(root,seed):
    root=Path(root);out=directory(root);cases=prepare(root);lock=json.loads((out/'frozen-checkpoints.json').read_text());entry=next(r for r in lock['models'] if r['seed']==seed);checkpoint=root/entry['path']
    assert sha(checkpoint)==entry['sha256']
    for split in cases:
        target=out/f's{seed}-{split}.json'
        if not target.exists():evaluate(root,checkpoint,cases[split],target,f'spatial-settled-s{seed}-{split}',batch=min(64,len(cases[split])),zero=split=='zero_cases',record_indices=(0,1,2,3) if split=='sealed_test' else ())
    assert sha(checkpoint)==entry['sha256']

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root)
    def load(n):return json.loads((out/n).read_text())
    models={s:load(f's{s}-sealed_test.json') for s in SEEDS};random=load('random-sealed.json');gate=acceptance(models,random,digest(cases['sealed_test']));rows=[]
    if load('rule-validation.json')['successes']<99:gate['reasons'].append('rule feasibility below99/100')
    lock=load('frozen-checkpoints.json')
    for seed,r in models.items():
        entry=next(x for x in lock['models'] if x['seed']==seed)
        assert r['checkpoint_sha256']==entry['sha256']==sha(root/entry['path'])
        for split in cases:
            x=load(f's{seed}-{split}.json');assert x['case_hash']==digest(cases[split]) and x['checkpoint_sha256']==entry['sha256']
        pairs=load(f's{seed}-instruction_pairs.json');boundary=load(f's{seed}-boundary.json');zero=load(f's{seed}-zero_cases.json')
        if pairs['successes']!=8:gate['reasons'].append(f'{seed}: instruction below8/8')
        if boundary['successes']<11:gate['reasons'].append(f'{seed}: boundary below11/12')
        if zero['successes']!=0:gate['reasons'].append(f'{seed}: zero feature succeeded')
        for name,p in r['profiles'].items():
            if p['success_rate']+1e-10<(.9 if name=='clean' else .85):gate['reasons'].append(f'{seed}: {name} below profile gate')
        for x in r['results']:
            if x['brain_tick']!=4*(1+x['physics_ticks']//12):gate['reasons'].append(f'{seed}: neural clock mismatch')
            if any(not p['physics_state_unchanged'] for p in x['phase_events']):gate['reasons'].append(f'{seed}: physical reset')
            if any(a not in SPATIAL_SPEC['allowed_actions'][k] for a,k in zip(x['actions'],x['action_skills'])):gate['reasons'].append(f'{seed}: phase action invalid')
            if x['success'] and not (x['final_phase']=='heading' and len(x['phase_events'])>=2 and x['height_error_m']<=.1 and x['vertical_speed_mps']<=.08 and x['horizontal_distance_m']<=.2 and x['horizontal_speed_mps']<=.08 and x['heading_error_deg']<=16 and x['angular_speed_rad_s']<=.08 and x['stable_hold_s']>=2-1e-8):gate['reasons'].append(f'{seed}: invalid full3D hold')
            if x['disturbance']['profile'] in ['force','combined'] and x['disturbance']['absolute_vertical_impulse_ns']<=0:gate['reasons'].append(f'{seed}: missing vertical force')
        for side in [-1,1]:
            if not any(x['navigation_height_range_m'] and side*(sum(x['navigation_height_range_m'])/2-1)>.15 for x in r['results']):gate['reasons'].append(f'{seed}: missing non1m navigation')
        rows.append(dict(seed=seed,validation=load(f's{seed}-validation.json')['successes'],sealed=r['successes'],profiles=r['profiles'],boundary=boundary['successes'],instruction=pairs['successes'],zero=zero['successes'],collisions=r['collision_or_bounds']))
    gate['passed']=not gate['reasons']
    summary=dict(stage='C2',status='evaluated',JOINT_3D_TASK_VERIFIED=gate['passed'],MODEL_READY_FOR_NEXT_STAGE=gate['passed'],FULL_TS1_READY=False,REAL_FLIGHT_READY=False,SINGLE_POLICY_JOINT_TRAINED=False,
        models=rows,acceptance=gate,new_training_actions=0,source_training_actions=16896,model_origin='parameter-identical frozen formally trained C2Q readouts',rule_successes=load('rule-validation.json')['successes'],random_successes=random['successes'],scope='simulated empty room, external pose mock, frozen MaleCNS, three jointly exercised readouts')
    atomic_json(out/'summary.json',summary);print(json.dumps(summary));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','transfer','evaluate','baselines','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command=='transfer':transfer(root)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    elif a.command=='baselines':
        cases=prepare(root);baseline(root,cases['validation'],'rule',out/'rule-validation.json');baseline(root,cases['sealed_test'],'random',out/'random-sealed.json')
    else:finalize(root)
if __name__=='__main__':main()
