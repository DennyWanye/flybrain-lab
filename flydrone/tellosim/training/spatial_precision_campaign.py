"""C2 preregistered continuous 3D curriculum, fixed budget, frozen held-out evaluation."""
import argparse,json,signal,time,math
from pathlib import Path
import numpy as np
import torch
from .spatial_precision import SpatialEnv,SpatialPool,SpatialDiagnostic,spatial_case,SPATIAL_SPEC,SKILLS,SOURCE_FILES,save_spatial,load_spatial,sha
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
SEEDS=(11,22,33);STAGES=(1536,2048,2048);BATCH=16
METHOD='C2P: new body-distance navigation encoder and independently initialized navigation readout; preservedC2R altitude/heading warm start;1536 teacher+2x2048 DAgger actions; frozen full MaleCNS; no PPO'
def directory(root):return Path(root)/'reports/ts1_spatial_precision'
def run_directory(root,seed,smoke=False):return Path(root)/f'runs/tellosim-sdk9/spatial-precision-{"smoke-" if smoke else ""}s{seed}'
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
        for name,base,n in [('validation',373000000,100),('sealed_test',374000000,300)]}
    cases['instruction_pairs']=[]
    for kind in ['height','yaw']:
        for pair in range(2):
            for sign in [-1,1]:
                c=with_profile(spatial_case(375000000+(0 if kind=='height' else 2)+pair,f'c2-{kind}-{pair}-{sign}'),'combined')
                if kind=='height':c['goal'][2]=1+sign*(.4+.2*pair)
                else:c['target_yaw_rad']=wrap_angle(c['initial_yaw_rad']+sign*math.pi/2)
                cases['instruction_pairs'].append(c)
    cases['boundary']=[]
    for i,d in enumerate([-.705,-.7,-.695,-.505,-.5,-.495,.495,.5,.505,.695,.7,.705]):
        c=with_profile(spatial_case(376000000+i,f'c2-boundary-{i}'),'clean');c['goal'][2]=1+d;cases['boundary'].append(c)
    cases['zero_cases']=[with_profile(spatial_case(377000000+i,f'c2-zero-{i}'),tuple(PROFILES)[i%4]) for i in range(12)]
    protocol=dict(stage='C2',seeds=list(SEEDS),stages=list(STAGES),batch=BATCH,epochs=80,learning_rate=.0003,weight_decay=.001,
        method=METHOD,task_spec=SPATIAL_SPEC,disturbance=DISTURBANCE_SPEC,initial_gate=sha(root/'reports/ts1_altitude_refined/summary.json'),
        training_seed_formula='380000000+model_seed*100000+stage*10000+lane*500+episode; smoke390000000',development='370000000..370000063',
        rehearsal='no navigation replay across encoders; every fourth C2R altitude/heading sample at half weight; not new actions',
        labels='real10Hz states at5Hz; navigation/heading STOP x2; altitude settled-only, abs(error).06-.14m symmetric x4',
        selection='fixed final weights, all3 seeds frozen before sealed; no heldout tuning',
        gate=dict(each_seed_success_min=.9,clean_success_min=.9,other_profile_min=.85,collision_or_bounds_max=.01,advantage_over_random_min=.2,
            instruction_successes=8,boundary_successes_min=11,zero_successes_max=0,rule_successes_min=99,
            physical_continuity=True,vertical_speed_tolerance=.08,non1m_navigation_both_sides=True),
        split_hashes={k:digest(v) for k,v in cases.items()})
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol);return cases

def train(root,seed,smoke=False):
    root=Path(root);prepare(root);out=run_directory(root,seed,smoke)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=SpatialPool(root,seed,batch=BATCH);sources=source_hashes(root);original=pool.original_sources
    pool.training_method=METHOD;rows={};opts={};initial={};provenance=[];stages=[];options=0;status='failed';envs=[]
    for skill in SKILLS:
        model=pool.models[skill]
        for p in model.parameters():p.requires_grad_(True)
        initial[skill]={k:v.detach().clone() for k,v in model.state_dict().items()}
        opts[skill]=torch.optim.AdamW(list(model.body.parameters())+list(model.actor.parameters()),lr=.0003,weight_decay=.001)
        if skill=='navigation':rows[skill]=[]
        else:
            data=np.load(root/f'runs/tellosim-sdk9/spatial-refined-s{seed}/{skill}/demonstrations.npz')
            rows[skill]=[(x.copy(),m.copy(),int(y),float(w)) for x,m,y,w in zip(data['features'][::4],data['masks'][::4],data['labels'][::4],data['weights'][::4]*.5)]
    old_counts={s:len(rows[s]) for s in SKILLS}
    save_spatial(out/'initial.pt',pool,None,root,0,dict(method='independent navigation initialization; altitude/heading from preservedC2R; no C2P training'))
    def annotate(env,x,kind):
        label=label_for(env);weight=(1. if kind=='decision' else .25)
        if env.phase=='altitude':weight*=4. if .06<=abs(float(env.observation[2])*3)<=.14 else 1.
        elif label==0 and env.previous_action is not None:weight*=2.
        rows[env.phase].append((x.copy(),env.mask.copy(),label,weight))
        provenance.append(dict(seed=env.case['seed'],stage=stage,skill=env.phase,step=env.steps,sample_id=env.agent.sample_id,
            profile=env.disturbance_profile,kind=kind,label=label,weight=weight,measured_height_m=env.device.latest_observation().position_m[2],target_height_m=env.case['goal'][2]))
        return label
    def stop(signum,frame):raise KeyboardInterrupt('SIGTERM')
    previous_signal=signal.signal(signal.SIGTERM,stop)
    try:
        for stage,count in enumerate((512,128) if smoke else STAGES):
            envs=[None]*BATCH;episodes=[0]*BATCH;done=[0]*BATCH;gens={};outcomes=[];ticks=0;stage_actions={s:0 for s in SKILLS}
            while any(n<count//BATCH for n in done):
                if time.monotonic()-started>5400:raise TimeoutError('stability training wall budget')
                pending={}
                for i in range(BATCH):
                    if done[i]>=count//BATCH:continue
                    if envs[i] is None:
                        if episodes[i]>=500:raise RuntimeError('training seed range exhausted')
                        sample=(390000000 if smoke else 380000000)+seed*100000+stage*10000+i*500+episodes[i];episodes[i]+=1
                        envs[i]=SpatialEnv(root,training_case(sample,i),pool.lanes[i])
                    env=envs[i]
                    if i not in gens:
                        label=annotate(env,env.features,'decision')
                        if stage==0:a,policy=label,None
                        else:a,_,_,policy=pool.lanes[i].decision(env.features,env.mask,deterministic=i%2==0)
                        stage_actions[env.phase]+=1;gens[i]=env.step_iter(a,policy)
                    try:next(gens[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                    except StopIteration:
                        del gens[i];done[i]+=1;options+=1
                        if env.terminated:outcomes.append(env.reason);env.close();envs[i]=None
                if pending:
                    features=pool.observe_lanes(pending);ticks+=len(pending)
                    for i in pending:
                        env=envs[i];env.features=features[i].copy()
                        if env.agent.sample_id%2==0 and (env.phase!='altitude' or (env.previous_duration>=.5 and (env.previous_action==0 or env.device.observed_operation()['client']!='sent'))):annotate(env,env.features,'intermediate')
            for env in envs:
                if env:env.close('collection_stage_complete')
            envs=[];metrics={};advances=pool.brain.advance_count
            for skill in SKILLS:
                metrics[skill]=fit(pool.models[skill],opts[skill],rows[skill],2 if smoke else 80)
            assert pool.brain.advance_count==advances
            save_spatial(out/f'stage-{stage}.pt',pool,opts,root,options,dict(method=METHOD))
            stages.append(dict(stage=stage,options=sum(done),base_ticks=ticks,action_skills=stage_actions,outcomes=outcomes,metrics=metrics))
            atomic_json(out/'progress.json',dict(options=options,stages=stages,elapsed_s=time.monotonic()-started))
            print(json.dumps(dict(seed=seed,options=options,metrics=metrics)),flush=True)
        checkpoint=out/'checkpoint.pt';save_spatial(checkpoint,pool,opts,root,options,dict(method=METHOD))
        expected={}
        for skill in SKILLS:
            data=rows[skill][-16:]
            with torch.no_grad():expected[skill]=distribution(pool.models[skill],torch.tensor(np.stack([r[0] for r in data])),np.stack([r[1] for r in data]))[0].probs.clone()
        load_spatial(checkpoint,pool,root);training={}
        for skill in SKILLS:
            data=rows[skill][-16:]
            with torch.no_grad():actual=distribution(pool.models[skill],torch.tensor(np.stack([r[0] for r in data])),np.stack([r[1] for r in data]))[0].probs
            assert torch.equal(expected[skill],actual)
            delta=float(torch.sqrt(sum((initial[skill][k]-v).square().sum() for k,v in pool.models[skill].state_dict().items())))
            assert delta>0
            training[skill]=dict(examples=len(rows[skill]),rehearsal_examples=old_counts[skill],new_examples=len(rows[skill])-old_counts[skill],
                options=sum(s['action_skills'][skill] for s in stages),parameter_delta_l2=delta,roundtrip_exact=True)
            if not smoke:assert training[skill]['options']>0
        assert source_hashes(root)==sources and all(sha(root/v['path'])==v['sha256'] for v in original.values())
        status='completed'
        atomic_json(out/'training.json',dict(status=status,smoke_only=smoke,seed=seed,options=options,method=METHOD,
            source_hashes=sources,original_sources=original,checkpoint_sha256=sha(checkpoint),skills=training,stages=stages,elapsed_s=time.monotonic()-started))
    except BaseException as exc:atomic_json(out/'failure.json',dict(error=repr(exc),options=options));raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        signal.signal(signal.SIGTERM,previous_signal);atomic_json(out/'provenance.json',provenance)
        for skill in SKILLS:
            (out/skill).mkdir(exist_ok=True)
            np.savez_compressed(out/skill/'demonstrations.npz',features=np.stack([r[0] for r in rows[skill]]),masks=np.stack([r[1] for r in rows[skill]]),labels=np.array([r[2] for r in rows[skill]]),weights=np.array([r[3] for r in rows[skill]]))

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

def freeze(root):
    root=Path(root);cases=prepare(root);models=[]
    for seed in SEEDS:
        folder=run_directory(root,seed);t=json.loads((folder/'training.json').read_text());checkpoint=folder/'checkpoint.pt'
        assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
        assert t['source_hashes']==source_hashes(root) and t['checkpoint_sha256']==sha(checkpoint)
        assert all(t['skills'][k]['parameter_delta_l2']>0 and t['skills'][k]['options']>0 for k in SKILLS)
        models.append(dict(seed=seed,path=str(checkpoint.relative_to(root)),sha256=sha(checkpoint)))
    frozen_json(directory(root)/'frozen-checkpoints.json',dict(models=models,case_hash=digest(cases['sealed_test'])))

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
        if not target.exists():evaluate(root,checkpoint,cases[split],target,f'spatial-precision-s{seed}-{split}',batch=min(64,len(cases[split])),zero=split=='zero_cases',record_indices=(0,1,2,3) if split=='sealed_test' else ())
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
        models=rows,acceptance=gate,new_training_actions=len(SEEDS)*sum(STAGES),rule_successes=load('rule-validation.json')['successes'],random_successes=random['successes'],scope='simulated empty room, external pose mock, frozen MaleCNS, three jointly exercised readouts')
    atomic_json(out/'summary.json',summary);print(json.dumps(summary));return summary

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
