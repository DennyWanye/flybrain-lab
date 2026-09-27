"""J2: train both skill readouts under mild perturbations; freeze then compare.

No teacher runs in learned evaluation. J1R inputs, physics, SDK actions and
acceptance are preserved. This is readout imitation/DAgger, not full-brain PPO.
"""
import argparse, copy, json, math, signal, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .robust_env import (PROFILES, DISTURBANCE_SPEC, with_profile,
                        RobustNavigationEnv, RobustHeadingEnv, RobustJointEnv)
from .joint import joint_case, JOINT_SPEC, sha
from .joint_refine import RefinedPool, refined_bundle, settling_label, fit_settling, training_case as navigation_case
from .heading import heading_case, teacher, wrap_angle, load_heading, save_heading
from .joint_campaign import frozen_json, result_row as joint_result, JointDiagnostic
from .parallel import ReservoirPool, collect_rollout
from .runtime import load_checkpoint, save_checkpoint, distribution
from .checkpoint import configure_exact_execution
from .contracts import digest
from .observer import TrainingObserver
from .c0_campaign import summarize, acceptance
from ..visual import atomic_json

SEEDS = (11,22,33)
SKILLS = ('navigation','heading')
STAGES = (512,768)
BATCH = 16
SOURCE_FILES = ('flydrone/tellosim/training/robust_env.py', 'flydrone/tellosim/training/robust_campaign.py')
METHOD = 'J2: each skill 512 teacher + 768 DAgger actions with four mild-disturbance profiles, old feature rehearsal; frozen MaleCNS; no PPO'

def directory(root): return Path(root)/'reports/ts1_robust'
def run_directory(root,seed,smoke=False): return Path(root)/f'runs/tellosim-sdk9/robust-{"smoke-" if smoke else ""}s{seed}'
def source_hashes(root): return {name:sha(Path(root)/name) for name in SOURCE_FILES}
def old_bundle(root,seed):
    path=Path(root)/f'runs/tellosim-sdk9/joint-refined-s{seed}/policy-bundle.json'
    value=json.loads(path.read_text())
    if value!=refined_bundle(root,seed): raise ValueError('J1R source bundle changed')
    return value

def training_case(seed,lane,skill):
    if skill=='navigation': case=navigation_case(seed,lane//4)
    elif skill=='heading':
        case=heading_case(seed);rng=np.random.default_rng(seed+81)
        if lane//4<2:
            delta=float(rng.choice([-1,1])*math.radians(rng.uniform(10,21)))
            case['target_yaw_rad']=wrap_angle(case['initial_yaw_rad']+delta)
    else: raise ValueError('unknown skill')
    return with_profile(case,tuple(PROFILES)[lane%4])

def case_set(base,count,label):
    return [with_profile(joint_case(base+i,f'j2-{label}-{i:03d}'),tuple(PROFILES)[i%4]) for i in range(count)]

def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(parents=True,exist_ok=True)
    cases={'validation':case_set(113000000,100,'val'), 'sealed_test':case_set(114000000,300,'test'), 'instruction_pairs':[]}
    for pair,yaw in enumerate((0.,math.radians(170))):
        for sign in (-1,1):
            c=joint_case(115000000+pair,f'j2-pair-{pair}-{sign}')
            c.update(start=[0.,0.,1.],goal=[1.,0.,1.],initial_yaw_rad=yaw,target_yaw_rad=wrap_angle(yaw+sign*math.pi/2))
            cases['instruction_pairs'].append(with_profile(c,'combined'))
    protocol={'seeds':list(SEEDS),'stages_per_skill':list(STAGES),'batch':BATCH,'epochs':150,'learning_rate':.0005,
        'new_actions_per_seed':2*sum(STAGES),'method':METHOD,'task_spec':JOINT_SPEC,'disturbance':DISTURBANCE_SPEC,
        'training_seed_formula':'120000000 + model_seed*100000 + skill_index*40000 + stage*10000 + lane*500 + episode',
        'training_curriculum':'navigation half near goal; heading half 10-21deg boundary; equal four disturbance profiles; all old feature examples rehearsed',
        'selection':'fixed final readouts; freeze all three seeds before any sealed evaluation; no sealed tuning',
        'gate':{'each_seed_success_min':.9,'each_profile_success_min':.85,'clean_profile_success_min':.9,
                'collision_or_bounds_max':.01,'advantage_over_uniform_min':.2,'instruction_pairs_required':4,
                'zero_features_success_max':0,'clean_retention_max_drop_pp':5.,'disturbed_mean_must_not_drop':True},
        'comparison':'J1R and J2 on identical 300 cases per seed; both use the same disturbances and unchanged J1 task',
        'split_hashes':{k:digest(v) for k,v in cases.items()},'source_hashes':source_hashes(root),
        'original_bundles':{str(s):old_bundle(root,s) for s in SEEDS}}
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol)
    return cases

def save_skill(path,pool,opt,root,skill,options,updates,extra):
    if skill=='heading':save_heading(path,pool,opt,root,options,extra)
    else:save_checkpoint(path,pool,opt,root,updates,options,extra)

def load_skill(path,pool,root,skill):
    return load_heading(path,pool,root) if skill=='heading' else load_checkpoint(path,pool,root)

def train_skill(root,seed,skill,smoke=False):
    root=Path(root);prepare(root);out=run_directory(root,seed,smoke)/skill
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();sources=source_hashes(root)
    torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=ReservoirPool(root/'data/male-v1.npz','cuda',seed=seed,profile='balanced_rate_v3',batch=BATCH,
                       physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
    original=old_bundle(root,seed)['sources'][skill];load_skill(root/original['path'],pool,root,skill)
    before={k:v.detach().clone() for k,v in pool.model.state_dict().items()}
    pool.training_method=METHOD;pool.value_trained=False
    previous_folder='joint-refined' if skill=='navigation' else 'heading'
    prior=np.load(root/f'runs/tellosim-sdk9/{previous_folder}-s{seed}/demonstrations.npz')
    prior_weights=prior['weights'] if 'weights' in prior else np.ones(len(prior['labels']))
    rows=[(x.copy(),m.copy(),int(y),float(w)) for x,m,y,w in zip(prior['features'],prior['masks'],prior['labels'],prior_weights)]
    opt=torch.optim.Adam(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0005)
    observer=TrainingObserver(root,f'robust-{"smoke-" if smoke else ""}s{seed}-{skill}',BATCH,METHOD)
    provenance=[];stages=[];options=0;updates=0;envs=[];status='failed'
    def stop(signum,frame):raise KeyboardInterrupt('SIGTERM')
    old_signal=signal.signal(signal.SIGTERM,stop)
    try:
        for stage,count in enumerate((16,) if smoke else STAGES):
            envs=[None]*BATCH;episodes=[0]*BATCH
            def make(i):
                if episodes[i]>=500:raise RuntimeError('training lane seed range exhausted')
                sample=(130000000 if smoke else 120000000)+seed*100000+SKILLS.index(skill)*40000+stage*10000+i*500+episodes[i];episodes[i]+=1
                cls=RobustNavigationEnv if skill=='navigation' else RobustHeadingEnv
                return cls(root,training_case(sample,i,skill),pool.lanes[i],observer=observer)
            def decide(env,i,x,mask):
                label=(settling_label if skill=='navigation' else teacher)(env.observation,mask,env.steps==0)
                weight=2. if label==0 and env.steps>0 else 1.
                rows.append((x.copy(),mask.copy(),label,weight))
                provenance.append({'seed':env.case['seed'],'skill':skill,'stage':stage,'step':env.steps,
                                   'profile':env.disturbance_profile,'label':label,'weight':weight})
                if stage==0:return label,0.,0.,None
                return pool.lanes[i].decision(x,mask,deterministic=i%2==0)
            try:collected,ticks,outcomes=collect_rollout(pool,envs,[count//BATCH]*BATCH,make,deadline=started+2400,decision_fn=decide)
            finally:
                for env in envs:
                    if env:env.close('collection_stage_complete')
                envs=[]
            options+=len(collected);metrics=fit_settling(pool,opt,rows,epochs=2 if smoke else 150);updates+=metrics['gradient_steps']
            stages.append({'stage':stage,'options':len(collected),'base_ticks':ticks,'outcomes':outcomes,**metrics})
            save_skill(out/f'stage-{stage}.pt',pool,opt,root,skill,options,updates,{'method':METHOD,'sources':sources,'source':original})
            observer.update({'options':options,'updates':updates,'elapsed_s':time.monotonic()-started})
            atomic_json(out/'progress.json',{'options':options,'stages':stages})
            print(json.dumps({'seed':seed,'skill':skill,'options':options,**metrics}),flush=True)
        checkpoint=out/'checkpoint.pt';save_skill(checkpoint,pool,opt,root,skill,options,updates,{'method':METHOD,'sources':sources,'source':original})
        expected=[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]]
        load_skill(checkpoint,pool,root,skill)
        assert expected==[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]]
        delta=float(torch.sqrt(sum((before[k]-v).square().sum() for k,v in pool.model.state_dict().items())))
        assert delta>0 and source_hashes(root)==sources and sha(root/original['path'])==original['sha256']
        status='completed';atomic_json(out/'training.json',{'status':status,'smoke_only':smoke,'seed':seed,'skill':skill,
            'options':options,'updates':updates,'new_examples':len(provenance),'rehearsal_examples':len(prior['labels']),
            'parameter_delta_l2':delta,'source_hashes':sources,'source':original,'method':METHOD,
            'checkpoint_sha256':sha(checkpoint),'roundtrip_exact':True,'stages':stages,'elapsed_s':time.monotonic()-started})
    except BaseException as exc:atomic_json(out/'failure.json',{'error':repr(exc),'options':options});raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        observer.close(status);signal.signal(signal.SIGTERM,old_signal)
        atomic_json(out/'provenance.json',provenance)
        np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]),
                            labels=np.array([r[2] for r in rows]),weights=np.array([r[3] for r in rows]))

def bundle_spec(root,seed):
    root=Path(root);old=old_bundle(root,seed);sources={};training={}
    for skill in SKILLS:
        folder=run_directory(root,seed)/skill;t=json.loads((folder/'training.json').read_text())
        assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
        assert t['source_hashes']==source_hashes(root) and t['checkpoint_sha256']==sha(folder/'checkpoint.pt')
        sources[skill]={'path':str((folder/'checkpoint.pt').relative_to(root)),'sha256':sha(folder/'checkpoint.pt')}
        training[skill]=sha(folder/'training.json')
    return {**old,'format':'tellosim.robust_skill_bundle/1','method':METHOD,'stage':'J2','sources':sources,
            'source_hashes':{**old['source_hashes'],**source_hashes(root)},'training_sha256':training,
            'new_training_actions':2*sum(STAGES),'disturbance_spec':DISTURBANCE_SPEC}

class RobustPool(RefinedPool):
    def __init__(self,root,bundle,batch=64):
        value=json.loads(Path(bundle).read_text()) if isinstance(bundle,(str,Path)) else bundle
        if value!=bundle_spec(root,value['seed']):raise ValueError('robust source contract mismatch')
        super().__init__(root,old_bundle(root,value['seed']),batch=batch)
        for skill in SKILLS:
            load_skill(Path(root)/value['sources'][skill]['path'],self,root,skill)
            self.models[skill]=copy.deepcopy(self.model).eval();self.value_flags[skill]=False
            for param in self.models[skill].parameters():param.requires_grad_(False)
        self.bundle=value;self.training_method=METHOD;self.checkpoint_sha256=digest(value)

def freeze(root):
    cases=prepare(root);models=[]
    for seed in SEEDS:
        path=run_directory(root,seed)/'policy-bundle.json';frozen_json(path,bundle_spec(root,seed))
        models.append({'seed':seed,'path':str(path.relative_to(root)),'sha256':sha(path)})
    frozen_json(directory(root)/'frozen-bundles.json',{'models':models,'case_hash':digest(cases['sealed_test'])})

def result_row(env,case,actions,run_id,skills):
    return {**joint_result(env,case,actions,run_id,skills),'disturbance':env.disturbance_evidence()}

def grouped(results):
    return {profile:{'episodes':len(rows),'successes':sum(r['success'] for r in rows),
                     'success_rate':sum(r['success'] for r in rows)/len(rows) if rows else 0.}
            for profile in PROFILES for rows in [[r for r in results if r['disturbance']['profile']==profile]]}

def evaluate(root,seed,cases,out,label,before=False,zero=False,record_indices=(),batch=64):
    root=Path(root);out=Path(out);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    bundle=old_bundle(root,seed) if before else bundle_spec(root,seed)
    pool=RefinedPool(root,bundle,batch=batch) if before else RobustPool(root,bundle,batch=batch)
    envs=[None]*batch;generators={};indices={};actions={};skills={};results={};cursor=0
    try:
        while len(results)<len(cases):
            if time.monotonic()-started>4200:raise TimeoutError('robust evaluation wall budget exhausted')
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;actions[index]=[];skills[index]=[]
                    envs[i]=RobustJointEnv(root,cases[index],pool.lanes[i],record=index in record_indices,
                        run_id=f'{label}-{index}' if index in record_indices else None,
                        policy_source='robust_ZERO_FEATURE_ABLATION' if zero else 'robust_argmax_malecns')
                env=envs[i]
                if i not in generators:
                    x=np.zeros_like(env.features) if zero else env.features
                    a,_,_,policy=pool.lanes[i].decision(x,env.mask,True)
                    actions[indices[i]].append(a);skills[indices[i]].append(env.phase);generators[i]=env.step_iter(a,policy)
                try:next(generators[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                except StopIteration:
                    del generators[i]
                    if env.terminated:
                        index=indices[i];results[index]=result_row(env,env.case,actions[index],env.session.run_id if env.live else None,skills[index])
                        env.close();envs[i]=None
                        if len(results)%10==0:
                            atomic_json(out.with_suffix('.progress.json'),summarize([results[k] for k in sorted(results)],label))
                            print(json.dumps({'label':label,'cases':len(results),'successes':sum(x['success'] for x in results.values())}),flush=True)
            if pending:
                features=pool.observe_lanes(pending)
                for i in pending:envs[i].features=features[i].copy();envs[i].publish()
    finally:
        for env in envs:
            if env:env.close()
    rows=[results[i] for i in range(len(cases))]
    result={**summarize(rows,label),'profiles':grouped(rows),'mode':'zero_features_ablation' if zero else 'argmax',
            'case_hash':digest(cases),'bundle_hash':digest(bundle),'source_weights':bundle['sources'],
            'runtime_source_hashes':source_hashes(root),'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result

def baseline(root,cases,mode,out):
    results=[];started=time.monotonic();agent=JointDiagnostic()
    for index,case in enumerate(cases):
        env=RobustJointEnv(root,case,agent);rng=np.random.default_rng(case['seed']+991);actions=[];skills=[]
        try:
            while not env.terminated:
                a=((settling_label(env.observation,env.mask,env.steps==0) if env.phase=='navigation' else teacher(env.observation,env.mask))
                   if mode=='rule' else int(rng.choice(np.flatnonzero(env.mask))))
                actions.append(a);skills.append(env.phase);env.step(a)
            results.append(result_row(env,case,actions,None,skills))
        finally:env.close()
        if (index+1)%10==0:
            atomic_json(out.with_suffix('.progress.json'),summarize(results,'robust-'+mode))
            print(json.dumps({'baseline':mode,'cases':index+1}),flush=True)
    result={**summarize(results,'robust-'+mode),'profiles':grouped(results),'mode':mode,
            'case_hash':digest(cases),'elapsed_s':time.monotonic()-started,'neural_model':False}
    atomic_json(out,result);return result

def evaluate_seed(root,seed):
    out=directory(root);cases=prepare(root)
    lock=json.loads((out/'frozen-bundles.json').read_text())
    entry=next(m for m in lock['models'] if m['seed']==seed)
    assert sha(Path(root)/entry['path'])==entry['sha256']
    for split in ('validation','sealed_test','instruction_pairs'):
        path=out/f's{seed}-{split}.json'
        if not path.exists():evaluate(root,seed,cases[split],path,f'robust-s{seed}-{split}',
            record_indices=(0,1,2,3) if split=='sealed_test' else (),batch=min(64,len(cases[split])))
    path=out/f's{seed}-zero-ablation.json'
    if not path.exists():evaluate(root,seed,cases['validation'][:12],path,f'robust-zero-s{seed}',zero=True,batch=12)
    path=out/f's{seed}-before-sealed.json'
    if not path.exists():evaluate(root,seed,cases['sealed_test'],path,f'robust-before-s{seed}',before=True,record_indices=(3,),batch=64)
    assert sha(Path(root)/entry['path'])==entry['sha256']

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root)
    def load(name):return json.loads((out/name).read_text())
    results={s:load(f's{s}-sealed_test.json') for s in SEEDS}
    random=load('random-sealed.json');gate=acceptance(results,random,digest(cases['sealed_test']));rows=[];old_disturbed=0;new_disturbed=0
    for seed,result in results.items():
        bundle=bundle_spec(root,seed);before=load(f's{seed}-before-sealed.json')
        assert result['bundle_hash']==digest(bundle) and before['case_hash']==result['case_hash']
        assert before['bundle_hash']==digest(old_bundle(root,seed))
        valid=load(f's{seed}-validation.json');pairs=load(f's{seed}-instruction_pairs.json');zero=load(f's{seed}-zero-ablation.json')
        if pairs['successes']!=4:gate['reasons'].append(f'{seed}: instruction pairs failed')
        if zero['successes']:gate['reasons'].append(f'{seed}: zero feature diagnostic succeeded')
        for profile,m in result['profiles'].items():
            threshold=.9 if profile=='clean' else .85
            if m['success_rate']+1e-10<threshold:gate['reasons'].append(f'{seed}: {profile} success below {threshold:.0%}')
            if profile!='clean':old_disturbed+=before['profiles'][profile]['successes'];new_disturbed+=m['successes']
        if result['profiles']['clean']['success_rate']+.05+1e-10<before['profiles']['clean']['success_rate']:
            gate['reasons'].append(f'{seed}: clean retention dropped more than 5pp')
        actions=[a for r in result['results'] for a in r['actions']]
        if not 7 in actions or not 8 in actions:gate['reasons'].append(f'{seed}: missing turn direction')
        for r in result['results']:
            if len(r['actions'])!=len(r['action_skills']) or any(a not in JOINT_SPEC['allowed_actions'][s] for a,s in zip(r['actions'],r['action_skills'])):gate['reasons'].append(f'{seed}: invalid phase action')
            if r['brain_tick']!=4*(1+r['physics_ticks']//12):gate['reasons'].append(f'{seed}: neural clock mismatch')
            if any(not e['physics_state_unchanged'] for e in r['phase_events']):gate['reasons'].append(f'{seed}: physical reset')
            if r['disturbance']['profile'] in ('force','combined') and r['duration_s']>7 and r['disturbance']['absolute_impulse_ns']<=0:gate['reasons'].append(f'{seed}: missing physical disturbance')
            if r['success'] and not (r['final_phase']=='heading' and r['phase_events'] and r['horizontal_distance_m']<=.2 and r['height_error_m']<=.1 and r['horizontal_speed_mps']<=.08 and r['heading_error_deg']<=16 and r['angular_speed_rad_s']<=.08 and r['stable_hold_s']>=2-1e-8):gate['reasons'].append(f'{seed}: invalid joint success')
        rows.append({'seed':seed,'validation':f"{valid['successes']}/100",'sealed':f"{result['successes']}/300",
            'success_rate':result['success_rate'],'before_sealed':f"{before['successes']}/300",'profiles':result['profiles'],
            'before_profiles':before['profiles'],'collision_or_bounds':result['collision_or_bounds'],
            'instruction_pairs_successes':pairs['successes'],'zero_features_successes':zero['successes'],
            'run_id':result['results'][3]['run_id'],'before_run_id':before['results'][3]['run_id'],
            'sample_outcome':result['results'][3]['reason'],'mean_duration_s':result['mean_duration_s']})
    if new_disturbed<old_disturbed:gate['reasons'].append('disturbed success declined versus J1R on paired cases')
    gate['passed']=not gate['reasons'];passed=gate['passed']
    summary={'stage':'J2','status':'evaluated','COMPOSED_JOINT_TASK_VERIFIED':passed,'MODEL_READY_FOR_NEXT_STAGE':passed,
        'SINGLE_POLICY_JOINT_TRAINED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False,
        'readiness_scope':'fixed-height empty-room composed task under four specified mild disturbance profiles',
        'models':rows,'acceptance':gate,'random_successes':random['successes'],'new_training_actions':len(SEEDS)*2*sum(STAGES),
        'disturbed_paired_successes':{'before':old_disturbed,'after':new_disturbed,'episodes':675},
        'note':'J2：导航和转向读出各新增1280个训练动作；共享冻结MaleCNS。四类轻微扰动，新旧模型同场对照，标准保持不变。快捷入口固定为封存第4场（延迟、噪声和外力叠加），未筛选样例。仍是两个技能组合。'}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','smoke','freeze','evaluate','baselines','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11)
    a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command in ('train','smoke'):
        for skill in SKILLS:train_skill(root,a.seed,skill,a.command=='smoke')
    elif a.command=='freeze':freeze(root)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    elif a.command=='baselines':
        cases=prepare(root)
        for split,mode in [('validation','rule'),('sealed_test','random')]:
            target=out/('rule-validation.json' if mode=='rule' else 'random-sealed.json')
            if not target.exists():baseline(root,cases[split],mode,target)
    else:finalize(root)
if __name__=='__main__':main()
