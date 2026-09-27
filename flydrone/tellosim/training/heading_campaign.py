"""Fixed-budget heading-only readout, preserved navigation skill, sealed acceptance."""
import argparse,hashlib,json,math,time,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .heading import HeadingEnv,HEADING_SPEC,heading_case,wrap_angle,teacher,save_heading,load_heading
from .parallel import ReservoirPool,collect_rollout
from .checkpoint import configure_exact_execution
from .bootstrap import fit
from .c0_campaign import summarize,acceptance
from .campaign_v2 import evaluate_batch
from .contracts import digest
from .observer import TrainingObserver
from ..visual import atomic_json
from ..physics.rigid import PROFILE

SEEDS=(11,22,33);STAGES=(384,448,448)
METHOD='Heading-only learned readout: 384 demonstrations + 2x448 DAgger; frozen original MaleCNS; navigation weights preserved; no PPO'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def directory(root):return Path(root)/'reports/ts1_heading'
def prepare(root):
    out=directory(root);out.mkdir(parents=True,exist_ok=True)
    cases={'schema':'tellosim.heading_cases/1','validation':[heading_case(51000000+i,f'h1-val-{i:03d}') for i in range(100)],
        'sealed_test':[heading_case(52000000+i,f'h1-test-{i:03d}') for i in range(300)],
        'retention':json.loads((Path(root)/'reports/ts1_c1/cases.json').read_text())['c0_retention']}
    cases['instruction_pairs']=[]
    for pair,yaw in enumerate((0.,math.radians(170))):
        for sign in (-1,1):
            case=heading_case(53000000+pair,f'instruction-{pair}-{sign}');case.update(initial_yaw_rad=yaw,target_yaw_rad=wrap_angle(yaw+sign*math.pi/2));cases['instruction_pairs'].append(case)
    protocol={'seeds':list(SEEDS),'stages':list(STAGES),'epochs':150,'learning_rate':.0005,'wall_seconds_per_seed':1800,
        'method':METHOD,'task_spec':HEADING_SPEC,'selection':'all fixed final checkpoints frozen before sealed results; no sealed tuning',
        'gate':{'each_seed_success_min':.9,'collision_or_bounds_max':.01,'advantage_over_uniform_min':.2,'must_use_cw_and_ccw':True,'retention':'same frozen navigation weights and same 100 cases must match prior results'},
        'split_hashes':{k:digest(v) for k,v in cases.items() if isinstance(v,list)},
        'navigation_sources':{str(s):{'path':f'runs/tellosim-sdk9/c1-s{s}/ppo/checkpoint.pt','sha256':sha(Path(root)/f'runs/tellosim-sdk9/c1-s{s}/ppo/checkpoint.pt')} for s in SEEDS}}
    for name,value in [('cases.json',cases),('protocol.json',protocol)]:
        p=out/name
        if p.exists() and json.loads(p.read_text())!=value:raise ValueError(f'frozen protocol changed: {name}')
        atomic_json(p,value)
    return cases

def pool_for(root,seed,batch):
    torch.set_num_threads(4);configure_exact_execution('cuda')
    return ReservoirPool(Path(root)/'data/male-v1.npz','cuda',seed=seed,profile='balanced_rate_v3',batch=batch,physics_profile=PROFILE,neural_backend='csr_fp64_accum')

def train(root,seed):
    root=Path(root).resolve();prepare(root);out=root/f'runs/tellosim-sdk9/heading-s{seed}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();pool=pool_for(root,seed,4);pool.training_method=METHOD
    save_heading(out/'initial.pt',pool,None,root,0,{'seed':seed,'status':'untrained heading readout'})
    optimizer=torch.optim.Adam(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0005)
    observer=TrainingObserver(root,f'heading-s{seed}',4,METHOD)
    rows=[];provenance=[];stages=[];total=0;envs=[];status='failed'
    def terminate(signum,frame):raise KeyboardInterrupt('SIGTERM')
    previous=signal.signal(signal.SIGTERM,terminate)
    try:
        for stage,count in enumerate(STAGES):
            envs=[None]*4;episodes=[0]*4
            def make_env(i):
                sample_seed=60000000+seed*100000+stage*10000+i*1000+episodes[i];episodes[i]+=1
                return HeadingEnv(root,heading_case(sample_seed),pool.lanes[i],observer=observer)
            def decide(env,i,x,mask):
                label=teacher(env.observation,mask,env.steps==0);rows.append((x.copy(),mask.copy(),label))
                provenance.append({'seed':env.case['seed'],'stage':stage,'step':env.steps,'teacher_label':label})
                if stage==0:return label,0.,0.,None
                return pool.lanes[i].decision(x,mask,deterministic=i%2==0)
            try:collected,ticks,outcomes=collect_rollout(pool,envs,[count//4]*4,make_env,deadline=started+1800,decision_fn=decide)
            finally:
                for env in envs:
                    if env:env.close('stage_boundary')
                envs=[]
            total+=len(collected);metrics=fit(pool,optimizer,rows,epochs=150)
            stages.append({'stage':stage,'options':len(collected),'base_ticks':ticks,'outcomes':outcomes,**metrics})
            save_heading(out/f'stage-{stage}.pt',pool,optimizer,root,total,{'seed':seed,'method':METHOD})
            observer.update({'options':total,'updates':sum(x['gradient_steps'] for x in stages),'elapsed_s':time.monotonic()-started})
            atomic_json(out/'progress.json',{'options':total,'elapsed_s':time.monotonic()-started,'stages':stages})
            print(json.dumps({'seed':seed,'phase':'heading-fit','options':total,**metrics}),flush=True)
        save_heading(out/'checkpoint.pt',pool,optimizer,root,total,{'seed':seed,'method':METHOD})
        expected=[pool.decision(x,m,True)[3] for x,m,_ in rows[:32]];load_heading(out/'checkpoint.pt',pool,root)
        assert expected==[pool.decision(x,m,True)[3] for x,m,_ in rows[:32]]
        status='completed';atomic_json(out/'training.json',{'status':status,'seed':seed,'options':total,'elapsed_s':time.monotonic()-started,'method':METHOD,'stages':stages,'roundtrip_exact':True,'checkpoint_sha256':sha(out/'checkpoint.pt')})
    except BaseException as exc:
        atomic_json(out/'failure.json',{'error':type(exc).__name__+': '+str(exc),'collected_examples':len(rows),'committed_stage_options':total});raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        observer.close(status);signal.signal(signal.SIGTERM,previous)
        if rows:np.savez_compressed(out/'demonstrations.npz',features=np.stack([x for x,_,_ in rows]),masks=np.stack([m for _,m,_ in rows]),labels=np.array([a for _,_,a in rows]))
        atomic_json(out/'provenance.json',provenance)

class RuleDiagnostic:
    """Explicit non-neural rule/random baseline, never saved as a brain model."""
    profile='balanced_rate_v3';stop_dwell_s=2.;physics_profile=PROFILE;feature_source='rule_or_random_baseline'
    def __init__(self):self.brain=SimpleNamespace(graph_sha256=None,mapping_sha256=None);self.reset()
    def reset(self):self.brain_tick=0;self.sample_id=-1
    def observe(self,vector,sample_id):self.brain_tick+=4;self.sample_id=sample_id;return np.asarray(vector).copy()

def result_row(env,case,actions,run_id):
    return {'case_id':case['case_id'],'seed':case['seed'],'reason':env.reason,'success':env.reason=='success','return':env.raw_return,
        'steps':env.steps,'duration_s':(env.session.tick-env.start_tick)/120,'distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
        'heading_error_deg':math.degrees(abs(wrap_angle(env.target_yaw-env.session.world.yaw_rad))),
        'initial_error_deg':math.degrees(wrap_angle(case['target_yaw_rad']-case['initial_yaw_rad'])),
        'angular_speed_rad_s':abs(float(env.session.world.data.qvel[5])),'stable_hold_s':env.hold,'run_id':run_id,'actions':actions}

def baseline(root,cases,mode,out):
    results=[];started=time.monotonic();agent=RuleDiagnostic()
    for index,case in enumerate(cases):
        env=HeadingEnv(root,case,agent);rng=np.random.default_rng(case['seed']+991);actions=[]
        try:
            while not env.terminated:
                a=teacher(env.observation,env.mask,env.steps==0) if mode=='rule' else int(rng.choice(np.flatnonzero(env.mask)))
                actions.append(a);env.step(a)
            results.append(result_row(env,case,actions,None))
        finally:env.close()
        if (index+1)%10==0:atomic_json(out.with_suffix('.progress.json'),summarize(results,'heading-'+mode));print(json.dumps({'baseline':mode,'cases':index+1,'successes':sum(x['success'] for x in results)}),flush=True)
    result={**summarize(results,'heading-'+mode),'mode':mode,'case_hash':digest(cases),'elapsed_s':time.monotonic()-started,'neural_model':False};atomic_json(out,result);return result

def evaluate(root,checkpoint,cases,out,label,record_indices=(0,1),batch=64,zero_features=False):
    root=Path(root).resolve();out=Path(out);started=time.monotonic();pool=pool_for(root,11,batch);load_heading(checkpoint,pool,root)
    envs=[None]*batch;generators={};indices={};actions={};results={};cursor=0
    try:
        while len(results)<len(cases):
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;actions[index]=[]
                    envs[i]=HeadingEnv(root,cases[index],pool.lanes[i],record=index in record_indices,run_id=f'{label}-{index}' if index in record_indices else None,policy_source='heading_argmax_malecns' if not zero_features else 'heading_ZERO_FEATURE_ABLATION')
                env=envs[i]
                if i not in generators:
                    x=np.zeros_like(env.features) if zero_features else env.features
                    a,_,_,policy=pool.lanes[i].decision(x,env.mask,True);actions[indices[i]].append(a);generators[i]=env.step_iter(a,policy)
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
    result={**summarize([results[i] for i in range(len(cases))],label),'mode':'argmax' if not zero_features else 'zero_features_ablation','case_hash':digest(cases),'checkpoint_sha256':sha(checkpoint),'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result

def freeze(root):
    root=Path(root);out=directory(root);cases=prepare(root);models=[]
    for seed in SEEDS:
        run=root/f'runs/tellosim-sdk9/heading-s{seed}';t=json.loads((run/'training.json').read_text());assert t['status']=='completed' and t['options']==sum(STAGES)
        checkpoint=run/'checkpoint.pt';models.append({'seed':seed,'path':str(checkpoint.relative_to(root)),'sha256':sha(checkpoint)})
    lock={'models':models,'case_hash':digest(cases['sealed_test'])};p=out/'frozen-checkpoints.json'
    if p.exists() and json.loads(p.read_text())!=lock:raise ValueError('frozen models changed')
    atomic_json(p,lock);return lock

def evaluate_seed(root,seed):
    root=Path(root);out=directory(root);cases=prepare(root);lock=json.loads((out/'frozen-checkpoints.json').read_text());m=next(x for x in lock['models'] if x['seed']==seed);checkpoint=root/m['path'];assert sha(checkpoint)==m['sha256']
    for split in ('validation','sealed_test'):
        target=out/f's{seed}-{split}.json'
        if not target.exists():evaluate(root,checkpoint,cases[split],target,f'heading-s{seed}-{split}',record_indices=(0,1) if split=='sealed_test' else ())
    target=out/f's{seed}-nav-retention.json'
    nav=root/f'runs/tellosim-sdk9/c1-s{seed}/ppo/checkpoint.pt'
    if not target.exists():evaluate_batch(root,root/'data/male-v1.npz',nav,cases['retention'],target,f'heading-nav-s{seed}',record_indices=(),batch=64)
    target=out/f's{seed}-zero-ablation.json'
    if not target.exists():evaluate(root,checkpoint,cases['validation'][:12],target,f'heading-zero-s{seed}',record_indices=(),batch=12,zero_features=True)
    target=out/f's{seed}-instruction-pairs.json'
    if not target.exists():evaluate(root,checkpoint,cases['instruction_pairs'],target,f'heading-instruction-s{seed}',record_indices=(),batch=4)
    if seed==11:
        for name,weight in [('untrained',checkpoint.parent/'initial.pt'),('trained',checkpoint)]:
            target=out/f'comparison-{name}.json'
            if not target.exists():evaluate(root,weight,cases['validation'][:1],target,f'heading-comparison-{name}',record_indices=(0,),batch=1)
    assert sha(checkpoint)==m['sha256']

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root);models={s:json.loads((out/f's{s}-sealed_test.json').read_text()) for s in SEEDS};random=json.loads((out/'random-sealed.json').read_text())
    gate=acceptance(models,random,digest(cases['sealed_test']));rows=[];retained=True
    protocol=json.loads((out/'protocol.json').read_text());lock=json.loads((out/'frozen-checkpoints.json').read_text())
    for seed,result in models.items():
        old=json.loads((root/f'reports/ts1_c1/s{seed}-c0_retention.json').read_text());nav=json.loads((out/f's{seed}-nav-retention.json').read_text())
        same=nav['results']==old['results'];retained &= same
        source=protocol['navigation_sources'][str(seed)];assert sha(root/source['path'])==source['sha256']
        assert sha(root/next(x['path'] for x in lock['models'] if x['seed']==seed))==result['checkpoint_sha256']
        cw=sum(x['actions'].count(7) for x in result['results']);ccw=sum(x['actions'].count(8) for x in result['results'])
        if not cw or not ccw:gate['reasons'].append(f'{seed}: both turn directions required')
        if not same:gate['reasons'].append(f'{seed}: navigation result differs')
        valid=json.loads((out/f's{seed}-validation.json').read_text());zero=json.loads((out/f's{seed}-zero-ablation.json').read_text())
        rows.append({'seed':seed,'validation':f"{valid['successes']}/100",'sealed':f"{result['successes']}/300",'success_rate':result['success_rate'],'collision_or_bounds':result['collision_or_bounds'],'cw_actions':cw,'ccw_actions':ccw,'retention':f"{nav['successes']}/100",'navigation_exact':same,'zero_features_successes':zero['successes'],'mean_duration_s':result['mean_duration_s'],'instruction_pairs_successes':json.loads((out/f's{seed}-instruction-pairs.json').read_text())['successes'],'run_id':result['results'][0]['run_id'],'sample_outcome':result['results'][0]['reason']})
    passed=not gate['reasons'];gate['passed']=passed
    summary={'status':'evaluated','HEADING_TASK_LEARNED':passed,'MODEL_READY_FOR_NEXT_STAGE':passed,'readiness_scope':'separate learned in-place heading skill, tolerance16deg, hold2s; navigation preserved via separate head','models':rows,'acceptance':gate,'random_successes':random['successes'],'navigation_preserved':retained,'JOINT_NAVIGATION_AND_HEADING_LEARNED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False,'note':'共享同一冻结MaleCNS图，独立转向读出；导航权重保留。30度转向动作，目标误差≤16度并保持2秒。独立转向技能不等于已学会导航与朝向联合任务。'}
    summary['random_mean_duration_s']=random['mean_duration_s']
    summary['comparison']=[{'label':name,'run_id':json.loads((out/f'comparison-{name}.json').read_text())['results'][0]['run_id']} for name in ('untrained','trained')]
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','baselines','freeze','evaluate','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command=='train':train(root,a.seed)
    elif a.command=='baselines':
        cases=prepare(root);rule=baseline(root,cases['validation'],'rule',out/'rule-validation.json');assert rule['successes']>=99
        baseline(root,cases['sealed_test'],'random',out/'random-sealed.json')
    elif a.command=='freeze':freeze(root)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    else:finalize(root)
if __name__=='__main__':main()
