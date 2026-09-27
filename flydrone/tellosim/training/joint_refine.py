"""J1R: train navigation settling, preserve H1, evaluate on a new sealed split.

Teachers label training data only. Deployment receives reservoir features only.
The J1 physics, phase manager, deadlines and success conditions are unchanged.
"""
import argparse,copy,json,math,time,signal
from pathlib import Path
import numpy as np
import torch
from .joint import JointEnv,JointPool,joint_case,bundle_spec,sha,JOINT_SPEC
from .joint_campaign import frozen_json,result_row,baseline
from .parallel import ReservoirPool,collect_rollout
from .env import TrainingEnv
from .runtime import load_checkpoint,save_checkpoint,distribution
from .checkpoint import configure_exact_execution
from .contracts import digest
from .observer import TrainingObserver
from .c0_campaign import summarize,acceptance
from .heading import wrap_angle
from ..visual import atomic_json
SEEDS=(11,22,33);STAGES=(768,1024,1024);BATCH=8
METHOD='J1R navigation readout: 768 demonstrations + 2x1024 DAgger options with near-goal settling and original C1 rehearsal; frozen MaleCNS and H1; no PPO'
def directory(root):return Path(root)/'reports/ts1_joint_refined'
def training_case(seed,lane):
    case=joint_case(seed);case['curriculum']='C1';rng=np.random.default_rng(seed+23)
    if lane%4<2:
        start=rng.uniform(-1.5,1.5,2);angle=rng.uniform(-math.pi,math.pi);distance=rng.uniform(.06,.55)
        case.update(start=[*start,1.],goal=[*(start+distance*np.array([math.cos(angle),math.sin(angle)])),1.])
    elif lane%4==3:
        yaw=float(rng.choice([0,math.pi/2,math.pi,-math.pi/2])+rng.uniform(-.12,.12));case['initial_yaw_rad']=wrap_angle(yaw)
    return case

def settling_label(observation,mask,first=False):
    error=np.asarray(observation[:2])*6
    if first or not observation[12] or np.linalg.norm(error)<=.18:return 0
    axis=int(np.argmax(np.abs(error)));action=(1 if error[0]>0 else 2) if axis==0 else (3 if error[1]>0 else 4)
    return action if mask[action] else 0

def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(parents=True,exist_ok=True)
    cases={'validation':[joint_case(83000000+i,f'j1r-val-{i:03d}') for i in range(100)],
        'sealed_test':[joint_case(84000000+i,f'j1r-test-{i:03d}') for i in range(300)],'instruction_pairs':[]}
    for pair,yaw in enumerate((0.,math.radians(170))):
        for sign in (-1,1):
            c=joint_case(85000000+pair,f'j1r-pair-{pair}-{sign}')
            c.update(start=[0.,0.,1.],goal=[1.,0.,1.],initial_yaw_rad=yaw,target_yaw_rad=wrap_angle(yaw+sign*math.pi/2));cases['instruction_pairs'].append(c)
    protocol={'seeds':list(SEEDS),'stages':list(STAGES),'new_options_per_seed':sum(STAGES),'batch':BATCH,'epochs':150,'learning_rate':.0005,
        'wall_seconds_per_seed':2700,'method':METHOD,'task_spec':JOINT_SPEC,'teacher_stop_radius_m':.18,
        'training_distribution':'50% near-goal .06-.55m; 25% full random yaw; 25% full near-cardinal yaw; original 1408 C1 examples replayed',
        'loss':'sqrt inverse-frequency class weights; noninitial near-goal STOP training examples weighted 2; no rule at inference',
        'selection':'fixed final checkpoint; freeze all seeds before sealed test; no sealed tuning',
        'gate':{'each_seed_success_min':.9,'collision_or_bounds_max':.01,'advantage_over_uniform_min':.2,'instruction_pairs_required':4,'zero_features_success_max':0},
        'split_hashes':{k:digest(v) for k,v in cases.items()},
        'original_bundles':{str(s):bundle_spec(root,s) for s in SEEDS}}
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol);return cases

def fit_settling(pool,optimizer,rows,epochs=150):
    x=torch.as_tensor(np.stack([r[0] for r in rows]),dtype=torch.float32);m=np.stack([r[1] for r in rows]);y=torch.tensor([r[2] for r in rows]);importance=torch.tensor([r[3] for r in rows])
    counts=torch.bincount(y,minlength=9).clamp_min(1).float();weights=(len(y)/(5*counts)).sqrt();updates=0;before=pool.brain.advance_count
    for _ in range(epochs):
        for ix in torch.randperm(len(rows)).split(128):
            pi,_=distribution(pool.model,x[ix],m[ix]);loss=-(pi.log_prob(y[ix])*weights[y[ix]]*importance[ix]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite settling loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(pool.model.parameters(),1.);optimizer.step();updates+=1
    assert pool.brain.advance_count==before
    with torch.no_grad():
        pi,_=distribution(pool.model,x,m);pred=pi.probs.argmax(1);stop=y==0
        metrics={'gradient_steps':updates,'examples':len(rows),'label_accuracy':float((pred==y).float().mean()),'stop_recall':float((pred[stop]==0).float().mean())}
    return metrics

def train(root,seed,smoke=False):
    root=Path(root);prepare(root);out=root/f'runs/tellosim-sdk9/joint-refined-s{seed}'
    if smoke:out=root/f'runs/tellosim-sdk9/joint-refined-smoke-s{seed}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();code_sha=sha(Path(__file__));torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=ReservoirPool(root/'data/male-v1.npz','cuda',seed=seed,profile='balanced_rate_v3',batch=BATCH,physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
    source=root/f'runs/tellosim-sdk9/c1-s{seed}/ppo/checkpoint.pt';source_sha=sha(source);load_checkpoint(source,pool,root);pool.training_method=METHOD;pool.value_trained=False
    before={k:v.detach().clone() for k,v in pool.model.state_dict().items()}
    old=np.load(root/f'runs/tellosim-sdk9/c1-s{seed}/demonstrations.npz');rows=[(x.copy(),m.copy(),int(y),1.) for x,m,y in zip(old['features'],old['masks'],old['labels'])]
    optimizer=torch.optim.Adam(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0005)
    observer=TrainingObserver(root,f'joint-refined-s{seed}'+('-smoke' if smoke else ''),BATCH,METHOD)
    metadata=[];stages=[];options=0;updates=0;envs=[];status='failed'
    def stop(signum,frame):raise KeyboardInterrupt('SIGTERM')
    prior=signal.signal(signal.SIGTERM,stop)
    try:
        for stage,count in enumerate((16,) if smoke else STAGES):
            envs=[None]*BATCH;episodes=[0]*BATCH
            def make(i):
                case_seed=(82000000 if smoke else 90000000)+seed*100000+stage*10000+i*1000+episodes[i];episodes[i]+=1
                return TrainingEnv(root,training_case(case_seed,i),pool.lanes[i],observer=observer)
            def choose(env,i,x,mask):
                label=settling_label(env.observation,mask,env.steps==0);weight=2. if label==0 and env.steps>0 else 1.
                rows.append((x.copy(),mask.copy(),label,weight));metadata.append({'seed':env.case['seed'],'stage':stage,'step':env.steps,'label':label,'weight':weight})
                if stage==0:return label,0.,0.,None
                return pool.lanes[i].decision(x,mask,deterministic=(i%2==0))
            try:collected,ticks,outcomes=collect_rollout(pool,envs,[count//BATCH]*BATCH,make,deadline=started+2700,decision_fn=choose)
            finally:
                for env in envs:
                    if env:env.close('collection_stage_complete')
                envs=[]
            options+=len(collected);metrics=fit_settling(pool,optimizer,rows,epochs=2 if smoke else 150);updates+=metrics['gradient_steps']
            stages.append({'stage':stage,'options':len(collected),'base_ticks':ticks,'outcomes':outcomes,**metrics})
            save_checkpoint(out/f'stage-{stage}.pt',pool,optimizer,root,updates,options,{'method':METHOD,'seed':seed,'source_sha256':source_sha})
            observer.update({'options':options,'updates':updates,'elapsed_s':time.monotonic()-started});atomic_json(out/'progress.json',{'options':options,'stages':stages});print(json.dumps({'seed':seed,'options':options,**metrics}),flush=True)
        save_checkpoint(out/'checkpoint.pt',pool,optimizer,root,updates,options,{'method':METHOD,'seed':seed,'source_sha256':source_sha})
        expected=[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]];load_checkpoint(out/'checkpoint.pt',pool,root);assert expected==[pool.decision(x,m,True)[3] for x,m,_,_ in rows[-16:]]
        delta=float(torch.sqrt(sum((before[k]-v).square().sum() for k,v in pool.model.state_dict().items())));assert delta>0 and sha(source)==source_sha and sha(Path(__file__))==code_sha
        status='completed';atomic_json(out/'training.json',{'status':status,'smoke_only':smoke,'training_source_sha256':code_sha,'seed':seed,'options':options,'rehearsal_examples':len(old['labels']),'new_examples':len(metadata),'updates':updates,'method':METHOD,'source_sha256':source_sha,'checkpoint_sha256':sha(out/'checkpoint.pt'),'parameter_delta_l2':delta,'roundtrip_exact':True,'stages':stages,'elapsed_s':time.monotonic()-started})
    except BaseException as exc:atomic_json(out/'failure.json',{'error':repr(exc),'options':options,'collected_examples':len(metadata)});raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        observer.close(status);signal.signal(signal.SIGTERM,prior);atomic_json(out/'provenance.json',metadata)
        np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]),labels=np.array([r[2] for r in rows]),weights=np.array([r[3] for r in rows]))

def refined_bundle(root,seed):
    root=Path(root);old=bundle_spec(root,seed);path=f'runs/tellosim-sdk9/joint-refined-s{seed}/checkpoint.pt'
    training=json.loads((root/f'runs/tellosim-sdk9/joint-refined-s{seed}/training.json').read_text())
    assert training['status']=='completed' and not training['smoke_only'] and training['options']==sum(STAGES) and training['training_source_sha256']==sha(root/'flydrone/tellosim/training/joint_refine.py')
    return {**old,'format':'tellosim.joint_refined_bundle/1','method':METHOD,'new_training_actions':sum(STAGES),
        'sources':{**old['sources'],'navigation':{'path':path,'sha256':sha(root/path)}},
        'source_hashes':{**old['source_hashes'],'flydrone/tellosim/training/joint_refine.py':sha(root/'flydrone/tellosim/training/joint_refine.py')},
        'training_sha256':sha(root/f'runs/tellosim-sdk9/joint-refined-s{seed}/training.json')}

class RefinedPool(JointPool):
    def __init__(self,root,bundle,device='cuda',batch=64):
        root=Path(root);value=json.loads(Path(bundle).read_text()) if isinstance(bundle,(str,Path)) else bundle
        if value!=refined_bundle(root,value['seed']):raise ValueError('refined bundle contract mismatch')
        super().__init__(root,bundle_spec(root,value['seed']),device,batch)
        load_checkpoint(root/value['sources']['navigation']['path'],self,root)
        self.models['navigation']=copy.deepcopy(self.model).eval();self.value_flags['navigation']=False
        for param in self.models['navigation'].parameters():param.requires_grad_(False)
        self.bundle=value;self.training_method=METHOD;self.checkpoint_sha256=digest(value)

def freeze(root):
    root=Path(root);cases=prepare(root);models=[]
    for seed in SEEDS:
        path=root/f'runs/tellosim-sdk9/joint-refined-s{seed}/policy-bundle.json';frozen_json(path,refined_bundle(root,seed));models.append({'seed':seed,'path':str(path.relative_to(root)),'sha256':sha(path)})
    frozen_json(directory(root)/'frozen-bundles.json',{'models':models,'case_hash':digest(cases['sealed_test'])})

def evaluate(root,bundle,cases,out,label,record_indices=(0,1),batch=64,zero_features=False):
    root=Path(root).resolve();out=Path(out);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda');pool=RefinedPool(root,bundle,batch=batch)
    envs=[None]*batch;generators={};indices={};actions={};skills={};results={};cursor=0
    try:
        while len(results)<len(cases):
            if time.monotonic()-started>3600:raise TimeoutError("joint evaluation wall budget exhausted")
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;actions[index]=[];skills[index]=[]
                    envs[i]=JointEnv(root,cases[index],pool.lanes[i],record=index in record_indices,run_id=f'{label}-{index}' if index in record_indices else None,policy_source='joint_argmax_malecns' if not zero_features else 'joint_ZERO_FEATURE_ABLATION')
                env=envs[i]
                if i not in generators:
                    x=np.zeros_like(env.features) if zero_features else env.features
                    a,_,_,policy=pool.lanes[i].decision(x,env.mask,True);actions[indices[i]].append(a);skills[indices[i]].append(env.phase);generators[i]=env.step_iter(a,policy)
                try:next(generators[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                except StopIteration:
                    del generators[i]
                    if env.terminated:
                        index=indices[i];results[index]=result_row(env,env.case,actions[index],env.session.run_id if env.live else None,skills[index]);env.close();envs[i]=None
                        if len(results)%10==0:atomic_json(out.with_suffix('.progress.json'),summarize([results[k] for k in sorted(results)],label));print(json.dumps({'label':label,'cases':len(results),'successes':sum(x['success'] for x in results.values())}),flush=True)
            if pending:
                features=pool.observe_lanes(pending)
                for i in pending:envs[i].features=features[i].copy();envs[i].publish()
    finally:
        for env in envs:
            if env:env.close()
    result={**summarize([results[i] for i in range(len(cases))],label),'mode':'argmax' if not zero_features else 'zero_features_ablation','case_hash':digest(cases),'bundle_sha256':sha(bundle),'source_weights':pool.bundle['sources'],'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result

def evaluate_seed(root,seed):
    root=Path(root);out=directory(root);cases=prepare(root);bundle=root/f'runs/tellosim-sdk9/joint-refined-s{seed}/policy-bundle.json'
    for split in ('validation','sealed_test','instruction_pairs'):
        target=out/f's{seed}-{split}.json'
        if not target.exists():evaluate(root,bundle,cases[split],target,f'joint-refined-s{seed}-{split}',record_indices=(0,1) if split!='validation' else (),batch=min(64,len(cases[split])))
    target=out/f's{seed}-zero-ablation.json'
    if not target.exists():evaluate(root,bundle,cases['validation'][:12],target,f'joint-zero-s{seed}',record_indices=(),batch=12,zero_features=True)
    assert json.loads(bundle.read_text())==refined_bundle(root,seed)

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root)
    models={s:json.loads((out/f's{s}-sealed_test.json').read_text()) for s in SEEDS}
    random=json.loads((out/'random-sealed.json').read_text());gate=acceptance(models,random,digest(cases['sealed_test']));rows=[]
    for seed,result in models.items():
        bundle=root/f'runs/tellosim-sdk9/joint-refined-s{seed}/policy-bundle.json'
        assert sha(bundle)==result['bundle_sha256'] and json.loads(bundle.read_text())==refined_bundle(root,seed)
        data=result['results'];cw=sum(r['actions'].count(7) for r in data);ccw=sum(r['actions'].count(8) for r in data)
        pairs=json.loads((out/f's{seed}-instruction_pairs.json').read_text());zero=json.loads((out/f's{seed}-zero-ablation.json').read_text());valid=json.loads((out/f's{seed}-validation.json').read_text())
        if not cw or not ccw:gate['reasons'].append(f'{seed}: missing turn direction')
        if pairs['successes']!=4:gate['reasons'].append(f'{seed}: instruction pairs failed')
        if zero['successes']!=0:gate['reasons'].append(f'{seed}: zero feature diagnostic succeeded')
        for r in data:
            if len(r['actions'])!=len(r['action_skills']) or any(a not in JOINT_SPEC['allowed_actions'][s] for a,s in zip(r['actions'],r['action_skills'])):gate['reasons'].append(f'{seed}: invalid phase action')
            if r['brain_tick']!=4*(1+r['physics_ticks']//12):gate['reasons'].append(f'{seed}: neural clock mismatch')
            if any(not e['physics_state_unchanged'] for e in r['phase_events']):gate['reasons'].append(f'{seed}: physical reset')
            if r['success'] and not (r['final_phase']=='heading' and r['phase_events'] and r['horizontal_distance_m']<=.2 and r['height_error_m']<=.1 and r['horizontal_speed_mps']<=.08 and r['heading_error_deg']<=16 and r['angular_speed_rad_s']<=.08 and r['stable_hold_s']>=2-1e-8):gate['reasons'].append(f'{seed}: invalid joint success')
        rows.append({'seed':seed,'validation':f"{valid['successes']}/100",'sealed':f"{result['successes']}/300",'success_rate':result['success_rate'],'collision_or_bounds':result['collision_or_bounds'],'cw_actions':cw,'ccw_actions':ccw,'instruction_pairs_successes':pairs['successes'],'zero_features_successes':zero['successes'],'run_id':data[0]['run_id'],'sample_outcome':data[0]['reason'],'mean_duration_s':result['mean_duration_s']})
    gate['passed']=not gate['reasons'];passed=gate['passed']
    summary={'status':'evaluated','COMPOSED_JOINT_TASK_VERIFIED':passed,'MODEL_READY_FOR_NEXT_STAGE':passed,
        'SINGLE_POLICY_JOINT_TRAINED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False,
        'readiness_scope':'Continuous fixed-height empty-room navigation, then heading within 16 degrees and joint stable hold for 2 seconds; composition of two frozen learned readouts',
        'models':rows,'acceptance':gate,'random_successes':random['successes'],'new_training_actions':sum(STAGES)*len(SEEDS),
        'note':'J1R：每种子新增2816个训练动作，重点改善导航到位停止；原转向权重保持。位置、朝向与稳定保持标准不变。仍是两个技能组合，不是单一策略端到端联合学习。'}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','smoke','freeze','baselines','evaluate','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command in ('train','smoke'):train(root,a.seed,a.command=='smoke')
    elif a.command=='freeze':freeze(root)
    elif a.command=='baselines':
        cases=prepare(root)
        for split,mode in [('validation','rule'),('sealed_test','random')]:
            target=out/('rule-validation.json' if mode=='rule' else 'random-sealed.json')
            if not target.exists():baseline(root,cases[split],mode,target)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    else:finalize(root)
if __name__=='__main__':main()
