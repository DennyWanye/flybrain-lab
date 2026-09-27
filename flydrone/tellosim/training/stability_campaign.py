"""J2R: dense neural-state supervision and continuous composed-task DAgger.

Teachers only annotate training observations. Learned evaluation uses frozen
MaleCNS features and argmax heads without any teacher or position-rule override.
Historical J2 source, representation, physics and acceptance remain unchanged.
"""
import argparse,copy,json,signal,time
from pathlib import Path
import numpy as np
import torch
from . import robust_campaign as prior
from .robust_env import PROFILES,DISTURBANCE_SPEC,RobustJointEnv,with_profile
from .joint import joint_case,JOINT_SPEC,sha
from .joint_refine import settling_label,training_case as navigation_case
from .heading import teacher,wrap_angle
from .joint_campaign import frozen_json
from .runtime import distribution
from .checkpoint import configure_exact_execution
from .contracts import digest
from .c0_campaign import summarize,acceptance
from ..visual import atomic_json

SEEDS=(11,22,33)
SKILLS=prior.SKILLS
STAGES=(512,768,768)
BATCH=16
SOURCE_FILES=(*prior.SOURCE_FILES,'flydrone/tellosim/training/stability_campaign.py')
METHOD='J2R: dense 5Hz training labels on real 10Hz neural states, continuous joint-task teacher then 2 DAgger stages; frozen MaleCNS, two learned readouts; no PPO'

def directory(root):return Path(root)/'reports/ts1_stability'
def run_directory(root,seed,smoke=False):return Path(root)/f'runs/tellosim-sdk9/stability-{"smoke-" if smoke else ""}s{seed}'
def source_hashes(root):return {name:sha(Path(root)/name) for name in SOURCE_FILES}
def old_bundle(root,seed):return prior.bundle_spec(root,seed)
def result_row(*args):return prior.result_row(*args)
grouped=prior.grouped
baseline=prior.baseline

def label_for(env):
    first=env.previous_action is None
    return (settling_label if env.phase=='navigation' else teacher)(env.observation,env.mask,first)

def training_case(seed,lane):
    c=joint_case(seed,f'j2r-train-{seed}')
    # Half of lanes start near the navigation handoff; heading still has the
    # independently sampled instruction. All four disturbance profiles remain.
    nav=navigation_case(seed,lane//4)
    c.update(start=nav['start'],goal=nav['goal'],initial_yaw_rad=nav['initial_yaw_rad'])
    return with_profile(c,tuple(PROFILES)[lane%4])

def prepare(root):
    out=directory(root);out.mkdir(parents=True,exist_ok=True)
    cases={name:[with_profile(joint_case(base+i,f'j2r-{name}-{i:03d}'),tuple(PROFILES)[i%4]) for i in range(count)]
           for name,base,count in [('validation',143000000,100),('sealed_test',144000000,300)]}
    cases['instruction_pairs']=[]
    for pair,yaw in enumerate((0.,np.deg2rad(170))):
        for sign in (-1,1):
            c=joint_case(145000000+pair,f'j2r-pair-{pair}-{sign}')
            c.update(start=[0.,0.,1.],goal=[1.,0.,1.],initial_yaw_rad=float(yaw),target_yaw_rad=wrap_angle(yaw+sign*np.pi/2))
            cases['instruction_pairs'].append(with_profile(c,'combined'))
    protocol=dict(seeds=list(SEEDS),stages=list(STAGES),batch=BATCH,epochs=60,learning_rate=.0003,
        new_actions_per_seed=sum(STAGES),method=METHOD,task_spec=JOINT_SPEC,disturbance=DISTURBANCE_SPEC,
        training_seed_formula='150000000 + model_seed*100000 + stage*10000 + lane*500 + episode; smoke uses 160000000',
        labels='decision weight 1, intermediate every second real sensor sample weight .25; noninitial STOP x2; original J2 data rehearsed',
        loss='weighted class-balanced cross entropy; AdamW weight decay .001, gradients clipped at 1; epochs fixed',
        selection='fixed final checkpoint; all seeds frozen before new sealed cases; no validation or sealed training examples',
        gate=json.loads((Path(root)/'reports/ts1_robust/protocol.json').read_text())['gate'],
        comparison='J2 and J2R on identical fresh cases; same physics, input representation, phase manager and gates',
        split_hashes={k:digest(v) for k,v in cases.items()},source_hashes=source_hashes(root),
        original_bundles={str(s):old_bundle(root,s) for s in SEEDS})
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol)
    return cases

def fit(model,opt,rows,epochs):
    x=torch.as_tensor(np.stack([r[0] for r in rows]),dtype=torch.float32)
    masks=np.stack([r[1] for r in rows]);y=torch.tensor([r[2] for r in rows]);importance=torch.tensor([r[3] for r in rows])
    counts=torch.zeros(9).scatter_add_(0,y,importance).clamp_min(1)
    weights=(importance.sum()/(int((counts>1).sum())*counts)).sqrt()
    updates=0
    for _ in range(epochs):
        for ix in torch.randperm(len(rows)).split(256):
            pi,_=distribution(model,x[ix],masks[ix]);loss=-(pi.log_prob(y[ix])*weights[y[ix]]*importance[ix]).sum()/importance[ix].sum()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite dense imitation loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();updates+=1
    with torch.no_grad():
        pi,_=distribution(model,x,masks);pred=pi.probs.argmax(1);stop=y==0
        return dict(gradient_steps=updates,examples=len(rows),label_accuracy=float((pred==y).float().mean()),stop_recall=float((pred[stop]==0).float().mean()))

def train(root,seed,smoke=False):
    root=Path(root);prepare(root);out=run_directory(root,seed,smoke)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    sources=source_hashes(root);original=old_bundle(root,seed);pool=prior.RobustPool(root,original,batch=BATCH)
    pool.training_method=METHOD;rows={};opts={};initial={};provenance=[];stages=[];options=0;status='failed';envs=[]
    for skill in SKILLS:
        model=pool.models[skill]
        for p in model.parameters():p.requires_grad_(True)
        initial[skill]={k:v.detach().clone() for k,v in model.state_dict().items()}
        opts[skill]=torch.optim.AdamW(list(model.body.parameters())+list(model.actor.parameters()),lr=.0003,weight_decay=.001)
        data=np.load(root/f'runs/tellosim-sdk9/robust-s{seed}/{skill}/demonstrations.npz')
        rows[skill]=[(x.copy(),m.copy(),int(y),float(w)) for x,m,y,w in zip(data['features'],data['masks'],data['labels'],data['weights'])]
    old_counts={s:len(rows[s]) for s in SKILLS}
    def annotate(env,x,kind):
        label=label_for(env);weight=(1. if kind=='decision' else .25)*(2. if label==0 and env.previous_action is not None else 1.)
        rows[env.phase].append((x.copy(),env.mask.copy(),label,weight))
        provenance.append(dict(seed=env.case['seed'],stage=stage,skill=env.phase,step=env.steps,sample_id=env.agent.sample_id,
            profile=env.disturbance_profile,kind=kind,label=label,weight=weight))
        return label
    def stop(signum,frame):raise KeyboardInterrupt('SIGTERM')
    previous_signal=signal.signal(signal.SIGTERM,stop)
    try:
        for stage,count in enumerate((64,64) if smoke else STAGES):
            envs=[None]*BATCH;episodes=[0]*BATCH;done=[0]*BATCH;gens={};outcomes=[];ticks=0;stage_actions={s:0 for s in SKILLS}
            while any(n<count//BATCH for n in done):
                if time.monotonic()-started>3600:raise TimeoutError('stability training wall budget')
                pending={}
                for i in range(BATCH):
                    if done[i]>=count//BATCH:continue
                    if envs[i] is None:
                        if episodes[i]>=500:raise RuntimeError('training seed range exhausted')
                        sample=(160000000 if smoke else 150000000)+seed*100000+stage*10000+i*500+episodes[i];episodes[i]+=1
                        envs[i]=RobustJointEnv(root,training_case(sample,i),pool.lanes[i])
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
                        if env.agent.sample_id%2==0:annotate(env,env.features,'intermediate')
            for env in envs:
                if env:env.close('collection_stage_complete')
            envs=[];metrics={};advances=pool.brain.advance_count
            for skill in SKILLS:
                metrics[skill]=fit(pool.models[skill],opts[skill],rows[skill],2 if smoke else 60)
                pool.model=pool.models[skill];pool.value_trained=False
                prior.save_skill(out/skill/f'stage-{stage}.pt',pool,opts[skill],root,skill,options,
                    sum(s['metrics'][skill]['gradient_steps'] for s in stages)+metrics[skill]['gradient_steps'],dict(method=METHOD,sources=sources,source=original['sources'][skill]))
            assert pool.brain.advance_count==advances
            stages.append(dict(stage=stage,options=sum(done),base_ticks=ticks,action_skills=stage_actions,outcomes=outcomes,metrics=metrics))
            atomic_json(out/'progress.json',dict(options=options,stages=stages,elapsed_s=time.monotonic()-started))
            print(json.dumps(dict(seed=seed,options=options,metrics=metrics)),flush=True)
        training={}
        for skill in SKILLS:
            pool.model=pool.models[skill];pool.value_trained=False
            path=out/skill/'checkpoint.pt';updates=sum(s['metrics'][skill]['gradient_steps'] for s in stages)
            prior.save_skill(path,pool,opts[skill],root,skill,options,updates,dict(method=METHOD,sources=sources,source=original['sources'][skill]))
            with torch.no_grad():expected=distribution(pool.model,torch.tensor(np.stack([r[0] for r in rows[skill][-16:]])),np.stack([r[1] for r in rows[skill][-16:]]))[0].probs.clone()
            prior.load_skill(path,pool,root,skill)
            with torch.no_grad():actual=distribution(pool.model,torch.tensor(np.stack([r[0] for r in rows[skill][-16:]])),np.stack([r[1] for r in rows[skill][-16:]]))[0].probs
            assert torch.equal(expected,actual)
            delta=float(torch.sqrt(sum((initial[skill][k]-v).square().sum() for k,v in pool.model.state_dict().items())))
            assert delta>0
            training[skill]=dict(examples=len(rows[skill]),rehearsal_examples=old_counts[skill],new_examples=len(rows[skill])-old_counts[skill],
                options=sum(s['action_skills'][skill] for s in stages),updates=updates,parameter_delta_l2=delta,roundtrip_exact=True,checkpoint_sha256=sha(path))
        assert source_hashes(root)==sources and old_bundle(root,seed)==original
        status='completed'
        atomic_json(out/'training.json',dict(status=status,smoke_only=smoke,seed=seed,options=options,method=METHOD,
            source_hashes=sources,original_bundle_hash=digest(original),skills=training,stages=stages,elapsed_s=time.monotonic()-started))
    except BaseException as exc:atomic_json(out/'failure.json',dict(error=repr(exc),options=options));raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        signal.signal(signal.SIGTERM,previous_signal);atomic_json(out/'provenance.json',provenance)
        for skill in SKILLS:
            (out/skill).mkdir(exist_ok=True)
            np.savez_compressed(out/skill/'demonstrations.npz',features=np.stack([r[0] for r in rows[skill]]),masks=np.stack([r[1] for r in rows[skill]]),labels=np.array([r[2] for r in rows[skill]]),weights=np.array([r[3] for r in rows[skill]]))

def bundle_spec(root,seed):
    root=Path(root);old=old_bundle(root,seed);folder=run_directory(root,seed);t=json.loads((folder/'training.json').read_text())
    assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
    assert t['source_hashes']==source_hashes(root) and t['original_bundle_hash']==digest(old)
    sources={}
    for skill in SKILLS:
        path=folder/skill/'checkpoint.pt';assert t['skills'][skill]['checkpoint_sha256']==sha(path)
        sources[skill]=dict(path=str(path.relative_to(root)),sha256=sha(path))
    return {**old,'format':'tellosim.stability_skill_bundle/1','method':METHOD,'stage':'J2R','sources':sources,
        'source_hashes':{**old['source_hashes'],**source_hashes(root)},'training_sha256':sha(folder/'training.json'),'new_training_actions':sum(STAGES)}

class StabilityPool(prior.RobustPool):
    def __init__(self,root,bundle,batch=64):
        value=json.loads(Path(bundle).read_text()) if isinstance(bundle,(str,Path)) else bundle
        if value!=bundle_spec(root,value['seed']):raise ValueError('stability source contract mismatch')
        super().__init__(root,old_bundle(root,value['seed']),batch=batch)
        for skill in SKILLS:
            prior.load_skill(Path(root)/value['sources'][skill]['path'],self,root,skill)
            self.models[skill]=copy.deepcopy(self.model).eval();self.value_flags[skill]=False
            for param in self.models[skill].parameters():param.requires_grad_(False)
        self.bundle=value;self.training_method=METHOD;self.checkpoint_sha256=digest(value)

def freeze(root):
    cases=prepare(root);models=[]
    for seed in SEEDS:
        path=run_directory(root,seed)/'policy-bundle.json';frozen_json(path,bundle_spec(root,seed))
        models.append({'seed':seed,'path':str(path.relative_to(root)),'sha256':sha(path)})
    frozen_json(directory(root)/'frozen-bundles.json',{'models':models,'case_hash':digest(cases['sealed_test'])})


def evaluate(root,seed,cases,out,label,before=False,zero=False,record_indices=(),batch=64):
    root=Path(root);out=Path(out);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    bundle=old_bundle(root,seed) if before else bundle_spec(root,seed)
    pool=prior.RobustPool(root,bundle,batch=batch) if before else StabilityPool(root,bundle,batch=batch)
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
                        policy_source='stability_ZERO_FEATURE_ABLATION' if zero else 'stability_argmax_malecns')
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


def evaluate_seed(root,seed):
    out=directory(root);cases=prepare(root)
    lock=json.loads((out/'frozen-bundles.json').read_text())
    entry=next(m for m in lock['models'] if m['seed']==seed)
    assert sha(Path(root)/entry['path'])==entry['sha256']
    for split in ('validation','sealed_test','instruction_pairs'):
        path=out/f's{seed}-{split}.json'
        if not path.exists():evaluate(root,seed,cases[split],path,f'stability-s{seed}-{split}',
            record_indices=(0,1,2,3) if split=='sealed_test' else (),batch=min(64,len(cases[split])))
    path=out/f's{seed}-zero-ablation.json'
    if not path.exists():evaluate(root,seed,cases['validation'][:12],path,f'stability-zero-s{seed}',zero=True,batch=12)
    path=out/f's{seed}-before-sealed.json'
    if not path.exists():evaluate(root,seed,cases['sealed_test'],path,f'stability-before-s{seed}',before=True,record_indices=(3,),batch=64)
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
    if new_disturbed<old_disturbed:gate['reasons'].append('disturbed success declined versus J2 on paired cases')
    gate['passed']=not gate['reasons'];passed=gate['passed']
    summary={'stage':'J2R','status':'evaluated','COMPOSED_JOINT_TASK_VERIFIED':passed,'MODEL_READY_FOR_NEXT_STAGE':passed,
        'SINGLE_POLICY_JOINT_TRAINED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False,
        'readiness_scope':'fixed-height empty-room composed task under four specified mild disturbance profiles',
        'models':rows,'acceptance':gate,'random_successes':random['successes'],'new_training_actions':len(SEEDS)*sum(STAGES),
        'disturbed_paired_successes':{'before':old_disturbed,'after':new_disturbed,'episodes':675},
        'note':'J2R：完整组合任务训练，连续神经状态密集标注；两种技能读出更新，冻结 MaleCNS。全新封存场景、相同扰动与验收标准；不代表完整果蝇大脑学习或真机就绪。'}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','smoke','freeze','evaluate','baselines','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11)
    a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command in ('train','smoke'):train(root,a.seed,a.command=='smoke')
    elif a.command=='freeze':freeze(root)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    elif a.command=='baselines':
        cases=prepare(root)
        for split,mode in [('validation','rule'),('sealed_test','random')]:
            target=out/('rule-validation.json' if mode=='rule' else 'random-sealed.json')
            if not target.exists():baseline(root,cases[split],mode,target)
    else:finalize(root)
if __name__=='__main__':main()
