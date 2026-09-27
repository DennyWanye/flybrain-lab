"""Frozen-skill composition evaluation. No training or rule-directed learned actions."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from .joint import JointEnv,JointPool,JOINT_SPEC,METHOD,joint_case,bundle_spec,sha
from .heading import wrap_angle,teacher
from .heading_campaign import RuleDiagnostic,result_row as heading_result_row
from .c0_campaign import expert,summarize,acceptance
from .checkpoint import configure_exact_execution
from .contracts import digest
from ..visual import atomic_json
SEEDS=(11,22,33)
def directory(root):return Path(root)/'reports/ts1_joint'
def frozen_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'frozen artifact changed: {path}')
        return
    atomic_json(path,value)
def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(parents=True,exist_ok=True)
    cases={'schema':'tellosim.joint_cases/1','validation':[joint_case(73000000+i,f'j1-val-{i:03d}') for i in range(100)],
        'sealed_test':[joint_case(74000000+i,f'j1-test-{i:03d}') for i in range(300)],'instruction_pairs':[]}
    for pair,yaw in enumerate((0.,math.radians(170))):
        for sign in (-1,1):
            case=joint_case(75000000+pair,f'j1-pair-{pair}-{sign}')
            case.update(start=[0.,0.,1.],goal=[1.,0.,1.],initial_yaw_rad=yaw,target_yaw_rad=wrap_angle(yaw+sign*math.pi/2))
            cases['instruction_pairs'].append(case)
    protocol={'seeds':list(SEEDS),'method':METHOD,'task_spec':JOINT_SPEC,'new_training_actions':0,
        'wall_seconds_per_evaluation':3600,'selection':'all six existing weights and composition source frozen before sealed evaluation; no sealed tuning',
        'gate':{'each_seed_success_min':.9,'collision_or_bounds_max':.01,'advantage_over_uniform_min':.2,'instruction_pairs_required':4,'both_turn_directions':True,'zero_features_success_max':0},
        'split_hashes':{k:digest(v) for k,v in cases.items() if isinstance(v,list)},
        'standalone_retention':'historical C1/H1 results only; unchanged source hashes checked, no claim of rerun'}
    frozen_json(out/'cases.json',cases);frozen_json(out/'protocol.json',protocol)
    models=[]
    for seed in SEEDS:
        bundle=bundle_spec(root,seed);path=root/f'runs/tellosim-sdk9/joint-s{seed}/policy-bundle.json'
        frozen_json(path,bundle);models.append({'seed':seed,'path':str(path.relative_to(root)),'sha256':sha(path)})
    frozen_json(out/'frozen-bundles.json',{'models':models,'case_hash':digest(cases['sealed_test'])})
    return cases
class JointDiagnostic(RuleDiagnostic):
    training_method='Explicit non-neural rule/random diagnostic'
    def reset(self):super().reset();self.skill='navigation'
    def switch_skill(self,skill):self.skill=skill

def result_row(env,case,actions,run_id,skills):
    return {**heading_result_row(env,case,actions,run_id),'action_skills':skills,'phase_events':list(env.phase_events),
        'final_phase':env.phase,'brain_tick':env.agent.brain_tick,'physics_ticks':env.session.tick-env.start_tick,
        'horizontal_distance_m':float(np.linalg.norm(env.session.world.position[:2]-np.asarray(case['goal'])[:2])),
        'height_error_m':abs(float(env.session.world.position[2]-case['goal'][2])),
        'horizontal_speed_mps':float(np.linalg.norm(env.session.world.velocity[:2]))}

def baseline(root,cases,mode,out):
    results=[];started=time.monotonic();agent=JointDiagnostic()
    for index,case in enumerate(cases):
        env=JointEnv(root,case,agent);rng=np.random.default_rng(case['seed']+991);actions=[];skills=[]
        try:
            while not env.terminated:
                a=(expert(env.observation,env.mask,env.steps==0) if env.phase=='navigation' else teacher(env.observation,env.mask)) if mode=='rule' else int(rng.choice(np.flatnonzero(env.mask)))
                actions.append(a);skills.append(env.phase);env.step(a)
            results.append(result_row(env,case,actions,None,skills))
        finally:env.close()
        if (index+1)%10==0:atomic_json(out.with_suffix('.progress.json'),summarize(results,'joint-'+mode));print(json.dumps({'baseline':mode,'cases':index+1,'successes':sum(x['success'] for x in results)}),flush=True)
    result={**summarize(results,'joint-'+mode),'mode':mode,'case_hash':digest(cases),'elapsed_s':time.monotonic()-started,'neural_model':False};atomic_json(out,result);return result

def evaluate(root,bundle,cases,out,label,record_indices=(0,1),batch=64,zero_features=False):
    root=Path(root).resolve();out=Path(out);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda');pool=JointPool(root,bundle,batch=batch)
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
    root=Path(root);out=directory(root);cases=prepare(root);bundle=root/f'runs/tellosim-sdk9/joint-s{seed}/policy-bundle.json'
    for split in ('validation','sealed_test','instruction_pairs'):
        target=out/f's{seed}-{split}.json'
        if not target.exists():evaluate(root,bundle,cases[split],target,f'joint-s{seed}-{split}',record_indices=(0,1) if split!='validation' else (),batch=min(64,len(cases[split])))
    target=out/f's{seed}-zero-ablation.json'
    if not target.exists():evaluate(root,bundle,cases['validation'][:12],target,f'joint-zero-s{seed}',record_indices=(),batch=12,zero_features=True)
    assert json.loads(bundle.read_text())==bundle_spec(root,seed)

def finalize(root):
    root=Path(root);out=directory(root);cases=prepare(root)
    models={s:json.loads((out/f's{s}-sealed_test.json').read_text()) for s in SEEDS}
    random=json.loads((out/'random-sealed.json').read_text());gate=acceptance(models,random,digest(cases['sealed_test']));rows=[]
    for seed,result in models.items():
        bundle=root/f'runs/tellosim-sdk9/joint-s{seed}/policy-bundle.json'
        assert sha(bundle)==result['bundle_sha256'] and json.loads(bundle.read_text())==bundle_spec(root,seed)
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
        'models':rows,'acceptance':gate,'random_successes':random['successes'],'new_training_actions':0,
        'note':'两个已训练技能在同一物理场景接力；位置、朝向、速度同时达标并保持2秒。没有进行单一模型的端到端联合训练；没有新增训练数据。'}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','baselines','evaluate','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command=='baselines':
        cases=prepare(root)
        for split,mode in [('validation','rule'),('sealed_test','random')]:
            target=out/('rule-validation.json' if mode=='rule' else 'random-sealed.json')
            if not target.exists():baseline(root,cases[split],mode,target)
    elif a.command=='evaluate':evaluate_seed(root,a.seed)
    else:finalize(root)
if __name__=='__main__':main()
