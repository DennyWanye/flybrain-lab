"""Bounded C0 curriculum and frozen, case-wise acceptance evaluation.

Teachers are used only for labelled training collection or explicitly named rule
baselines. Deployment and evaluation always call the saved neural policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .runtime import ReservoirAgent, load_checkpoint, save_checkpoint, update
from .bootstrap import fit
from .critic_warmup import fit_value_head, discounted_returns
from .env import TrainingEnv, curriculum_case
from .boundary_repair import validation_cases
from .contracts import digest
from ..visual import atomic_json

SEEDS=(11,22,33)
METHOD='Four-direction demonstrations + DAgger + conservative PPO; frozen MaleCNS'


def expert(observation, mask, first=False):
    error=np.asarray(observation[:2])*6
    if first or not observation[12] or np.linalg.norm(error)<=.165:return 0
    axis=int(np.argmax(np.abs(error)))
    action=(1 if error[0]>0 else 2) if axis==0 else (3 if error[1]>0 else 4)
    return action if mask[action] else 0


def training_case(seed, near=False):
    if not near:return curriculum_case(seed,curriculum='C0')
    rng=np.random.default_rng(seed)
    case=curriculum_case(seed,curriculum='C0-near-xy')
    # Include diagonal approaches and the success boundary, not just axis routes.
    if seed%3!=0:
        angle=rng.uniform(-math.pi,math.pi);distance=rng.uniform(.205,.7)
        case['goal'][:2]=(np.asarray(case['start'][:2])+distance*np.array([math.cos(angle),math.sin(angle)])).tolist()
    return case


def make_splits(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    validation=[curriculum_case(4100000+i,f'c0-validation-{i:03d}') for i in range(100)]
    sealed=[curriculum_case(4200000+i,f'c0-sealed-{i:03d}') for i in range(300)]
    value={'schema':'sdk9.c0_cases/1.0','validation':validation,'sealed_test':sealed,
        'formal_seeds':list(SEEDS),'selection':'validation only; seal opened only after all checkpoints frozen'}
    target=out/'cases.json'
    if target.exists():
        if json.loads(target.read_text())!=value:raise ValueError('existing case manifest mismatch')
    else:atomic_json(target,value)
    atomic_json(out/'case-hashes.json',{'validation':digest(validation),'sealed_test':digest(sealed)})
    return value


def wilson(successes,n):
    if n<=0:return None
    z=1.959963984540054;p=successes/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,center-half),min(1,center+half)]


def summarize(results,label):
    n=len(results);successes=sum(r['success'] for r in results)
    return {'label':label,'episodes':n,'successes':successes,'success_rate':successes/n if n else None,
        'collision_or_bounds':sum(r['reason'] in ('collision','out_of_bounds') for r in results),
        'wilson95':wilson(successes,n),'mean_duration_s':float(np.mean([r['duration_s'] for r in results])) if n else None,
        'mean_return':float(np.mean([r['return'] for r in results])) if n else None,'results':results}


def evaluate(root,agent,cases,out,label,mode='argmax',record_indices=(0,)):
    out=Path(out);results=[];started=time.monotonic()
    for index,case in enumerate(cases):
        torch.manual_seed(4800000+index);rng=np.random.default_rng(4800000+index)
        recording=index in record_indices
        env=TrainingEnv(root,case,agent,record=recording,
            run_id=f'{out.parent.name}-{label}-{index}' if recording else None,
            policy_source=f'{label}_{mode}_sdk9')
        try:
            while not env.terminated:
                if mode=='rule':action=expert(env.observation,env.mask,env.steps==0);policy=None
                elif mode=='random':action=int(rng.choice(np.flatnonzero(env.mask)));policy=None
                else:action,_,_,policy=agent.decision(env.features,env.mask,mode=='argmax')
                env.step(action,policy)
            results.append({'case_id':case['case_id'],'seed':case['seed'],'reason':env.reason,
                'success':env.reason=='success','return':env.raw_return,'steps':env.steps,
                'duration_s':(env.session.tick-env.start_tick)/120,
                'distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
                'run_id':env.session.run_id if recording else None})
        finally:env.close()
        if (index+1)%10==0 or index+1==len(cases):
            atomic_json(out.with_suffix('.progress.json'),summarize(results,label))
            print(json.dumps({'phase':'evaluate','label':label,'cases':len(results),'total':len(cases),
                'successes':sum(r['success'] for r in results)}),flush=True)
    result={**summarize(results,label),'mode':mode,'case_hash':digest(cases),
        'checkpoint_sha256':getattr(agent,'checkpoint_sha256',None),'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result


class BudgetExceeded(RuntimeError):pass

class Budget:
    def __init__(self,options=10000,seconds=7200):
        self.options=options;self.seconds=seconds;self.used=0;self.start=time.monotonic()
    def take(self):
        if self.used>=self.options or time.monotonic()-self.start>=self.seconds:raise BudgetExceeded('training budget exhausted')
        self.used+=1


def collect(root,agent,cases,rows,metadata,budget,stage):
    outcomes=[]
    for index,case in enumerate(cases):
        rng=np.random.default_rng(case['seed']+9);env=TrainingEnv(root,case,agent)
        try:
            while not env.terminated:
                budget.take();label=expert(env.observation,env.mask,env.steps==0)
                rows.append((env.features.copy(),env.mask.copy(),label))
                metadata.append({'case_seed':case['seed'],'stage':stage,'step':env.steps,'teacher_label':label})
                if stage<2:action=label;policy=None
                else:action,_,_,policy=agent.decision(env.features,env.mask,index%2==0)
                # Off-trajectory recovery states only in training, with expert labels
                # kept separate from the exploratory action actually executed.
                if index%4==0 and 1<=env.steps<=4:
                    action=int(rng.choice(np.flatnonzero(env.mask)));policy=None
                elif index%4==1 and env.steps==2:action=0;policy=None
                env.step(action,policy)
            outcomes.append(env.reason)
        finally:env.close()
        if (index+1)%16==0:print(json.dumps({'phase':'collect','stage':stage,'episodes':index+1,
            'total':len(cases),'options':budget.used,'examples':len(rows)}),flush=True)
    return {'episodes':len(cases),'collection_successes':outcomes.count('success')}


def critic_fit(root,agent,out,seeds,budget):
    features=[];targets=[];outcomes=[]
    for seed in seeds:
        env=TrainingEnv(root,curriculum_case(seed),agent);x=[];rewards=[];gammas=[]
        try:
            while not env.terminated:
                budget.take();x.append(env.features.copy())
                action,_,_,policy=agent.decision(env.features,env.mask)
                row=env.step(action,policy);rewards.append(row['reward']);gammas.append(row['Gamma'])
            features.extend(x);targets.extend(discounted_returns(rewards,gammas));outcomes.append(env.reason)
        finally:env.close()
    np.savez_compressed(out/'critic-data.npz',features=np.asarray(features),returns=np.asarray(targets))
    result=fit_value_head(agent,features,targets)
    atomic_json(out/'critic-warmup.json',{**result,'outcomes':outcomes,'seeds':seeds})
    return result


def ppo_fit(root,agent,out,seed,budget,count=512):
    for p in agent.model.body.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in agent.model.parameters() if p.requires_grad],lr=1e-5)
    total=0;updates=0;episode=0;env=None;metrics=[]
    try:
        while total<count:
            rows=[]
            while len(rows)<min(128,count-total):
                if env is None:env=TrainingEnv(root,curriculum_case(3600000+seed*1000+episode),agent)
                budget.take();features=env.features.copy();mask=env.mask.copy()
                action,logp,value,policy=agent.decision(features,mask);result=env.step(action,policy)
                truncated=total+len(rows)+1==count and not result['terminal']
                _,_,next_value,_=agent.decision(result['next_features'],result['next_mask'],True)
                rows.append({'features':features,'mask':mask,'action':action,'logp':logp,'value':value,
                    'reward':result['reward'],'Gamma':result['Gamma'],'terminal':result['terminal'],
                    'truncated':truncated,'next_value':next_value})
                if env.terminated:env.close();env=None;episode+=1
            before=agent.brain.advance_count;metric=update(agent,optimizer,rows)
            assert before==agent.brain.advance_count
            total+=len(rows);updates+=1;metrics.append(metric)
            print(json.dumps({'phase':'ppo','options':total,'updates':updates}),flush=True)
    finally:
        if env is not None:env.close('external_budget_truncation')
        for p in agent.model.body.parameters():p.requires_grad_(True)
        atomic_json(out/'ppo.json',{'options':total,'updates':updates,'learning_rate':1e-5,'frozen_body':True,'metrics':metrics,'complete':total==count})
    atomic_json(out/'ppo.json',{'options':total,'updates':updates,'learning_rate':1e-5,'frozen_body':True,'metrics':metrics})
    return optimizer,updates


def train(root,graph,out,seed,counts=(128,128,192,192),epochs=200,source='reservoir'):
    root=Path(root).resolve();out=Path(out).resolve()
    if len(counts)!=4 or any(type(n) is not int or not 1<=n<=512 for n in counts) or not 1<=epochs<=500:raise ValueError('four stage sizes in 1..512 and epochs in 1..500 required')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4)
    agent=ReservoirAgent(graph,'cuda',seed,profile='balanced_rate_v3',feature_source=source)
    budget=Budget();rows=[];metadata=[];stages=[];updates=0;status='running'
    training=[[training_case(3000000+seed*10000+stage*1000+i,near=stage==0 or (stage>=2 and i%4==0))
        for i in range(count)] for stage,count in enumerate(counts)]
    critic_seeds=list(range(3700000+seed*1000,3700000+seed*1000+16))
    protocol={'schema':'sdk9.c0_training/1.0','seed':seed,'feature_source':source,'method':METHOD,
        'counts':list(counts),'epochs':epochs,'lr':.0003,'ppo_options':512,'ppo_lr':1e-5,
        'training_stages':training,'critic_seeds':critic_seeds,'option_limit':10000,'wall_limit_s':7200,
        'selection':'fixed final checkpoint; validation never used as training data',
        'source':'randomly initialized action/value readout; frozen original connectome',
        'resume':'explicit warm-start only; exact resume unsupported',
        'source_hashes':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root/'flydrone/tellosim').rglob('*.py'))}}
    atomic_json(out/'protocol.json',protocol)
    save_checkpoint(out/'initial.pt',agent,None,root,0,0,{'method':'untrained','seed':seed,'curriculum':'C0'})
    agent.training_method=METHOD if source=='reservoir' else METHOD.replace('frozen MaleCNS','RAW OBSERVATION MLP CONTROL')
    optimizer=torch.optim.Adam(list(agent.model.body.parameters())+list(agent.model.actor.parameters()),lr=.0003)
    try:
        for stage,cases in enumerate(training):
            collection=collect(root,agent,cases,rows,metadata,budget,stage)
            result=fit(agent,optimizer,rows,epochs=epochs);updates+=result['gradient_steps']
            stages.append({'stage':stage,**collection,**result})
            save_checkpoint(out/f'stage-{stage}.pt',agent,optimizer,root,updates,budget.used,{'method':agent.training_method})
            atomic_json(out/'progress.json',{'status':'training','stages':stages,'options':budget.used})
            print(json.dumps({'phase':'fit','seed':seed,**stages[-1]}),flush=True)
        critic_fit(root,agent,out,critic_seeds,budget)
        save_checkpoint(out/'supervised.pt',agent,optimizer,root,updates,budget.used,{'method':agent.training_method,'phase':'before PPO'})
        optimizer,ppo_updates=ppo_fit(root,agent,out,seed,budget)
        status='completed'
    except BudgetExceeded:
        status='budget_exhausted';optimizer=None
        ppo_updates=json.loads((out/'ppo.json').read_text())['updates'] if (out/'ppo.json').exists() else 0
    except BaseException:
        save_checkpoint(out/'interrupted.pt',agent,optimizer,root,updates,budget.used,{'method':agent.training_method,'status':'interrupted'})
        raise
    finally:
        if rows:
            np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),
                masks=np.stack([r[1] for r in rows]),teacher_actions=np.asarray([r[2] for r in rows]))
            atomic_json(out/'data-provenance.json',metadata)
    save_checkpoint(out/'checkpoint.pt',agent,optimizer,root,updates,budget.used,
        {'method':agent.training_method,'seed':seed,'curriculum':'C0','status':status,'protocol_hash':digest(protocol)})
    probes=rows[:32];expected=[agent.decision(x,m,True)[3] for x,m,_ in probes]
    load_checkpoint(out/'checkpoint.pt',agent,root)
    assert expected==[agent.decision(x,m,True)[3] for x,m,_ in probes]
    result={'status':status,'seed':seed,'stages':stages,'options':budget.used,'supervised_updates':updates,
        'ppo_updates':ppo_updates,'elapsed_s':time.monotonic()-budget.start,'checkpoint_sha256':agent.checkpoint_sha256,
        'graph_sha256':agent.brain.graph_sha256,'mapping_sha256':agent.brain.mapping_sha256,
        'checkpoint_roundtrip_exact':True,'training_method':agent.training_method,'MODEL_READY_FOR_NEXT_STAGE':False}
    atomic_json(out/'training.json',result);print(json.dumps(result),flush=True)
    return result


def acceptance(models,random_result,case_hash):
    reasons=[]
    if sorted(models)!=list(SEEDS):reasons.append('requires formal seeds 11,22,33')
    if random_result.get('mode')!='random':reasons.append('baseline must be uniform random')
    if any(result.get('mode')!='argmax' for result in models.values()):reasons.append('frozen deployment evaluation must use argmax')
    for name,result in [('random',random_result)]+list(models.items()):
        rows=result['results']
        if result.get('case_hash')!=case_hash or len(rows)!=300 or len({r['case_id'] for r in rows})!=300:
            reasons.append(f'{name}: incomplete or mismatched sealed cases')
    if reasons:return {'passed':False,'reasons':reasons}
    reference=[r['case_id'] for r in random_result['results']]
    for seed,result in models.items():
        rows=result['results'];n=len(rows);success=sum(r['success'] for r in rows)/n
        if [r['case_id'] for r in rows]!=reference:reasons.append(f'{seed}: case order differs')
        if success<.9:reasons.append(f'{seed}: success below 90%')
        if sum(r['reason'] in ('collision','out_of_bounds') for r in rows)/n>.01:reasons.append(f'{seed}: collision/bounds above 1%')
        if success-sum(r['success'] for r in random_result['results'])/300<.2:reasons.append(f'{seed}: improvement below 20 percentage points')
    return {'passed':not reasons,'reasons':reasons}


def load_agent(root,graph,checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);contract=state['contract']
    torch.set_num_threads(4)
    if contract.get('neural_backend') in ('coo_deterministic','csr_fp64_accum'):
        from .checkpoint import configure_exact_execution
        configure_exact_execution('cuda')
    agent=ReservoirAgent(graph,'cuda',neural_backend=contract.get('neural_backend','csr'),profile=contract.get('profile','legacy'),feature_source=contract.get('feature_source','reservoir'),physics_profile=contract.get('physics',{}).get('profile','bounded_level_body_surrogate'))
    load_checkpoint(checkpoint,agent,root);return agent


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['cases','train','evaluate'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--graph',type=Path,default=Path('data/male-v1.npz'))
    p.add_argument('--seed',type=int,default=11);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cases',type=Path,default=Path('reports/sdk9_c0/cases.json'))
    p.add_argument('--split',choices=['validation','sealed_test','near-regression','faults'],default='validation')
    p.add_argument('--mode',choices=['argmax','sampled','rule','random'],default='argmax')
    p.add_argument('--limit',type=int);p.add_argument('--counts',type=int,nargs=4,default=[128,128,192,192]);p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--source',choices=['reservoir','raw_observation_control'],default='reservoir')
    a=p.parse_args();root=Path('.').resolve()
    if a.command=='cases':make_splits(a.out);return
    if a.command=='train':train(root,a.graph,a.out,a.seed,a.counts,a.epochs,a.source);return
    if not a.checkpoint:raise ValueError('checkpoint required')
    if a.out.exists():raise FileExistsError(a.out)
    if a.split=='sealed_test' and a.limit is not None:raise ValueError('sealed evaluation must include all 300 cases')
    if a.split=='sealed_test':
        lock=json.loads((a.cases.parent/'frozen-checkpoints.json').read_text())
        sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        if sha not in lock['checkpoint_hashes'] or digest(json.loads(a.cases.read_text())['sealed_test'])!=lock['case_hash']:
            raise ValueError('checkpoint/cases not frozen before sealed evaluation')
    if a.split=='near-regression':cases=validation_cases(True)
    elif a.split=='faults':
        cases=[]
        for i in range(8):
            for label,overrides in [('pose-noise',{'noise':.01}),('pose-delay',{'delay':2}),
                ('pose-mild',{'noise':.01,'delay':1,'dropout':.05}),('pose-lost',{'dropout':1.}),
                ('reply-lost',{'reply_loss':1.}),('request-lost',{'channel_profile':{'request_drop':1.}}),('channel-delay',{'channel_profile':{'request_delay_ticks':6,'reply_delay_ticks':6}})]:
                cases.append(curriculum_case(4300000+i,f'{label}-{i}',**overrides))
    else:cases=json.loads(a.cases.read_text())[a.split]
    if a.limit:cases=cases[:a.limit]
    agent=load_agent(root,a.graph,a.checkpoint)
    evaluate(root,agent,cases,a.out,a.out.stem,a.mode,record_indices=(0,1) if a.split!='sealed_test' else (0,))

if __name__=='__main__':main()
