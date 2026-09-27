from __future__ import annotations
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch

from ..visual import atomic_json
from .contracts import SPEC,digest
from .env import TrainingEnv,sample_case,curriculum_case
from .runtime import ReservoirAgent,save_checkpoint,load_checkpoint,update


def parameter_hash(agent):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in agent.model.parameters())).hexdigest()


def evaluate(root,agent,cases,label,run_prefix,record_first=True,ablate=False):
    results=[]
    for index,case in enumerate(cases):
        env=TrainingEnv(root,case,agent,record=record_first and index==0,
            run_id=f'{run_prefix}-{label}-{index}',policy_source=f'{label}_sdk9')
        rng=np.random.default_rng(case['seed']+300)
        try:
            while not env.terminated:
                if label=='rule':
                    error=env.observation[:2]*6
                    action=0 if np.linalg.norm(error)<=.17 else (1 if error[0]>0 else 2) if abs(error[0])>=abs(error[1]) else (3 if error[1]>0 else 4)
                    if not env.mask[action]:action=0
                    policy=None
                elif label=='random':
                    action=int(rng.choice(np.flatnonzero(env.mask)));policy=None
                else:
                    features=np.zeros_like(env.features) if ablate else env.features
                    action,_,_,policy=agent.decision(features,env.mask,True)
                env.step(action,policy)
            results.append({'case_id':case['case_id'],'seed':case['seed'],'reason':env.reason,
                'success':env.reason=='success','return':env.raw_return,'steps':env.steps,
                'duration_s':(env.session.tick-env.start_tick)/120,
                'distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
                'run_id':env.session.run_id if index==0 and record_first else None})
        finally:env.close()
        print(json.dumps({'evaluation':label,'case':index+1,'of':len(cases),'result':results[-1]['reason']}),flush=True)
    return {'label':label,'episodes':len(results),'successes':sum(r['success'] for r in results),
            'success_rate':float(np.mean([r['success'] for r in results])),
            'mean_return':float(np.mean([r['return'] for r in results])),'results':results}


def run(root,graph,out,options=128,rollout=32,eval_cases=4,seed=11,device='cpu',init_from=None,profile='legacy',curriculum='C0',feature_source='reservoir',critic_episodes=0,freeze_body=False,learning_rate=3e-4,publish=True):
    if not 0<=critic_episodes<=64 or not 0<learning_rate<=1e-2:
        raise ValueError('critic episodes 0..64 and learning rate (0, .01] required')
    if (critic_episodes or freeze_body) and not init_from:
        raise ValueError('critic warmup/frozen body require an explicit source checkpoint')
    if type(options) is not int or not 8<=options<=10000 or not 2<=rollout<=256 or not 1<=eval_cases<=100:
        raise ValueError('options 8..10000, rollout 2..256, diagnostic cases 1..100 required')
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('output directory must be new')
    out.mkdir(parents=True,exist_ok=True)
    prefix=out.name
    if not all(c.isalnum() or c in '-_' for c in prefix):raise ValueError('output name must be alphanumeric/hyphen/underscore')
    torch.set_num_threads(4)
    started=time.monotonic()
    agent=ReservoirAgent(graph,device,seed,profile=profile,feature_source=feature_source)
    lineage=None
    if init_from:
        load_checkpoint(init_from,agent,root)
        lineage={'mode':'warm_start','path':str(Path(init_from).resolve()),
                 'sha256':agent.checkpoint_sha256,'optimizer_reset':True,'episode_state_reset':True}
    agent.training_method=('PPO warm-start' if init_from else 'PPO from random initialization')+'; '+feature_source
    if freeze_body:
        for parameter in agent.model.body.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in agent.model.parameters() if p.requires_grad],lr=learning_rate)
    validation=[curriculum_case(910000+i,f'validation-{i:03d}',curriculum) for i in range(100)]
    sealed=[curriculum_case(920000+i,f'sealed-{i:03d}',curriculum) for i in range(300)]
    splits={'schema':'sdk9.cases/1.0','train_seed_start':110000+seed*10000,
            'critic_seeds':list(range(700000+seed*1000,700000+seed*1000+critic_episodes)),
            'curriculum':curriculum,'validation':validation,'sealed_test':sealed,'sealed_test_opened':False}
    atomic_json(out/'cases.json',splits)
    save_checkpoint(out/'initial.pt',agent,optimizer,root,0,0,{'seed':seed,'case_hash':digest(splits),'initialization':lineage,'curriculum':curriculum,'feature_source':feature_source,'method':agent.training_method})
    load_checkpoint(out/'initial.pt',agent,root)
    original_parameters=[p.detach().clone() for p in agent.model.parameters()]
    initial_hash=parameter_hash(agent)
    baselines=[]
    # Small environment checks, not a claimed 99% formal rule acceptance run.
    for label in ('rule','random','warm-start' if init_from else 'untrained'):
        baselines.append(evaluate(root,agent,validation[:eval_cases],label,prefix,record_first=label!='random'))
    if not all(r['success'] for r in baselines[0]['results']):
        atomic_json(out/'summary.json',{'status':'environment_check_failed','baselines':baselines})
        raise RuntimeError('nominal rule check failed; fix the environment before training')
    load_checkpoint(out/'initial.pt',agent,root)
    critic_metrics=None
    if critic_episodes:
        from .critic_warmup import warmup
        critic_metrics=warmup(root,agent,out,splits['critic_seeds'],curriculum)
    agent.training_method=('PPO warm-start' if init_from else 'PPO from random initialization')+'; '+feature_source
    if freeze_body:agent.training_method+='; frozen shared body; actor and critic heads only'
    run_extra={'seed':seed,'case_hash':digest(splits),'initialization':lineage,'curriculum':curriculum,
        'feature_source':feature_source,'method':agent.training_method,'critic_warmup_episodes':critic_episodes,
        'freeze_body':freeze_body,'learning_rate':learning_rate}
    if critic_episodes:save_checkpoint(out/'critic-ready.pt',agent,optimizer,root,0,0,run_extra)
    train_start=time.monotonic();total=0;updates=0;episodes=0;total_base=0;training_results=[];probe=[]
    env=None
    try:
        with (out/'updates.jsonl').open('w') as log, (out/'transitions.jsonl').open('w') as trace:
            while total<options:
                rows=[]
                if env is None:
                    case=curriculum_case(splits['train_seed_start']+episodes,curriculum=curriculum)
                    env=TrainingEnv(root,case,agent,record=updates==0,run_id=f'{prefix}-training-{episodes}')
                while len(rows)<min(rollout,options-total):
                    features=env.features.copy();mask=env.mask.copy()
                    action,logp,value,policy=agent.decision(features,mask)
                    result=env.step(action,policy)
                    if total+len(rows)+1>=options and not result['terminal']:
                        result['truncated']=True
                    _,_,next_value,_=agent.decision(result['next_features'],result['next_mask'],True)
                    row={'features':features,'mask':mask,'action':action,'logp':logp,'value':value,
                         **{k:result[k] for k in ('reward','Gamma','terminal','truncated')},'next_value':next_value}
                    rows.append(row);total_base+=result['k']
                    if len(probe)<16:probe.append((features.copy(),mask.copy()))
                    recorded={k:v for k,v in result.items() if k not in ('next_features','next_observation','next_mask','reward_base_terms')}
                    recorded.update(option=total+len(rows),case_id=env.case['case_id'],action=action,mask=mask.tolist(),
                                    old_logp=logp,value=value,next_value=next_value,
                                    feature_sha256=hashlib.sha256(features.tobytes()).hexdigest())
                    if total+len(rows)<=8:recorded['reward_base_terms']=result['reward_base_terms']
                    trace.write(json.dumps(recorded)+'\n')
                    if env.terminated:
                        training_results.append({'episode':episodes,'return':env.raw_return,'reason':env.reason,'steps':env.steps})
                        env.close();episodes+=1;env=None
                        if len(rows)<min(rollout,options-total):
                            case=curriculum_case(splits['train_seed_start']+episodes,curriculum=curriculum)
                            env=TrainingEnv(root,case,agent)
                    if (total+len(rows))%8==0:
                        print(json.dumps({'phase':'sampling','options':total+len(rows),'target':options,'base_ticks':total_base}),flush=True)
                brain_advances=agent.brain.advance_count
                metrics=update(agent,optimizer,rows)
                assert brain_advances==agent.brain.advance_count,'PPO must not advance the reservoir'
                total+=len(rows);updates+=1
                metrics.update(update=updates,options=total,base_ticks=total_base,elapsed_s=time.monotonic()-train_start)
                log.write(json.dumps(metrics)+'\n');log.flush();trace.flush()
                save_checkpoint(out/'checkpoint.pt',agent,optimizer,root,updates,total,run_extra)
                atomic_json(out/'progress.json',metrics)
                print(json.dumps({'phase':'update',**metrics}),flush=True)
    finally:
        if env is not None:env.close('external_budget_truncation')
    parameter_delta=float(torch.sqrt(sum((a-b.detach()).square().sum() for a,b in zip(original_parameters,agent.model.parameters()))))
    trained_hash=parameter_hash(agent)
    train_elapsed=time.monotonic()-train_start
    # Check that a newly loaded artifact, not the in-memory optimizer object,
    # reproduces decisions exactly before evaluating it on held-out cases.
    expected=[agent.decision(x,m,True)[3] for x,m in probe]
    for p in agent.model.parameters():p.data.zero_()
    load_checkpoint(out/'checkpoint.pt',agent,root)
    restored=[agent.decision(x,m,True)[3] for x,m in probe]
    roundtrip=expected==restored
    if not roundtrip:raise AssertionError('checkpoint decision roundtrip mismatch')
    evaluation=evaluate(root,agent,validation[:eval_cases],'trained',prefix)
    ablated=evaluate(root,agent,validation[:min(eval_cases,2)],'brain-zero',prefix,record_first=False,ablate=True)
    faults=[sample_case(930000,'fault-noisy-pose',noise=.01,delay=2),
            sample_case(930001,'fault-reply-loss',reply_loss=1.)]
    fault_results=evaluate(root,agent,faults,'fault-diagnostic',prefix)
    summary={'schema':'tellosim.training_pilot/1.0','status':'completed','seed':seed,'device':device,
        'critic_warmup':critic_metrics,'freeze_body':freeze_body,'learning_rate':learning_rate,
        'profile':profile,'curriculum':curriculum,'feature_source':feature_source,'options':total,'updates':updates,'base_ticks':total_base,'episodes':training_results,'initialization':lineage,
        'train_elapsed_s':train_elapsed,'options_per_s':total/train_elapsed,'base_ticks_per_s':total_base/train_elapsed,
        'elapsed_s':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0,
        'parameter_delta_l2':parameter_delta,'initial_parameter_hash':initial_hash,'trained_parameter_hash':trained_hash,
        'checkpoint_roundtrip_exact':roundtrip,'frozen_graph_sha256':agent.brain.graph_sha256,
        'contract_hash':digest(SPEC),'case_hash':digest(splits),'baselines':baselines,'trained':evaluation,
        'brain_zero_ablation':ablated,'fault_diagnostics':fault_results,
        'checkpoint_sha256':hashlib.sha256((out/'checkpoint.pt').read_bytes()).hexdigest(),
        'TRAINING_PIPELINE_READY':True,'TS1_C0_TASK_LEARNED':False,'MODEL_READY_FOR_NEXT_STAGE':False,
        'scope':'Small single-seed diagnostic pilot. Not formal 3-seed x 300 sealed acceptance. Sealed cases unused.',
        'resume_support':'No exact resume; compatible inference/warm-start only',
        'training_method':agent.training_method}
    atomic_json(out/'summary.json',summary)
    summary['artifact_directory']=str(out.relative_to(root)) if out.is_relative_to(root) else str(out)
    if publish and feature_source=='reservoir':atomic_json(root/'reports/vis/tellosim/training-summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description='Train frozen MaleCNS SDK9 policy in the shared 3D simulator')
    parser.add_argument('--project-root',type=Path,default=Path('.'));parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--options',type=int,default=128)
    parser.add_argument('--rollout',type=int,default=32);parser.add_argument('--eval-cases',type=int,default=4)
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--device',default='cpu')
    parser.add_argument('--init-from',type=Path)
    parser.add_argument('--profile',choices=['legacy','balanced_rate_v2','balanced_rate_v3'],default='legacy')
    parser.add_argument('--feature-source',choices=['reservoir','raw_observation_control','zero_brain_control'],default='reservoir')
    parser.add_argument('--curriculum',choices=['C0','C0-near-x','C0-near-xy'],default='C0')
    parser.add_argument('--critic-episodes',type=int,default=0)
    parser.add_argument('--freeze-body',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--no-publish',action='store_true')
    args=parser.parse_args()
    result=run(args.project_root,args.graph,args.out,args.options,args.rollout,args.eval_cases,args.seed,args.device,args.init_from,args.profile,args.curriculum,args.feature_source,args.critic_episodes,args.freeze_body,args.learning_rate,not args.no_publish)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baselines','episodes','trained','brain_zero_ablation','fault_diagnostics')},indent=2),flush=True)


if __name__=='__main__':main()
