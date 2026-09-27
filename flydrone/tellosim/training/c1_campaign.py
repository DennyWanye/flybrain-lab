"""Fixed-budget C1 curriculum, disjoint cases and frozen-model acceptance."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from .parallel import ReservoirPool,collect_rollout
from .continuation import initialize_from,run as ppo_run
from .checkpoint import configure_exact_execution
from .runtime import save_checkpoint
from .bootstrap import fit
from .env import TrainingEnv,curriculum_case
from .c0_campaign import expert,acceptance
from .campaign_v2 import evaluate_batch
from .contracts import digest
from .observer import TrainingObserver
from ..physics.rigid import PROFILE
from ..visual import atomic_json

SEEDS=(11,22,33)
METHOD='C1: 384 demonstrations + 2x512 DAgger options + 128 PPO; 25% C0 rehearsal; frozen MaleCNS'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(root):
    out=Path(root)/'reports/ts1_c1';out.mkdir(parents=True,exist_ok=True)
    cases={'schema':'tellosim.c1_cases/1','validation':[curriculum_case(21000000+i,f'c1-val-{i:03d}',curriculum='C1') for i in range(100)],
        'sealed_test':[curriculum_case(22000000+i,f'c1-test-{i:03d}',curriculum='C1') for i in range(300)],
        'c0_retention':[curriculum_case(23000000+i,f'c0-retention-{i:03d}') for i in range(100)]}
    path=out/'cases.json'
    if path.exists() and json.loads(path.read_text())!=cases:raise ValueError('case manifest already exists and differs')
    atomic_json(path,cases)
    protocol={'seeds':list(SEEDS),'method':METHOD,'stages':[384,512,512],'epochs':100,'learning_rate':.0001,'ppo_options':128,
        'options_per_seed':1536,'wall_seconds_per_seed':1800,'selection':'fixed final checkpoint; validation is never training data; no tuning on sealed results',
        'c1_gate':{'success_min':.9,'collision_or_bounds_max':.01,'advantage_over_uniform_min':.2},
        'c0_retention_gate':{'success_min':.9,'max_drop_from_starting_model':.05},
        'graph_sha256':'badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9',
        'split_hashes':{k:digest(v) for k,v in cases.items() if isinstance(v,list)}}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('protocol changed')
    atomic_json(out/'protocol.json',protocol);return cases

def train(root,graph,seed,source,out):
    root=Path(root).resolve();out=Path(out).resolve();prepare(root)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=ReservoirPool(graph,'cuda',seed=seed,profile='balanced_rate_v3',batch=4,physics_profile=PROFILE,neural_backend='csr_fp64_accum')
    lineage=initialize_from(source,pool,root);pool.training_method=METHOD
    save_checkpoint(out/'initial.pt',pool,None,root,0,0,{'lineage':lineage,'method':'C0 weights on C1 runtime, before new training'})
    observer=TrainingObserver(root,f'c1-s{seed}-learn',4,METHOD)
    rows=[];metadata=[];stages=[];options=0;updates=0;status='failed'
    optimizer=torch.optim.Adam(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0001)
    try:
        for stage,count in enumerate((384,512,512)):
            envs=[None]*4;episodes=[0]*4
            def make_env(i):
                episode=episodes[i];episodes[i]+=1
                sample_seed=30000000+seed*100000+stage*10000+i*1000+episode
                course='C0' if episode%4==3 else 'C1'
                case=curriculum_case(sample_seed,curriculum=course)
                return TrainingEnv(root,case,pool.lanes[i],observer=observer)
            def choose(env,i,features,mask):
                label=expert(env.observation,mask,env.steps==0)
                rows.append((features.copy(),mask.copy(),label));metadata.append({'seed':env.case['seed'],'curriculum':env.case.get('curriculum','C0'),'stage':stage,'step':env.steps,'label':label})
                if stage==0:return label,0.,0.,None
                # Scheduled training-only turn perturbation; teacher labels remain independent.
                if env.case.get('curriculum')=='C1' and env.steps==2 and i%2==0:return (7 if i==0 else 8),0.,0.,None
                return pool.lanes[i].decision(features,mask,deterministic=(i%2==0))
            try:collected,ticks,outcomes=collect_rollout(pool,envs,[count//4]*4,make_env,deadline=started+1800,decision_fn=choose)
            finally:
                for env in envs:
                    if env:env.close('collection_stage_complete')
            options+=len(collected);metrics=fit(pool,optimizer,rows,epochs=100);updates+=metrics['gradient_steps'];pool.value_trained=False
            stage_result={'stage':stage,'options':len(collected),'base_ticks':ticks,'outcomes':outcomes,**metrics};stages.append(stage_result)
            save_checkpoint(out/f'stage-{stage}.pt',pool,optimizer,root,updates,options,{'method':METHOD,'lineage':lineage})
            observer.update({'options':options,'updates':updates,'elapsed_s':time.monotonic()-started})
            atomic_json(out/'progress.json',{'status':'learning','options':options,'stages':stages,'elapsed_s':time.monotonic()-started})
            print(json.dumps({'seed':seed,'phase':'fit','options':options,**metrics}),flush=True)
        save_checkpoint(out/'supervised.pt',pool,optimizer,root,updates,options,{'method':METHOD,'lineage':lineage})
        status='supervised_complete'
    except BaseException as exc:
        atomic_json(out/'failure.json',{'error':type(exc).__name__+': '+str(exc),'completed_options':options,'collected_examples':len(rows),'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'})
        raise
    finally:
        observer.close(status)
        if rows:np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]),labels=np.asarray([r[2] for r in rows]))
        atomic_json(out/'provenance.json',metadata)
    del pool,optimizer;torch.cuda.empty_cache()
    remaining=1800-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('C1 training budget exhausted')
    ppo=ppo_run(root,graph,out/'ppo',options=128,rollout=32,batch=4,seed=seed,init_from=out/'supervised.pt',wall_seconds=remaining,curriculum='C1',observe=True,observer_id=f'c1-s{seed}-ppo')
    result={'status':ppo['status'],'seed':seed,'method':METHOD,'lineage':lineage,'options':options+ppo['options'],'elapsed_s':time.monotonic()-started,'stages':stages,'ppo':ppo,'checkpoint':str((out/'ppo/checkpoint.pt').relative_to(root)),'MODEL_READY_FOR_NEXT_STAGE':False}
    atomic_json(out/'training.json',result);return result

def freeze(root):
    out=Path(root)/'reports/ts1_c1';cases=prepare(root);models=[]
    for seed in SEEDS:
        run=Path(root)/f'runs/tellosim-sdk9/c1-s{seed}'
        status=json.loads((run/'training.json').read_text());assert status['status']=='completed' and status['options']==1536
        checkpoint=run/'ppo/checkpoint.pt';models.append({'seed':seed,'path':str(checkpoint.relative_to(root)),'sha256':sha(checkpoint)})
    lock={'models':models,'case_hash':digest(cases['sealed_test']),'selection':'all final checkpoints fixed before sealed evaluation'}
    if (out/'frozen-checkpoints.json').exists() and json.loads((out/'frozen-checkpoints.json').read_text())!=lock:raise ValueError('frozen candidates cannot be changed')
    atomic_json(out/'frozen-checkpoints.json',lock);return lock

def evaluate_seed(root,graph,seed):
    root=Path(root).resolve();out=root/'reports/ts1_c1';cases=prepare(root);lock=json.loads((out/'frozen-checkpoints.json').read_text())
    model=next(x for x in lock['models'] if x['seed']==seed);checkpoint=root/model['path'];assert sha(checkpoint)==model['sha256']
    for split in ('validation','c0_retention','sealed_test'):
        target=out/f's{seed}-{split}.json'
        if target.exists():
            result=json.loads(target.read_text());assert result['case_hash']==digest(cases[split]) and result['checkpoint_sha256']==model['sha256'];continue
        evaluate_batch(root,graph,checkpoint,cases[split],target,f'c1-s{seed}-{split}',record_indices=(0,1) if split=='sealed_test' else (),batch=64)
    target=out/f's{seed}-before-retention.json'
    if not target.exists():evaluate_batch(root,graph,root/f'runs/tellosim-sdk9/c1-s{seed}/initial.pt',cases['c0_retention'],target,f'c1-before-s{seed}-retention',record_indices=(),batch=64)

def finalize(root):
    root=Path(root);out=root/'reports/ts1_c1';cases=prepare(root);lock=json.loads((out/'frozen-checkpoints.json').read_text())
    models={seed:json.loads((out/f's{seed}-sealed_test.json').read_text()) for seed in SEEDS}
    random=json.loads((out/'random-sealed.json').read_text());gate=acceptance(models,random,digest(cases['sealed_test']))
    retained=True;rows=[]
    for seed,model in models.items():
        before=json.loads((out/f's{seed}-before-retention.json').read_text());after=json.loads((out/f's{seed}-c0_retention.json').read_text());validation=json.loads((out/f's{seed}-validation.json').read_text())
        valid=after['success_rate']>=.9 and after['success_rate']+1e-8>=before['success_rate']-.05;retained &= valid
        rows.append({'seed':seed,'validation':f"{validation['successes']}/100",'sealed':f"{model['successes']}/300",'success_rate':model['success_rate'],'wilson95':model['wilson95'],'collision_or_bounds':model['collision_or_bounds'],'retention_before':before['successes'],'retention_after':after['successes'],'retention_passed':valid,'run_id':next(r['run_id'] for r in model['results'] if r['run_id'])})
        assert sha(root/next(x['path'] for x in lock['models'] if x['seed']==seed))==model['checkpoint_sha256']
    passed=gate['passed'] and retained
    note='C1 随机朝向、允许转向；三个模型封存：'+'、'.join(f"{r['seed']}={r['sealed']}" for r in rows)+'。'+('C1 与 C0 保持门槛均通过。' if passed else '尚未全部通过 C1 与 C0 保持门槛，暂不推进 C2。')+'起降由任务管理器执行；连接图冻结；不是学会真实飞行。'
    summary={'schema':'tellosim.c1_result/1','status':'evaluated','C1_TASK_LEARNED':passed,'MODEL_READY_FOR_NEXT_STAGE':passed,'C0_RETENTION_PASSED':retained,'C1_ACCEPTANCE':gate,'models':rows,'random_successes':random['successes'],'C2':'NOT_RUN','REAL_FLIGHT_READY':False,'FULL_TS1_READY':False,'note':note}
    atomic_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','freeze','evaluate','finalize']);p.add_argument('--seed',type=int,choices=SEEDS,default=11);a=p.parse_args()
    root=Path('.').resolve();graph=root/'data/male-v1.npz'
    if a.command=='prepare':prepare(root)
    elif a.command=='train':train(root,graph,a.seed,root/f'runs/tellosim-sdk9/rigid-final-s{a.seed}/checkpoint.pt',root/f'runs/tellosim-sdk9/c1-s{a.seed}')
    elif a.command=='freeze':freeze(root)
    elif a.command=='evaluate':evaluate_seed(root,graph,a.seed)
    else:finalize(root)
if __name__=='__main__':main()
