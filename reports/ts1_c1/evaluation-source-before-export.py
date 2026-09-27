"""Bounded six-DOF readout adaptation and versioned, batched evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .parallel import ReservoirPool,collect_rollout
from .runtime import load_checkpoint,save_checkpoint
from .checkpoint import configure_exact_execution
from .continuation import initialize_from,run as ppo_run
from .bootstrap import fit
from .c0_campaign import expert,training_case,summarize
from .contracts import digest
from .env import TrainingEnv
from ..physics.rigid import PROFILE
from ..visual import atomic_json

METHOD='Six-DOF adaptation: 256 rule demonstrations + 256 DAgger options + 128 PPO options; frozen MaleCNS'

def adapt(root,graph,out,seed,source):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(4);configure_exact_execution('cuda')
    pool=ReservoirPool(graph,'cuda',seed=seed,profile='balanced_rate_v3',batch=4,physics_profile=PROFILE,neural_backend='csr_fp64_accum')
    lineage=initialize_from(source,pool,root);pool.training_method=METHOD
    rows=[];metadata=[];stages=[];options=0
    optimizer=torch.optim.Adam(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=5e-5)
    for stage in range(2):
        envs=[None]*4;episodes=[0]*4
        def make_env(i):
            j=episodes[i];episodes[i]+=1
            sample_seed=7000000+seed*10000+stage*1000+i*200+j
            case=training_case(sample_seed,near=(j%2==0));return TrainingEnv(root,case,pool.lanes[i])
        def choose(env,i,features,mask):
            label=expert(env.observation,mask,env.steps==0)
            rows.append((features.copy(),mask.copy(),label));metadata.append({'case_seed':env.case['seed'],'step':env.steps,'stage':stage,'teacher_action':label})
            if stage==0:return label,0.,0.,None
            return pool.lanes[i].decision(features,mask,deterministic=(i%2==0))
        try:collected,ticks,outcomes=collect_rollout(pool,envs,[64]*4,make_env,deadline=started+1800,decision_fn=choose)
        finally:
            for env in envs:
                if env:env.close('supervised_collection_boundary')
        options+=len(collected);metrics=fit(pool,optimizer,rows,epochs=100)
        pool.value_trained=False
        stages.append({'stage':stage,'options':len(collected),'base_ticks':ticks,'outcomes':outcomes,**metrics})
        save_checkpoint(out/f'stage-{stage}.pt',pool,optimizer,root,sum(s['gradient_steps'] for s in stages),options,{'method':METHOD,'lineage':lineage,'stage':stage})
        atomic_json(out/'progress.json',{'stages':stages,'options':options,'elapsed_s':time.monotonic()-started})
        print(json.dumps({'phase':'adapt','seed':seed,'stage':stage,'options':options,**metrics}),flush=True)
    np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),masks=np.stack([r[1] for r in rows]),teacher_actions=np.array([r[2] for r in rows]))
    atomic_json(out/'data-provenance.json',metadata)
    # PPO below trains value as well; do not label the old critic calibrated to new dynamics.
    supervised=out/'supervised.pt';save_checkpoint(supervised,pool,optimizer,root,sum(s['gradient_steps'] for s in stages),options,{'method':METHOD,'lineage':lineage})
    remaining=1800-(time.monotonic()-started)
    if remaining<=0:raise TimeoutError('adaptation budget exhausted before PPO')
    del pool,optimizer;torch.cuda.empty_cache()
    result=ppo_run(root,graph,out/'ppo',options=128,rollout=32,batch=4,seed=seed,device='cuda',init_from=supervised,wall_seconds=remaining)
    report={'method':METHOD,'lineage':lineage,'stages':stages,'ppo':result,'options':options+result['options'],
        'elapsed_s':time.monotonic()-started,'checkpoint':str((out/'ppo/checkpoint.pt').relative_to(root)),
        'status':result['status'],'MODEL_READY_FOR_NEXT_STAGE':False}
    atomic_json(out/'training.json',report);return report


def evaluate_batch(root,graph,checkpoint,cases,out,label,record_indices=(0,),batch=4,device="cuda"):
    root=Path(root).resolve();out=Path(out);started=time.monotonic();torch.set_num_threads(4)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);c=state['contract']
    if c.get('neural_backend') in ('coo_deterministic','csr_fp64_accum'):configure_exact_execution(device)
    pool=ReservoirPool(graph,device,profile=c['profile'],batch=batch,physics_profile=PROFILE,neural_backend=c.get('neural_backend','csr'))
    load_checkpoint(checkpoint,pool,root)
    envs=[None]*batch;indices={};generators={};cursor=0;results={};decisions={}
    try:
        while len(results)<len(cases):
            pending={}
            for i in range(batch):
                if envs[i] is None:
                    if cursor>=len(cases):continue
                    index=cursor;cursor+=1;indices[i]=index;decisions[index]=[]
                    # Each recorded environment carries its lane and episode identity.
                    record=index in record_indices
                    envs[i]=TrainingEnv(root,cases[index],pool.lanes[i],record=record,
                        run_id=f'{label}-{index}' if record else None,policy_source=f'{label}_argmax_sdk9')
                env=envs[i]
                if i not in generators:
                    action,_,_,policy=pool.decision(env.features,env.mask,True);decisions[indices[i]].append(action)
                    generators[i]=env.step_iter(action,policy)
                try:
                    next(generators[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
                except StopIteration:
                    del generators[i]
                    if env.terminated:
                        index=indices[i];case=env.case
                        results[index]={'case_id':case['case_id'],'seed':case['seed'],'reason':env.reason,'success':env.reason=='success',
                            'return':env.raw_return,'steps':env.steps,'duration_s':(env.session.tick-env.start_tick)/120,
                            'distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
                            'run_id':env.session.run_id if env.live else None,'actions':decisions[index]}
                        env.close();envs[i]=None
                        if len(results)%10==0:
                            print(json.dumps({'phase':'batch-eval','label':label,'cases':len(results),'successes':sum(r['success'] for r in results.values())}),flush=True)
                            atomic_json(out.with_suffix('.progress.json'),summarize([results[k] for k in sorted(results)],label))
            if pending:
                features=pool.observe_lanes(pending)
                for i in pending:envs[i].features=features[i].copy();envs[i].publish()
    finally:
        for env in envs:
            if env:env.close()
    result={**summarize([results[k] for k in range(len(cases))],label),'mode':'argmax','batch':batch,'case_hash':digest(cases),
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-started}
    atomic_json(out,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['adapt','evaluate']);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--graph',type=Path,default=Path('data/male-v1.npz'));p.add_argument('--cases',type=Path,default=Path('reports/ts1_rigid_v2/cases.json'))
    p.add_argument('--split',choices=['validation','sealed_test'],default='validation');p.add_argument('--batch',type=int,default=4)
    a=p.parse_args();root=Path('.').resolve()
    if a.command=='adapt':adapt(root,a.graph,a.out,a.seed,a.checkpoint)
    else:
        if a.out.exists():raise FileExistsError(a.out)
        cases=json.loads(a.cases.read_text())[a.split]
        if a.split=='sealed_test':
            lock=json.loads((a.cases.parent/'frozen-checkpoints.json').read_text())
            if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() not in lock['checkpoint_hashes'] or digest(cases)!=lock['case_hash']:raise ValueError('sealed evaluation requires frozen models/cases')
        evaluate_batch(root,a.graph,a.checkpoint,cases,a.out,a.out.stem,batch=a.batch)
if __name__=='__main__':main()
