"""Bounded real MaleCNS PPO with 1/4 independent environments and exact resume."""
import argparse,copy,hashlib,json,resource,time,signal,threading
from pathlib import Path
import numpy as np
import torch
from .parallel import ReservoirPool,collect_rollout
from .runtime import checkpoint_contract,save_checkpoint,update
from .checkpoint import save_training,load_training,configure_exact_execution
from .env import TrainingEnv,curriculum_case
from ..visual import atomic_json
from ..physics.rigid import PROFILE

def initialize_from(path,pool,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.sdk9.reservoir/1.0':raise ValueError('warm-start needs SDK9 policy artifact')
    source=dict(state['contract']);target=dict(checkpoint_contract(pool,root))
    source.pop('physics',None);target.pop('physics',None)
    source.pop('neural_backend',None);target.pop('neural_backend',None)
    if source!=target:raise ValueError('warm-start observation/graph/mapping/model contract mismatch')
    pool.model.load_state_dict(state['policy']);pool.value_trained=state.get('value_trained',False)
    return {'mode':'explicit_warm_start_new_physics','source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_physics':state['contract'].get('physics',{}).get('profile','bounded_level_body_surrogate'),'optimizer_reset':True,'source_neural_backend':state['contract'].get('neural_backend','csr'),'target_neural_backend':pool.neural_backend}

def run(root,graph,out,options=128,rollout=32,batch=4,seed=11,device='cuda',init_from=None,resume=None,stop_after_updates=None,wall_seconds=7200,curriculum='C0',observe=False,observer_id=None):
    if batch not in (1,4) or not batch<=rollout<=256 or not batch<=options<=10000 or not 0<wall_seconds<=7200:raise ValueError('bounded pilot: batch 1/4, options <=10000, wall <=7200s')
    if curriculum not in ('C0','C1'):raise ValueError('unknown curriculum')
    root=Path(root).resolve();out=Path(out).resolve()
    if not resume and out.exists() and any(out.iterdir()):raise FileExistsError('new run requires an empty output directory')
    if init_from and resume:raise ValueError('init-from and resume are different operations')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);started=time.monotonic()
    configure_exact_execution(device)
    pool=ReservoirPool(graph,device,seed,profile='balanced_rate_v3',batch=batch,physics_profile=PROFILE,neural_backend='csr_fp64_accum' if str(device).startswith('cuda') else 'csr')
    pool.training_method='PPO on six-DOF physics; frozen MaleCNS; exact rollout-boundary resume'
    optimizer=torch.optim.Adam(pool.model.parameters(),lr=1e-5)
    settings={'options':options,'rollout':rollout,'batch':batch,'seed':seed,'wall_seconds':wall_seconds,'curriculum':curriculum}
    if resume:
        envs,counters=load_training(resume,pool,optimizer,root)
        if counters['settings']!=settings:raise ValueError('resume run settings differ')
    else:
        lineage=initialize_from(init_from,pool,root) if init_from else None
        envs=[None]*batch
        counters={'settings':settings,'options':0,'updates':0,'episodes':[0]*batch,'base_ticks':0,'elapsed_s':0.,'lineage':lineage,'outcomes':[]}
        save_checkpoint(out/'initial.pt',pool,optimizer,root,0,0,{'lineage':lineage,'seed':seed,'physics':PROFILE})
        save_training(out/'resume.pt',pool,optimizer,envs,counters,root)
    from .observer import TrainingObserver
    observer=TrainingObserver(root,observer_id or out.name,batch,pool.training_method) if observe else None
    if observer:
        for env in envs:
            if env:env.observer=observer
    old_term=None
    if threading.current_thread() is threading.main_thread():
        old_term=signal.getsignal(signal.SIGTERM)
        def interrupted(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupted)
    prior=counters['elapsed_s']
    if resume and (out/'summary.json').exists():
        previous_report=json.loads((out/'summary.json').read_text())
        if previous_report.get('settings')==settings:prior=max(prior,previous_report['elapsed_s'])
    status='completed';committed=copy.deepcopy(counters);activity={'attempted_options':0,'executed_base_ticks':0}
    def make_env(i):
        episode=counters['episodes'][i];counters['episodes'][i]+=1
        case=curriculum_case(10000000+seed*100000+i*20000+episode,curriculum=('C0' if curriculum=='C1' and episode%4==3 else curriculum))
        return TrainingEnv(root,case,pool.lanes[i],observer=observer)
    try:
        while counters['options']<options:
            if prior+time.monotonic()-started>=wall_seconds:status='wall_budget';break
            n=min(rollout,options-counters['options']);quotas=[n//batch+(i<n%batch) for i in range(batch)]
            rows,ticks,outcomes=collect_rollout(pool,envs,quotas,make_env,deadline=started+wall_seconds-prior,activity=activity)
            metrics=update(pool,optimizer,rows)
            counters['options']+=len(rows);counters['base_ticks']+=ticks;counters['updates']+=1;counters['outcomes']+=outcomes
            counters['elapsed_s']=prior+time.monotonic()-started
            save_checkpoint(out/'checkpoint.pt',pool,optimizer,root,counters['updates'],counters['options'],{'lineage':counters['lineage'],'seed':seed,'physics':PROFILE})
            save_training(out/'resume.pt',pool,optimizer,envs,counters,root)
            committed=copy.deepcopy(counters)
            atomic_json(out/'progress.json',{**counters,**metrics})
            if observer:observer.update(counters)
            print(json.dumps({'options':counters['options'],'updates':counters['updates'],'base_ticks':counters['base_ticks'],'elapsed_s':counters['elapsed_s']}),flush=True)
            if stop_after_updates and counters['updates']>=stop_after_updates:status='stopped_at_saved_boundary';break
    except TimeoutError:
        status='wall_budget_partial_rollout_discarded'
    except KeyboardInterrupt:
        status='interrupted_use_last_committed_boundary'
    except BaseException as exc:
        status='failed_use_last_committed_boundary'
        atomic_json(out/'failure.json',{'error':type(exc).__name__+': '+str(exc),'committed_options':committed['options'],'activity':activity})
        raise
    finally:
        if old_term is not None:signal.signal(signal.SIGTERM,old_term)
        for env in envs:
            if env is not None:env.close('runner_shutdown_after_snapshot')
        if observer:observer.update(committed,status);observer.close(status)
    elapsed=prior+time.monotonic()-started
    counters=committed
    report={**counters,'status':status,'elapsed_s':elapsed,'options_per_s':counters['options']/max(elapsed,1e-9),
        'base_ticks_per_s':counters['base_ticks']/max(elapsed,1e-9),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/2**20 if pool.brain.device.type=='cuda' else 0,
        'this_process_activity':activity,'committed_options':counters['options'],
        'resume_support':'exact committed PPO update boundary, same execution contract; partial rollout discarded',
        'physics':PROFILE,'MODEL_READY_FOR_NEXT_STAGE':False}
    atomic_json(out/'summary.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path('.'));p.add_argument('--graph',type=Path,default=Path('data/male-v1.npz'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--options',type=int,default=128);p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--seed',type=int,default=11);p.add_argument('--device',default='cuda')
    p.add_argument('--init-from',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop-after-updates',type=int)
    p.add_argument('--wall-seconds',type=float,default=7200)
    p.add_argument('--curriculum',choices=['C0','C1'],default='C0');p.add_argument('--observe',action='store_true');p.add_argument('--observer-id')
    a=p.parse_args();print(json.dumps(run(a.project_root,a.graph,a.out,a.options,a.rollout,a.batch,a.seed,a.device,a.init_from,a.resume,a.stop_after_updates,a.wall_seconds,a.curriculum,a.observe,a.observer_id)))
if __name__=='__main__':main()
