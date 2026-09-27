"""Exact synchronous rollout-boundary snapshots. Never restore partial files."""
import copy,hashlib,os,random,sys
from pathlib import Path
import numpy as np
import torch
from .runtime import checkpoint_contract
from .env import TrainingEnv

def execution_contract(pool,root):
    root=Path(root);code_root=Path(__file__).resolve().parents[1]
    names=['../brain.py','../encoder.py','../policy.py','training/checkpoint.py','training/parallel.py','training/continuation.py','training/runtime.py','training/contracts.py','training/env.py','training/observer.py','visual.py','physics/world.py','physics/rigid.py','sdk/channels.py','sdk/codec.py']
    return {'policy':checkpoint_contract(pool,root),'batch':pool.brain.batch,'device':str(pool.brain.device),
        'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'threads':torch.get_num_threads(),
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'interop_threads':torch.get_num_interop_threads(),
        'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name(pool.brain.device) if pool.brain.device.type=='cuda' else 'cpu',
        'sources':{name:hashlib.sha256((code_root/name).read_bytes()).hexdigest() for name in names}}

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])

def save_training(path,pool,optimizer,envs,counters,root):
    state={'format':'tellosim.exact_training/3','boundary':'after_ppo_update_no_active_option',
        'contract':execution_contract(pool,root),'policy':pool.model.state_dict(),'optimizer':optimizer.state_dict(),
        'brain':pool.brain.state_dict(),'lanes':[{'brain_tick':a.brain_tick,'sample_id':a.sample_id,'action_rng':a.action_rng.get_state()} for a in pool.lanes],
        'envs':[env.export_state() if env else None for env in envs],
        'counters':copy.deepcopy(counters),'value_trained':pool.value_trained,'rng':rng_state()}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return state

def load_training(path,pool,optimizer,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.exact_training/3' or state.get('boundary')!='after_ppo_update_no_active_option':raise ValueError('not an exact boundary snapshot')
    if state['contract']!=execution_contract(pool,root):raise ValueError('resume execution contract changed; use explicit warm-start for a new run')
    envs=[]
    for i,saved in enumerate(state['envs']):
        env=None
        if saved is not None:
            env=TrainingEnv(root,saved['fields']['case'],pool.lanes[i])
            env.restore_state(saved)
        envs.append(env)
    pool.model.load_state_dict(state['policy']);optimizer.load_state_dict(state['optimizer'])
    pool.brain.load_state_dict(state['brain']);pool.value_trained=state['value_trained']
    for lane,values in zip(pool.lanes,state['lanes']):
        lane.brain_tick=values['brain_tick'];lane.sample_id=values['sample_id'];lane.action_rng.set_state(values['action_rng'])
    restore_rng(state['rng'])
    return envs,copy.deepcopy(state['counters'])


def configure_exact_execution(device):
    """Set the execution contract before CUDA initialization, never silently downgrade."""
    if str(device).startswith('cuda'):
        if torch.cuda.is_initialized() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):
            raise RuntimeError('exact CUDA training requires a fresh process with CUBLAS_WORKSPACE_CONFIG')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8',':16:8'):
            raise ValueError('unsupported CUBLAS_WORKSPACE_CONFIG for exact resume')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
