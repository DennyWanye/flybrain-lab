from pathlib import Path
import copy,json,time
from collections import deque
import numpy as np
import torch
from flydrone.tellosim.training.continuation import run
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2';GRAPH=ROOT/'data/male-v1.npz';INIT=ROOT/'runs/tellosim-sdk9/c0-s11-20260927/checkpoint.pt'
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple,deque)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def main():
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    args=dict(root=ROOT,graph=GRAPH,options=32,rollout=16,batch=4,seed=11,device='cuda',wall_seconds=300)
    import shutil
    from flydrone.tellosim.training import continuation
    original_save=continuation.save_training
    def capture(path,pool,optimizer,envs,counters,root):
        state=original_save(path,pool,optimizer,envs,counters,root)
        if counters['updates']==1:shutil.copy2(path,OUT/'fp64-boundary.pt')
        return state
    continuation.save_training=capture
    try:continuous=run(out=OUT/'perf-fp64-b4-continuous',init_from=INIT,**args)
    finally:continuation.save_training=original_save
    resumed=run(out=OUT/'perf-fp64-b4-resumed',resume=OUT/'fp64-boundary.pt',**args)
    a=torch.load(OUT/'perf-fp64-b4-continuous/resume.pt',weights_only=False);b=torch.load(OUT/'perf-fp64-b4-resumed/resume.pt',weights_only=False)
    checks={key:equal(a[key],b[key]) for key in ('policy','optimizer','brain','lanes','rng')}
    checks['physics_all_lanes']=all(equal(x['session']['world'],y['session']['world']) for x,y in zip(a['envs'],b['envs']))
    checks['sensors_all_lanes']=all(equal(x['sensor'],y['sensor']) and equal(x['sensor_rng'],y['sensor_rng']) for x,y in zip(a['envs'],b['envs']))
    assert all(checks.values()),checks
    args['batch']=1;single=run(out=OUT/'perf-fp64-b1',init_from=INIT,**args)
    report={'graph':'real MaleCNS','graph_sha256':a['contract']['policy']['graph_sha256'],'resume_exact':checks,
        'batch1':single,'batch4':continuous,'resumed':resumed,'benchmark_scope':'32 options each; identical saved-boundary fork; includes graph load/checkpoints, concurrent adaptation load; diagnostic PPO, not convergence or a long throughput test',
        'base_tick_speed_ratio_4_over_1':continuous['base_ticks_per_s']/single['base_ticks_per_s']}
    atomic_json(OUT/'full-graph-check.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
