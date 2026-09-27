"""Targeted development-only traces; never consumes sealed cases."""
import json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from flydrone.tellosim.training.robust_campaign import RobustPool,bundle_spec,result_row
from flydrone.tellosim.training.robust_env import RobustJointEnv
from flydrone.tellosim.training.joint_refine import settling_label
from flydrone.tellosim.training.heading import teacher
from flydrone.tellosim.training.checkpoint import configure_exact_execution
from flydrone.tellosim.visual import atomic_json
root=Path.cwd();out=root/'reports/ts1_stability';torch.set_num_threads(4);configure_exact_execution('cuda')
indices=[3,17,21,63,72,85,0,1]
cases=json.loads((root/'reports/ts1_robust/cases.json').read_text())['validation']
pool=RobustPool(root,bundle_spec(root,11),batch=len(indices));envs=[None]*len(indices);gens={};traces=[];results=[];xs=[];started=time.monotonic()
try:
    for i,index in enumerate(indices):envs[i]=RobustJointEnv(root,cases[index],pool.lanes[i])
    while any(envs):
        if time.monotonic()-started>900:raise TimeoutError('diagnostic budget')
        pending={}
        for i,env in enumerate(envs):
            if env is None:continue
            if i not in gens:
                a,_,_,policy=pool.lanes[i].decision(env.features,env.mask,True)
                label=(settling_label if env.phase=='navigation' else teacher)(env.observation,env.mask,env.steps==0 or env.previous_action is None)
                traces.append(dict(case_index=indices[i],phase=env.phase,step=env.steps,action=a,label=label,probabilities=policy['probabilities'],observation=env.observation.tolist(),mask=env.mask.tolist(),feature_index=len(xs),time_s=(env.session.tick-env.start_tick)/120))
                xs.append(env.features.copy());gens[i]=env.step_iter(a,policy)
            try:next(gens[i]);pending[i]=(env.observation,env.device.latest_observation().sample_id)
            except StopIteration:
                del gens[i]
                if env.terminated:
                    ts=[t for t in traces if t['case_index']==indices[i]]
                    results.append(result_row(env,env.case,[t['action'] for t in ts],None,[t['phase'] for t in ts]));env.close();envs[i]=None
        if pending:
            f=pool.observe_lanes(pending)
            for i in pending:envs[i].features=f[i].copy()
finally:
    for env in envs:
        if env:env.close()
atomic_json(out/'development-traces.json',dict(seed=11,split='J2 validation only',indices=indices,traces=traces,results=results,elapsed_s=time.monotonic()-started))
np.savez_compressed(out/'development-features.npz',features=np.stack(xs))
print(json.dumps(dict(episodes=len(results),successes=sum(r['success'] for r in results),decisions=len(traces),elapsed_s=time.monotonic()-started)),flush=True)
