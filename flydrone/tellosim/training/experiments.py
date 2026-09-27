"""Finite multi-seed comparison. Child processes are tracked and cleaned up."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from ..visual import atomic_json

SOURCES={'reservoir':'','raw_observation_control':'raw-','zero_brain_control':'zero-'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path('.'))
    p.add_argument('--options',type=int,default=1536);p.add_argument('--eval-cases',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]);p.add_argument('--tag',default='20260926')
    args=p.parse_args();root=args.project_root.resolve();reports=root/'reports/sdk9_learning'
    reports.mkdir(parents=True,exist_ok=True);results=[];children=[];handles=[]
    try:
        for seed in args.seeds:
            group=[]
            for source,label in SOURCES.items():
                name=f'nearx-v2-{label}s{seed}-{args.tag}';out=root/'runs/tellosim-sdk9'/name
                if (out/'summary.json').exists():
                    saved=json.loads((out/'summary.json').read_text())
                    assert saved['status']=='completed' and saved['options']==args.options
                    assert saved['seed']==seed and saved['feature_source']==source
                    assert saved['profile']=='balanced_rate_v2' and saved['curriculum']=='C0-near-x'
                    assert saved['trained']['episodes']==args.eval_cases
                    results.append(saved);continue
                if out.exists():raise RuntimeError(f'incomplete existing run: {out}')
                handle=(reports/(name+'.log')).open('w');handles.append(handle)
                command=[sys.executable,'-u','-m','flydrone.tellosim.training.runner',
                    '--graph','data/male-v1.npz','--device','cuda','--profile','balanced_rate_v2',
                    '--curriculum','C0-near-x','--feature-source',source,'--seed',str(seed),
                    '--out',str(out),'--options',str(args.options),'--rollout','128',
                    '--eval-cases',str(args.eval_cases)]
                proc=subprocess.Popen(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT)
                children.append(proc);group.append((proc,out))
                print(json.dumps({'started':name,'pid':proc.pid}),flush=True)
                # A single GPU runs these small sparse kernels faster serially.
                if proc.wait():raise RuntimeError(f'run failed: {out}')
            while any(proc.poll() is None for proc,_ in group):time.sleep(1)
            for proc,out in group:
                if proc.returncode:raise RuntimeError(f'run failed {out}: exit {proc.returncode}')
                results.append(json.loads((out/'summary.json').read_text()))
            print(json.dumps({'seed_completed':seed}),flush=True)
    finally:
        for proc in children:
            if proc.poll() is None:proc.terminate()
        for proc in children:
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for handle in handles:handle.close()
    rows=[{'seed':x['seed'],'source':x['feature_source'],'options':x['options'],
        'trained_successes':x['trained']['successes'],'evaluation_episodes':x['trained']['episodes'],
        'initial_successes':x['baselines'][2]['successes'],
        'training_successes':sum(y['reason']=='success' for y in x['episodes']),
        'training_episodes':len(x['episodes']),'parameter_delta_l2':x['parameter_delta_l2'],
        'checkpoint_roundtrip_exact':x['checkpoint_roundtrip_exact'],
        'checkpoint_sha256':x['checkpoint_sha256'],'case_hash':x['case_hash']} for x in results]
    assert len({x['case_hash'] for x in results})==len(args.seeds)  # seed only changes training split
    for seed in args.seeds:
        group=[x for x in results if x['seed']==seed]
        assert len({x['case_hash'] for x in group})==1
        assert len({x['initial_parameter_hash'] for x in group})==1
    report={'schema':'sdk9.learning_comparison/1.0','curriculum':'C0-near-x','seeds':args.seeds,
        'options_per_run':args.options,'rows':rows,'MODEL_READY_FOR_NEXT_STAGE':False,
        'scope':'Exploratory held-out validation, same initial parameters and task cases per seed; sealed tests unused.'}
    atomic_json(reports/'comparison.json',report)
    atomic_json(root/'reports/vis/tellosim/learning-summary.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
