"""Evaluate all completed comparison checkpoints with matched action RNGs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from ..visual import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-seconds',type=int,default=0)
    p.add_argument('--tag',default='20260926');args=p.parse_args()
    root=Path('.').resolve();reports=root/'reports/sdk9_learning';source=reports/'comparison.json'
    deadline=time.monotonic()+args.wait_seconds
    while not source.exists():
        if time.monotonic()>deadline:raise RuntimeError('completed comparison not available')
        time.sleep(5)
    report=json.loads(source.read_text());children=[];handles=[]
    labels={'reservoir':'','raw_observation_control':'raw-','zero_brain_control':'zero-'}
    try:
        for seed in report['seeds']:
            group=[]
            for row in [r for r in report['rows'] if r['seed']==seed]:
                name=f"nearx-v2-{labels[row['source']]}s{seed}-{args.tag}"
                output=reports/(name+'-sampled.json')
                handle=(reports/(name+'-sampled.log')).open('w');handles.append(handle)
                command=[sys.executable,'-u','-m','flydrone.tellosim.training.evaluation',
                    '--graph','data/male-v1.npz','--checkpoint',f'runs/tellosim-sdk9/{name}/checkpoint.pt',
                    '--out',str(output),'--repeats','1','--sample-only',
                    '--case-count',str(row['evaluation_episodes'])]
                proc=subprocess.Popen(command,cwd=root,stdout=handle,stderr=subprocess.STDOUT)
                children.append(proc);group.append((proc,output,row))
                if proc.wait():raise RuntimeError(f'evaluation failed: {output}')
            while any(proc.poll() is None for proc,_,_ in group):time.sleep(1)
            for proc,output,row in group:
                if proc.returncode:raise RuntimeError(f'evaluation failed: {output}')
                data=json.loads(output.read_text())
                assert data['checkpoint_sha256']==row['checkpoint_sha256']
                score=data['summary']['sampled']
                row.update(sampled_successes=score['successes'],sampled_episodes=score['episodes'],
                           sampled_evaluation_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            print(json.dumps({'sampled_seed_completed':seed}),flush=True)
    finally:
        for proc in children:
            if proc.poll() is None:proc.terminate()
        for proc in children:
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for handle in handles:handle.close()
    report['sampled_scope']='One matched action RNG per each of 8 validation cases and 3 training seeds. Exploratory, not independent 24-case acceptance.'
    atomic_json(reports/'comparison-sampled.json',report)
    atomic_json(root/'reports/vis/tellosim/learning-summary.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
