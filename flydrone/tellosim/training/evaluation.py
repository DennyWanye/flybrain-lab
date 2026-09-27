"""Seeded sampled-vs-argmax evaluation of a frozen checkpoint."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
from .runtime import ReservoirAgent,load_checkpoint
from .env import TrainingEnv
from ..visual import atomic_json


def evaluate_checkpoint(root,graph,checkpoint,out,case_count=8,repeats=3,device='cuda',sample_only=False):
    checkpoint=Path(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    contract=state['contract'];torch.set_num_threads(4)
    agent=ReservoirAgent(graph,device,profile=contract.get('profile','legacy'),
                         feature_source=contract.get('feature_source','reservoir'))
    load_checkpoint(checkpoint,agent,root)
    cases=json.loads((checkpoint.parent/'cases.json').read_text())['validation'][:case_count]
    results=[]
    modes=('sampled',) if sample_only else ('argmax','sampled')
    for mode in modes:
        for rep in range(1 if mode=='argmax' else repeats):
            for index,case in enumerate(cases):
                torch.manual_seed(960000+rep*1000+index)
                env=TrainingEnv(root,case,agent);actions=collections.Counter();discount=1.;total=0.
                try:
                    while not env.terminated:
                        action,_,_,policy=agent.decision(env.features,env.mask,mode=='argmax')
                        actions[action]+=1;row=env.step(action,policy)
                        total+=discount*row['reward'];discount*=row['Gamma']
                    results.append({'mode':mode,'repeat':rep,'case_id':case['case_id'],
                        'reason':env.reason,'success':env.reason=='success','actions':dict(actions),
                        'discounted_return':total,'duration_s':(env.session.tick-env.start_tick)/120})
                finally:env.close()
            print(json.dumps({'mode':mode,'repeat':rep,'completed':len(results)}),flush=True)
    report={'checkpoint_sha256':agent.checkpoint_sha256,'feature_source':agent.feature_source,
        'scope':'Exploratory validation. Sampling seeds fixed, no training or checkpoint selection.',
        'results':results,'summary':{m:{'successes':sum(r['success'] for r in results if r['mode']==m),
        'episodes':sum(r['mode']==m for r in results)} for m in modes}}
    atomic_json(out,report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=Path('.'))
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--case-count',type=int,default=8)
    p.add_argument('--sample-only',action='store_true');p.add_argument('--repeats',type=int,default=3);p.add_argument('--device',default='cuda')
    a=p.parse_args();result=evaluate_checkpoint(a.project_root,a.graph,a.checkpoint,a.out,a.case_count,a.repeats,a.device,a.sample_only)
    print(json.dumps(result['summary']),flush=True)

if __name__=='__main__':main()
