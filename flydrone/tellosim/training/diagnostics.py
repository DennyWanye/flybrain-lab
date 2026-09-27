"""Bounded, read-only signal and reward diagnostics; no optimizer updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import SensorObservation
from .runtime import ReservoirAgent, load_checkpoint
from .env import TrainingEnv
from ..visual import atomic_json


def signal_probe(agent, samples=40):
    results=[]; arrays=[]
    sensor=SensorObservation(0,0,0,'room_map',(0,0,1),0.,(0,0,0),1.,1.,0.,0.)
    for distance in (.15,.3,.6,1.2):
        features=[]
        for direction,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
            obs=sensor.vector((distance*dx,distance*dy,1),None,0,1)
            agent.reset();history=[]
            for step in range(samples):history.append(agent.observe(obs,step))
            features.append(np.asarray(history))
        a=np.asarray(features);arrays.append(a)
        diffs=[float(np.linalg.norm(a[i]-a[j])) for i in range(4) for j in range(i)]
        results.append({'distance_m':distance,'min_direction_difference_l2':min(diffs),
            'max_direction_difference_l2':max(diffs),'distinct_direction_histories':len({x.tobytes() for x in a}),
            'nonconstant_features':int(np.count_nonzero(np.ptp(a.reshape(-1,a.shape[-1]),axis=0)>1e-7)),
            'feature_min':float(a.min()),'feature_max':float(a.max())})
    return results,np.asarray(arrays)


def reward_probe(root,agent):
    results=[]
    for strategy in ('stop','toward','away'):
        env=TrainingEnv(root,{'case_id':'reward-'+strategy,'seed':940001,'start':[0,0,1],'goal':[.6,0,1]},agent)
        discount=1.;total=0.;first=None
        try:
            while not env.terminated:
                action=0 if strategy=='stop' else 2 if strategy=='away' else (0 if np.linalg.norm(env.observation[:2]*6)<.17 else 1)
                row=env.step(action)
                if first is None:first={k:row[k] for k in ('reward','raw_reward','k','Gamma')}
                total+=discount*row['reward'];discount*=row['Gamma']
            results.append({'strategy':strategy,'discounted_episode_return':total,'raw_return':env.raw_return,
                'reason':env.reason,'duration_s':(env.session.tick-env.start_tick)/120,'first_option':first})
        finally:env.close()
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--profile',default='legacy');p.add_argument('--device',default='cuda');p.add_argument('--project-root',type=Path,default=Path('.'))
    args=p.parse_args();agent=ReservoirAgent(args.graph,args.device,profile=args.profile)
    if args.checkpoint:load_checkpoint(args.checkpoint,agent,args.project_root)
    args.out.mkdir(parents=True,exist_ok=True)
    probe,features=signal_probe(agent)
    np.savez_compressed(args.out/'direction-features.npz',features=features)
    report={'signal':probe,'reward':reward_probe(args.project_root,agent),'graph_sha256':agent.brain.graph_sha256,
        'mapping_sha256':agent.brain.mapping_sha256,'training_updates':0}
    atomic_json(args.out/'diagnostic.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
