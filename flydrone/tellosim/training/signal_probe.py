"""Supervised diagnostic decoder only; never saves a flight policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import SensorObservation
from .runtime import ReservoirAgent
from ..visual import atomic_json


def probe(graph,profile,device='cuda'):
    torch.set_num_threads(4);agent=ReservoirAgent(graph,device,profile=profile)
    rng=np.random.default_rng(970001);x=[];y=[]
    for index in range(192):
        label=index%3
        distance=rng.uniform(-.15,.15) if label==0 else rng.uniform(.3,.65)*(1 if label==1 else -1)
        sensor=SensorObservation(0,0,0,'room_map',(0,0,1),0.,(float(rng.uniform(-.08,.08)),0,0),
            1.,float(rng.uniform(.8,1)),0.,0.)
        observation=sensor.vector((distance,0,1),int(rng.integers(3)),float(rng.uniform(.5,3)),float(rng.uniform(.2,1)))
        agent.reset()
        for tick in range(16):features=agent.observe(observation,tick)
        x.append(features.copy());y.append(label)
    x=np.asarray(x);y=np.asarray(y);train=128
    mean=x[:train].mean(0);std=np.maximum(x[:train].std(0),.01)
    z=np.c_[np.clip((x-mean)/std,-5,5),np.ones(len(x))]
    w=np.linalg.solve(z[:train].T@z[:train]+np.eye(z.shape[1]),z[:train].T@np.eye(3)[y[:train]])
    predicted=(z@w).argmax(1)
    confusion=np.zeros((3,3),int)
    for truth,guess in zip(y[train:],predicted[train:]):confusion[truth,guess]+=1
    return {'profile':profile,'train_contexts':train,'test_contexts':len(x)-train,
        'train_accuracy':float(np.mean(predicted[:train]==y[:train])),
        'heldout_accuracy':float(np.mean(predicted[train:]==y[train:])),
        'labels':['inside_target','forward','back'],'heldout_confusion':confusion.tolist(),
        'scope':'Diagnostic linear decoder on synthetic measured observations with nuisance variables. Not flight success; no policy trained.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    rows=[probe(a.graph,profile,a.device) for profile in ('legacy','balanced_rate_v2')]
    atomic_json(a.out,{'rows':rows});print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
