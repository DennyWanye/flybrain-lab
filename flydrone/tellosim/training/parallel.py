"""Synchronous, masked neural batches; independent physics and sensor lanes."""
import time
import numpy as np
import torch
from .runtime import ReservoirAgent

class AgentLane:
    def __init__(self,pool,index):
        self.pool=pool;self.index=index;self.brain_tick=0;self.sample_id=-1
        self.action_rng=torch.Generator(device="cpu").manual_seed(pool.action_seed+1000003*index)
    def __getattr__(self,name):return getattr(self.pool,name)
    def decision(self,features,mask,deterministic=False):
        return self.pool.decision(features,mask,deterministic,generator=self.action_rng)
    def reset(self):
        mask=np.zeros(self.brain.batch,bool);mask[self.index]=True
        self.brain.reset(mask);self.brain_tick=0;self.sample_id=-1
    def observe(self,vector,sample_id):
        return self.pool.observe_lanes({self.index:(vector,sample_id)})[self.index]
    def snapshot(self):
        b=self.brain;ids=b.outputs[:64];i=self.index
        v=b.v[ids,i].cpu().tolist();s=b.s[ids,i].cpu().tolist();tr=b.trace[ids,i].cpu().tolist()
        return {'status':'recorded','sample_phase':'after_substep_3','neurons':[
            {'neuron_id':b.readout_ids[k],'neuron_index':int(ids[k]),'v':v[k],'spike':int(s[k]),'trace':tr[k]} for k in range(len(v))]}

class ReservoirPool(ReservoirAgent):
    def __init__(self,*args,batch=4,**kwargs):
        self.action_seed=int(kwargs.get("seed",args[2] if len(args)>2 else 11))
        super().__init__(*args,batch=batch,**kwargs)
        self.lanes=[AgentLane(self,i) for i in range(batch)]
    def observe_lanes(self,pending):
        observations=np.zeros((self.brain.batch,26),np.float32);mask=np.zeros(self.brain.batch,bool)
        for i,(vector,sample_id) in pending.items():
            lane=self.lanes[i]
            if sample_id<=lane.sample_id:raise ValueError('stale lane observation')
            observations[i]=vector;mask[i]=True
        self.brain.advance(self.brain.encoder(observations),mask)
        for i,(_,sample_id) in pending.items():self.lanes[i].brain_tick+=4;self.lanes[i].sample_id=sample_id
        features=self.brain.current_features()
        if self.feature_source=='zero_brain_control':features[:]=0
        elif self.feature_source=='raw_observation_control':
            features[:]=0;features[:,:26]=observations;features[:,:3]*=np.array([6,6,3])
        return features

def collect_rollout(pool,envs,quotas,make_env,deadline=None,activity=None,decision_fn=None):
    """Do not compute GAE across lanes. Cut traces at every lane boundary."""
    rows=[[] for _ in envs];generators={};pending_rows={};base_ticks=0;episodes=[]
    while any(len(rows[i])<quotas[i] for i in range(len(envs))):
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError("rollout wall budget exhausted")
        pending={}
        for i in range(len(envs)):
            if len(rows[i])>=quotas[i]:continue
            while True:
                if i not in generators:
                    if envs[i] is None:envs[i]=make_env(i)
                    env=envs[i];features=env.features.copy();mask=env.mask.copy()
                    action,logp,value,policy=pool.lanes[i].decision(features,mask) if decision_fn is None else decision_fn(env,i,features,mask)
                    pending_rows[i]={'features':features,'mask':mask,'action':action,'logp':logp,'value':value}
                    generators[i]=env.step_iter(action,policy)
                    if activity is not None:activity['attempted_options']+=1
                try:
                    env=next(generators[i]);
                    if activity is not None:activity['executed_base_ticks']+=1
                    pending[i]=(env.observation,env.device.latest_observation().sample_id);break
                except StopIteration as done:
                    result=done.value;env=envs[i];base_ticks+=result['k']
                    _,_,value,_=pool.lanes[i].decision(result['next_features'],result['next_mask'],True)
                    row={**pending_rows.pop(i),**{k:result[k] for k in ('reward','Gamma','terminal','truncated')},'next_value':value,
                        'lane':i,'case_id':env.case['case_id'],'duration_base_ticks':result['k']}
                    rows[i].append(row);del generators[i]
                    if env.terminated:
                        episodes.append({'lane':i,'case_id':env.case['case_id'],'reason':env.reason,'return':env.raw_return})
                        env.close();envs[i]=None
                    if len(rows[i])>=quotas[i]:break
        if pending:
            features=pool.observe_lanes(pending)
            for i in pending:envs[i].features=features[i].copy();envs[i].publish()
    for lane in rows:
        if lane and not lane[-1]['terminal']:lane[-1]['truncated']=True
    return [r for lane in rows for r in lane],base_ticks,episodes
