from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.distributions import Categorical
import mujoco

from ...brain import FrozenReservoir
from ...policy import ActorCritic
from ...encoder import FeatureNormalizer
from .contracts import SPEC, Encoder34, digest


def distribution(model,features,masks):
    policy,value=model(features)
    masks=torch.as_tensor(np.asarray(masks),dtype=torch.bool,device=features.device)
    if not masks.any(-1).all():raise ValueError('empty action mask')
    return Categorical(logits=policy.logits.masked_fill(~masks,-1e9)),value


class ReservoirAgent:
    def __init__(self,graph,device='cpu',seed=11,readout=64,profile='legacy',feature_source='reservoir',batch=1,physics_profile='bounded_level_body_surrogate',neural_backend='csr'):
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        if profile not in ('legacy','balanced_rate_v2','balanced_rate_v3'):raise ValueError('unknown profile')
        if physics_profile not in ('bounded_level_body_surrogate','rigid_body_thrust_v2'):raise ValueError('unknown physics profile')
        if neural_backend not in ("csr","coo_deterministic","csr_fp64_accum"):raise ValueError("unknown neural backend")
        self.neural_backend=neural_backend
        self.physics_profile=physics_profile
        self.profile=profile
        self.value_trained=False
        self.stop_dwell_s=2. if profile=='balanced_rate_v3' else .5
        if feature_source not in ('reservoir','raw_observation_control','zero_brain_control'):raise ValueError('unknown feature source')
        self.feature_source=feature_source
        self.brain=FrozenReservoir(str(graph),batch,device=device,readout_neurons=readout,
                                   encoded_dim=34,encoder=Encoder34('legacy' if profile=='legacy' else 'balanced_rate_v2'),
                                   readout_strategy='channel_balanced' if profile!='legacy' else 'random_neighbors')
        if neural_backend=='csr_fp64_accum':
            self.brain.w=self.brain.w.to(dtype=torch.float64)
        if neural_backend=='coo_deterministic':
            if not torch.are_deterministic_algorithms_enabled():raise RuntimeError('COO resume backend requires deterministic algorithms')
            self.brain.w=self.brain.w.to_sparse_coo().coalesce()
        if profile!='legacy':
            # Fixed physical resting potential, not validation-derived statistics.
            mean=np.r_[np.full(readout,.14/(1-np.exp(-.020/.100))),np.zeros(readout)].astype(np.float32)
            std=np.r_[np.full(readout,.05),np.full(readout,.1)].astype(np.float32)
            self.brain.normalizer=FeatureNormalizer(mean,std)
        self.model=ActorCritic(self.brain.feature_dim,9)
        self.brain_tick=0;self.sample_id=-1

    def reset(self):
        self.brain.reset();self.brain_tick=0;self.sample_id=-1

    def observe(self,vector,sample_id):
        if sample_id<=self.sample_id:raise ValueError('duplicate or stale sensor sample')
        self.brain.advance(self.brain.encoder(np.asarray([vector],np.float32)))
        self.brain_tick+=4;self.sample_id=sample_id
        features=self.brain.current_features()[0]
        if self.feature_source=='zero_brain_control':return np.zeros_like(features)
        if self.feature_source=='raw_observation_control':
            # Deliberately separate diagnostic baseline, never the fly policy.
            features=np.zeros_like(features);features[:26]=vector
            features[:3]*=np.asarray([6,6,3],np.float32)
        return features

    @torch.no_grad()
    def decision(self,features,mask,deterministic=False,generator=None):
        policy,value=distribution(self.model,torch.as_tensor(features[None],dtype=torch.float32),[mask])
        action=policy.probs.argmax(-1) if deterministic else (policy.sample() if generator is None else torch.multinomial(policy.probs,1,generator=generator).squeeze(-1))
        return int(action.item()),float(policy.log_prob(action).item()),float(value.item()),{
            'selected_action':int(action.item()),'probabilities':policy.probs[0].tolist(),
            'value_estimate':float(value.item()),'value_trained':self.value_trained,'mask':np.asarray(mask,bool).tolist(),
            'input_source':'reservoir_v_trace' if self.feature_source=='reservoir' else self.feature_source,'feature_dim':self.brain.feature_dim}

    def snapshot(self):
        b=self.brain;ids=b.outputs[:64]
        v=b.v[ids,0].cpu().tolist();s=b.s[ids,0].cpu().tolist();tr=b.trace[ids,0].cpu().tolist()
        return {'status':'recorded','sample_phase':'after_substep_3','neurons':[
            {'neuron_id':b.readout_ids[i],'neuron_index':int(ids[i]),'v':v[i],'spike':int(s[i]),'trace':tr[i]}
            for i in range(len(v))]}


def checkpoint_contract(agent,root):
    root=Path(root)
    worlds={p.name:digest(json.loads(p.read_text())) for p in (root/'configs/tellosim').glob('world_*.json')}
    contract={'spec':SPEC,'spec_hash':digest(SPEC),'worlds':worlds,'mujoco_version':mujoco.__version__,
            'graph_sha256':agent.brain.graph_sha256,'mapping_sha256':agent.brain.mapping_sha256,
            'feature_dim':agent.brain.feature_dim,'encoded_dim':34,'observation_dim':26,'action_dim':9,
            'readout':len(agent.brain.readout_ids),'normalizer':{'kind':'identity','clip_abs':5.},
            'brain_parameters':{'internal_steps':4,'gain':1.,'tonic':.14,'drive_gain':.8,'mapping_seed':64}}
    if agent.profile!='legacy':
        contract.update(profile=agent.profile,readout_strategy=agent.brain.readout_strategy,
            encoder={'kind':'rate_offset34/2.0','offset':.12,'amplitude':.88,'goal_gains':[6,6,6,6,3,3]},
            normalizer={'kind':'fixed_lif_rest/2.0','mean':agent.brain.normalizer.mean.tolist(),
                        'std':agent.brain.normalizer.std.tolist(),'clip_abs':5.})
    if agent.profile=='balanced_rate_v3':contract['stop_min_s']=agent.stop_dwell_s
    if agent.feature_source!='reservoir':contract['feature_source']=agent.feature_source
    if agent.neural_backend!='csr':contract['neural_backend']=agent.neural_backend
    if agent.physics_profile!='bounded_level_body_surrogate':
        from ..physics.world import WorldConfig
        from dataclasses import asdict
        gains=json.loads((root/'configs/tellosim/world_room6.json').read_text()).get('low_level',{})
        code_root=Path(__file__).resolve().parents[3]
        contract['physics']={'profile':agent.physics_profile,'config':asdict(WorldConfig(controller_profile=agent.physics_profile,position_kp=float(gains.get('position_kp',.7)),velocity_kd=float(gains.get('velocity_kd',1.8)))),
            'source_hashes':{name:__import__('hashlib').sha256((code_root/name).read_bytes()).hexdigest() for name in ('flydrone/tellosim/physics/world.py','flydrone/tellosim/physics/rigid.py','flydrone/tellosim/visual.py','flydrone/tellosim/training/env.py')}}
    return contract


def save_checkpoint(path,agent,optimizer,root,updates,options,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'format':'tellosim.sdk9.reservoir/1.0','contract':checkpoint_contract(agent,root),
           'policy':agent.model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,
           'updates':updates,'options':options,'value_trained':agent.value_trained,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
               'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
           'mapping':{'input_indices':agent.brain.input_indices,'input_channels':agent.brain.input_channels,
               'readout_indices':agent.brain.readout_indices,'input_ids':agent.brain.input_ids,'readout_ids':agent.brain.readout_ids},
           'restore_mode':'inference_or_explicit_warm_start_only_no_mid_episode_resume',
           'extra':extra or {}}
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def load_checkpoint(path,agent,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.sdk9.reservoir/1.0':raise ValueError('incompatible checkpoint format')
    if state['contract']!=checkpoint_contract(agent,root):raise ValueError('checkpoint contract/graph/mapping/world mismatch')
    agent.model.load_state_dict(state['policy']);agent.reset()
    agent.value_trained=state.get('value_trained',state.get('updates',0)>0)
    if 'method' in state.get('extra',{}):agent.training_method=state['extra']['method']
    import hashlib
    agent.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return state


def smdp_gae(rewards,values,next_values,gammas,terminals,truncated,lam=.95):
    advantage=np.zeros(len(rewards),np.float32);running=0.
    for i in range(len(rewards)-1,-1,-1):
        delta=rewards[i]+gammas[i]*(not terminals[i])*next_values[i]-values[i]
        running=delta+gammas[i]*lam*(not (terminals[i] or truncated[i]))*running
        advantage[i]=running
    return advantage,advantage+np.asarray(values,np.float32)


def update(agent,optimizer,rows,epochs=4):
    features=torch.as_tensor(np.stack([r['features'] for r in rows]),dtype=torch.float32)
    masks=np.stack([r['mask'] for r in rows]);actions=torch.tensor([r['action'] for r in rows])
    old=torch.tensor([r['logp'] for r in rows]);values=np.asarray([r['value'] for r in rows])
    advantage,returns=smdp_gae([r['reward'] for r in rows],values,[r['next_value'] for r in rows],
        [r['Gamma'] for r in rows],[r['terminal'] for r in rows],[r['truncated'] for r in rows])
    a=torch.tensor(advantage);a=(a-a.mean())/(a.std(unbiased=False)+1e-8);ret=torch.tensor(returns)
    metrics=[]
    for _ in range(epochs):
        for indices in torch.randperm(len(rows)).split(32):
            policy,value=distribution(agent.model,features[indices],masks[indices])
            logp=policy.log_prob(actions[indices]);ratio=(logp-old[indices]).exp()
            pi=-torch.minimum(ratio*a[indices],ratio.clamp(.8,1.2)*a[indices]).mean()
            vf=.5*(value-ret[indices]).square().mean();entropy=policy.entropy().mean()
            loss=pi+.5*vf-.01*entropy
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
            optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(agent.model.parameters(),.5);optimizer.step();agent.value_trained=True
            kl=((ratio-1)-(logp-old[indices])).mean().item()
            metrics.append({'policy_loss':pi.item(),'value_loss':vf.item(),'entropy':entropy.item(),
                            'approx_kl':kl,'gradient_norm':float(grad)})
            if kl>.03:return {k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    return {k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
