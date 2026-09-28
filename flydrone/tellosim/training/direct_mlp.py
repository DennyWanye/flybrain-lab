"""T44 ordinary MLP pilot: no connectome, no fake neural state or fly claim."""
import argparse,json,time,copy,signal
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ...policy import ActorCritic
from .runtime import distribution
from .heading_campaign import RuleDiagnostic
from .altitude_refined import AltitudeEnv,ALTITUDE_SPEC,DISTURBANCE_SPEC,altitude_case,with_profile,PROFILES,teacher
from .altitude_refined_campaign import training_case,result_row,grouped,evaluate as fly_evaluate
from .stability_campaign import fit
from .contracts import digest
from .joint import sha
from .joint_campaign import frozen_json
from .c0_campaign import summarize
from ..visual import atomic_json
STAGES=(768,1024,1024);SEED=11
METHOD='direct MLP26->128->128->9, raw measured observation only; no MaleCNS or reservoir; single-seed engineering pilot'
class DirectAgent(RuleDiagnostic):
    feature_source='raw_observation_direct_mlp';training_method=METHOD;value_trained=False
    def __init__(self,model,seed):
        self.model=model;self.action_rng=torch.Generator().manual_seed(seed);super().__init__()
    def observe(self,vector,sample_id):
        if sample_id<=self.sample_id:raise ValueError('stale direct-MLP sample')
        # Legacy Env clock compatibility only: no neural substeps are executed.
        self.brain_tick+=4;self.sample_id=sample_id
        x=np.asarray(vector,np.float32).copy();x[:3]*=[6,6,3];return x
    def snapshot(self):return {'status':'not_applicable_direct_mlp','neurons':[],'neural_updates':0}
    @torch.no_grad()
    def decision(self,features,mask,deterministic=True):
        pi,value=distribution(self.model,torch.tensor(features[None]),[mask])
        action=pi.probs.argmax(-1) if deterministic else torch.multinomial(pi.probs,1,generator=self.action_rng).squeeze(-1)
        return int(action.item()),{'selected_action':int(action.item()),'probabilities':pi.probs[0].tolist(),'input_source':self.feature_source,'value_trained':False,'feature_dim':26,'neural_updates':0}
def directory(root):return Path(root)/'reports/ts1_mlp_pilot'
def run_directory(root,smoke=False):return Path(root)/f'runs/tellosim-sdk9/direct-mlp-{"smoke-" if smoke else ""}s11'
def prepare(root):
    root=Path(root);out=directory(root);out.mkdir(exist_ok=True)
    cases=[with_profile(altitude_case(270000000+i,f'mlp-paired-{i}',direction=1 if i//4%2==0 else -1),tuple(PROFILES)[i%4]) for i in range(60)]
    protocol=dict(method=METHOD,seed=SEED,task=ALTITUDE_SPEC,disturbance=DISTURBANCE_SPEC,stages=list(STAGES),epochs=80,learning_rate=.0003,
        training_seed_formula='280000000+stage*10000+lane*500+episode; smoke290000000',development='271000000..271000007',paired_case_hash=digest(cases),
        comparison='fixed V1R seed11 chosen before evaluation; same60 cases, observations, SDK9 masks, physics and task limits; separate training datasets; no claim of connectome superiority or matched compute cost',
        model_selection='one fixed final MLP; held-out scores never used for selection',neural_graph_used=False,neural_updates=0,
        clock_note='inherited Env brain_tick is a compatibility scheduler count, never neural work; exported MLP results rename it')
    frozen_json(out/'protocol.json',protocol);frozen_json(out/'cases.json',cases);return cases

def save(path,model,options,root):
    torch.save({'format':'tellosim.direct_mlp_readout/1','model':model.state_dict(),'options':options,'neural_graph_used':False,'source_sha256':sha(Path(root)/'flydrone/tellosim/training/direct_mlp.py')},path)

def train(root,smoke=False):
    root=Path(root);prepare(root);out=run_directory(root,smoke)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(SEED)
    model=ActorCritic(26,9);initial=copy.deepcopy(model.state_dict());save(out/'initial.pt',model,0,root)
    opt=torch.optim.AdamW(list(model.body.parameters())+list(model.actor.parameters()),lr=.0003,weight_decay=.001)
    rows=[];provenance=[];options=0;stages=[];started=time.monotonic();envs=[]
    def stop(signum,frame):raise KeyboardInterrupt('scoped MLP interrupted')
    old=signal.signal(signal.SIGTERM,stop)
    def annotate(env,kind,stage):
        label=teacher(env.observation,env.mask,env.previous_action is None);error=abs(float(env.observation[2])*3)
        weight=(1 if kind=='decision' else .25)*(4 if .06<=error<=.14 and env.previous_action is not None else 1)
        rows.append((env.features.copy(),env.mask.copy(),label,weight));provenance.append(dict(seed=env.case['seed'],stage=stage,kind=kind,label=label,weight=weight));return label
    try:
        for stage,count in enumerate((64,64) if smoke else STAGES):
            envs=[None]*16;episodes=[0]*16;agents=[DirectAgent(model,SEED+1000003*i) for i in range(16)]
            for option in range(count):
                if time.monotonic()-started>3600:raise TimeoutError('MLP pilot budget')
                i=option%16
                if envs[i] is None:
                    seed=(290000000 if smoke else 280000000)+stage*10000+i*500+episodes[i];episodes[i]+=1
                    envs[i]=AltitudeEnv(root,training_case(seed,i),agents[i])
                env=envs[i];label=annotate(env,'decision',stage)
                action,policy=(label,None) if stage==0 else agents[i].decision(env.features,env.mask,deterministic=i%2==0)
                iterator=env.step_iter(action,policy)
                for _ in iterator:
                    env.features=agents[i].observe(env.observation,env.device.latest_observation().sample_id)
                    if agents[i].sample_id%2==0 and env.previous_duration>=.5 and (env.previous_action==0 or env.device.observed_operation()['client']!='sent'):annotate(env,'settled',stage)
                options+=1
                if env.terminated:env.close();envs[i]=None
            for env in envs:
                if env:env.close('stage_complete')
            envs=[];metrics=fit(model,opt,rows,2 if smoke else 80);stages.append(dict(stage=stage,options=count,**metrics))
            save(out/f'stage-{stage}.pt',model,options,root);atomic_json(out/'progress.json',dict(options=options,stages=stages));print(json.dumps(dict(options=options,**metrics)),flush=True)
        save(out/'checkpoint.pt',model,options,root)
        with torch.no_grad():before=model(torch.tensor(np.stack([r[0] for r in rows[-16:]])))[0].probs.clone()
        state=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu');model.load_state_dict(state['model'])
        with torch.no_grad():after=model(torch.tensor(np.stack([r[0] for r in rows[-16:]])))[0].probs
        assert torch.equal(before,after)
        delta=float(torch.sqrt(sum((initial[k]-v).square().sum() for k,v in model.state_dict().items())));assert delta>0
        atomic_json(out/'training.json',dict(status='completed',smoke_only=smoke,options=options,examples=len(rows),parameter_delta_l2=delta,roundtrip_exact=True,method=METHOD,
            checkpoint_sha256=sha(out/'checkpoint.pt'),elapsed_s=time.monotonic()-started,neural_graph_used=False,neural_updates=0,stages=stages))
    except BaseException as exc:atomic_json(out/'failure.json',dict(error=repr(exc),options=options));raise
    finally:
        for env in envs:
            if env:env.close('interrupted')
        atomic_json(out/'provenance.json',provenance);signal.signal(signal.SIGTERM,old)
        if rows:np.savez_compressed(out/'demonstrations.npz',features=np.stack([r[0] for r in rows]),labels=[r[2] for r in rows],masks=np.stack([r[1] for r in rows]),weights=[r[3] for r in rows])

def evaluate_mlp(root,cases,out,smoke=False):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    path=run_directory(root,smoke)/'checkpoint.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['format']=='tellosim.direct_mlp_readout/1' and not state['neural_graph_used'] and state['source_sha256']==sha(Path(root)/'flydrone/tellosim/training/direct_mlp.py')
    model=ActorCritic(26,9);model.load_state_dict(state['model']);agent=DirectAgent(model,SEED);rows=[]
    for case in cases:
        env=AltitudeEnv(root,case,agent);actions=[]
        try:
            while not env.terminated:
                action,policy=agent.decision(env.features,env.mask);actions.append(action);env.step(action,policy)
            r=result_row(env,case,actions,None);r['scheduler_substep_count']=r.pop('brain_tick');r['neural_updates']=0;rows.append(r)
        finally:env.close()
    result={**summarize(rows,'direct_mlp_pilot'),'profiles':grouped(rows),'case_hash':digest(cases),'checkpoint_sha256':sha(path),'neural_graph_used':False,'method':METHOD}
    atomic_json(out,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','train','evaluate']);a=p.parse_args();root=Path.cwd();out=directory(root)
    if a.command=='prepare':prepare(root)
    elif a.command in ['smoke','train']:train(root,a.command=='smoke')
    else:
        cases=prepare(root);training=json.loads((run_directory(root)/'training.json').read_text());assert training['status']=='completed' and not training['smoke_only'] and training['options']==sum(STAGES)
        frozen_json(out/'frozen.json',{'mlp':training['checkpoint_sha256'],'fly':sha(root/'runs/tellosim-sdk9/altitude-refined-s11/checkpoint.pt'),'case_hash':digest(cases)})
        mlp=evaluate_mlp(root,cases,out/'mlp-paired.json')
        fly=fly_evaluate(root,root/'runs/tellosim-sdk9/altitude-refined-s11/checkpoint.pt',cases,out/'malecns-paired.json','mlp-pilot-malecns-control',batch=60,record_indices=())
        summary={'status':'pilot_completed','software_acceptance_T44':True,'mlp_successes':mlp['successes'],'malecns_successes':fly['successes'],'episodes_each':60,'seed':11,'case_hash':digest(cases),'CONNECTOME_ADVANTAGE_PROVEN':False,'notes':'different training data; same observations/actions/physics/paired evaluation. One-seed pilot, not a superiority study.'}
        atomic_json(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
