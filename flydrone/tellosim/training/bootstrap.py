"""Explicit rule-demonstration initialization and DAgger on neural features.

No raw-observation policy bypass and no PPO improvement claim. The expert uses
only the same measured goal error and mask available through the sensor contract.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .runtime import ReservoirAgent,distribution,save_checkpoint,load_checkpoint
from .runner import evaluate,parameter_hash
from .env import TrainingEnv,curriculum_case
from .contracts import digest
from ..visual import atomic_json


def teacher(env):
    if env.steps==0 or np.linalg.norm(env.observation[:2]*6)<=.17:return 0
    action=1 if env.observation[0]>0 else 2
    return action if env.mask[action] else 0


def collect(root,agent,seeds,rows,execute_teacher,record_prefix=None):
    outcomes=[]
    for index,seed in enumerate(seeds):
        env=TrainingEnv(root,curriculum_case(seed,curriculum='C0-near-x'),agent,
            record=record_prefix is not None and index==0,run_id=f'{record_prefix}-{index}' if record_prefix else None,
            policy_source='rule_demonstration_sdk9' if execute_teacher else 'dagger_learner_sdk9')
        try:
            while not env.terminated:
                label=teacher(env)
                rows.append((env.features.copy(),env.mask.copy(),label))
                if execute_teacher:action=label;policy=None
                else:action,_,_,policy=agent.decision(env.features,env.mask,True)
                env.step(action,policy)
            outcomes.append(env.reason)
        finally:env.close()
    return outcomes


def fit(agent,optimizer,rows,epochs=60):
    features=torch.as_tensor(np.stack([r[0] for r in rows]),dtype=torch.float32)
    masks=np.stack([r[1] for r in rows]);labels=torch.tensor([r[2] for r in rows])
    counts=torch.bincount(labels,minlength=9).clamp_min(1).float()
    weights=len(labels)/(3*counts)
    before=agent.brain.advance_count;steps=0
    for _ in range(epochs):
        for indices in torch.randperm(len(rows)).split(128):
            policy,_=distribution(agent.model,features[indices],masks[indices])
            loss=-(policy.log_prob(labels[indices])*weights[labels[indices]]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite imitation loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(agent.model.parameters(),1.);optimizer.step();steps+=1
    assert agent.brain.advance_count==before
    with torch.no_grad():
        policy,_=distribution(agent.model,features,masks)
        accuracy=float((policy.probs.argmax(1)==labels).float().mean())
    return {'gradient_steps':steps,'training_examples':len(rows),'training_label_accuracy':accuracy}


def run(root,graph,out,seed):
    root=Path(root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4);started=time.monotonic()
    agent=ReservoirAgent(graph,'cuda',seed,profile='balanced_rate_v3')
    original=[p.detach().clone() for p in agent.model.parameters()];initial_hash=parameter_hash(agent)
    validation=[curriculum_case(910000+i,f'validation-{i:03d}','C0-near-x') for i in range(100)]
    train_seeds=list(range(300000+seed*1000,300000+seed*1000+64))
    splits={'curriculum':'C0-near-x','training_seeds':train_seeds,'validation':validation,'sealed_test_opened':False}
    atomic_json(out/'cases.json',splits)
    optimizer=torch.optim.Adam(agent.model.parameters(),lr=1e-3)
    save_checkpoint(out/'initial.pt',agent,optimizer,root,0,0,{'method':'untrained'})
    load_checkpoint(out/'initial.pt',agent,root)
    baseline=evaluate(root,agent,validation[:8],'untrained',out.name)
    agent.training_method='Rule demonstrations + 2 DAgger rounds; zero PPO updates'
    rows=[];stages=[];total_steps=0
    for stage in range(3):
        chosen=train_seeds[:32] if stage==0 else train_seeds[32+(stage-1)*16:32+stage*16]
        outcomes=collect(root,agent,chosen,rows,stage==0,out.name+'-teacher' if stage==0 else None)
        stats=fit(agent,optimizer,rows);total_steps+=stats['gradient_steps']
        stages.append({'stage':stage,'source':'expert' if stage==0 else 'learner_with_expert_labels',
            'collection_successes':outcomes.count('success'),'collection_episodes':len(outcomes),**stats})
        atomic_json(out/'progress.json',{'stages':stages})
        print(json.dumps({'seed':seed,**stages[-1]}),flush=True)
    x=np.stack([r[0] for r in rows]);m=np.stack([r[1] for r in rows]);y=np.asarray([r[2] for r in rows])
    np.savez_compressed(out/'demonstrations.npz',features=x,masks=m,teacher_actions=y)
    extra={'method':agent.training_method,'seed':seed,'curriculum':'C0-near-x',
           'case_hash':digest(splits),'teacher':'first STOP, then measured goal error <=0.17 m STOP, else forward/back',
           'demonstrations_sha256':hashlib.sha256((out/'demonstrations.npz').read_bytes()).hexdigest()}
    save_checkpoint(out/'checkpoint.pt',agent,optimizer,root,total_steps,len(rows),extra)
    expected=[agent.decision(a,b,True)[3] for a,b in zip(x[:16],m[:16])]
    for p in agent.model.parameters():p.data.zero_()
    load_checkpoint(out/'checkpoint.pt',agent,root)
    assert expected==[agent.decision(a,b,True)[3] for a,b in zip(x[:16],m[:16])]
    trained=evaluate(root,agent,validation[:8],'supervised-bootstrap',out.name)
    ablated=evaluate(root,agent,validation[:8],'brain-zero',out.name,record_first=False,ablate=True)
    delta=float(torch.sqrt(sum((a-b.detach()).square().sum() for a,b in zip(original,agent.model.parameters()))))
    summary={'schema':'sdk9.supervised_bootstrap/1.0','status':'completed','seed':seed,
        'profile':agent.profile,'feature_source':'reservoir','curriculum':'C0-near-x',
        'training_method':agent.training_method,'ppo_updates':0,'options':len(rows),'updates':total_steps,
        'stages':stages,'parameter_delta_l2':delta,'initial_parameter_hash':initial_hash,
        'checkpoint_roundtrip_exact':True,'checkpoint_sha256':agent.checkpoint_sha256,
        'baselines':[baseline],'trained':trained,'brain_zero_ablation':ablated,
        'case_hash':digest(splits),'graph_sha256':agent.brain.graph_sha256,
        'mapping_sha256':agent.brain.mapping_sha256,'elapsed_s':time.monotonic()-started,
        'MODEL_READY_FOR_NEXT_STAGE':False,'TS1_C0_TASK_LEARNED':False,
        'scope':'Supervised near-axis initialization only. Not PPO-only, full C0, 3x300 acceptance or real flight.',
        'artifact_directory':str(out.relative_to(root))}
    atomic_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37]);p.add_argument('--tag',default='20260926')
    p.add_argument('--wait-for',type=Path);a=p.parse_args()
    if a.wait_for:
        deadline=time.monotonic()+1800
        while not a.wait_for.exists():
            if time.monotonic()>deadline:raise RuntimeError('prerequisite did not complete')
            time.sleep(5)
    results=[run(Path('.'),a.graph,Path('runs/tellosim-sdk9')/f'nearx-bootstrap-s{s}-{a.tag}',s) for s in a.seeds]
    report={'schema':'sdk9.bootstrap_comparison/1.0','method':'Rule demonstrations + DAgger, no PPO',
        'results':results,'MODEL_READY_FOR_NEXT_STAGE':False}
    atomic_json(Path('reports/sdk9_learning/bootstrap-comparison.json'),report)
    # Fixed seed 11 is the displayed representative, never best-case selection.
    atomic_json(Path('reports/vis/tellosim/training-summary.json'),results[0])
    print(json.dumps({'bootstrap':[{'seed':r['seed'],'successes':r['trained']['successes'],
        'zero_successes':r['brain_zero_ablation']['successes']} for r in results]}),flush=True)

if __name__=='__main__':main()
