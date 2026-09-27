"""Fit only the value head from complete on-policy training episodes."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import torch
from .env import TrainingEnv, curriculum_case
from ..visual import atomic_json


def actor_hash(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
        for name,p in model.named_parameters() if not name.startswith("critic."))).hexdigest()


def discounted_returns(rewards, gammas):
    result=np.empty(len(rewards),np.float32); tail=0.
    for i in reversed(range(len(rewards))):
        tail=float(rewards[i])+float(gammas[i])*tail
        result[i]=tail
    return result


def fit_value_head(agent, features, targets, epochs=100):
    x=torch.as_tensor(np.asarray(features),dtype=torch.float32)
    y=torch.as_tensor(np.asarray(targets),dtype=torch.float32)
    if len(x)==0 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("finite nonempty critic examples required")
    before=actor_hash(agent.model); advances=agent.brain.advance_count
    # Detached hidden states ensure value fitting cannot alter learned actions.
    with torch.no_grad(): hidden=agent.model.body(x)
    optimizer=torch.optim.Adam(agent.model.critic.parameters(),lr=1e-3)
    def mse():
        with torch.no_grad():return float((agent.model.critic(hidden).squeeze(-1)-y).square().mean())
    initial=mse(); steps=0
    for _ in range(epochs):
        for indices in torch.randperm(len(x)).split(64):
            prediction=agent.model.critic(hidden[indices]).squeeze(-1)
            loss=(prediction-y[indices]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite value loss")
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.critic.parameters(),.5)
            optimizer.step();steps+=1
    assert before==actor_hash(agent.model), "critic warmup changed actor"
    assert advances==agent.brain.advance_count, "critic fitting advanced reservoir"
    agent.value_trained=True
    return {"examples":len(x),"gradient_steps":steps,"epochs":epochs,
        "training_mse_before":initial,"training_mse_after":mse(),
        "actor_exactly_unchanged":True,"actor_sha256":before,
        "target":"complete-episode discounted on-policy returns; no validation data"}


def warmup(root,agent,out,seeds,curriculum):
    features=[];targets=[];outcomes=[]
    for seed in seeds:
        env=TrainingEnv(root,curriculum_case(seed,curriculum=curriculum),agent)
        episode_features=[];rewards=[];gammas=[]
        try:
            while not env.terminated:
                episode_features.append(env.features.copy())
                action,_,_,policy=agent.decision(env.features,env.mask)
                row=env.step(action,policy)
                rewards.append(row["reward"]);gammas.append(row["Gamma"])
            features.extend(episode_features)
            targets.extend(discounted_returns(rewards,gammas))
            outcomes.append({"seed":seed,"reason":env.reason,"steps":env.steps})
        finally:env.close()
        print(json.dumps({"phase":"critic_collect","episodes":len(outcomes),"target":len(seeds)}),flush=True)
    path=out/"critic-data.npz"
    np.savez_compressed(path,features=np.asarray(features),returns=np.asarray(targets))
    metrics=fit_value_head(agent,features,targets)
    metrics.update(episodes=outcomes,data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic_json(out/"critic-warmup.json",metrics)
    print(json.dumps({"phase":"critic_fit",**{k:v for k,v in metrics.items() if k!="episodes"}}),flush=True)
    return metrics
