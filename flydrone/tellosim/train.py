from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from ..policy import ActorCritic
from ..ppo import compute_gae, ppo_update
from .physics import SimWorld, WorldConfig
from .sim import SimTello
from .env import TelloSimEnv


def smoke_train(out: str | Path, total_steps: int = 64, seed: int = 11) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    world = SimWorld(WorldConfig())
    sim = SimTello(world)
    env = TelloSimEnv(sim)
    model = ActorCritic(26, action_dim=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    rows = []
    for update in range(max(1, total_steps // 16)):
        features=[]; actions=[]; old_log_probs=[]; values=[]; rewards=[]; next_values=[]; terms=[]; truncs=[]
        for _ in range(min(16, total_steps - len(rows))):
            ft=torch.as_tensor(obs[None],dtype=torch.float32)
            action, logp, value, _ = model.act_with_probs(ft)
            no, reward, term, trunc, info = env.step(int(action.item()))
            with torch.no_grad():
                _, nv = model(torch.as_tensor(no[None],dtype=torch.float32))
            features.append(ft[0]); actions.append(action[0]); old_log_probs.append(logp[0]); values.append(value[0]); rewards.append(reward); next_values.append(nv[0]); terms.append(term); truncs.append(trunc)
            obs = no
            rows.append({"duration_s": info["duration_s"], "reward": float(reward), "distance_m": info["distance_m"]})
            if term or trunc:
                obs, _ = env.reset(seed=seed + len(rows))
        if not features: break
        adv, ret = compute_gae(np.asarray(rewards), np.asarray([v.detach().item() for v in values], dtype=np.float32), np.asarray([v.detach().item() for v in next_values], dtype=np.float32), np.asarray(terms), np.asarray(truncs))
        metrics = ppo_update(model, optimizer, {"features":torch.stack(features),"actions":torch.stack(actions),"old_log_probs":torch.stack(old_log_probs),"advantages":torch.as_tensor(adv),"returns":torch.as_tensor(ret)}, epochs=1, minibatch=16)
        metrics["env_steps"] = len(rows); metrics["mean_reward"] = float(np.mean(rewards)); rows[-1]["update"] = metrics
    result={"schema_version":"tellosim.smoke_train/1.0","sim_only":True,"real_flight_authorized":False,"observation_dim":26,"action_dim":9,"total_env_steps":len(rows),"seed":seed,"metrics":rows}
    dest=Path(out); dest.mkdir(parents=True,exist_ok=True); (dest/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    torch.save({"format_version":1,"observation_dim":26,"action_dim":9,"seed":seed,"sim_only":True,"real_flight_authorized":False,"policy_state_dict":model.state_dict()},dest/"checkpoint.pt")
    return result
