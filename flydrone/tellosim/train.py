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


def smoke_train(out: str | Path, total_steps: int = 64, seed: int = 11, brain=None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    world = SimWorld(WorldConfig())
    sim = SimTello(world)
    env = TelloSimEnv(sim)
    model = ActorCritic(brain.feature_dim if brain is not None else 26, action_dim=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    rows = []
    for update in range(max(1, total_steps // 16)):
        features=[]; actions=[]; old_log_probs=[]; values=[]; rewards=[]; next_values=[]; terms=[]; truncs=[]
        for _ in range(min(16, total_steps - len(rows))):
            current = brain.current_features()[0] if brain is not None else obs; ft=torch.as_tensor(current[None],dtype=torch.float32)
            action, logp, value, _ = model.act_with_probs(ft)
            no, reward, term, trunc, info = env.step(int(action.item()))
            with torch.no_grad():
                _, nv = model(torch.as_tensor(no[None],dtype=torch.float32))
            features.append(ft[0]); actions.append(action[0]); old_log_probs.append(logp[0]); values.append(value[0]); rewards.append(reward); next_values.append(nv[0]); terms.append(term); truncs.append(trunc)
            obs = no
            rows.append({"step":len(rows),"duration_s":info["duration_s"],"reward":float(reward),"distance_m":info["distance_m"],"action":int(action.item()),"command":info["command"],"command_args":info["command_args"],"sim_tick":info["sim_tick"]})
            if term or trunc:
                obs, _ = env.reset(seed=seed + len(rows))
        if not features: break
        adv, ret = compute_gae(np.asarray(rewards), np.asarray([v.detach().item() for v in values], dtype=np.float32), np.asarray([v.detach().item() for v in next_values], dtype=np.float32), np.asarray(terms), np.asarray(truncs))
        metrics = ppo_update(model, optimizer, {"features":torch.stack(features),"actions":torch.stack(actions),"old_log_probs":torch.stack(old_log_probs),"advantages":torch.as_tensor(adv),"returns":torch.as_tensor(ret)}, epochs=1, minibatch=16)
        metrics["env_steps"] = len(rows); metrics["mean_reward"] = float(np.mean(rewards)); rows[-1]["update"] = metrics
    result={"schema_version":"tellosim.smoke_train/1.0","sim_only":True,"real_flight_authorized":False,"observation_dim":26,"action_dim":9,"feature_dim":int(model.feature_dim),"total_env_steps":len(rows),"seed":seed,"metrics":rows}
    dest=Path(out); dest.mkdir(parents=True,exist_ok=True); (dest/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    torch.save({"format_version":1,"observation_dim":26,"action_dim":9,"seed":seed,"sim_only":True,"real_flight_authorized":False,"policy_state_dict":model.state_dict()},dest/"checkpoint.pt")
    return result


def run_config(path: str | Path, out: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("sim_only") is not True or config.get("real_flight_authorized") is not False:
        raise ValueError("TelloSim config must remain simulation-only")
    result = smoke_train(out, int(config.get("total_env_steps", 64)), int(config.get("seed", 11)))
    result["config"] = str(path)
    Path(out, "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
