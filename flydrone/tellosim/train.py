from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import math

from ..policy import ActorCritic
from ..ppo import compute_gae, ppo_update
from .physics import SimWorld, WorldConfig
from .sim import SimTello
from .env import TelloSimEnv, VariableDurationConfig
from ..brain import FrozenReservoir


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


def brain_train(out: str | Path, graph: str | Path, total_steps: int = 256,
                seed: int = 11, readout_neurons: int = 64, device: str = "cpu") -> dict:
    """Train a FlyBrain reservoir policy on the TelloSim telemetry contract."""
    torch.manual_seed(seed); np.random.seed(seed)
    world = SimWorld(WorldConfig()); sim = SimTello(world)
    env = TelloSimEnv(sim, VariableDurationConfig(goal_m=(1.5, 1.0, 1.0), start_m=(-1.5, -1.0, 1.0), max_episode_steps=40))
    brain = FrozenReservoir(str(graph), batch=1, device=device, readout_neurons=readout_neurons)
    model = ActorCritic(brain.feature_dim, action_dim=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed)
    brain.reset(); brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
    rows = []
    feature_rows = []; action_rows = []; logp_rows = []; value_rows = []
    reward_rows = []; next_value_rows = []; term_rows = []; trunc_rows = []
    while len(rows) < total_steps:
        features = torch.as_tensor(brain.current_features(), dtype=torch.float32)
        action, logp, value, _ = model.act_with_probs(features)
        next_obs, reward, term, trunc, info = env.step(int(action.item()))
        brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
        with torch.no_grad():
            _, next_value = model(torch.as_tensor(brain.current_features(), dtype=torch.float32))
        feature_rows.append(features[0].detach()); action_rows.append(action[0].detach()); logp_rows.append(logp[0].detach()); value_rows.append(value[0].detach())
        reward_rows.append(float(reward)); next_value_rows.append(next_value[0]); term_rows.append(bool(term)); trunc_rows.append(bool(trunc))
        rows.append({"step": len(rows), "reward": float(reward), "distance_m": info["distance_m"], "action": int(action.item()), "command": info["command"], "sim_tick": info["sim_tick"]})
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset(seed=seed + len(rows)); brain.reset(); brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
    advantages, returns = compute_gae(np.asarray(reward_rows), np.asarray([v.detach().item() for v in value_rows], dtype=np.float32),
                                      np.asarray([v.detach().item() for v in next_value_rows], dtype=np.float32),
                                      np.asarray(term_rows), np.asarray(trunc_rows))
    metrics = ppo_update(model, optimizer, {"features": torch.stack(feature_rows), "actions": torch.stack(action_rows),
        "old_log_probs": torch.stack(logp_rows), "advantages": torch.as_tensor(advantages), "returns": torch.as_tensor(returns)},
        epochs=2, minibatch=min(64, len(feature_rows)))
    result = {"schema_version": "tellosim.brain_train/1.0", "sim_only": True, "real_flight_authorized": False,
              "graph_sha256": brain.graph_sha256, "mapping_sha256": brain.mapping_sha256, "feature_dim": brain.feature_dim,
              "action_dim": 9, "total_env_steps": len(rows), "seed": seed, "metrics": rows, "ppo_update": metrics}
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"format_version": 1, "observation_dim": 26, "action_dim": 9, "feature_dim": brain.feature_dim,
                "seed": seed, "sim_only": True, "real_flight_authorized": False, "mode": "tellosim_brain",
                "graph_sha256": brain.graph_sha256, "mapping_sha256": brain.mapping_sha256,
                "policy_state_dict": model.state_dict()}, dest / "checkpoint.pt")
    return result


def brain_imitation_train(out: str | Path, graph: str | Path, episodes: int = 16,
                          seed: int = 11, readout_neurons: int = 64, device: str = "cpu") -> dict:
    """Train a reservoir policy from the documented TelloSim reference controller."""
    torch.manual_seed(seed); np.random.seed(seed)
    feature_rows, action_rows, episode_rows = [], [], []
    for episode in range(episodes):
        world = SimWorld(WorldConfig()); sim = SimTello(world)
        if episode == 0:
            start, goal = (-1.5, -1.0, 1.2), (1.5, 1.0, 1.2)
        else:
            rng = np.random.default_rng(seed + episode)
            start = (float(rng.uniform(-1.8, -1.0)), float(rng.uniform(-1.5, -0.5)), 1.2)
            goal = (float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.5, 1.5)), 1.2)
        env = TelloSimEnv(sim, VariableDurationConfig(goal_m=goal, start_m=start, max_episode_steps=60, stable_hold_s=1.0, target_radius_m=0.3))
        brain = FrozenReservoir(str(graph), batch=1, device=device, readout_neurons=readout_neurons)
        obs, _ = env.reset(seed=seed + episode)
        brain.reset(); brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
        rows = []
        for step in range(env.config.max_episode_steps):
            features = torch.as_tensor(brain.current_features()[0], dtype=torch.float32)
            position = np.asarray(env.pose.snapshot().position_m); delta = env.goal - position
            desired = math.atan2(float(delta[1]), float(delta[0])); error = math.atan2(math.sin(desired - sim.world.yaw_rad), math.cos(desired - sim.world.yaw_rad))
            if abs(float(delta[2])) > 0.12: teacher_action = 5 if delta[2] > 0 else 6
            elif abs(error) > 0.20: teacher_action = 7 if error < 0 else 8
            else: teacher_action = 1
            next_obs, reward, term, trunc, info = env.step(teacher_action)
            feature_rows.append(features); action_rows.append(teacher_action)
            rows.append({"step": step, "action": teacher_action, "distance_m": info["distance_m"], "reward": float(reward)})
            brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
            obs = next_obs
            if term or trunc: break
        episode_rows.append({"episode": episode, "start": start, "goal": goal, "steps": len(rows), "success": bool(rows and rows[-1]["distance_m"] <= 0.3), "rows": rows})
    model = ActorCritic(readout_neurons * 2, action_dim=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.stack(feature_rows); y = torch.as_tensor(action_rows, dtype=torch.long)
    for _ in range(200):
        policy, _ = model(x); loss = torch.nn.functional.cross_entropy(policy.logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = {"schema_version": "tellosim.brain_imitation/1.0", "sim_only": True, "real_flight_authorized": False,
              "graph_sha256": brain.graph_sha256, "mapping_sha256": brain.mapping_sha256, "feature_dim": model.feature_dim,
              "action_dim": 9, "seed": seed, "episodes": episodes, "samples": len(action_rows),
              "teacher_successes": sum(int(r["success"]) for r in episode_rows), "training_loss": float(loss.item())}
    dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps({**result, "episodes": episode_rows}, indent=2), encoding="utf-8")
    torch.save({"format_version": 1, "observation_dim": 26, "action_dim": 9, "feature_dim": model.feature_dim,
                "seed": seed, "sim_only": True, "real_flight_authorized": False, "mode": "tellosim_brain_imitation",
                "graph_sha256": brain.graph_sha256, "mapping_sha256": brain.mapping_sha256,
                "policy_state_dict": model.state_dict()}, dest / "checkpoint.pt")
    return result
