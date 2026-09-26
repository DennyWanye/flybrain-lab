from __future__ import annotations
import hashlib, json, math, shutil
from pathlib import Path
import numpy as np
import torch
from ..policy import ActorCritic
from .physics import SimWorld, WorldConfig
from .sim import SimTello
from .env import TelloSimEnv, VariableDurationConfig


def sha256(path):
 h=hashlib.sha256();
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()


def record_golden(out: str|Path, seed=11, checkpoint: str|Path|None = None):
 out=Path(out); out.mkdir(parents=True,exist_ok=True)
 world=SimWorld(WorldConfig()); sim=SimTello(world)
 env=TelloSimEnv(sim,VariableDurationConfig(goal_m=(0.8,0.0,1.0),action_duration_s=(0.5,1.0,2.0),max_episode_steps=120))
 obs,info=env.reset(seed=seed); rows=[]; total=0.0; hold=0.0; success=False
 model=ActorCritic(26, action_dim=9)
 payload=torch.load(checkpoint, map_location="cpu", weights_only=False)
 model.load_state_dict(payload["policy_state_dict"])
 model.eval()
 for step in range(120):
  with torch.no_grad(): action=int(model.act(torch.as_tensor(obs[None],dtype=torch.float32), deterministic=True)[0].item())
  before=env.pose.snapshot(); next_obs,reward,term,trunc,meta=env.step(action); after=env.pose.snapshot()
  dist=meta["distance_m"]; speed=float(np.linalg.norm(after.velocity_mps)); total+=reward
  hold=hold+meta["duration_s"] if dist<=0.30 and speed<=0.15 else 0.0
  success=hold>=1.0
  rows.append({"run_id":"golden-script-eval","episode_id":"golden-0001","scenario_id":"room6_c0_fixed_target","seed":seed,"decision_step":step,"sim_time_s":after.sim_tick*world.config.dt,"observation":next_obs.tolist(),"action":action,"command":meta["command"],"command_args":meta["command_args"],"position_xyz_m":list(after.position_m),"velocity_xyz_mps":list(after.velocity_mps),"yaw_rad":after.yaw_rad,"target_xyz_m":list(env.goal),"distance_to_target_m":dist,"reward_total":reward,"reward_components":{"progress_reward":reward,"time_penalty":-0.01*meta["duration_s"]},"stable_hold_s":hold,"collision":False,"out_of_bounds":bool(np.max(np.abs(np.asarray(after.position_m)[:2]))>world.config.room_half_extent_m),"neural_capture":"not_recorded"})
  if success or trunc: break
 replay=out/"replay.jsonl"; replay.write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n")
 ck=Path(checkpoint) if checkpoint else None;
 if ck is None or not ck.is_file(): raise FileNotFoundError("a real policy checkpoint is required")
 manifest={"schema_version":"golden_episode/1.0","run_id":"golden-script-eval","episode_id":"golden-0001","scenario_id":"room6_c0_fixed_target","seed":seed,"policy_checkpoint_sha256":sha256(ck),"graph_sha256":None,"brain_mapping_sha256":None,"success":success,"episode_return":total,"sim_duration_s":rows[-1]["sim_time_s"],"final_distance_m":rows[-1]["distance_to_target_m"],"stable_hold_s":hold,"collision":False,"out_of_bounds":any(x["out_of_bounds"] for x in rows),"replay_sha256":sha256(replay),"policy_source":"checkpoint_deterministic_policy","neural_capture":"not_recorded"}
 shutil.copy2(ck, out/"checkpoint.pt")
 (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 (out/"checksums.sha256").write_text(manifest["replay_sha256"]+"  replay.jsonl\n"+manifest["policy_checkpoint_sha256"]+"  checkpoint.pt\n")
 return manifest
