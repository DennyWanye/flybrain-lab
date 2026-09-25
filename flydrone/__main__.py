from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .brain import FrozenReservoir, graph_hash
from .contracts import ACTION_SCHEMA, FlightAction, FlightConfig, OBSERVATION_SCHEMA
from .encoder import FeatureNormalizer
from .env import FlyToTargetEnv
from .dynamics import local_waypoint
from .policy import ActorCritic
from .ppo import compute_gae, ppo_update
from .vis.live import create_live_writer, utc_now


def root_path(path: str, config_path: str | None = None) -> Path:
    p = Path(path)
    if p.is_absolute(): return p
    base = Path(config_path).resolve().parents[2] if config_path else Path.cwd()
    return base / p


def load_config(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    allowed = {"schema_version", "sim_only", "env", "reward", "brain", "normalizer", "train", "evaluation", "recording"}
    unknown = set(data) - allowed
    if unknown: raise ValueError(f"unknown config fields: {sorted(unknown)}")
    return data


def env_config(cfg, curriculum=None):
    values = dict(cfg["env"])
    if curriculum: values["curriculum"] = curriculum
    values.pop("id", None); return FlightConfig(**values)


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def make_cases(args):
    cfg = load_config(args.config); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env_cfg = env_config(cfg)
    rows = {}
    for name, count, namespace in [("validation", args.validation_count, 100000), ("test", args.test_count, 200000)]:
        path = out / f"{name}.jsonl"; rows[name] = []
        with path.open("w") as f:
            for i in range(count):
                seed = namespace + i; env = FlyToTargetEnv(env_cfg); _, info = env.reset(seed=seed)
                row = {"case_id": f"{name}-{i:04d}", "seed": seed, "curriculum": env_cfg.curriculum,
                       "initial_state": {"position_xy": env.state.position_xy.tolist(), "goal_xy": env.state.goal_xy.tolist()},
                       "env_config_hash": digest_json(env_cfg.to_dict())}
                f.write(json.dumps(row) + "\n"); rows[name].append(row)
    manifest = {"config_hash": digest_json(cfg), "counts": {k: len(v) for k,v in rows.items()},
                "files": {k: hashlib.sha256((out/f"{k}.jsonl").read_bytes()).hexdigest() for k in rows}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def make_normalizer(args):
    cfg = load_config(args.config)
    brain_cfg = cfg["brain"]; graph = root_path(args.graph or brain_cfg["graph"], args.config)
    env_cfg = env_config(cfg); envs = args.envs
    brain = FrozenReservoir(str(graph), envs, device=args.device, mapping_seed=brain_cfg["mapping_seed"], inputs_per_channel=brain_cfg["inputs_per_channel"], readout_neurons=brain_cfg["readout_neurons"], internal_steps=brain_cfg["internal_steps"], gain=brain_cfg["gain"], tonic=brain_cfg["tonic"], input_gain=brain_cfg["input_gain"])
    observations=[]; obs=[]; env_list=[]
    for i in range(envs):
        e=FlyToTargetEnv(env_cfg); o,_=e.reset(seed=500000+i); env_list.append(e); obs.append(o)
    for _ in range((args.calibration_steps + envs - 1)//envs):
        encoded=brain.encoder(np.asarray(obs)); brain.advance(encoded); observations.append(brain.current_features())
        for i,e in enumerate(env_list):
            o,_,term,_,_=e.step(int(i%5));
            if term: o,_=e.reset(seed=500000+i+len(observations))
            obs[i]=o
    features=np.concatenate(observations, axis=0)[:args.calibration_steps]
    normalizer=FeatureNormalizer(features.mean(0), features.std(0), cfg["normalizer"]["std_floor"], cfg["normalizer"]["clip_abs"])
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True); np.savez(out/"artifact.npz", mean=normalizer.mean, std=normalizer.std)
    (out/"metadata.json").write_text(json.dumps({"graph_sha256":brain.graph_sha256,"mapping_sha256":brain.mapping_sha256,"feature_dim":brain.feature_dim,"steps":len(features)}, indent=2))
    print(json.dumps({"graph_sha256":brain.graph_sha256,"mapping_sha256":brain.mapping_sha256,"feature_dim":brain.feature_dim,"steps":len(features),"mean_spike_fraction":float(np.mean(brain.s.cpu().numpy()))}, indent=2))


def load_normalizer(path, dim):
    if not path or not Path(path).exists(): return FeatureNormalizer.identity(dim)
    with np.load(path) as z: return FeatureNormalizer(z["mean"], z["std"])


def make_model(cfg, args, mode, envs):
    if mode == "direct": return None, ActorCritic(8), 8
    b=cfg["brain"]; graph=root_path(args.graph or b["graph"], args.config); normalizer=load_normalizer(args.normalizer, 2*b["readout_neurons"])
    brain=FrozenReservoir(str(graph), envs, device=args.device, mapping_seed=b["mapping_seed"], inputs_per_channel=b["inputs_per_channel"], readout_neurons=b["readout_neurons"], internal_steps=b["internal_steps"], gain=b["gain"], tonic=b["tonic"], input_gain=b["input_gain"], normalizer=normalizer)
    return brain, ActorCritic(brain.feature_dim), brain.feature_dim


def reset_batch(envs, env_cfg, seed):
    env_list=[]; obs=[]
    for i in range(envs):
        e=FlyToTargetEnv(env_cfg); o,_=e.reset(seed=seed+i); env_list.append(e); obs.append(o)
    return env_list, np.asarray(obs, np.float32)


def checkpoint_payload(model, brain, cfg, mode, seed, steps, optimizer=None):
    return {"format_version":1,"sim_only":True,"real_flight_authorized":False,"env_id":"FlyToTarget2D-v1","action_schema":ACTION_SCHEMA,"observation_schema":OBSERVATION_SCHEMA,"mode":mode,"seed":seed,"global_env_steps":steps,"policy_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict() if optimizer else None,"brain_contract":brain.contract() if brain else None,"graph_sha256":brain.graph_sha256 if brain else None,"mapping_sha256":brain.mapping_sha256 if brain else None,"feature_dim":model.feature_dim}


def train(args, resume=False):
    cfg=load_config(args.config); mode=args.mode or cfg["train"]["mode"]; seed=args.seed; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    envs=args.envs or cfg["train"]["envs"]; env_cfg=env_config(cfg,args.curriculum); brain,model,_=make_model(cfg,args,mode,envs); optimizer=torch.optim.Adam(model.parameters(),lr=cfg["train"]["learning_rate"])
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    live_writer = None
    if getattr(args, "telemetry_config", None):
        telemetry_path = Path(args.telemetry_config).resolve()
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if not telemetry.get("telemetry_enabled", False):
            raise ValueError("telemetry config must explicitly set telemetry_enabled=true")
        project_root = Path(args.config).resolve().parents[2]
        live_dir = Path(telemetry.get("live_dir", str(out / "live")))
        if not live_dir.is_absolute():
            live_dir = project_root / live_dir
        live_writer = create_live_writer(
            live_dir,
            run_id=f"train-seed-{seed}",
            brain_contract=brain.contract() if brain else None,
            config_hash=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            publish_hz=float(telemetry.get("live_publish_hz_max", 10)),
        )
        live_writer.publish_status({"status": "starting", "updated_at_utc": utc_now(), "last_seq": -1,
                                    "dropped_frames": 0, "trace_complete": False})
    if getattr(args, "init_from", None):
        init_payload=torch.load(args.init_from, map_location="cpu", weights_only=False)
        if init_payload.get("action_schema") != ACTION_SCHEMA or init_payload.get("mode") != mode or int(init_payload.get("feature_dim")) != model.feature_dim:
            raise ValueError("init checkpoint is incompatible with this training contract")
        if mode == "brain" and init_payload.get("graph_sha256") != brain.graph_sha256:
            raise ValueError("init checkpoint graph hash mismatch")
        model.load_state_dict(init_payload["policy_state_dict"])
    torch.save(checkpoint_payload(model,brain,cfg,mode,seed,0,optimizer),out/"initial.pt")
    total=0; history=[]; env_list,obs=reset_batch(envs,env_cfg,seed*1000)
    if brain: brain.reset(); brain.advance(brain.encoder(obs)); features=brain.current_features()
    else: features=obs.copy()
    while total < args.total_env_steps:
        T=min(cfg["train"]["rollout_steps"], max(1,(args.total_env_steps-total+envs-1)//envs)); buf={k:[] for k in ["features","actions","old_log_probs","values","rewards","next_values","terminated","truncated"]}
        for _ in range(T):
            ft=torch.as_tensor(features,dtype=torch.float32); action,logp,value,probabilities=model.act_with_probs(ft)
            before_positions=[e.state.position_xy.copy() for e in env_list]
            before_velocities=[e.state.velocity_xy.copy() for e in env_list]
            before_ticks=[e.state.physics_tick for e in env_list]
            next_obs=[]; rewards=[]; terms=[]; truncs=[]; infos=[]; next_vals=[]
            for i,e in enumerate(env_list):
                no,r,t,tr,info=e.step(int(action[i])); next_obs.append(no); rewards.append(r); terms.append(t); truncs.append(tr); infos.append(info)
            next_obs=np.asarray(next_obs,np.float32)
            if live_writer is not None:
                i = 0
                state = env_list[i].state
                neural = []
                if brain is not None:
                    selected = brain.outputs[:min(64, len(brain.outputs))].detach().cpu().numpy().tolist()
                    values_v = brain.v[selected, i].detach().cpu().numpy().tolist()
                    values_s = brain.s[selected, i].detach().cpu().numpy().tolist()
                    values_t = brain.trace[selected, i].detach().cpu().numpy().tolist()
                    neural = [{"neuron_id": str(brain.readout_ids[j]), "neuron_index": int(selected[j]),
                               "v_before_reset": float(values_v[j]), "spike": int(values_s[j]),
                               "trace_after_update": float(values_t[j])} for j in range(len(selected))]
                live_writer.publish({
                    "schema_version": "1.0.0", "source_kind": "simulation_live", "run_id": f"train-seed-{seed}",
                    "source_epoch": live_writer.source_epoch, "seq": live_writer._written_seq + live_writer.dropped + 1,
                    "emitted_at_utc": utc_now(), "kind": "transition",
                    "payload": {"env_index": i, "episode_id": f"env-{i}", "case_id": f"live-env-{i}",
                                "decision_step": int(before_ticks[i] // env_cfg.action_ticks), "policy_update": len(history),
                                "state_before": {"position_xy_m": before_positions[i].tolist(), "velocity_xy_mps": before_velocities[i].tolist(), "goal_xy_m": state.goal_xy.tolist()},
                                "observation": obs[i].tolist(), "encoded_observation": None,
                                "action_id": int(action[i]), "action_name": FlightAction(int(action[i])).name, "action_selection": "sample",
                                "action_probabilities": probabilities[i].detach().cpu().numpy().tolist(), "value_estimate": float(value[i].item()),
                                "waypoint_xy_m": local_waypoint(before_positions[i], FlightAction(int(action[i]))).tolist(), "state_after": {"position_xy_m": state.position_xy.tolist(), "velocity_xy_mps": state.velocity_xy.tolist(), "goal_xy_m": state.goal_xy.tolist()},
                                "reward_parts": infos[i]["reward_parts"], "reward": float(rewards[i]), "terminated": bool(terms[i]), "truncated": bool(truncs[i]),
                                "end_reason": infos[i]["end_reason"], "sim_tick_before": int(before_ticks[i]), "sim_tick_after": int(state.physics_tick),
                                "physics_path": None, "readout_snapshot": {"phase": "post_reset_final_substep", "neurons": neural} if neural else None}
                })
            if brain: brain.advance(brain.encoder(next_obs)); next_features=brain.current_features()
            else: next_features=next_obs.copy()
            with torch.no_grad(): _, nv=model(torch.as_tensor(next_features,dtype=torch.float32)); next_vals=nv.numpy()
            for i,t in enumerate(terms):
                if t:
                    no,_=env_list[i].reset(seed=seed*1000+total+i+1); next_obs[i]=no
            if brain and any(terms):
                brain.reset(np.asarray(terms, dtype=bool))
                brain.advance(brain.encoder(next_obs), mask=np.asarray(terms, dtype=bool))
                next_features = brain.current_features()
            buf["features"].append(features.copy()); buf["actions"].append(action.numpy()); buf["old_log_probs"].append(logp.detach().numpy()); buf["values"].append(value.detach().numpy()); buf["rewards"].append(np.asarray(rewards)); buf["next_values"].append(next_vals); buf["terminated"].append(np.asarray(terms)); buf["truncated"].append(np.asarray(truncs)); features=next_features; obs=next_obs; total += envs
        adv,ret=compute_gae(np.asarray(buf["rewards"]),np.asarray(buf["values"]),np.asarray(buf["next_values"]),np.asarray(buf["terminated"]),np.asarray(buf["truncated"]),cfg["train"]["gamma"],cfg["train"]["gae_lambda"])
        batch={"features":torch.as_tensor(np.concatenate(buf["features"])),"actions":torch.as_tensor(np.concatenate(buf["actions"])),"old_log_probs":torch.as_tensor(np.concatenate(buf["old_log_probs"])),"advantages":torch.as_tensor(adv.reshape(-1)),"returns":torch.as_tensor(ret.reshape(-1))}
        metrics=ppo_update(model,optimizer,batch,cfg["train"]["epochs"],cfg["train"]["minibatch"],cfg["train"]["clip_ratio"],cfg["train"]["value_coef"],cfg["train"]["entropy_coef"],cfg["train"]["max_grad_norm"],cfg["train"]["target_kl"]); metrics["env_steps"]=total; history.append(metrics)
        if live_writer is not None:
            live_writer.publish_metrics({"kind": "training_metric", "env_steps": total, "metrics": metrics, "updated_at_utc": utc_now()})
        torch.save(checkpoint_payload(model,brain,cfg,mode,seed,total,optimizer),out/"latest.pt")
    (out/"train.jsonl").write_text("\n".join(json.dumps(x) for x in history)+"\n"); torch.save(checkpoint_payload(model,brain,cfg,mode,seed,total,optimizer),out/"best.pt"); print(json.dumps({"total_env_steps":total,"mode":mode,"out":str(out)},indent=2))
    if live_writer is not None:
        live_writer.close()


def check_env(args):
    from gymnasium.utils.env_checker import check_env
    check_env(FlyToTargetEnv(env_config(load_config(args.config))), skip_render_check=True); print(json.dumps({"env_id":"FlyToTarget2D-v1","checker":"PASS"}))


def baseline(args):
    cfg=load_config(args.config); path=Path(args.cases); rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; e_cfg=env_config(cfg); results=[]
    for row in rows:
        e=FlyToTargetEnv(e_cfg); obs,_=e.reset(seed=row["seed"],options={"initial_state":row["initial_state"],"case_id":row["case_id"]}); total=0
        for _ in range(e_cfg.deadline_ticks//e_cfg.action_ticks+2):
            if args.policy=="random": action=int(e.action_space.sample())
            else:
                error = e.state.goal_xy - e.state.position_xy
                if float(np.linalg.norm(error)) <= 0.15:
                    action = 0
                else:
                    delta=error-.25*e.state.velocity_xy; action=int(np.argmax([0,delta[0],-delta[0],delta[1],-delta[1]]))
            obs,r,t,_,info=e.step(action); total+=r
            if t: break
        results.append({"case_id":row["case_id"],"return":total,"end_reason":info["end_reason"]})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/"episodes.jsonl").write_text("\n".join(json.dumps(x) for x in results)+"\n");(out/"summary.json").write_text(json.dumps({"count":len(results),"success_rate":sum(x["end_reason"]=="success" for x in results)/len(results)},indent=2));print(json.dumps({"count":len(results)}))


def evaluate(args):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("checkpoint action schema is incompatible")
    cfg = load_config(args.config); mode = payload["mode"]; env_cfg = env_config(cfg)
    model = ActorCritic(int(payload["feature_dim"])); model.load_state_dict(payload["policy_state_dict"]); model.eval()
    brain = None
    if mode == "brain":
        b = cfg["brain"]; graph = root_path(args.graph or b["graph"], args.config)
        if payload.get("graph_sha256") != graph_hash(graph): raise ValueError("checkpoint graph hash mismatch")
        brain = FrozenReservoir(str(graph), 1, device=args.device, mapping_seed=b["mapping_seed"], inputs_per_channel=b["inputs_per_channel"], readout_neurons=b["readout_neurons"], internal_steps=b["internal_steps"], gain=b["gain"], tonic=b["tonic"], input_gain=b["input_gain"], normalizer=load_normalizer(args.normalizer, int(payload["feature_dim"])))
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]
    results=[]
    for row in cases:
        env=FlyToTargetEnv(env_cfg); obs,_=env.reset(seed=row["seed"],options={"initial_state":row["initial_state"],"case_id":row["case_id"]})
        if brain: brain.reset(); brain.advance(brain.encoder(obs[None,:])); feat=brain.current_features()
        else: feat=obs[None,:]
        total=0.; steps=0
        while not env.state.ended and steps < env_cfg.deadline_ticks // env_cfg.action_ticks + 2:
            with torch.no_grad(): action,_,_=model.act(torch.as_tensor(feat), deterministic=args.action_selection=="greedy")
            next_obs,r,term,_,info=env.step(int(action.item())); total += r; steps += 1
            if not term:
                if brain: brain.advance(brain.encoder(next_obs[None,:])); feat=brain.current_features()
                else: feat=next_obs[None,:]
        results.append({"case_id":row["case_id"],"return":total,"steps":steps,"end_reason":info["end_reason"]})
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/"episodes.jsonl").write_text("\n".join(json.dumps(x) for x in results)+"\n")
    summary={"count":len(results),"success_rate":sum(x["end_reason"]=="success" for x in results)/max(1,len(results)),"end_reasons":{r:sum(x["end_reason"]==r for x in results) for r in ["success","boundary","deadline"]}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


def trace(args):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config); env_cfg = env_config(cfg); mode = payload["mode"]
    model = ActorCritic(int(payload["feature_dim"])); model.load_state_dict(payload["policy_state_dict"]); model.eval()
    brain = None
    if mode == "brain":
        b=cfg["brain"]; graph=root_path(args.graph or b["graph"], args.config)
        brain=FrozenReservoir(str(graph), 1, device=args.device, mapping_seed=b["mapping_seed"], inputs_per_channel=b["inputs_per_channel"], readout_neurons=b["readout_neurons"], internal_steps=b["internal_steps"], gain=b["gain"], tonic=b["tonic"], input_gain=b["input_gain"], normalizer=load_normalizer(args.normalizer, int(payload["feature_dim"])))
    cases=[json.loads(x) for x in Path(args.cases).read_text().splitlines() if x.strip()]; row=cases[args.case_index]
    env=FlyToTargetEnv(env_cfg); obs,_=env.reset(seed=row["seed"], options={"initial_state":row["initial_state"],"case_id":row["case_id"]})
    if brain: brain.reset(); brain.advance(brain.encoder(obs[None,:])); feat=brain.current_features()
    else: feat=obs[None,:]
    records=[]
    while not env.state.ended and len(records) < args.max_steps:
        with torch.no_grad():
            policy, value = model(torch.as_tensor(feat))
            action = policy.probs.argmax(-1)
            probabilities = policy.probs[0].cpu().numpy().tolist()
            value_estimate = float(value[0].item())
        action_i=int(action.item()); before=env.state.position_xy.copy(); before_velocity=env.state.velocity_xy.copy(); before_tick=env.state.physics_tick
        waypoint=local_waypoint(before, FlightAction(action_i)).tolist(); next_obs,reward,term,_,info=env.step(action_i)
        row_out={"case_id":row["case_id"],"decision_step":len(records),"obs":obs.tolist(),"action":action_i,
                 "action_name":FlightAction(action_i).name,"action_selection":"greedy","action_probabilities":probabilities,
                 "value_estimate":value_estimate,"position_xy":env.state.position_xy.tolist(),"goal_xy":env.state.goal_xy.tolist(),
                 "velocity_xy":env.state.velocity_xy.tolist(),"state_before":{"position_xy":before.tolist(),"velocity_xy":before_velocity.tolist()},
                 "state_after":{"position_xy":env.state.position_xy.tolist(),"velocity_xy":env.state.velocity_xy.tolist()},"waypoint_xy_m":waypoint,
                 "sim_tick_before":before_tick,"sim_tick_after":env.state.physics_tick,"reward":reward,
                 "reward_parts":info["reward_parts"],"end_reason":info["end_reason"]}
        if brain:
            selected=brain.outputs[:min(args.neurons, len(brain.outputs))].detach().cpu().numpy().tolist(); row_out["neuron_indices"]=selected; row_out["v_before_reset"]=brain.v[selected,0].detach().cpu().numpy().tolist(); row_out["spike"]=brain.s[selected,0].detach().cpu().numpy().tolist(); row_out["trace_after_update"]=brain.trace[selected,0].detach().cpu().numpy().tolist()
        records.append(row_out); obs=next_obs
        if not term:
            if brain: brain.advance(brain.encoder(obs[None,:])); feat=brain.current_features()
            else: feat=obs[None,:]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/"trace.json").write_text(json.dumps({"sim_only":True,"case":row,"records":records},indent=2)); print(json.dumps({"case_id":row["case_id"],"steps":len(records),"out":str(out)}))


def replay(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data=json.loads((Path(args.trace_dir)/"trace.json").read_text()); records=data["records"]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    xs=[r["position_xy"][0] for r in records]; ys=[r["position_xy"][1] for r in records]; goal=data["case"]["initial_state"]["goal_xy"]
    fig,ax=plt.subplots(); ax.plot(xs,ys,"-o",ms=2); ax.scatter([goal[0]],[goal[1]],marker="x",s=80); ax.set_title("SIMULATION ONLY"); ax.set_aspect("equal"); fig.savefig(out/"trajectory.png",dpi=120); plt.close(fig)
    payload={"title":"SIMULATION ONLY","goal":goal,"records":records}; (out/"replay.html").write_text("<!doctype html><meta charset=utf-8><title>SIMULATION ONLY</title><pre id=p></pre><script>const d="+json.dumps(payload)+";let i=0;function show(){p.textContent=JSON.stringify(d.records[i]||{},null,2)}show();document.body.onclick=()=>{i=(i+1)%d.records.length;show()}</script>")
    (out/"trajectory.json").write_text(json.dumps(payload,indent=2)); print(json.dumps({"out":str(out),"steps":len(records)}))


def accept(args):
    root=Path(args.root); tests=root.parent.parent/"logs/p1/final-tests.log"; report=root/"P1_ACCEPTANCE.md"; software=bool((root/"STARTUP.md").exists() and (root/"calibration/artifact.npz").exists() and (root/"real-probe-b1/metadata.json").exists())
    text="# P1 FlyToTarget2D Acceptance\n\n## Status\n\nP0_ENVIRONMENT_READY = YES\nP1_SOFTWARE_READY = "+("YES" if software else "NO")+"\nP1_SIM_TASK_LEARNED = NOT_RUN\nP1_REAL_FLIGHT_READY = NO\n\n## Evidence\n\n- Offline simulation only; no Tello, UDP, takeoff, or landing code.\n- Environment, brain, PPO, smoke checkpoint, and independent evaluation are implemented.\n- Formal three-seed C1 learning gate is not yet run.\n- Final software test log: `"+str(tests)+"`\n\n## Known gaps\n\n- Exact resume state restoration, formal C1 evaluation, and multi-seed learning threshold remain pending.\n"
    report.write_text(text); print(text)


def main():
    p=argparse.ArgumentParser(prog="python -m flydrone"); sub=p.add_subparsers(dest="command",required=True)
    for name in ["check-env","make-cases","baseline","probe","train","resume","eval","trace","replay","accept"]: sub.add_parser(name)
    for name in ["check-env","make-cases","baseline","probe","train"]: sub.choices[name].add_argument("--config",required=True)
    sub.choices["check-env"].set_defaults(func=check_env)
    sub.choices["make-cases"].add_argument("--validation-count",type=int,default=100);sub.choices["make-cases"].add_argument("--test-count",type=int,default=500);sub.choices["make-cases"].add_argument("--out",required=True);sub.choices["make-cases"].set_defaults(func=make_cases)
    for x in ["baseline"]: sub.choices[x].add_argument("--policy",choices=["reference","random"],required=True);sub.choices[x].add_argument("--cases",required=True);sub.choices[x].add_argument("--out",required=True);sub.choices[x].set_defaults(func=baseline)
    sub.choices["probe"].add_argument("--graph");sub.choices["probe"].add_argument("--device",default="cpu");sub.choices["probe"].add_argument("--envs",type=int,default=1);sub.choices["probe"].add_argument("--calibration-steps",type=int,default=256);sub.choices["probe"].add_argument("--out",required=True);sub.choices["probe"].add_argument("--fit-normalizer",action="store_true");sub.choices["probe"].set_defaults(func=make_normalizer)
    for x in ["train"]: sub.choices[x].add_argument("--mode",choices=["brain","direct"]);sub.choices[x].add_argument("--seed",type=int,default=11);sub.choices[x].add_argument("--device",default="cpu");sub.choices[x].add_argument("--envs",type=int);sub.choices[x].add_argument("--curriculum");sub.choices[x].add_argument("--graph");sub.choices[x].add_argument("--normalizer");sub.choices[x].add_argument("--init-from");sub.choices[x].add_argument("--telemetry-config");sub.choices[x].add_argument("--total-env-steps",type=int,default=1024);sub.choices[x].add_argument("--out",required=True);sub.choices[x].set_defaults(func=train)
    ev=sub.choices["eval"]; ev.add_argument("--config",required=True); ev.add_argument("--checkpoint",required=True); ev.add_argument("--cases",required=True); ev.add_argument("--out",required=True); ev.add_argument("--device",default="cpu"); ev.add_argument("--graph"); ev.add_argument("--normalizer"); ev.add_argument("--action-selection",choices=["greedy","sample"],default="greedy"); ev.set_defaults(func=evaluate)
    tr=sub.choices["trace"]; tr.add_argument("--config",required=True); tr.add_argument("--checkpoint",required=True); tr.add_argument("--cases",required=True); tr.add_argument("--case-index",type=int,default=0); tr.add_argument("--neurons",type=int,default=64); tr.add_argument("--max-steps",type=int,default=128); tr.add_argument("--out",required=True); tr.add_argument("--device",default="cpu"); tr.add_argument("--graph"); tr.add_argument("--normalizer"); tr.set_defaults(func=trace)
    rp=sub.choices["replay"]; rp.add_argument("--trace-dir",required=True); rp.add_argument("--out",required=True); rp.set_defaults(func=replay)
    ac=sub.choices["accept"]; ac.add_argument("--root",required=True); ac.add_argument("--runs-root",required=True); ac.add_argument("--out",required=False); ac.set_defaults(func=accept)
    args=p.parse_args();
    if hasattr(args,"func"): args.func(args)


if __name__ == "__main__": main()
