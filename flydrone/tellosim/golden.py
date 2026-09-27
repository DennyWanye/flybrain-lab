from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..brain import FrozenReservoir
from ..policy import ActorCritic
from .env import TelloSimEnv, VariableDurationConfig
from .physics import SimWorld, WorldConfig
from .sim import SimTello

ACTION_NAMES = ["STOP_HOLD_500MS", "FORWARD_20", "BACK_20", "LEFT_20", "RIGHT_20", "UP_20", "DOWN_20", "CW_30", "CCW_30"]
ALLOWED_COMMANDS = {"stop", "forward", "back", "left", "right", "up", "down", "cw", "ccw"}

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")

def _brain_snapshot(brain: FrozenReservoir | None) -> dict[str, Any]:
    if brain is None:
        return {"status": "not_recorded", "neurons": []}
    selected = brain.outputs[: min(64, len(brain.outputs))].detach().cpu().numpy().tolist()
    values_v = brain.v[selected, 0].detach().cpu().numpy().tolist()
    values_s = brain.s[selected, 0].detach().cpu().numpy().tolist()
    values_t = brain.trace[selected, 0].detach().cpu().numpy().tolist()
    top = torch.topk(brain.trace[:, 0], 32).indices.cpu().tolist()
    return {"status": "recorded", "sample_phase": "after_final_neural_substep_before_action", "top_n_basis": "trace", "top_n": [{"neuron_index": i, "trace": float(brain.trace[i, 0]), "spike": int(brain.s[i, 0])} for i in top], "whole_brain_spike_fraction": float(brain.s[:, 0].mean().item()),
            "neurons": [{"neuron_id": str(brain.readout_ids[i]), "neuron_index": int(selected[i]),
                          "v": float(values_v[i]), "spike": int(values_s[i]), "trace": float(values_t[i])}
                         for i in range(len(selected))]}

def load_runtime(checkpoint, graph, readout=64, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(int(payload.get("feature_dim", 26)), action_dim=int(payload.get("action_dim", 9)))
    if int(payload.get("action_dim", 9)) != 9:
        raise ValueError("incompatible action space")
    model.load_state_dict(payload["policy_state_dict"])
    model.eval()
    brain = FrozenReservoir(str(graph), batch=1, readout_neurons=readout, device=device) if graph else None
    if brain:
        if payload.get("graph_sha256") != brain.graph_sha256 or payload.get("mapping_sha256") != brain.mapping_sha256:
            raise ValueError("checkpoint graph/mapping mismatch")
        if model.feature_dim != brain.feature_dim:
            raise ValueError("checkpoint feature dimension mismatch")
    elif model.feature_dim != 26:
        raise ValueError("reservoir checkpoint requires its brain graph")
    return model, brain, sha256(checkpoint)


def record_golden(out: str | Path, seed: int = 11, checkpoint: str | Path | None = None,
                  scenario: str | Path | None = None, brain_graph: str | Path | None = None,
                  brain_readout_neurons: int = 64, device: str = "cpu",
                  runtime=None, write_artifact: bool = True) -> dict[str, Any]:
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("a real policy checkpoint is required")
    out = Path(out)
    if write_artifact:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "replay.jsonl").exists():
            raise FileExistsError("preserve sealed replay; choose a new output directory")
    scenario_data = scenario if isinstance(scenario, dict) else json.loads(Path(scenario).read_text(encoding="utf-8")) if scenario else {"scenario_id": "room6_c0_fixed_target", "world": {}, "success": {}}
    world_data, success_data = scenario_data.get("world", {}), scenario_data.get("success", {})
    start = tuple(world_data.get("start_xyz_m", [0.0, 0.0, 1.0]))
    goal = tuple(world_data.get("target_xyz_m", [0.8, 0.0, 1.0]))
    if world_data.get("room_size_m", [6, 6, 3]) != [6, 6, 3] or world_data.get("yaw_rad", 0) != 0 or world_data.get("obstacles"):
        raise ValueError("Golden runner currently supports only the empty 6x6x3 room and initial yaw=0")
    if int(scenario_data.get("max_episode_steps", 120)) < 1:
        raise ValueError("max_episode_steps must be positive")
    world = SimWorld(WorldConfig())
    sim = SimTello(world)
    env = TelloSimEnv(sim, VariableDurationConfig(
        goal_m=goal, start_m=start,
        action_duration_s=tuple(scenario_data.get("duration_schedule_s", [0.5, 1.0, 2.0])),
        max_episode_steps=int(scenario_data.get("max_episode_steps", 120)),
        target_radius_m=float(success_data.get("target_radius_m", 0.15)),
        stable_hold_s=float(success_data.get("stable_hold_s", 0.0)),
        stable_speed_mps=float(success_data.get("speed_mps", 0.15))))
    runtime = runtime or load_runtime(checkpoint, brain_graph, brain_readout_neurons, device)
    model, brain, policy_hash = runtime
    feature_dim = model.feature_dim
    graph_digest = brain.graph_sha256 if brain else None
    mapping_digest = brain.mapping_sha256 if brain else None
    if brain is not None:
        brain.reset()
    obs, _ = env.reset(seed=seed)
    rows, total_return = [], 0.0
    for step in range(env.config.max_episode_steps):
        brain_obs = env.brain_observation()
        encoded = brain.encoder(np.asarray([brain_obs], dtype=np.float32)) if brain else None
        if brain is not None:
            brain.advance(encoded)
        feature_array = brain.current_features() if brain is not None and feature_dim == brain.feature_dim else np.asarray([obs], dtype=np.float32)
        features = torch.as_tensor(feature_array, dtype=torch.float32)
        if features.shape[1] != feature_dim:
            raise ValueError(f"checkpoint feature_dim={feature_dim} cannot consume TelloSim observation_dim={features.shape[1]}")
        with torch.no_grad():
            action, _, value, probabilities = model.act_with_probs(features, deterministic=True)
            probs = probabilities[0].cpu().numpy()
            entropy = float(-(probabilities[0] * torch.log(probabilities[0].clamp_min(1e-12))).sum().item())
        before = env.pose.snapshot()
        next_obs, reward, terminated, truncated, meta = env.step(int(action.item()))
        after = env.pose.snapshot()
        total_return += float(reward)
        rows.append({
            "run_id": scenario_data.get("run_id", "golden-checkpoint-eval"), "episode_id": scenario_data.get("episode_id", "golden-0001"),
            "scenario_id": scenario_data.get("scenario_id", "tellosim-golden"),
            "policy_checkpoint_sha256": policy_hash, "graph_sha256": graph_digest, "brain_mapping_sha256": mapping_digest,
            "seed": seed, "decision_step": step, "neural_substep": brain.internal_steps - 1 if brain else None,
            "physics_dt_s": world.config.dt, "decision_time_s": before.sim_tick * world.config.dt,
            "brain_observation": brain_obs.tolist() if brain else None,
            "encoded_observation": encoded[0].tolist() if brain else None, "policy_features": features[0].tolist(),
            "physics_samples": meta["physics_samples"],
            "sim_time_s": after.sim_tick * world.config.dt, "observation_schema_version": "tellosim.pose/1.0",
            "observation": obs.tolist(),
            "world": {"position_xyz_m": list(after.position_m), "velocity_xyz_mps": list(after.velocity_mps), "yaw_rad": after.yaw_rad,
                      "is_flying": after.airborne, "collision": meta["collision"],
                      "out_of_bounds": meta["out_of_bounds"],
                      "target_xyz_m": list(env.goal), "distance_to_target_m": meta["distance_m"]},
            "brain": _brain_snapshot(brain) if write_artifact else {"status": "recorded" if brain else "not_recorded"},
            "policy": {"action_space_version": "tellosim.actions9/1.0", "selected_action": int(action.item()),
                       "selected_action_name": ACTION_NAMES[int(action.item())] if int(action.item()) < len(ACTION_NAMES) else "UNKNOWN",
                       "probabilities": [float(v) for v in probs], "value_estimate": float(value.item()), "entropy": entropy,
                       "input_source": "direct_observation" if brain is None else "flybrain_reservoir_features"},
            "command": {"proposed_action": int(action.item()), "approved_action": int(action.item()), "transformation_reason": "none", "command_id": meta["command_id"], "command_type": meta["command"], "command_args": list(meta["command_args"]),
                        "issued_at_sim_time": meta["issued_sim_tick"] * world.config.dt,
                        "started_at_sim_time": meta["started_sim_tick"] * world.config.dt,
                        "completed_at_sim_time": meta["completed_sim_tick"] * world.config.dt, "result": meta["command_result"]},
            "reward": {"reward_total": float(reward), "reward_components": dict(meta["reward_components"])},
            "state_before": {"position_xyz_m": list(before.position_m), "velocity_xyz_mps": list(before.velocity_mps)},
            "state_after_observation": next_obs.tolist(), "stable_hold_s": meta["stable_hold_s"],
            "success": bool(meta["success"]), "terminated": bool(terminated), "truncated": bool(truncated)})
        obs = next_obs
        sim.events.clear()
        if terminated or truncated:
            break
    final = rows[-1]
    manifest = {"schema_version": "golden_episode/1.2", "run_id": scenario_data.get("run_id", "golden-checkpoint-eval"), "episode_id": scenario_data.get("episode_id", "golden-0001"),
        "scenario_id": scenario_data.get("scenario_id", "tellosim-golden"), "seed": seed, "policy_checkpoint_sha256": policy_hash,
        "graph_sha256": graph_digest, "brain_mapping_sha256": mapping_digest, "success": bool(final["success"]),
        "episode_return": total_return, "episode_steps": len(rows), "sim_duration_s": final["sim_time_s"],
        "final_distance_m": final["world"]["distance_to_target_m"], "stable_hold_s": final["stable_hold_s"],
        "collision": any(r["world"]["collision"] for r in rows), "out_of_bounds": any(r["world"]["out_of_bounds"] for r in rows),
        "replay_sha256": None, "policy_source": final["policy"]["input_source"],
        "neural_capture": "recorded" if brain else "not_recorded", "replay_mode": "recorded_replay",
        "scenario": {"start_xyz_m": list(start), "target_xyz_m": list(goal)},
        "scenario_config": scenario_data, "timeout": final["truncated"],
        "success_contract": {"target_radius_m": env.config.target_radius_m, "stable_hold_s": env.config.stable_hold_s, "speed_mps": env.config.stable_speed_mps},
        "brain": {"model": "frozen_LIF", "n": brain.n, "nnz": brain.w._nnz(), "internal_steps": brain.internal_steps, "device": str(brain.device)} if brain else None,
        "observation_note": "brain_observation is the actual 8D encoder input from simulated pose; observation is 26D telemetry; policy_features is the exact policy input",
        "initial_state": "initialized airborne at known pose, no takeoff task",
        "physics_dt_s": world.config.dt}
    if not write_artifact:
        manifest["neural_capture"] = "not_recorded"
        manifest["replay_mode"] = "evaluation_summary"
        manifest["decisions"] = [{"decision_step": r["decision_step"], "action": r["policy"]["selected_action"], "position_xyz_m": r["world"]["position_xyz_m"], "sim_time_s": r["sim_time_s"], "distance_m": r["world"]["distance_to_target_m"], "reward": r["reward"]["reward_total"], "stable_hold_s": r["stable_hold_s"]} for r in rows]
        return manifest
    replay = out / "replay.jsonl"
    _jsonl(replay, rows)
    for name, key in [("world.jsonl", "world"), ("policy.jsonl", "policy"), ("commands.jsonl", "command"), ("rewards.jsonl", "reward"), ("brain_summary.jsonl", "brain")]:
        _jsonl(out / name, [{**r[key], "run_id": r["run_id"], "episode_id": r["episode_id"], "decision_step": r["decision_step"], "sim_time_s": r["sim_time_s"]} for r in rows])
    ck = out / "checkpoint.pt"
    shutil.copy2(checkpoint, ck)
    manifest["replay_sha256"] = sha256(replay)
    if brain:
        mapping = {"input_indices": brain.input_indices.tolist(), "input_channels": brain.input_channels.tolist(), "readout_indices": brain.readout_indices.tolist()}
        (out / "brain_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in {"checksums.sha256", "GOLDEN_EPISODE_REPORT.md"})
    (out / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    (out / "GOLDEN_EPISODE_REPORT.md").write_text("# Golden Episode Report\n\n" +
        f"- success: {manifest['success']}\n- steps: {manifest['episode_steps']}\n" +
        f"- final distance: {manifest['final_distance_m']:.4f} m\n- policy source: {manifest['policy_source']}\n" +
        f"- neural capture: {manifest['neural_capture']}\n- replay sha256: {manifest['replay_sha256']}\n\n" +
        "This artifact is a recorded replay. The viewer does not re-run the policy.\n", encoding="utf-8")
    return manifest

def validate_golden(path: str | Path, graph: str | Path | None = None) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "replay.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {}
    contract = manifest.get("success_contract", {})
    samples = [p for r in rows for p in r.get("physics_samples", [])]
    hold = 0.0
    goal = np.asarray(manifest["scenario"]["target_xyz_m"])
    dt = manifest.get("physics_dt_s", 1/120)
    for p in samples:
        good = np.linalg.norm(np.asarray([p["x_m"], p["y_m"], p["z_m"]])-goal) <= contract.get("target_radius_m", .3) and p["speed_mps"] <= contract.get("speed_mps", .15)
        hold = hold + dt if good else 0.0
    checks["GE-01"] = bool(samples and manifest["success"] and rows[-1].get("success") and hold + 1e-9 >= max(1., contract.get("stable_hold_s", 1.)) and abs(hold-manifest["stable_hold_s"]) < 1e-6 and not any(p["collision"] or p["out_of_bounds"] for p in samples) and not manifest.get("timeout"))
    verbs = ["stop", "forward", "back", "left", "right", "up", "down", "cw", "ccw"]
    checks["GE-02"] = all(0 <= r["policy"]["selected_action"] < 9 and r["command"]["command_type"] == verbs[r["policy"]["selected_action"]] for r in rows)
    checks["GE-03"] = bool(rows) and all(r["command"]["issued_at_sim_time"] <= r["command"]["started_at_sim_time"] < r["command"]["completed_at_sim_time"] == r["sim_time_s"] for r in rows) and all(a["sim_time_s"] <= b["command"]["issued_at_sim_time"] for a,b in zip(rows, rows[1:]))
    checks["GE-04"] = bool(rows) and all(r["decision_step"] == i and all(r[k] == manifest[k] for k in ("run_id", "episode_id", "scenario_id", "policy_checkpoint_sha256", "graph_sha256", "brain_mapping_sha256")) and r.get("neural_substep") == (manifest.get("brain") or {}).get("internal_steps", 0)-1 for i,r in enumerate(rows))
    checks["GE-05"] = None  # browser must display missing data explicitly
    for key in ("GE-06", "GE-07", "GE-08", "GE-15"):
        checks[key] = None  # actual browser interaction evidence is required
    checks["GE-09"] = all(abs(sum(r["reward"]["reward_components"].values())-r["reward"]["reward_total"]) < 1e-6 for r in rows)
    checks["GE-10"] = all(len(r["policy"]["probabilities"]) == 9 and np.isfinite(r["policy"]["probabilities"]).all() and all(0 <= v <= 1 for v in r["policy"]["probabilities"]) and abs(sum(r["policy"]["probabilities"])-1) < 1e-5 and 0 <= r["policy"]["selected_action"] < 9 for r in rows)
    checks["GE-11"] = checks["GE-02"] and all(r["command"]["command_args"] == ([] if r["policy"]["selected_action"] == 0 else [30] if r["policy"]["selected_action"] >= 7 else [20]) and r["command"]["result"].get("device_execution") == "completed" for r in rows)
    continuous = bool(samples) and all(b["sim_tick"] == a["sim_tick"]+1 and np.linalg.norm(np.asarray([b["x_m"],b["y_m"],b["z_m"]])-np.asarray([a["x_m"],a["y_m"],a["z_m"]])) <= max(a["speed_mps"], b["speed_mps"])*dt+1e-5 for a,b in zip(samples,samples[1:]))
    checks["GE-12"] = continuous and any(sum(p["speed_mps"] > .001 for p in r.get("physics_samples", [])) > 2 for r in rows)
    checks["GE-13"] = sha256(root / "checkpoint.pt") == manifest.get("policy_checkpoint_sha256") and sha256(root / "replay.jsonl") == manifest.get("replay_sha256")
    mapping_file = root / "brain_mapping.json"
    mapping_ok = False
    if mapping_file.exists():
        m = json.loads(mapping_file.read_text())
        digest = hashlib.sha256(b"".join(np.asarray(m[k], dtype=np.int64).tobytes() for k in ("input_indices", "input_channels", "readout_indices"))).hexdigest()
        mapping_ok = digest == manifest["brain_mapping_sha256"]
    checks["GE-14"] = bool(graph and sha256(graph) == manifest.get("graph_sha256") and mapping_ok)
    browser = root / "browser_acceptance.json"
    if browser.exists():
        evidence = json.loads(browser.read_text())
        project = Path(__file__).resolve().parents[2]
        viewer_path = project / "flyview/static/index.html"
        events_path = project / "reports/vis/views/golden-episode/events.jsonl"
        evidence_matches = evidence.get("replay_sha256") == sha256(root / "replay.jsonl") and evidence.get("viewer_sha256") == sha256(viewer_path) and events_path.exists() and evidence.get("events_sha256") == sha256(events_path)
        if evidence_matches:
            for key in ("GE-05", "GE-06", "GE-07", "GE-08", "GE-15"):
                checks[key] = evidence.get("checks", {}).get(key) is True
    return {"GOLDEN_EPISODE_READY": bool(rows) and all(v is True for v in checks.values()), "checks": checks, "manifest": manifest, "rows": len(rows), "pending": [k for k,v in checks.items() if v is None]}
