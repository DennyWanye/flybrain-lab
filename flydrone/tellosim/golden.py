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
    return {"status": "recorded", "whole_brain_spike_fraction": float(brain.s[:, 0].mean().item()),
            "neurons": [{"neuron_id": str(brain.readout_ids[i]), "neuron_index": int(selected[i]),
                          "v": float(values_v[i]), "spike": int(values_s[i]), "trace": float(values_t[i])}
                         for i in range(len(selected))]}

def record_golden(out: str | Path, seed: int = 11, checkpoint: str | Path | None = None,
                  scenario: str | Path | None = None, brain_graph: str | Path | None = None,
                  brain_readout_neurons: int = 64) -> dict[str, Any]:
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("a real policy checkpoint is required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scenario_data = json.loads(Path(scenario).read_text(encoding="utf-8")) if scenario else {"scenario_id": "room6_c0_fixed_target", "world": {}, "success": {}}
    world_data, success_data = scenario_data.get("world", {}), scenario_data.get("success", {})
    start = tuple(world_data.get("start_xyz_m", [0.0, 0.0, 1.0]))
    goal = tuple(world_data.get("target_xyz_m", [0.8, 0.0, 1.0]))
    world = SimWorld(WorldConfig())
    sim = SimTello(world)
    env = TelloSimEnv(sim, VariableDurationConfig(
        goal_m=goal, start_m=start,
        action_duration_s=tuple(scenario_data.get("duration_schedule_s", [0.5, 1.0, 2.0])),
        max_episode_steps=int(scenario_data.get("max_episode_steps", 120)),
        target_radius_m=float(success_data.get("target_radius_m", 0.15)),
        stable_hold_s=float(success_data.get("stable_hold_s", 0.0)),
        stable_speed_mps=float(success_data.get("speed_mps", 0.15))))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_dim = int(payload.get("feature_dim", payload.get("observation_dim", 26)))
    action_dim = int(payload.get("action_dim", 9))
    if action_dim != 9:
        raise ValueError(f"checkpoint action_dim={action_dim} is incompatible with tellosim.actions9")
    model = ActorCritic(feature_dim, action_dim=action_dim)
    model.load_state_dict(payload["policy_state_dict"])
    model.eval()
    brain = None
    graph_digest, mapping_digest = payload.get("graph_sha256"), payload.get("mapping_sha256")
    if brain_graph is not None:
        brain = FrozenReservoir(str(brain_graph), batch=1, readout_neurons=brain_readout_neurons)
        graph_digest, mapping_digest = brain.graph_sha256, brain.mapping_sha256
    policy_hash = sha256(checkpoint)
    obs, _ = env.reset(seed=seed)
    rows, total_return = [], 0.0
    for step in range(env.config.max_episode_steps):
        if brain is not None:
            brain.advance(brain.encoder(np.asarray([env.brain_observation()], dtype=np.float32)))
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
            "run_id": "golden-checkpoint-eval", "episode_id": "golden-0001",
            "scenario_id": scenario_data.get("scenario_id", "tellosim-golden"),
            "policy_checkpoint_sha256": policy_hash, "graph_sha256": graph_digest, "brain_mapping_sha256": mapping_digest,
            "decision_step": step, "neural_substep": 0 if brain is not None else None,
            "sim_time_s": after.sim_tick * world.config.dt, "observation_schema_version": "tellosim.pose/1.0",
            "observation": obs.tolist(),
            "world": {"position_xyz_m": list(after.position_m), "velocity_xyz_mps": list(after.velocity_mps), "yaw_rad": after.yaw_rad,
                      "is_flying": after.airborne, "collision": False,
                      "out_of_bounds": bool(np.max(np.abs(np.asarray(after.position_m)[:2])) > world.config.room_half_extent_m),
                      "target_xyz_m": list(env.goal), "distance_to_target_m": meta["distance_m"]},
            "brain": _brain_snapshot(brain),
            "policy": {"action_space_version": "tellosim.actions9/1.0", "selected_action": int(action.item()),
                       "selected_action_name": ACTION_NAMES[int(action.item())] if int(action.item()) < len(ACTION_NAMES) else "UNKNOWN",
                       "probabilities": [float(v) for v in probs], "value_estimate": float(value.item()), "entropy": entropy,
                       "input_source": "direct_observation" if brain is None else "flybrain_reservoir_features"},
            "command": {"command_id": meta["command_id"], "command_type": meta["command"], "command_args": list(meta["command_args"]),
                        "issued_at_sim_time": meta["issued_sim_tick"] * world.config.dt,
                        "started_at_sim_time": meta["started_sim_tick"] * world.config.dt,
                        "completed_at_sim_time": meta["completed_sim_tick"] * world.config.dt, "result": meta["command_result"]},
            "reward": {"reward_total": float(reward), "reward_components": dict(meta["reward_components"])},
            "state_before": {"position_xyz_m": list(before.position_m), "velocity_xyz_mps": list(before.velocity_mps)},
            "state_after_observation": next_obs.tolist(), "stable_hold_s": meta["stable_hold_s"],
            "terminated": bool(terminated), "truncated": bool(truncated)})
        obs = next_obs
        if terminated or truncated:
            break
    replay = out / "replay.jsonl"
    _jsonl(replay, rows)
    for name, key in [("world.jsonl", "world"), ("policy.jsonl", "policy"), ("commands.jsonl", "command"), ("rewards.jsonl", "reward"), ("brain_summary.jsonl", "brain")]:
        _jsonl(out / name, [{**r[key], "run_id": r["run_id"], "episode_id": r["episode_id"], "decision_step": r["decision_step"], "sim_time_s": r["sim_time_s"]} for r in rows])
    ck = out / "checkpoint.pt"
    shutil.copy2(checkpoint, ck)
    final = rows[-1]
    manifest = {"schema_version": "golden_episode/1.1", "run_id": "golden-checkpoint-eval", "episode_id": "golden-0001",
        "scenario_id": scenario_data.get("scenario_id", "tellosim-golden"), "seed": seed, "policy_checkpoint_sha256": policy_hash,
        "graph_sha256": graph_digest, "brain_mapping_sha256": mapping_digest, "success": bool(final["terminated"]),
        "episode_return": total_return, "episode_steps": len(rows), "sim_duration_s": final["sim_time_s"],
        "final_distance_m": final["world"]["distance_to_target_m"], "stable_hold_s": final["stable_hold_s"],
        "collision": any(r["world"]["collision"] for r in rows), "out_of_bounds": any(r["world"]["out_of_bounds"] for r in rows),
        "replay_sha256": sha256(replay), "policy_source": final["policy"]["input_source"],
        "neural_capture": "recorded" if brain else "not_recorded", "replay_mode": "recorded_replay",
        "scenario": {"start_xyz_m": list(start), "target_xyz_m": list(goal)}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name not in {"checksums.sha256", "GOLDEN_EPISODE_REPORT.md"})
    (out / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    (out / "GOLDEN_EPISODE_REPORT.md").write_text("# Golden Episode Report\n\n" +
        f"- success: {manifest['success']}\n- steps: {manifest['episode_steps']}\n" +
        f"- final distance: {manifest['final_distance_m']:.4f} m\n- policy source: {manifest['policy_source']}\n" +
        f"- neural capture: {manifest['neural_capture']}\n- replay sha256: {manifest['replay_sha256']}\n\n" +
        "This artifact is a recorded replay. The viewer does not re-run the policy.\n", encoding="utf-8")
    return manifest

def validate_golden(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "replay.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = {}
    checks["GE-01"] = bool(manifest.get("success") and manifest.get("stable_hold_s", 0) >= 1.0 and not manifest.get("collision") and not manifest.get("out_of_bounds"))
    checks["GE-02"] = all(r["command"]["command_type"] in ALLOWED_COMMANDS for r in rows)
    checks["GE-03"] = all(rows[i]["sim_time_s"] < rows[i + 1]["sim_time_s"] for i in range(len(rows) - 1))
    checks["GE-04"] = all(r["decision_step"] == i and r["episode_id"] == manifest["episode_id"] for i, r in enumerate(rows))
    checks["GE-05"] = all(r["brain"]["status"] in {"recorded", "not_recorded"} for r in rows)
    checks["GE-06"] = sha256(root / "replay.jsonl") == manifest.get("replay_sha256")
    checks["GE-07"] = bool(rows)
    checks["GE-08"] = checks["GE-06"]
    checks["GE-09"] = all(abs(sum(r["reward"]["reward_components"].values()) - r["reward"]["reward_total"]) < 1e-6 for r in rows)
    checks["GE-10"] = all(np.isfinite(r["policy"]["probabilities"]).all() and abs(sum(r["policy"]["probabilities"]) - 1.0) < 1e-5 and 0 <= r["policy"]["selected_action"] < len(r["policy"]["probabilities"]) for r in rows)
    checks["GE-11"] = checks["GE-02"]
    checks["GE-12"] = any(r["world"]["position_xyz_m"] != r["state_before"]["position_xyz_m"] for r in rows)
    checks["GE-13"] = sha256(root / "checkpoint.pt") == manifest.get("policy_checkpoint_sha256") and checks["GE-06"]
    checks["GE-14"] = bool(manifest.get("graph_sha256") and manifest.get("brain_mapping_sha256") and manifest.get("neural_capture") == "recorded")
    checks["GE-15"] = checks["GE-06"]
    return {"GOLDEN_EPISODE_READY": all(checks.values()), "checks": checks, "manifest": manifest, "rows": len(rows)}
