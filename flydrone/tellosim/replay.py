"""Recorded Golden replay export and independent checkpoint evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .golden import ACTION_NAMES, _jsonl, sha256, load_runtime, record_golden


def export_golden(episode, out, view_id="golden-episode", registry=None):
    source, dest = Path(episode).resolve(), Path(out).resolve()
    m = json.loads((source / "manifest.json").read_text())
    if sha256(source / "replay.jsonl") != m["replay_sha256"] or sha256(source / "checkpoint.pt") != m["policy_checkpoint_sha256"]:
        raise ValueError("sealed artifact hash mismatch")
    rows = [json.loads(line) for line in (source / "replay.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty replay")
    events, cumulative = [], 0.0
    for i, r in enumerate(rows):
        w, p = r["world"], r["policy"]
        cumulative += r["reward"]["reward_total"]
        dt = r.get("physics_dt_s")
        payload = {
            "episode_id": r["episode_id"], "case_id": r["scenario_id"], "decision_step": r["decision_step"],
            "sim_time_s": r["sim_time_s"], "decision_time_s": r["command"]["issued_at_sim_time"],
            "physics_dt_s": dt, "state_before": r["state_before"],
            "state_after": {"position_xy_m": w["position_xyz_m"][:2], "velocity_xy_mps": w["velocity_xyz_mps"][:2],
                "goal_xy_m": w["target_xyz_m"][:2], "z_m": w["position_xyz_m"][2], "yaw_rad": w["yaw_rad"], "airborne": w["is_flying"]},
            "observation": r["observation"], "action_id": p["selected_action"], "action_name": p["selected_action_name"],
            "action_probabilities": p["probabilities"], "value_estimate": p["value_estimate"], "action_selection": "recorded",
            "reward_parts": r["reward"]["reward_components"], "reward_total": r["reward"]["reward_total"], "cumulative_return": cumulative,
            "readout_snapshot": r["brain"], "neural_capture": r["brain"]["status"],
            "terminated": r["terminated"], "truncated": r["truncated"],
            "end_reason": ("collision" if w["collision"] else "out_of_bounds" if w["out_of_bounds"] else "success" if r["terminated"] and m["success"] else "timeout" if r["truncated"] else None),
            "golden": r,  # lossless original record; never invoke inference on export or playback
        }
        events.append({"schema_version": "golden_view/1.0", "source_kind": "simulation_recorded", "run_id": r["run_id"],
                       "source_epoch": m["replay_sha256"], "seq": i, "kind": "transition", "payload": payload})
    dest.mkdir(parents=True, exist_ok=True)
    _jsonl(dest / "events.jsonl", events)
    neuron_ids = sorted({n["neuron_id"] for r in rows for n in r["brain"].get("neurons", [])})
    (dest / "neurons.json").write_text(json.dumps({"dynamic_recorded": neuron_ids, "nodes": []}))
    manifest = {"schema_version": "golden_view/1.0", "view_id": view_id, "source_kind": "simulation_recorded", "mode": "REPLAY",
                "run_id": m["run_id"], "episode_id": m["episode_id"], "scenario_id": m["scenario_id"],
                "event_count": len(events), "trace_complete": True, "source_directory": str(source),
                "replay_sha256": m["replay_sha256"], "policy_checkpoint_sha256": m["policy_checkpoint_sha256"],
                "graph_sha256": m["graph_sha256"], "brain_mapping_sha256": m["brain_mapping_sha256"],
                "events_sha256": sha256(dest / "events.jsonl"), "actions": ACTION_NAMES, "recorded_neuron_ids": neuron_ids,
                "capabilities": {"topology": False, "anatomy": False, "substep_spikes": False, "physics_path": all(r.get("physics_samples") for r in rows)},
                "world_half_extent_m": 3, "target_radius_m": m.get("success_contract", {}).get("target_radius_m", .3),
                "limitations": ["Only final neural substep is recorded; other substeps: not_recorded", "No anatomical labels or local topology recorded", "MuJoCo surrogate, initialized at altitude; no real drone"],
                "source_manifest": m}
    (dest / "view_manifest.json").write_text(json.dumps(manifest, indent=2))
    if registry:
        path = Path(registry)
        index = json.loads(path.read_text()) if path.exists() else {}
        index[view_id] = str(dest)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(path)
    return manifest


def evaluate(checkpoint, graph, cases_path, out, device="cpu", readout=64):
    """Evaluate a predeclared case set without updating model weights or selecting a checkpoint."""
    spec = json.loads(Path(cases_path).read_text())
    cases = spec["cases"]
    if not cases or len({c["scenario_id"] for c in cases}) != len(cases):
        raise ValueError("case IDs must be nonempty and unique")
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=False)
    runtime = load_runtime(checkpoint, graph, readout, device)
    provenance = {"case_set_sha256": sha256(cases_path), "checkpoint_sha256": sha256(checkpoint),
                  "graph_sha256": runtime[1].graph_sha256, "mapping_sha256": runtime[1].mapping_sha256,
                  "device": device, "scope": spec["scope"], "criteria": spec["criteria"]}
    (dest / "manifest.json").write_text(json.dumps(provenance, indent=2))
    results = []
    for i, case in enumerate(cases):
        case = {**case, "episode_id": case["scenario_id"], "run_id": "golden-multiscenario-eval"}
        result = record_golden(dest, case["seed"], checkpoint, case, graph, readout, device, runtime=runtime, write_artifact=False)
        result["group"] = case["group"]
        results.append(result)
        _jsonl(dest / "episodes.jsonl", results)
        print(json.dumps({"completed": i+1, "total": len(cases), "case": case["scenario_id"], "success": result["success"], "distance": result["final_distance_m"]}), flush=True)
    groups = {}
    for name in sorted({r["group"] for r in results}):
        subset = [r for r in results if r["group"] == name]
        groups[name] = {"episodes": len(subset), "successes": sum(r["success"] for r in subset), "success_rate": sum(r["success"] for r in subset)/len(subset)}
    summary = {**provenance, "episodes": len(results), "successes": sum(r["success"] for r in results),
               "success_rate": sum(r["success"] for r in results)/len(results), "mean_return": float(np.mean([r["episode_return"] for r in results])),
               "collisions": sum(r["collision"] for r in results), "out_of_bounds": sum(r["out_of_bounds"] for r in results),
               "timeouts": sum(r["timeout"] for r in results), "groups": groups,
               "MODEL_READY_FOR_NEXT_STAGE": False,
               "reason": "Diagnostic single-checkpoint evaluation only. TS1 stage gate requires 3 independently trained seeds, 300 sealed cases per seed, >=90% success, <=1% collision/boundary and >=20 percentage point gain over random for each seed."}
    summary["diagnostic_pass"] = summary["success_rate"] >= spec["criteria"]["diagnostic_success_rate"] and (summary["collisions"] + summary["out_of_bounds"])/len(results) <= spec["criteria"]["diagnostic_collision_boundary_rate_max"]
    if not summary["diagnostic_pass"]:
        summary["reason"] = "Diagnostic success/safety criteria failed. " + summary["reason"]
    if sha256(checkpoint) != provenance["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during evaluation")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
