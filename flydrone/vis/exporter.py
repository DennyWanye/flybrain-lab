from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


ACTION_NAMES = ["HOLD", "POS_X", "NEG_X", "POS_Y", "NEG_Y"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_time() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _state(position: list[float], velocity: list[float], goal: list[float]) -> dict[str, Any]:
    return {
        "position_xy_m": position,
        "velocity_xy_mps": velocity,
        "goal_xy_m": goal,
    }


def _export_topology(graph_path: Path, center_indices: list[int], destination: Path) -> bool:
    with np.load(graph_path, allow_pickle=False) as archive:
        n = int(archive["n"])
        matrix = sp.csr_matrix((archive["data"], archive["indices"], archive["indptr"]), shape=(n, n))
        ids = archive["ids"].astype(str)
    csc = matrix.tocsc()
    nodes: set[int] = set(center_indices)
    edges: list[dict[str, Any]] = []
    for center in center_indices[:64]:
        incoming = matrix.getrow(center)
        outgoing = csc.getcol(center)
        incoming_order = np.argsort(-np.abs(incoming.data))[:64]
        outgoing_order = np.argsort(-np.abs(outgoing.data))[:64]
        for idx in incoming_order:
            pre = int(incoming.indices[idx]); nodes.add(pre)
            edges.append({"source_id": str(ids[pre]), "target_id": str(ids[center]), "signed_weight": float(incoming.data[idx]), "direction": "upstream"})
        for idx in outgoing_order:
            post = int(outgoing.indices[idx]); nodes.add(post)
            edges.append({"source_id": str(ids[center]), "target_id": str(ids[post]), "signed_weight": float(outgoing.data[idx]), "direction": "downstream"})
    edges = edges[:512]
    allowed_indices = set(center_indices[:64])
    limited_edges: list[dict[str, Any]] = []
    for edge in edges:
        source = next((index for index in nodes if str(ids[index]) == edge["source_id"]), None)
        target = next((index for index in nodes if str(ids[index]) == edge["target_id"]), None)
        if source is None or target is None:
            continue
        new_nodes = {source, target} - allowed_indices
        if len(allowed_indices) + len(new_nodes) > 129:
            continue
        allowed_indices.update(new_nodes)
        limited_edges.append(edge)
    node_ids = {str(ids[index]) for index in allowed_indices}
    topology = {
        "schema_version": "1.0.0",
        "dataset_id": "male-v1",
        "graph_sha256": _sha256(graph_path),
        "node_ids": sorted(node_ids),
        "edges": limited_edges,
        "truncated": len(limited_edges) < len(edges) or len(edges) >= 512,
        "total_edges_in_source": int(matrix.nnz),
    }
    (destination / "topology.json").write_text(json.dumps(topology, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def export_trace(trace_dir: str | Path, out_dir: str | Path, view_id: str | None = None,
                 graph: str | Path | None = None) -> dict[str, Any]:
    """Convert the legacy P1 trace into an immutable viewer view.

    A trace generated after this adapter was added can carry richer policy fields.
    Older traces remain valid and expose missing fields as explicit nulls.
    """
    trace_path = Path(trace_dir) / "trace.json"
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    case = data["case"]
    records = data.get("records", [])
    view_id = view_id or f"p1-{case.get('case_id', 'trace')}"
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=False)

    events: list[dict[str, Any]] = []
    neuron_ids: dict[str, int] = {}
    previous_position = list(case["initial_state"]["position_xy"])
    previous_velocity = list(case["initial_state"].get("velocity_xy", [0.0, 0.0]))
    goal = list(case["initial_state"]["goal_xy"])
    for seq, record in enumerate(records):
        position = list(record.get("position_xy", previous_position))
        velocity = list(record.get("velocity_xy", previous_velocity))
        action_id = int(record.get("action", 0))
        neural = []
        for index, membrane, spike, trace in zip(
            record.get("neuron_indices", []),
            record.get("v_before_reset", []),
            record.get("spike", []),
            record.get("trace_after_update", []),
        ):
            key = str(index)
            neuron_ids[key] = int(index)
            neural.append({
                "neuron_id": key,
                "neuron_index": int(index),
                "v_before_reset": float(membrane),
                "spike": int(spike),
                "trace_after_update": float(trace),
            })
        payload = {
            "env_index": 0,
            "episode_id": str(case.get("case_id", "episode-0")),
            "case_id": str(record.get("case_id", case.get("case_id", "case-0"))),
            "decision_step": int(record.get("decision_step", seq)),
            "policy_update": 0,
            "state_before": _state(previous_position, previous_velocity, goal),
            "observation": record.get("obs"),
            "encoded_observation": record.get("encoded_observation"),
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else "UNKNOWN",
            "action_selection": record.get("action_selection", "greedy"),
            "action_probabilities": record.get("action_probabilities"),
            "value_estimate": record.get("value_estimate"),
            "waypoint_xy_m": record.get("waypoint_xy_m"),
            "state_after": _state(position, velocity, goal),
            "reward_parts": record.get("reward_parts", {}),
            "reward": float(record.get("reward", 0.0)),
            "terminated": record.get("end_reason") in {"success", "boundary", "deadline"},
            "truncated": False,
            "end_reason": record.get("end_reason"),
            "sim_tick_before": int(record.get("sim_tick_before", seq * 50)),
            "sim_tick_after": int(record.get("sim_tick_after", (seq + 1) * 50)),
            "physics_path": record.get("physics_path"),
            "readout_snapshot": {
                "phase": "post_reset_final_substep",
                "neurons": neural,
            } if neural else None,
        }
        events.append({
            "schema_version": "1.0.0",
            "source_kind": "simulation_recorded",
            "run_id": f"trace-{case.get('case_id', 'unknown')}",
            "source_epoch": "trace-0",
            "seq": seq,
            "emitted_at_utc": _event_time(),
            "kind": "transition",
            "payload": payload,
        })
        previous_position, previous_velocity = position, velocity

    events_path = destination / "events.jsonl"
    events_path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
    neurons = {
        "schema_version": "1.0.0",
        "dataset_id": "male-v1",
        "dynamic_recorded": [
            {"neuron_id": key, "neuron_index": index, "roles": ["engineered_readout"],
             "cell_type": None, "region_labels": None, "position": None}
            for key, index in sorted(neuron_ids.items(), key=lambda item: item[1])
        ],
        "nodes": [],
    }
    (destination / "neurons.json").write_text(json.dumps(neurons, ensure_ascii=False, indent=2), encoding="utf-8")
    topology_enabled = False
    if graph and neuron_ids:
        topology_enabled = _export_topology(Path(graph), list(neuron_ids.values()), destination)
    manifest = {
        "schema_version": "1.0.0",
        "view_id": view_id,
        "source_kind": "simulation_recorded",
        "run_id": f"trace-{case.get('case_id', 'unknown')}",
        "source_epoch": "trace-0",
        "env_index": 0,
        "episode_count": 1,
        "event_count": len(events),
        "trace_complete": bool(events),
        "capabilities": {
            "live": False,
            "replay": bool(events),
            "neural_substeps": bool(neuron_ids),
            "physics_path": any(event["payload"]["physics_path"] for event in events),
            "topology": topology_enabled,
        },
        "files": {
            "events.jsonl": {"sha256": _sha256(events_path), "bytes": events_path.stat().st_size},
            "neurons.json": {"sha256": _sha256(destination / "neurons.json"), "bytes": (destination / "neurons.json").stat().st_size},
        },
    }
    if topology_enabled:
        topology_path = destination / "topology.json"
        manifest["files"]["topology.json"] = {"sha256": _sha256(topology_path), "bytes": topology_path.stat().st_size}
    (destination / "view_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
