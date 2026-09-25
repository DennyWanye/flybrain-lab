from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps({k: str(v) for k, v in payload.items() if k != "policy_state_dict"}, sort_keys=True).encode()).hexdigest()


def save_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload); body["manifest_sha256"] = _digest(body)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(body, tmp)
    os.replace(tmp, path)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({"manifest_sha256": body["manifest_sha256"]}, indent=2))


def load_checkpoint(path: str | Path, *, action_schema: dict | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if not payload.get("sim_only") or payload.get("real_flight_authorized"):
        raise ValueError("checkpoint is not an offline simulation checkpoint")
    if action_schema is not None and payload.get("action_schema") != action_schema:
        raise ValueError("checkpoint action schema mismatch")
    expected = payload.get("manifest_sha256")
    if expected and expected != _digest(payload):
        raise ValueError("checkpoint manifest mismatch")
    return payload
