from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .physics import SimWorld, WorldConfig
from .sdk import CommandError, OperationManager, decode_command, encode_command


def run_spike(world_config: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    config = WorldConfig.from_json(world_config) if world_config else WorldConfig()
    accepted = ["command", "takeoff", "speed 20", "forward 40", "cw 90", "stop", "land"]
    commands = [encode_command(decode_command(value)) for value in accepted]
    assert encode_command(decode_command("  forward   40  ")) == "forward 40"
    rejected: dict[str, str] = {}
    for value in ("forward 15", "go 20 0 0 20", "rc 0 0 0 0"):
        try:
            decode_command(value)
        except CommandError as exc:
            rejected[value] = exc.code
        else:
            raise AssertionError(f"command unexpectedly accepted: {value}")

    operations = OperationManager()
    operation = operations.submit(decode_command("forward 40"), request_id="request-1", sim_tick=0)
    operations.device_started(operation.operation_id, sim_tick=1)
    operations.device_completed(operation.operation_id, sim_tick=240)
    completed_without_ack = operations.mark_unknown(operation.operation_id, sim_tick=241)
    second = operations.submit(decode_command("stop"), request_id="request-2", sim_tick=241)
    unknown = operations.mark_unknown(second.operation_id, sim_tick=250)

    world = SimWorld(config)
    world.set_target((0.8, 0.0, config.takeoff_height_m))
    trajectory = world.step(config.physics_hz * 5)
    final = trajectory[-1]
    assert final["x_m"] > 0.2, final
    assert abs(final["z_m"] - config.takeoff_height_m) < 0.35, final
    result: dict[str, Any] = {
        "sdk_codec": {"accepted": commands, "rejected": rejected},
        "operation_state": {
            "completed_without_ack": {
                "operation_id": operation.operation_id,
                "device_execution": completed_without_ack.device_execution.value,
                "phase": completed_without_ack.phase.value,
            },
            "reply_loss": {
                "operation_id": unknown.operation_id,
                "device_execution": unknown.device_execution.value,
                "phase": unknown.phase.value,
            },
        },
        "physics": {
            "mujoco_version": __import__("mujoco").__version__,
            "physics_hz": config.physics_hz,
            "steps": len(trajectory),
            "final": final,
            "target": world.target.tolist(),
            "controller_model": "bounded_body_axis_thrust_surrogate",
            "calibration": "NOT_TELLO_CALIBRATED",
        },
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
