from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from .physics import SimWorld, WorldConfig
from .sdk import OperationManager, OperationPhase, TelloCommand, decode_command


class SimTello:
    """Script-only Tello-like adapter backed by one MuJoCo world."""

    def __init__(self, world: SimWorld):
        self.world = world
        self.operations = OperationManager()
        self.sim_tick = 0
        self.sdk_mode = False
        self.airborne = False
        self.phase = "grounded"
        self.speed_cm_s = 20
        self.source_kind = "simulation_recorded"
        self.events: list[dict[str, Any]] = []

    def _event(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, "sim_tick": self.sim_tick, **payload})

    def submit(self, text: str, request_id: str) -> dict[str, Any]:
        command = decode_command(text)
        if command.verb == "command":
            self.sdk_mode = True
            self._event("command", request_id=request_id, command=text, reply="ok")
            return {"phase": "ack_ok", "device_execution": "completed", "raw_response": "ok"}
        if not self.sdk_mode:
            raise RuntimeError("sdk_mode_required")
        if command.verb == "takeoff" and self.airborne:
            raise RuntimeError("already_airborne")
        if command.verb not in {"takeoff", "land", "stop", "speed", "forward", "back", "left", "right", "cw", "ccw"}:
            raise RuntimeError("unsupported_command")
        if command.verb in {"forward", "back", "left", "right", "cw", "ccw"} and not self.airborne:
            raise RuntimeError("not_airborne")
        operation = self.operations.submit(command, request_id=request_id, sim_tick=self.sim_tick)
        self._event("operation", operation_id=operation.operation_id, request_id=request_id,
                    command=text, phase=operation.phase.value, device_execution=operation.device_execution.value)
        self.operations.device_started(operation.operation_id, sim_tick=self.sim_tick)
        self._execute(operation.operation_id, command)
        completed = self.operations.device_completed(operation.operation_id, sim_tick=self.sim_tick)
        final = self.operations.receive_ack(operation.operation_id, sim_tick=self.sim_tick, ok=True, raw_response="ok")
        self._event("operation", operation_id=final.operation_id, request_id=request_id,
                    command=text, phase=final.phase.value, device_execution=completed.device_execution.value,
                    sent_tick=final.sent_tick, event_tick=final.event_tick, raw_response=final.raw_response)
        return {"operation_id": final.operation_id, "phase": final.phase.value,
                "device_execution": completed.device_execution.value, "raw_response": final.raw_response}

    def _execute(self, operation_id: str, command: TelloCommand) -> None:
        if command.verb == "takeoff":
            self.airborne = True
            self.phase = "taking_off"
            self.world.set_target((float(self.world.position[0]), float(self.world.position[1]), self.world.config.takeoff_height_m))
            self._step_for_seconds(2.0)
            self.phase = "airborne"
            return
        if command.verb == "land":
            self.phase = "landing"
            self.world.set_target((float(self.world.position[0]), float(self.world.position[1]), self.world.config.floor_z_m + 0.045))
            self._step_for_seconds(5.0)
            self.world.hold()
            self._step_for_seconds(1.0)
            self.airborne = False
            self.phase = "grounded"
            return
        if command.verb == "stop":
            self.world.hold()
            self._step_for_seconds(0.5)
            return
        if command.verb == "speed":
            self.speed_cm_s = command.args[0]
            return
        if command.verb in {"forward", "back", "left", "right"}:
            distance = command.args[0] / 100.0
            yaw = self.world.yaw_rad
            signs = {"forward": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
            local_x, local_y = signs[command.verb]
            dx = math.cos(yaw) * local_x - math.sin(yaw) * local_y
            dy = math.sin(yaw) * local_x + math.cos(yaw) * local_y
            position = self.world.position
            self.world.set_target((position[0] + dx * distance, position[1] + dy * distance, self.world.config.takeoff_height_m))
            self._step_for_seconds(max(distance / max(self.speed_cm_s / 100.0, 0.01) + 1.0, 1.0))
            return
        if command.verb in {"cw", "ccw"}:
            delta = math.radians(command.args[0]) * (-1.0 if command.verb == "cw" else 1.0)
            target = self.world.yaw_rad + delta
            self.world.set_yaw(target)
            self._step_for_seconds(abs(delta) / math.radians(45.0) + 0.5)

    def _step_for_seconds(self, seconds: float) -> None:
        ticks = max(1, math.ceil(seconds * self.world.config.physics_hz))
        for row in self.world.step(ticks):
            self.sim_tick = int(row["sim_tick"])
            self._event("trajectory", **row, yaw_rad=self.world.yaw_rad, airborne=self.airborne)


def run_script(commands_path: str | Path, world_config: str | Path, out: str | Path) -> dict[str, Any]:
    commands_data = json.loads(Path(commands_path).read_text(encoding="utf-8"))
    config = WorldConfig.from_json(world_config)
    world = SimWorld(config)
    device = SimTello(world)
    for index, item in enumerate(commands_data["commands"]):
        command = TelloCommand(str(item["verb"]), tuple(int(value) for value in item.get("args", [])))
        device.submit(" ".join([command.verb, *(str(value) for value in command.args)]), f"script-{index}")
    result = {
        "schema_version": "tellosim.script_run/1.0",
        "source_kind": "simulation_recorded",
        "policy_source": "script_not_learned_policy",
        "real_device_allowed": False,
        "world_config": str(world_config),
        "events": device.events,
        "final_state": {"position_m": world.position.tolist(), "velocity_mps": world.velocity.tolist(),
                         "yaw_rad": world.yaw_rad, "airborne": device.airborne, "sim_tick": device.sim_tick},
    }
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
