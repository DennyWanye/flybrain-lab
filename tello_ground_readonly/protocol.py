"""Strict protocol helpers for the Tello ground-observation boundary.

This module deliberately does not contain movement, landing, emergency, or
stream-control operations. Mission-pad fields are parsed from telemetry only;
the collector never enables mission-pad mode on the aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any


# ``command`` only enters SDK mode and does not request motion. It is kept
# separate in the audit record from the actual read-only query commands.
READ_ONLY_COMMANDS = frozenset({"command", "sdk?", "battery?", "hardware?", "time?", "speed?"})
MISSION_PAD_SETUP_COMMANDS = frozenset({"mon", "mdirection 0"})
FORBIDDEN_COMMANDS = frozenset({
    "takeoff", "land", "stop", "emergency", "rc", "up", "down", "left", "right",
    "forward", "back", "backward", "cw", "ccw", "go", "curve", "flip", "throwfly",
    "mon", "moff", "mdirection", "streamon", "streamoff", "wifi", "ap", "reboot",
    "motoron", "motoroff", "speed", "port", "EXT",
})

_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def command_is_allowed(command: str) -> bool:
    """Return whether *command* is in the exact ground-only allowlist."""
    return isinstance(command, str) and command.strip() in READ_ONLY_COMMANDS


def validate_mission_pad_command(command: str) -> str:
    if not isinstance(command, str) or command != command.strip() or "\n" in command or "\r" in command:
        raise ValueError("mission-pad command must be one ASCII line")
    if command not in MISSION_PAD_SETUP_COMMANDS:
        raise PermissionError(f"mission-pad setup command denied: {command!r}")
    command.encode("ascii")
    return command


def validate_command(command: str) -> str:
    if not isinstance(command, str) or command != command.strip() or "\n" in command or "\r" in command:
        raise ValueError("command must be one ASCII line without surrounding whitespace")
    if not command_is_allowed(command):
        raise PermissionError(f"command denied by ground-readonly policy: {command!r}")
    try:
        command.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("command must be ASCII") from exc
    return command


def _number(text: str) -> int | float | str:
    value = text.strip()
    if not value:
        return value
    try:
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
            return int(value)
        number = float(value)
        if math.isfinite(number):
            return number
    except ValueError:
        pass
    return value


def parse_state_fields(payload: str) -> dict[str, Any]:
    """Parse semicolon-delimited Tello state while retaining raw values.

    Unknown fields are preserved as typed values. Duplicate fields are rejected
    because accepting the last value would hide a malformed or spoofed packet.
    """
    if not isinstance(payload, str):
        raise TypeError("state payload must be text")
    fields: dict[str, Any] = {}
    for item in payload.strip().split(";"):
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"malformed state field: {item!r}")
        key, value = item.split(":", 1)
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid state field name: {key!r}")
        if key in fields:
            raise ValueError(f"duplicate state field: {key}")
        fields[key] = _number(value)
    if not fields:
        raise ValueError("empty state payload")
    return fields


def _finite_number(fields: dict[str, Any], key: str) -> float | None:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class MissionPadObservation:
    valid: bool
    card_id: int | None
    position_m: tuple[float, float, float] | None
    frame_id: str
    reason: str | None


@dataclass(frozen=True)
class TelloState:
    raw: str
    fields: dict[str, Any]
    mission_pad: MissionPadObservation
    height_m: float | None
    battery_fraction: float | None
    airborne: bool

    def shadow_pose(self) -> dict[str, Any]:
        """Return a deliberately invalid room pose for Shadow V1.

        Mission-pad XYZ is relative to the card and is not a calibrated room
        frame, so it must never be presented as global position to the model.
        """
        return {
            "captured_ns": None,
            "received_ns": None,
            "valid": False,
            "frame_id": "tello_unmapped",
            "position_m": None,
            "velocity_mps": None,
            "yaw_rad": None,
            "mission_pad": {
                "valid": self.mission_pad.valid,
                "card_id": self.mission_pad.card_id,
                "position_m": self.mission_pad.position_m,
                "frame_id": self.mission_pad.frame_id,
                "reason": self.mission_pad.reason,
            },
        }


def parse_state_packet(payload: str | bytes) -> TelloState:
    """Parse one raw state datagram without inventing unavailable telemetry."""
    if isinstance(payload, bytes):
        raw = payload.decode("ascii", errors="strict")
    else:
        raw = payload
    fields = parse_state_fields(raw)
    mid = _finite_number(fields, "mid")
    x, y, z = (_finite_number(fields, key) for key in ("x", "y", "z"))
    if mid is None:
        pad = MissionPadObservation(False, None, None, "mission_pad_relative_cm", "mid_missing")
    elif mid != int(mid):
        pad = MissionPadObservation(False, None, None, "mission_pad_relative_cm", "card_id_noninteger")
    elif mid < 0:
        pad = MissionPadObservation(False, int(mid), None, "mission_pad_relative_cm", "no_card")
    elif any(v is None for v in (x, y, z)):
        pad = MissionPadObservation(False, int(mid), None, "mission_pad_relative_cm", "card_coordinates_missing")
    else:
        # Tello mission-pad XYZ is centimetres relative to the detected pad.
        pad = MissionPadObservation(True, int(mid), (x / 100.0, y / 100.0, z / 100.0),
                                    "mission_pad_relative", None)
    height = _finite_number(fields, "h")
    battery = _finite_number(fields, "bat")
    height_m = None if height is None else height / 100.0
    battery_fraction = None if battery is None else max(0.0, min(1.0, battery / 100.0))
    return TelloState(raw=raw, fields=fields, mission_pad=pad, height_m=height_m,
                      battery_fraction=battery_fraction, airborne=bool(height_m is not None and height_m > 0.03))
