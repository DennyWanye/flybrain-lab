"""UDP collector for read-only RoboMaster TT ground observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any, Iterable

from .protocol import TelloState, parse_state_packet, validate_command, validate_mission_pad_command


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CollectorConfig:
    host: str = "192.168.10.1"
    command_port: int = 8889
    state_port: int = 8890
    local_state_host: str = "0.0.0.0"
    timeout_s: float = 2.0
    state_window_s: float = 5.0
    allowed_peer: str = "192.168.10.1"

    def __post_init__(self) -> None:
        if self.host != self.allowed_peer:
            raise ValueError("collector is pinned to the TT peer 192.168.10.1")
        if self.command_port != 8889 or self.state_port != 8890:
            raise ValueError("unexpected Tello ports")
        if not 0.05 <= self.timeout_s <= 10 or not 0.1 <= self.state_window_s <= 3600:
            raise ValueError("timeouts out of range")


class ReadOnlyCollector:
    """Collect command replies and state datagrams without any flight control."""

    def __init__(self, config: CollectorConfig | None = None):
        self.config = config or CollectorConfig()
        self._closed = False

    def _peer_check(self, address: tuple[str, int]) -> None:
        if address[0] != self.config.allowed_peer:
            raise PermissionError(f"unexpected UDP peer {address[0]}; packet ignored")

    def query(self, command: str, *, session_dir: Path | None = None,
              mission_pad_setup: bool = False) -> dict[str, Any]:
        command = validate_mission_pad_command(command) if mission_pad_setup else validate_command(command)
        sent_ns = time.time_ns()
        received_ns: int | None = None
        raw_response: bytes = b""
        error: str | None = None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(self.config.timeout_s)
            sock.sendto(command.encode("ascii"), (self.config.host, self.config.command_port))
            try:
                while True:
                    payload, address = sock.recvfrom(4096)
                    self._peer_check(address)
                    raw_response = payload
                    received_ns = time.time_ns()
                    break
            except socket.timeout:
                error = "timeout"
        row = {
            "kind": "command_reply",
            "command": command,
            "sent_ns": sent_ns,
            "received_ns": received_ns,
            "peer": self.config.host,
            "raw_response_hex": raw_response.hex(),
            "raw_response_ascii": raw_response.decode("ascii", errors="replace"),
            "error": error,
            "transmitted": True,
            "flight_control": False,
            "mission_pad_setup": mission_pad_setup,
        }
        self._append(session_dir, row)
        return row

    def collect_state(self, *, duration_s: float | None = None,
                      session_dir: Path | None = None) -> list[dict[str, Any]]:
        duration = self.config.state_window_s if duration_s is None else duration_s
        if not 0.1 <= duration <= 3600:
            raise ValueError("state collection duration out of range")
        rows: list[dict[str, Any]] = []
        deadline = time.monotonic() + duration
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.config.local_state_host, self.config.state_port))
            sock.settimeout(min(0.5, duration))
            while time.monotonic() < deadline:
                try:
                    payload, address = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                received_ns = time.time_ns()
                if address[0] != self.config.allowed_peer:
                    row = {"kind": "state_packet", "received_ns": received_ns,
                           "peer": address[0], "accepted": False,
                           "error": "unexpected_peer", "raw_response_hex": payload.hex()}
                    self._append(session_dir, row)
                    continue
                try:
                    state = parse_state_packet(payload)
                    row = self._state_row(state, payload, received_ns, address)
                except Exception as exc:
                    row = {"kind": "state_packet", "received_ns": received_ns,
                           "peer": address[0], "accepted": False,
                           "error": f"{type(exc).__name__}: {exc}",
                           "raw_response_hex": payload.hex(),
                           "raw_response_ascii": payload.decode("ascii", errors="replace")}
                self._append(session_dir, row)
                rows.append(row)
        return rows

    @staticmethod
    def _state_row(state: TelloState, payload: bytes, received_ns: int,
                   address: tuple[str, int]) -> dict[str, Any]:
        return {
            "kind": "state_packet",
            "received_ns": received_ns,
            "peer": address[0],
            "peer_port": address[1],
            "accepted": True,
            "raw_response_hex": payload.hex(),
            "raw_response_ascii": state.raw,
            "fields": state.fields,
            "height_m": state.height_m,
            "battery_fraction": state.battery_fraction,
            "airborne": state.airborne,
            "mission_pad": asdict(state.mission_pad),
            "world_pose": None,
            "real_flight_ready": False,
        }

    @staticmethod
    def _append(session_dir: Path | None, row: dict[str, Any]) -> None:
        if session_dir is None:
            return
        session_dir.mkdir(parents=True, exist_ok=True)
        with (session_dir / "udp_raw.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(row) + "\n")
            stream.flush()

    def run(self, *, session_dir: Path, commands: Iterable[str] = ("command", "sdk?", "battery?", "hardware?"),
            state_duration_s: float | None = None, handshake: bool = True,
            enable_mission_pad: bool = False) -> dict[str, Any]:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=False)
        started_ns = time.time_ns()
        manifest = {
            "schema": "flybrain.tello.ground-readonly.session/1",
            "started_ns": started_ns,
            "config": asdict(self.config),
            "policy": {
                "real_flight_ready": False,
                "transmitted_flight_commands": 0,
                "mission_pad_mode_enabled": False,
                "world_pose_available": False,
                "shadow_actions_executed": False,
            },
        }
        (session_dir / "session.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        replies: list[dict[str, Any]] = []
        # Bind state reception before SDK mode/query traffic so the first
        # telemetry datagrams cannot be lost during handshake.
        state_rows: list[dict[str, Any]] = []
        state_error: list[BaseException] = []
        state_thread = threading.Thread(
            target=self._collect_state_thread,
            args=(state_rows, state_error, state_duration_s, session_dir),
            daemon=True,
            name="tello-readonly-state",
        )
        state_thread.start()
        time.sleep(0.05)
        if handshake:
            replies.append(self.query("command", session_dir=session_dir))
            # Let the device switch its command/state services before the
            # read-only queries. This is not a retry and does not send motion.
            time.sleep(0.25)
        if enable_mission_pad:
            replies.append(self.query("mon", session_dir=session_dir, mission_pad_setup=True))
            replies.append(self.query("mdirection 0", session_dir=session_dir, mission_pad_setup=True))
            manifest["policy"]["mission_pad_mode_enabled"] = True
            (session_dir / "session.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        for command in commands:
            if command == "command" and handshake:
                continue
            replies.append(self.query(command, session_dir=session_dir))
        state_thread.join()
        if state_error:
            raise state_error[0]
        states = state_rows
        result = {
            "schema": "flybrain.tello.ground-readonly.result/1",
            "finished_ns": time.time_ns(),
            "replies": replies,
            "accepted_state_packets": len(states),
            "mission_pad_packets": sum(int(bool(r.get("mission_pad", {}).get("valid"))) for r in states),
            "mission_pad_mode_enabled": enable_mission_pad,
            "transmitted_flight_commands": 0,
            "shadow_transmitted": 0,
            "real_flight_ready": False,
        }
        (session_dir / "result.json").write_text(_json(result) + "\n", encoding="utf-8")
        return result

    def _collect_state_thread(self, rows: list[dict[str, Any]], errors: list[BaseException],
                              duration_s: float | None, session_dir: Path) -> None:
        try:
            self.collect_state(duration_s=duration_s, session_dir=session_dir)
            # collect_state appends records as they arrive. Keep the same list
            # for the final summary without re-reading the raw file.
            with (session_dir / "udp_raw.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    if item.get("kind") == "state_packet":
                        rows.append(item)
        except BaseException as exc:  # propagated by run after join
            errors.append(exc)


def state_to_shadow_sample(row: dict[str, Any], *, seq: int, now_ns: int,
                           goal_m: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> dict[str, Any]:
    """Convert a ground state row to a safe Shadow V1 sample.

    The Tello's mission-pad-relative coordinates remain diagnostic metadata and
    are intentionally not copied into ``pose``. Consequently every sample is
    blocked until an independently calibrated room pose provider is attached.
    """
    if not row.get("accepted"):
        raise ValueError("cannot make Shadow sample from rejected state packet")
    captured_ns = int(row["received_ns"])
    return {
        "schema": "flybrain.shadow.sample/1",
        "seq": seq,
        "time_ns": now_ns,
        "pose": {
            "captured_ns": captured_ns,
            "received_ns": now_ns,
            "valid": False,
            "frame_id": "tello_unmapped",
            "position_m": None,
            "velocity_mps": None,
            "yaw_rad": None,
        },
        "state": {
            "captured_ns": captured_ns,
            "received_ns": now_ns,
            "valid": row.get("height_m") is not None and row.get("battery_fraction") is not None,
            "height_m": row.get("height_m"),
            "battery_fraction": row.get("battery_fraction"),
            "airborne": bool(row.get("airborne", False)),
        },
        "context": {
            "phase": "altitude",
            "goal_m": list(goal_m),
            "target_yaw_rad": 0.0,
            "previous_action": None,
            "previous_duration_s": 0.0,
            "remaining_s": 60.0,
            "operation_status": "idle",
        },
    }
