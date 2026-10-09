"""CLI for ground-only TT/Tello observation collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .collector import CollectorConfig, ReadOnlyCollector, state_to_shadow_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-command")
    check.add_argument("value")
    collect = sub.add_parser("collect")
    collect.add_argument("--output", required=True, type=Path)
    collect.add_argument("--host", default="192.168.10.1")
    collect.add_argument("--duration-s", type=float, default=5.0)
    collect.add_argument("--timeout-s", type=float, default=2.0)
    collect.add_argument("--no-command", action="store_true", help="do not enter SDK mode")
    collect.add_argument("--no-hardware-query", action="store_true")
    collect.add_argument("--enable-mission-pad", action="store_true")
    shadow = sub.add_parser("shadow-jsonl")
    shadow.add_argument("--session", required=True, type=Path)
    shadow.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "check-command":
        from .protocol import validate_command
        try:
            print(json.dumps({"allowed": True, "command": validate_command(args.value)}))
        except Exception as exc:
            print(json.dumps({"allowed": False, "error": f"{type(exc).__name__}: {exc}"}))
            return 2
        return 0
    if args.command == "collect":
        config = CollectorConfig(host=args.host, timeout_s=args.timeout_s, state_window_s=args.duration_s)
        commands = ["sdk?", "battery?"]
        if not args.no_hardware_query:
            commands.append("hardware?")
        result = ReadOnlyCollector(config).run(session_dir=args.output, commands=commands,
                                               state_duration_s=args.duration_s,
                                               handshake=not args.no_command,
                                               enable_mission_pad=args.enable_mission_pad)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    if args.command == "shadow-jsonl":
        raw_path = args.session / "udp_raw.jsonl"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        profile_path = args.output.with_name("shadow-profile.json")
        count = 0
        base_now_ns = time.time_ns()
        with raw_path.open(encoding="utf-8") as source, args.output.open("x", encoding="utf-8") as dest:
            for line in source:
                row = json.loads(line)
                if row.get("kind") != "state_packet" or not row.get("accepted"):
                    continue
                # Replay preserves the Shadow contract's 10 Hz chronology;
                # wall-clock time is irrelevant to an offline recording.
                now_ns = base_now_ns + count * 100_000_000
                sample = state_to_shadow_sample(row, seq=count, now_ns=now_ns)
                dest.write(json.dumps(sample, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                count += 1
        profile_path.write_text(json.dumps({
            "source_id": "tello-ground-readonly-v1",
            "provenance": "real_recording",
            "source_frame": "tello_unmapped",
            "yaw_to_room_rad": 0.0,
            "translation_m": [0.0, 0.0, 0.0],
            "max_age_s": 0.2,
            "clock_uncertainty_s": 0.02,
        }, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "profile": str(profile_path), "samples": count,
                          "transmitted": False, "real_flight_ready": False}))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
