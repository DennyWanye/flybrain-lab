from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spike import run_spike


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m flydrone.tellosim")
    sub = parser.add_subparsers(dest="command", required=True)
    spike = sub.add_parser("spike", help="run SDK, operation, and headless physics spikes")
    spike.add_argument("--world-config", type=Path)
    spike.add_argument("--out", type=Path, default=Path("reports/tellosim/spike.json"))
    args = parser.parse_args()
    if args.command == "spike":
        print(json.dumps(run_spike(args.world_config, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
