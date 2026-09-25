#!/usr/bin/env python3
"""Run the first TS1 SDK and MuJoCo spikes without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flydrone.tellosim.spike import run_spike


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/tellosim/spike.json"))
    args = parser.parse_args()
    print(json.dumps(run_spike(args.world_config, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
