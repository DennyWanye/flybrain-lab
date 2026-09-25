from __future__ import annotations

import argparse
import json
from pathlib import Path

from .exporter import export_trace


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m flydrone.vis")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export a P1 trace as a read-only browser view")
    export.add_argument("--trace-dir", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--view-id")
    export.add_argument("--graph")
    args = parser.parse_args()
    if args.command == "export":
        destination = Path(args.out)
        manifest = export_trace(args.trace_dir, destination, args.view_id, args.graph)
        print(json.dumps({"view_id": manifest["view_id"], "out": str(destination), "events": manifest["event_count"]}, indent=2))


if __name__ == "__main__":
    main()
