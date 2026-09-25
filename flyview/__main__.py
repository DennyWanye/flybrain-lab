from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .server import serve


def register(project_root: Path, view_id: str, source_dir: Path | None = None, live_dir: Path | None = None) -> None:
    if not view_id.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("view id must contain only letters, numbers, '_' or '-'")
    if bool(source_dir) == bool(live_dir):
        raise SystemExit("provide exactly one of --source-dir or --live-dir")
    source_dir = source_dir.resolve() if source_dir else live_dir.resolve()
    if live_dir is None and not (source_dir / "view_manifest.json").is_file():
        raise SystemExit("source directory must contain view_manifest.json")
    if live_dir is not None and not (source_dir / "live_descriptor.json").is_file():
        raise SystemExit("live directory must contain live_descriptor.json")
    root = (project_root / "reports" / "vis" / "views").resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / view_id
    if destination.exists():
        raise SystemExit(f"refusing to overwrite registered view: {view_id}")
    if live_dir is None:
        shutil.copytree(source_dir, destination)
        registered_path = destination
    else:
        registered_path = source_dir
    index = root / "index.json"
    registry = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    registry[view_id] = str(registered_path)
    index.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"view_id": view_id, "path": str(destination)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m flyview")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_cmd = sub.add_parser("serve")
    serve_cmd.add_argument("--project-root", default=".")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8765)
    reg = sub.add_parser("register")
    reg.add_argument("--project-root", default=".")
    reg.add_argument("--view-id", required=True)
    reg.add_argument("--source-dir")
    reg.add_argument("--live-dir")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args.project_root, args.host, args.port)
    else:
        register(Path(args.project_root), args.view_id,
                 Path(args.source_dir) if args.source_dir else None,
                 Path(args.live_dir) if args.live_dir else None)


if __name__ == "__main__":
    main()
