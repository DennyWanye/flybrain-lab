from __future__ import annotations

import json
import mimetypes
import os
import re
import base64
import hashlib
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


VIEW_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Viewer:
    def __init__(self, project_root: Path, registry_root: Path):
        self.project_root = project_root.resolve()
        self.registry_root = registry_root.resolve()
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_root / "index.json"

    def registry(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid registry: {exc}") from exc
        return value if isinstance(value, dict) else {}

    def view_dir(self, view_id: str) -> Path:
        if not VIEW_ID_RE.fullmatch(view_id):
            raise KeyError("invalid view id")
        raw = self.registry().get(view_id)
        if not isinstance(raw, str):
            raise KeyError("unknown view")
        path = Path(raw).resolve()
        allowed_roots = (self.registry_root, self.project_root / "runs", self.project_root / "reports" / "vis")
        if not any(path.is_relative_to(root.resolve()) for root in allowed_roots):
            raise PermissionError("view path is outside registry root")
        if not path.is_dir():
            raise FileNotFoundError("view directory does not exist")
        return path

    def manifest(self, view_id: str) -> dict:
        directory = self.view_dir(view_id)
        path = directory / ("live_descriptor.json" if (directory / "live_descriptor.json").exists() else "view_manifest.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def events(self, view_id: str, limit: int = 256, kind: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 256))
        path = self.view_dir(view_id) / "events.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    event = json.loads(line)
                    if kind and event.get("kind") != kind:
                        continue
                    rows.append(event)
                    if len(rows) >= limit:
                        break
        return rows

    def neurons(self, view_id: str) -> dict:
        path = self.view_dir(view_id) / "neurons.json"
        if not path.exists():
            return {"dynamic_recorded": [], "nodes": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def topology(self, view_id: str, center_id: str | None = None) -> dict:
        path = self.view_dir(view_id) / "topology.json"
        topology = json.loads(path.read_text(encoding="utf-8"))
        if center_id is None:
            return topology
        center_id = str(center_id)
        node_ids = {str(node_id) for node_id in topology.get("node_ids", [])}
        if center_id not in node_ids:
            return {**topology, "center_id": center_id, "node_ids": [], "edges": [], "truncated": False,
                    "selection_status": "not_in_export"}
        edges = [edge for edge in topology.get("edges", [])
                 if str(edge.get("source_id")) == center_id or str(edge.get("target_id")) == center_id]
        selected_nodes = {center_id}
        for edge in edges:
            selected_nodes.add(str(edge.get("source_id")))
            selected_nodes.add(str(edge.get("target_id")))
        return {**topology, "center_id": center_id, "node_ids": sorted(selected_nodes), "edges": edges,
                "selection_status": "matched"}

    def metrics(self, view_id: str) -> dict:
        directory = self.view_dir(view_id)
        result: dict = {"view": {"status": "not_recorded"}}
        for name in ("metrics.json", "live.metrics.json"):
            path = directory / name
            if path.exists():
                result["view"] = json.loads(path.read_text(encoding="utf-8"))
                break

        validation: list[dict] = []
        report_root = self.project_root / "reports" / "p1"
        for summary_path in sorted(report_root.glob("*-validation/summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                end_reasons = summary.get("end_reasons") or {}
                count = int(summary.get("count", 0))
                successes = int(end_reasons.get("success", round(float(summary.get("success_rate", 0)) * count)))
                validation.append({
                    "run_id": summary_path.parent.name.removesuffix("-validation"),
                    "split": "validation",
                    "successes": successes,
                    "episodes": count,
                    "success_rate": (successes / count) if count else None,
                    "end_reasons": end_reasons,
                    "source": str(summary_path.relative_to(self.project_root)),
                })
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

        training: list[dict] = []
        run_root = self.project_root / "runs" / "flydrone"
        for train_path in sorted(run_root.glob("*/train.jsonl")):
            updates = 0
            last: dict = {}
            try:
                with train_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            value = json.loads(line)
                            if isinstance(value, dict):
                                updates += 1
                                last = value
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if updates:
                training.append({
                    "run_id": train_path.parent.name,
                    "updates": updates,
                    "env_steps": last.get("env_steps"),
                    "mean_return": last.get("mean_return"),
                    "source": str(train_path.relative_to(self.project_root)),
                })

        sealed_test: dict = {"status": "not_recorded"}
        case_manifest = report_root / "cases" / "manifest.json"
        if case_manifest.exists():
            try:
                value = json.loads(case_manifest.read_text(encoding="utf-8"))
                sealed_test = {
                    "status": "sealed",
                    "split": "test",
                    "episodes": int((value.get("counts") or {}).get("test", 0)),
                    "source": str(case_manifest.relative_to(self.project_root)),
                    "note": "封存测试集只记录规模和哈希；页面不触发评估。",
                }
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        result["training_windows"] = training
        result["validation"] = validation
        result["sealed_test"] = sealed_test
        return result


STATIC_ROOT = Path(__file__).with_name("static")


class Handler(BaseHTTPRequestHandler):
    server_version = "FlyBrainView/0.1"

    @property
    def viewer(self) -> Viewer:
        return self.server.viewer  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, _json_bytes({"error": message}))

    def _ws_frame(self, payload: bytes) -> bytes:
        length = len(payload)
        if length < 126:
            return bytes([0x81, length]) + payload
        if length < 65536:
            return bytes([0x81, 126]) + length.to_bytes(2, "big") + payload
        return bytes([0x81, 127]) + length.to_bytes(8, "big") + payload

    def _websocket(self, view_id: str) -> None:
        if self.headers.get("Upgrade", "").lower() != "websocket":
            self._error(400, "websocket upgrade required")
            return
        origin = self.headers.get("Origin", "")
        if origin and origin not in {"http://127.0.0.1:8765", "http://localhost:8765"}:
            self._error(403, "origin is not allowed")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self._error(400, "missing websocket key")
            return
        try:
            self.viewer.view_dir(view_id)
        except (KeyError, FileNotFoundError, PermissionError) as exc:
            self._error(404, str(exc))
            return
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        last_seq = None
        deadline = time.monotonic() + 3600
        try:
            while time.monotonic() < deadline:
                directory = self.viewer.view_dir(view_id)
                latest = directory / "live.latest.json"
                if latest.exists():
                    event = json.loads(latest.read_text(encoding="utf-8"))
                    if event.get("seq") != last_seq:
                        self.connection.sendall(self._ws_frame(_json_bytes(event)))
                        last_seq = event.get("seq")
                time.sleep(0.2)
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError, json.JSONDecodeError):
            return

    def do_POST(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "viewer is read-only")

    do_PUT = do_POST
    do_DELETE = do_POST

    def do_GET(self) -> None:  # noqa: N802
        host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
        if host not in {"127.0.0.1", "localhost"}:
            self._error(403, "host is not allowed")
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        ws_match = re.fullmatch(r"/ws/views/([^/]+)", path)
        if ws_match:
            self._websocket(ws_match.group(1))
            return
        try:
            if path == "/healthz":
                self._send(200, _json_bytes({"ok": True, "service": "flyview", "read_only": True}))
                return
            if path == "/api/views":
                views = []
                for view_id in sorted(self.viewer.registry()):
                    try:
                        manifest = self.viewer.manifest(view_id)
                        views.append({"view_id": view_id, "source_kind": manifest.get("source_kind"),
                                      "capabilities": manifest.get("capabilities", {}),
                                      "event_count": manifest.get("event_count", 0)})
                    except Exception as exc:  # corrupt views appear as explicit errors
                        views.append({"view_id": view_id, "status": "ERROR", "error": str(exc)})
                self._send(200, _json_bytes({"views": views}))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/manifest", path)
            if match:
                self._send(200, _json_bytes(self.viewer.manifest(match.group(1))))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/events", path)
            if match:
                query = parse_qs(parsed.query)
                limit = query.get("limit", ["256"])[0]
                kind = query.get("kind", [None])[0]
                self._send(200, _json_bytes({"events": self.viewer.events(match.group(1), int(limit), kind), "source_complete": True}))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/latest", path)
            if match:
                directory = self.viewer.view_dir(match.group(1))
                live = directory / "live.latest.json"
                if live.exists():
                    self._send(200, live.read_bytes())
                else:
                    rows = self.viewer.events(match.group(1), 1, "transition")
                    self._send(200, _json_bytes(rows[0] if rows else {"event": None}))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/neurons", path)
            if match:
                self._send(200, _json_bytes(self.viewer.neurons(match.group(1))))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/topology", path)
            if match:
                query = parse_qs(parsed.query)
                center_id = query.get("center_id", [None])[0]
                self._send(200, _json_bytes(self.viewer.topology(match.group(1), center_id)))
                return
            match = re.fullmatch(r"/api/views/([^/]+)/metrics", path)
            if match:
                self._send(200, _json_bytes(self.viewer.metrics(match.group(1))))
                return
            if path == "/" or path == "/index.html":
                body = (STATIC_ROOT / "index.html").read_bytes()
                self._send(200, body, "text/html; charset=utf-8")
                return
            if path.startswith("/static/"):
                asset = (STATIC_ROOT / path.removeprefix("/static/")).resolve()
                if not asset.is_relative_to(STATIC_ROOT.resolve()) or not asset.is_file():
                    raise FileNotFoundError
                self._send(200, asset.read_bytes(), mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
                return
            self._error(404, "not found")
        except KeyError as exc:
            self._error(404, str(exc))
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(422, str(exc))
        except PermissionError as exc:
            self._error(403, str(exc))
        except FileNotFoundError:
            self._error(404, "view artifact not found")
        except Exception as exc:
            self._error(500, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        print(f"[flyview] {format % args}", flush=True)


def serve(project_root: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = Path(project_root).resolve()
    viewer = Viewer(root, root / "reports" / "vis" / "views")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.viewer = viewer  # type: ignore[attr-defined]
    print(f"FlyBrain viewer: http://{host}:{port}", flush=True)
    print("Read-only mode: browser requests cannot control training or hardware.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
