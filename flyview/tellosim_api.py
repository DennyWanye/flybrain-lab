"""Authenticated local sandbox APIs, isolated from the legacy read-only viewer."""
from __future__ import annotations
import hashlib
import json
import re
import secrets
import threading
import os
from pathlib import Path
from urllib.parse import urlparse

from flydrone.tellosim.visual import VisualSession, import_golden, packed, atomic_json

ID = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


class Observatory:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "reports/vis/tellosim"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.sessions = {}
        self.lock = threading.RLock()
        for path in self.directory.glob("*/manifest.json"):
            previous=json.loads(path.read_text())
            if not previous.get("complete") and not self.owner_alive(previous):
                ticks=[c["last_tick"] for stream in previous.get("streams",{}).values() for c in stream]
                previous.update(partial=True,partial_reason="server_interrupted",outcome="interrupted",
                                duration_s=max(ticks,default=0)/120)
                atomic_json(path,previous)
        import_golden(root)

    def manifest(self, run_id):
        if not ID.fullmatch(run_id):
            raise ValueError("invalid run id")
        return json.loads((self.directory / run_id / "manifest.json").read_text())

    @staticmethod
    def owner_alive(manifest):
        pid=manifest.get('owner_pid')
        if not isinstance(pid,int) or pid<=0:return False
        try:os.kill(pid,0);return True
        except (ProcessLookupError,PermissionError):return False

    def catalog(self):
        rows = []
        for path in sorted(self.directory.glob("*/manifest.json")):
            if not ID.fullmatch(path.parent.name):
                continue
            m = json.loads(path.read_text())
            rows.append({**{k:m.get(k) for k in ("run_id", "epoch", "policy_source", "complete", "outcome", "duration_s", "partial", "observer_only", "training_group", "env_id", "env_count")},
                         "active":(m["run_id"] in self.sessions and not self.sessions[m["run_id"]].finished)
                                    or (not m.get('complete') and self.owner_alive(m)),
                         "external_live":bool(m.get('owner_pid'))})
        evaluation_path = self.root / "reports/golden_evaluation_final/summary.json"
        comparison = self.directory / "comparison.json"
        training = self.directory / 'training-summary.json'
        learning = self.directory / 'learning-summary.json'
        rigid = self.root / 'reports/ts1_rigid_v2/summary.json'
        joint=self.root/'reports/ts1_stability/summary.json'
        if not joint.exists():joint=self.root/'reports/ts1_robust/summary.json'
        if not joint.exists():joint=self.root/'reports/ts1_joint_refined/summary.json'
        if not joint.exists():joint=self.root/'reports/ts1_joint/summary.json'
        heading=self.root/'reports/ts1_heading/summary.json'
        c1=self.root/'reports/ts1_c1/summary.json'
        altitude=self.root/'reports/ts1_altitude/summary.json'
        return {"altitude":json.loads(altitude.read_text()) if altitude.exists() else None,"joint":json.loads(joint.read_text()) if joint.exists() else None,"heading":json.loads(heading.read_text()) if heading.exists() else None,"c1":json.loads(c1.read_text()) if c1.exists() else None,"rigid":json.loads(rigid.read_text()) if rigid.exists() else None,"runs":rows, "evaluation":json.loads(evaluation_path.read_text()) if evaluation_path.exists() else None,
                "comparison":json.loads(comparison.read_text()) if comparison.exists() else None,
                "training":json.loads(training.read_text()) if training.exists() else None,
                "learning":json.loads(learning.read_text()) if learning.exists() else None}

    def close(self):
        for session in list(self.sessions.values()):
            session.close("server_shutdown")
            if session.thread:
                session.thread.join(timeout=2)

    def handle(self, handler, post=False):
        path = urlparse(handler.path).path
        if not path.startswith("/api/tellosim/"):
            return False
        def send(value):
            handler._send(200, packed(value))
        try:
            port = handler.server.server_address[1]
            hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
            if handler.headers.get("Host") not in hosts:
                raise PermissionError("host is not allowed")
            origin = handler.headers.get("Origin")
            if origin is not None and origin not in {f"http://{h}" for h in hosts}:
                raise PermissionError("origin is not allowed")
            parts = path.removeprefix("/api/tellosim/").split("/")
            if post:
                if origin not in {f"http://{h}" for h in hosts}:
                    raise PermissionError("same-origin POST required")
                if not secrets.compare_digest(handler.headers.get("X-Sandbox-Token", ""), self.token):
                    raise PermissionError("sandbox token required")
                if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("JSON body required")
                length = int(handler.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid request length")
                def unique_object(pairs):
                    result={}
                    for key,value in pairs:
                        if key in result:raise ValueError("duplicate JSON key")
                        result[key]=value
                    return result
                def reject_constant(value):raise ValueError("nonfinite JSON number")
                body = json.loads(handler.rfile.read(length),object_pairs_hook=unique_object,parse_constant=reject_constant)
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                with self.lock:
                    if parts == ["sessions"]:
                        if set(body)-{"mode","seed","obstacles"}:raise ValueError("unknown session field")
                        if type(body.get("obstacles",False)) is not bool:raise ValueError("obstacles must be boolean")
                        mode = body.get("mode", "manual")
                        if mode not in {"manual", "script"}:
                            raise ValueError("mode must be manual or script")
                        active = [s for s in self.sessions.values() if not s.finished]
                        if len(active) >= 4:
                            raise ValueError("four sessions already active; close one first")
                        seed = body.get("seed",11)
                        if type(seed) is not int or not 0 <= seed < 2**32:
                            raise ValueError("invalid seed")
                        session = VisualSession(self.root, mode, bool(body.get("obstacles",False)), seed)
                        # Finished sessions remain on disk, not in producer memory.
                        self.sessions = {k:s for k,s in self.sessions.items() if not s.finished}
                        self.sessions[session.run_id] = session
                        session.start()
                        send({"run_id":session.run_id, "epoch":session.epoch})
                    elif len(parts) == 3 and parts[0] == "sessions" and parts[2] in {"command", "close"}:
                        allowed={"epoch"} if parts[2]=="close" else {"epoch","wire","request_id","lose_reply"}
                        if set(body)-allowed:raise ValueError("unknown command field")
                        if type(body.get("lose_reply",False)) is not bool:raise ValueError("lose_reply must be boolean")
                        session = self.sessions[parts[1]]
                        if body.get("epoch") != session.epoch:
                            raise ValueError("stale epoch")
                        if parts[2] == "close":
                            session.close()
                            send({"closed":True})
                        else:
                            if session.mode != "manual":
                                raise PermissionError("script and policy runs are read-only")
                            send(session.command(body.get("wire"),body.get("request_id"),body.get("lose_reply",False)))
                    else:
                        raise KeyError("unknown sandbox endpoint")
            elif parts == ["bootstrap"]:
                send({"token":self.token, "sim_only":True, "max_frame_bytes":65536, "live_hz":10})
            elif parts == ["runs"]:
                send(self.catalog())
            elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "manifest":
                send(self.manifest(parts[1]))
            elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "latest":
                self.manifest(parts[1])
                send(json.loads((self.directory / parts[1] / 'live.latest.json').read_text()))
            elif len(parts) == 4 and parts[0] == "runs" and parts[2] == "chunks":
                manifest = self.manifest(parts[1])
                chunk = next((c for chunks in manifest["streams"].values() for c in chunks if c["file"] == parts[3]),None)
                if chunk is None:
                    raise KeyError("unregistered chunk")
                file = self.directory / parts[1] / chunk["file"]
                data = file.read_bytes()
                if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                    raise ValueError("chunk hash mismatch")
                handler._send(200,data,"application/x-ndjson")
            elif len(parts) == 3 and parts[0] == "sessions" and parts[2] == "latest":
                session = self.sessions[parts[1]]
                # Snapshot replacement: consumers never queue work on the producer.
                send(session.snapshot)
            else:
                raise KeyError("unknown endpoint")
        except PermissionError as exc:
            handler._error(403,str(exc))
        except (KeyError,FileNotFoundError) as exc:
            handler._error(404,str(exc))
        except (ValueError,TypeError) as exc:
            handler._error(422,str(exc))
        return True
