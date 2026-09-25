from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LatestWriter:
    """A bounded latest-only writer; publishing never waits on disk I/O."""

    def __init__(self, directory: str | Path, descriptor: dict[str, Any], publish_hz: float = 10.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.publish_interval = 1.0 / max(0.1, min(float(publish_hz), 10.0))
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._dropped = 0
        self._written_seq = -1
        self._stop = threading.Event()
        self.source_epoch = str(descriptor["source_epoch"])
        self._write_json("live_descriptor.json", descriptor)
        self._thread = threading.Thread(target=self._run, name="flybrain-live-writer", daemon=True)
        self._thread.start()

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._latest is not None:
                self._dropped += 1
            self._latest = event

    def publish_status(self, status: dict[str, Any]) -> None:
        self._write_json("live.status.json", status)

    def publish_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json("live.metrics.json", metrics)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            latest = self._latest
            self._latest = None
        if latest is not None:
            self._write_latest(latest)
        self.publish_status({"status": "finished", "updated_at_utc": utc_now(), "last_seq": self._written_seq,
                             "dropped_frames": self.dropped, "trace_complete": True})

    def _write_json(self, name: str, value: dict[str, Any]) -> None:
        temporary = self.directory / f".{name}.tmp"
        target = self.directory / name
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, target)

    def _write_latest(self, event: dict[str, Any]) -> None:
        self._write_json("live.latest.json", event)
        self._written_seq = int(event.get("seq", self._written_seq))

    def _run(self) -> None:
        while not self._stop.wait(self.publish_interval):
            with self._lock:
                latest = self._latest
                self._latest = None
            if latest is not None:
                try:
                    self._write_latest(latest)
                    self.publish_status({"status": "running", "updated_at_utc": utc_now(), "last_seq": self._written_seq,
                                         "dropped_frames": self.dropped, "trace_complete": False})
                except (OSError, TypeError, ValueError) as exc:
                    self.publish_status({"status": "degraded", "updated_at_utc": utc_now(), "last_seq": self._written_seq,
                                         "dropped_frames": self.dropped, "trace_complete": False, "error": str(exc)})


def create_live_writer(directory: str | Path, *, run_id: str, brain_contract: dict[str, Any] | None,
                       config_hash: str, publish_hz: float = 10.0) -> LatestWriter:
    descriptor = {
        "schema_version": "1.0.0",
        "descriptor_kind": "live",
        "source_kind": "simulation_live",
        "view_id": Path(directory).name,
        "run_id": run_id,
        "source_epoch": uuid.uuid4().hex,
        "config_sha256": config_hash,
        "graph_sha256": brain_contract.get("graph_sha256") if brain_contract else None,
        "mapping_sha256": brain_contract.get("mapping_sha256") if brain_contract else None,
        "capabilities": {"live_readout": brain_contract is not None, "substep_spikes": False,
                          "physics_path": False, "topology": False, "anatomy": False},
    }
    return LatestWriter(directory, descriptor, publish_hz)
