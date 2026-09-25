from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .codec import TelloCommand


class OperationPhase(str, Enum):
    CREATED = "created"
    SENT = "sent"
    ACK_OK = "ack_ok"
    ACK_ERROR = "ack_error"
    UNKNOWN_EXECUTION = "unknown_execution"
    CANCELLED_LOCAL = "cancelled_local"


class DeviceExecution(str, Enum):
    NOT_OBSERVED = "not_observed"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class OperationSnapshot:
    operation_id: str
    request_id: str
    command: TelloCommand
    phase: OperationPhase
    device_execution: DeviceExecution
    sent_tick: int
    event_tick: int
    raw_response: str | None = None


class OperationManager:
    """Single-device operation semantics shared by simulated transports.

    The manager intentionally does not retry relative motion after an unknown
    response. ``operation_id`` is local runtime metadata, never a wire field.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._active: OperationSnapshot | None = None
        self._history: dict[str, OperationSnapshot] = {}

    @property
    def active(self) -> OperationSnapshot | None:
        return self._active

    def submit(self, command: TelloCommand, *, request_id: str, sim_tick: int) -> OperationSnapshot:
        if self._active is not None and self._active.phase in {
            OperationPhase.CREATED,
            OperationPhase.SENT,
            OperationPhase.UNKNOWN_EXECUTION,
        } and command.verb != "stop":
            raise RuntimeError("busy: one unresolved operation is already active")
        operation_id = f"op-{self._next_id}"
        self._next_id += 1
        snapshot = OperationSnapshot(
            operation_id=operation_id,
            request_id=request_id,
            command=command,
            phase=OperationPhase.SENT,
            device_execution=DeviceExecution.NOT_OBSERVED,
            sent_tick=int(sim_tick),
            event_tick=int(sim_tick),
        )
        self._active = snapshot
        self._history[operation_id] = snapshot
        return snapshot

    def _update(self, operation_id: str, **changes: object) -> OperationSnapshot:
        current = self._history[operation_id]
        updated = replace(current, **changes)
        self._history[operation_id] = updated
        if self._active and self._active.operation_id == operation_id:
            self._active = updated
        return updated

    def device_started(self, operation_id: str, *, sim_tick: int) -> OperationSnapshot:
        current = self._history[operation_id]
        if current.phase not in {OperationPhase.SENT, OperationPhase.UNKNOWN_EXECUTION}:
            raise RuntimeError("operation is not executable")
        return self._update(operation_id, device_execution=DeviceExecution.RUNNING, event_tick=int(sim_tick))

    def device_completed(self, operation_id: str, *, sim_tick: int, failed: bool = False) -> OperationSnapshot:
        current = self._history[operation_id]
        if current.device_execution not in {DeviceExecution.RUNNING, DeviceExecution.NOT_OBSERVED}:
            raise RuntimeError("device execution is already terminal")
        return self._update(
            operation_id,
            device_execution=DeviceExecution.FAILED if failed else DeviceExecution.COMPLETED,
            event_tick=int(sim_tick),
        )

    def receive_ack(self, operation_id: str, *, sim_tick: int, ok: bool, raw_response: str) -> OperationSnapshot:
        current = self._history[operation_id]
        if current.phase in {OperationPhase.ACK_OK, OperationPhase.ACK_ERROR, OperationPhase.CANCELLED_LOCAL}:
            raise RuntimeError("late acknowledgement cannot rewrite a terminal operation")
        return self._update(
            operation_id,
            phase=OperationPhase.ACK_OK if ok else OperationPhase.ACK_ERROR,
            event_tick=int(sim_tick),
            raw_response=raw_response,
        )

    def mark_unknown(self, operation_id: str, *, sim_tick: int) -> OperationSnapshot:
        current = self._history[operation_id]
        if current.phase not in {OperationPhase.SENT, OperationPhase.UNKNOWN_EXECUTION}:
            raise RuntimeError("only sent operations can become unknown")
        return self._update(operation_id, phase=OperationPhase.UNKNOWN_EXECUTION, event_tick=int(sim_tick))

    def cancel_locally(self, operation_id: str, *, sim_tick: int) -> OperationSnapshot:
        current = self._history[operation_id]
        if current.phase in {OperationPhase.ACK_OK, OperationPhase.ACK_ERROR, OperationPhase.CANCELLED_LOCAL}:
            raise RuntimeError("operation is already terminal")
        return self._update(operation_id, phase=OperationPhase.CANCELLED_LOCAL, event_tick=int(sim_tick))

    def get(self, operation_id: str) -> OperationSnapshot:
        return self._history[operation_id]
