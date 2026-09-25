"""TS1 TelloSim foundations.

The package is simulation-only. It does not open sockets to a real Tello.
"""

from .sdk import (
    CommandError, DeviceExecution, OperationManager, OperationPhase,
    OperationSnapshot, TelloCommand, decode_command, encode_command,
)

__all__ = [
    "CommandError", "DeviceExecution", "OperationManager", "OperationPhase",
    "OperationSnapshot", "TelloCommand", "decode_command", "encode_command",
]
