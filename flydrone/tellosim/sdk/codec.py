from __future__ import annotations

from dataclasses import dataclass
import re


class CommandError(ValueError):
    """A command rejected before it can reach the simulated device."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TelloCommand:
    verb: str
    args: tuple[int, ...] = ()


_INT = re.compile(r"-?(?:0|[1-9][0-9]*)$")
_NO_ARG = {"command", "takeoff", "land", "stop"}
_QUERY = {"speed?", "battery?", "time?", "sdk?", "sn?", "hardware?"}
_DISTANCE = {"forward", "back", "left", "right", "up", "down"}
_ROTATION = {"cw", "ccw"}


def _parse_int(text: str) -> int:
    if not _INT.fullmatch(text):
        raise CommandError("syntax_error", "integer arguments must use canonical decimal syntax")
    return int(text)


def decode_command(text: str) -> TelloCommand:
    if not isinstance(text, str) or not text:
        raise CommandError("syntax_error", "command must be a non-empty ASCII string")
    if len(text.encode("ascii", errors="ignore")) != len(text) or len(text.encode("ascii")) > 256:
        raise CommandError("syntax_error", "command must be at most 256 ASCII bytes")
    if any(char in text for char in "\r\n\t;\x00"):
        raise CommandError("syntax_error", "control characters and separators are not allowed")
    parts = text.strip().split()
    verb = parts[0]
    if verb in _NO_ARG or verb in _QUERY:
        if len(parts) != 1:
            raise CommandError("syntax_error", f"{verb} does not take arguments")
        return TelloCommand(verb)
    if verb in _DISTANCE:
        if len(parts) != 2:
            raise CommandError("syntax_error", f"{verb} requires one distance")
        value = _parse_int(parts[1])
        if not 20 <= value <= 500:
            raise CommandError("range_error", "distance must be in [20, 500] cm")
        return TelloCommand(verb, (value,))
    if verb in _ROTATION:
        if len(parts) != 2:
            raise CommandError("syntax_error", f"{verb} requires one angle")
        value = _parse_int(parts[1])
        if not 1 <= value <= 360:
            raise CommandError("range_error", "rotation must be in [1, 360] degrees")
        return TelloCommand(verb, (value,))
    if verb == "speed":
        if len(parts) != 2:
            raise CommandError("syntax_error", "speed requires one value")
        value = _parse_int(parts[1])
        if not 10 <= value <= 100:
            raise CommandError("range_error", "speed must be in [10, 100] cm/s")
        return TelloCommand(verb, (value,))
    if verb == "go":
        if len(parts) != 5:
            raise CommandError("syntax_error", "go requires x y z speed")
        values = tuple(_parse_int(value) for value in parts[1:])
        if any(abs(value) > 500 for value in values[:3]):
            raise CommandError("range_error", "go axes must be in [-500, 500] cm")
        if max(abs(value) for value in values[:3]) <= 20:
            raise CommandError("range_error", "go requires one axis outside the 20 cm dead zone")
        if not 10 <= values[3] <= 100:
            raise CommandError("range_error", "go speed must be in [10, 100] cm/s")
        return TelloCommand(verb, values)
    raise CommandError("unsupported_command", f"unsupported command: {verb}")


def encode_command(command: TelloCommand) -> str:
    if not isinstance(command, TelloCommand):
        raise TypeError("command must be TelloCommand")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in command.args):
        raise CommandError("syntax_error", "command arguments must be integers")
    decoded = decode_command(" ".join([command.verb, *(str(value) for value in command.args)]))
    return " ".join([decoded.verb, *(str(value) for value in decoded.args)])
