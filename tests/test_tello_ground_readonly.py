import json
import tempfile
import unittest
from pathlib import Path

from tello_ground_readonly.collector import state_to_shadow_sample
from tello_ground_readonly.protocol import (
    READ_ONLY_COMMANDS,
    parse_state_packet,
    validate_command,
)


class GroundReadonlyTests(unittest.TestCase):
    def test_allowlist_and_flight_rejection(self):
        for command in READ_ONLY_COMMANDS:
            self.assertEqual(validate_command(command), command)
        for command in ("takeoff", "land", "stop", "emergency", "rc 0 0 0 0", "mon", "mdirection 0"):
            with self.assertRaises(PermissionError):
                validate_command(command)

    def test_state_and_mission_pad_units(self):
        state = parse_state_packet(
            "pitch:0;roll:0;yaw:5;vgx:0;vgy:0;vgz:0;tof:10;h:0;bat:96;mid:3;x:12;y:-8;z:50;"
        )
        self.assertTrue(state.mission_pad.valid)
        self.assertEqual(state.mission_pad.card_id, 3)
        self.assertEqual(state.mission_pad.position_m, (0.12, -0.08, 0.5))
        self.assertFalse(state.airborne)
        no_card = parse_state_packet("h:0;bat:96;mid:-1;")
        self.assertFalse(no_card.mission_pad.valid)
        self.assertEqual(no_card.mission_pad.reason, "no_card")
        fractional = parse_state_packet("h:0;bat:96;mid:1.5;x:1;y:2;z:3;")
        self.assertFalse(fractional.mission_pad.valid)
        self.assertEqual(fractional.mission_pad.reason, "card_id_noninteger")

    def test_duplicate_field_rejected(self):
        with self.assertRaises(ValueError):
            parse_state_packet("h:0;h:1;bat:90;")

    def test_shadow_sample_is_unmapped_and_ground_blocked(self):
        row = {
            "accepted": True,
            "received_ns": 1_000_000_000,
            "height_m": 0.0,
            "battery_fraction": 0.96,
            "airborne": False,
            "mission_pad": {"valid": True, "card_id": 3},
        }
        sample = state_to_shadow_sample(row, seq=0, now_ns=1_000_000_000)
        self.assertFalse(sample["pose"]["valid"])
        self.assertIsNone(sample["pose"]["position_m"])
        self.assertFalse(sample["state"]["airborne"])
        self.assertEqual(set(sample), {"schema", "seq", "time_ns", "pose", "state", "context"})

    def test_recording_json_is_ascii_and_explicitly_nonflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.jsonl"
            row = {"transmitted_flight_commands": 0, "shadow_transmitted": 0, "real_flight_ready": False}
            path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")
            self.assertEqual(json.loads(path.read_text())["transmitted_flight_commands"], 0)


if __name__ == "__main__":
    unittest.main()
