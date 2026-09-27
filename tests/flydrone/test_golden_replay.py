import json
import shutil
from pathlib import Path

import pytest

from flydrone.tellosim.env import TelloSimEnv, VariableDurationConfig
from flydrone.tellosim.physics import SimWorld
from flydrone.tellosim.sim import SimTello
from flydrone.tellosim.golden import sha256, validate_golden
from flydrone.tellosim.replay import export_golden
from flyview.server import Viewer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/golden_episode_verified"


def test_hold_uses_physics_time_not_schedule():
    env = TelloSimEnv(SimTello(SimWorld()), VariableDurationConfig(
        start_m=(0, 0, 1), goal_m=(0, 0, 1), stable_hold_s=1,
        action_duration_s=(99,)))
    env.reset()
    _, reward, term, _, info = env.step(0)
    assert not term
    assert info["stable_hold_s"] == pytest.approx(.5)
    assert info["duration_s"] == pytest.approx(.5)
    assert reward == pytest.approx(-.005)
    assert len(info["physics_samples"]) == 60
    assert env.step(0)[2]


def test_crossing_boundary_is_failure_even_near_goal():
    env = TelloSimEnv(SimTello(SimWorld()), VariableDurationConfig(
        start_m=(2.99, 0, 1), goal_m=(3.1, 0, 1), target_radius_m=.3))
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert terminated and not truncated and info["out_of_bounds"]
    assert not info["success"]


def test_export_is_lossless_readonly_and_repeatable(tmp_path):
    before = sha256(SOURCE / "replay.jsonl")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"existing": "/other/view"}))
    dest = tmp_path / "golden"
    manifest = export_golden(SOURCE, dest, registry=index)
    raw = [json.loads(s) for s in (SOURCE / "replay.jsonl").read_text().splitlines()]
    events = [json.loads(s) for s in (dest / "events.jsonl").read_text().splitlines()]
    assert [e["payload"]["golden"] for e in events] == raw
    assert manifest["event_count"] == len(raw)
    assert events[-1]["payload"]["end_reason"] == "success"
    assert json.loads(index.read_text())["existing"] == "/other/view"
    digest = sha256(dest / "events.jsonl")
    export_golden(SOURCE, dest, registry=index)
    assert sha256(dest / "events.jsonl") == digest
    assert sha256(SOURCE / "replay.jsonl") == before


def test_validator_does_not_invent_browser_acceptance(tmp_path):
    for name in ("manifest.json", "replay.jsonl", "checkpoint.pt", "brain_mapping.json"):
        shutil.copyfile(SOURCE / name, tmp_path / name)
    result = validate_golden(tmp_path, ROOT / "data/male-v1.npz")
    assert not result["GOLDEN_EPISODE_READY"]
    assert set(result["pending"]) == {"GE-05", "GE-06", "GE-07", "GE-08", "GE-15"}
    assert all(v is not False for v in result["checks"].values())
    with (tmp_path / "replay.jsonl").open("a") as handle:
        handle.write("\n")
    assert not validate_golden(tmp_path)["checks"]["GE-13"]
    with pytest.raises(ValueError, match="hash mismatch"):
        export_golden(tmp_path, tmp_path / "view")


def test_viewer_pages_every_event_without_duplicates(tmp_path):
    registry = tmp_path / "reports/vis/views"
    dest = registry / "long"
    dest.mkdir(parents=True)
    (registry / "index.json").write_text(json.dumps({"long": str(dest)}))
    (dest / "events.jsonl").write_text("".join(json.dumps({"kind": "transition", "seq": i})+"\n" for i in range(600)))
    viewer = Viewer(tmp_path, registry)
    rows = sum((viewer.events("long", offset=i) for i in (0, 256, 512)), [])
    assert [r["seq"] for r in rows] == list(range(600))
    assert not viewer.events("long", offset=600)
