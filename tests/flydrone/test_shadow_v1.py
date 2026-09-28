"""Focused boundary tests; fake policies test plumbing, never model capability."""
import copy
import json
import math
from dataclasses import asdict
import numpy as np
import pytest
from flydrone.shadow.__main__ import fixture, main
from flydrone.shadow.inputs import Profile, ObservationBridge
from flydrone.shadow.recording import Recorder, verify, samples, read_json
from flydrone.shadow.runtime import ShadowSession
from flydrone.tellosim.training.contracts import SensorObservation, RealDeviceDisabled


def profile(**changes):
    return Profile(**{**asdict(Profile('fixture', 'synthetic_fixture', 'fixture_z_up')), **changes})


def packet():
    return next(fixture())


class PlumbingPolicy:
    metadata = {'model': 'TEST_DOUBLE_NOT_MODEL_EVIDENCE'}
    brain_tick = 0
    def infer(self, prepared):
        self.brain_tick += 4
        return 0, {'selected_action': 0}, 'plumbing-only'


def test_transform_matches_existing_observation_contract():
    p = packet()
    p['context']['phase'] = 'navigation'
    p['pose']['velocity_mps'] = [.2, .1, -.1]
    bridge = ObservationBridge(profile(yaw_to_room_rad=math.pi/2, translation_m=(1., 2., .3)))
    actual = bridge.prepare(p)
    expected = SensorObservation(0, 120, 120, 'room_map', (1., 2., 1.3), .2+math.pi/2,
                                 (-.1, .2, -.1), 1., .8, 0., 0.)
    np.testing.assert_allclose(actual.observation, expected.vector(p['context']['goal_m'], None, 0., 1.), atol=1e-7)


@pytest.mark.parametrize('phase', ['altitude', 'navigation', 'heading'])
def test_identity_vectors_match_frozen_contract(phase):
    p = packet(); p['context']['phase'] = phase
    actual = ObservationBridge(profile()).prepare(p)
    expected = actual.sensor.vector(p['context']['goal_m'], None, 0., 1.)
    if phase == 'heading':
        expected[6:8] = [math.sin(.8), math.cos(.8)]
    np.testing.assert_array_equal(actual.observation, expected)


@pytest.mark.parametrize('fault', ['frame', 'future', 'nan', 'missing_velocity', 'units', 'bool_seq', 'phase', 'extra_truth', 'state_missing'])
def test_malformed_rejected_before_clock_commit(fault):
    p = packet()
    if fault == 'frame': p['pose']['frame_id'] = 'wrong'
    elif fault == 'future': p['pose']['captured_ns'] += 1
    elif fault == 'nan': p['pose']['position_m'][0] = float('nan')
    elif fault == 'missing_velocity': p['pose']['velocity_mps'] = None
    elif fault == 'units': p['state']['battery_fraction'] = 80
    elif fault == 'bool_seq': p['seq'] = True
    elif fault == 'phase': p['context']['phase'] = 'takeoff'
    elif fault == 'extra_truth': p['truth'] = {'position_m': [1, 2, 3]}
    elif fault == 'state_missing': p['state']['height_m'] = None
    bridge = ObservationBridge(profile())
    with pytest.raises(ValueError): bridge.prepare(p)
    assert bridge.last_seq is None


@pytest.mark.parametrize('fault', ['duplicate', 'gap', 'time_gap', 'capture_reordered'])
def test_discontinuities_rejected(fault):
    bridge = ObservationBridge(profile())
    first, second = list(fixture())[:2]
    bridge.prepare(first)
    if fault == 'duplicate': second = first
    elif fault == 'gap': second['seq'] += 1
    elif fault == 'time_gap': second['time_ns'] += 1
    elif fault == 'capture_reordered': second['pose']['captured_ns'] = 0
    with pytest.raises(ValueError): bridge.prepare(second)


@pytest.mark.parametrize('fault,reason', [('stale','pose_stale'), ('invalid','pose_invalid'), ('state','state_stale'), ('ground','not_airborne'), ('busy','operation_busy'), ('unknown','operation_unknown'), ('deadline','phase_deadline')])
def test_unusable_input_suppresses_every_proposal(tmp_path, fault, reason):
    p = packet()
    if fault == 'stale': p['pose']['captured_ns'] = 0
    elif fault == 'invalid': p['pose']['valid'] = False; p['pose']['position_m'] = None
    elif fault == 'state': p['state']['captured_ns'] = 0
    elif fault == 'ground': p['state']['airborne'] = False
    elif fault in ('busy', 'unknown'): p['context']['operation_status'] = fault
    elif fault == 'deadline': p['context']['remaining_s'] = 0.
    with ShadowSession(profile(), PlumbingPolicy(), tmp_path/'shadow') as session:
        row = session.step(p)
    assert reason in row['blocked_reasons']
    assert row['proposed_command'] is None and row['proposed_action'] is None
    assert row['transmitted'] is False and row['brain_tick'] == 4


def test_session_latches_fault_and_keeps_evidence(tmp_path):
    with ShadowSession(profile(), PlumbingPolicy(), tmp_path/'shadow') as session:
        session.step(packet())
        with pytest.raises(ValueError): session.step(packet())
        with pytest.raises(RuntimeError): session.step(list(fixture())[1])
    result = json.loads((tmp_path/'shadow/result.json').read_text())
    assert not result['completed'] and result['samples'] == 1
    assert result['brain_substeps'] == 4 and result['transmitted_commands'] == 0


def test_record_replay_hashes_and_no_overwrite(tmp_path):
    out = tmp_path/'record'
    rows = list(fixture())
    with Recorder(out, profile()) as recorder:
        for p in rows: recorder.append(p)
    assert list(samples(out)) == rows
    assert verify(out)['samples'] == 24
    with pytest.raises(FileExistsError): Recorder(out, profile())
    text = (out/'samples.jsonl').read_text().replace('"height_m":1.0', '"height_m":1.1', 1)
    (out/'samples.jsonl').write_text(text)
    with pytest.raises(ValueError): verify(out)


@pytest.mark.parametrize('fault', ['truncated', 'incomplete', 'missing_footer'])
def test_partial_recording_not_accepted(tmp_path, fault):
    out = tmp_path/'record'
    with Recorder(out, profile()) as recorder:
        for p in list(fixture())[:2]: recorder.append(p)
    if fault == 'truncated': (out/'samples.jsonl').write_text((out/'samples.jsonl').read_text().splitlines()[0]+'\n')
    elif fault == 'incomplete':
        footer=json.loads((out/'footer.json').read_text()); footer['completed']=False
        (out/'footer.json').write_text(json.dumps(footer))
    else: (out/'footer.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)): verify(out)


def test_duplicate_json_and_nan_rejected():
    with pytest.raises(ValueError): read_json('{"seq":0,"seq":1}')
    with pytest.raises(ValueError): read_json('{"seq":NaN}')


def test_real_adapter_remains_disabled():
    with pytest.raises(RuntimeError, match='Real flight is disabled'): RealDeviceDisabled()


def test_cli_example_and_record(tmp_path):
    main(['example', '--output', str(tmp_path/'example')])
    main(['record', '--profile', str(tmp_path/'example/profile.json'), '--input', str(tmp_path/'example/samples.jsonl'), '--output', str(tmp_path/'record')])
    assert verify(tmp_path/'record')['samples'] == 24


def test_live_backlog_stops_before_neural_advance(tmp_path):
    with ShadowSession(profile(), PlumbingPolicy(), tmp_path/'shadow') as session:
        with pytest.raises(ValueError, match='backlog'):
            session.step(packet(), received_now_ns=2_000_000_000)
    assert session.policy.brain_tick == 0


def test_stdin_stall_has_bounded_failure(monkeypatch):
    import threading
    from flydrone.shadow.__main__ import packets
    release = threading.Event()
    class Stalled:
        def __iter__(self):
            release.wait(2)
            return iter([])
    monkeypatch.setattr('sys.stdin', Stalled())
    try:
        with pytest.raises(TimeoutError, match='stalled'):
            list(packets('-', .02))
    finally:
        release.set()


@pytest.mark.parametrize('change', ['row', 'header', 'footer', 'truncate'])
def test_shadow_audit_detects_mutation(tmp_path, change):
    from flydrone.shadow.recording import verify_shadow
    out=tmp_path/'shadow'
    with ShadowSession(profile(), PlumbingPolicy(), out) as session:
        for row in list(fixture())[:2]: session.step(row)
    assert verify_shadow(out)['samples']==2
    if change=='row':
        p=out/'decisions.jsonl';p.write_text(p.read_text().replace('"transmitted":false','"transmitted":true',1))
    elif change=='header':
        p=out/'session.json';v=json.loads(p.read_text());v['profile']['source_id']='changed';p.write_text(json.dumps(v))
    elif change=='footer':
        p=out/'result.json';v=json.loads(p.read_text());v['samples']=3;p.write_text(json.dumps(v))
    else:
        p=out/'decisions.jsonl';p.write_text(p.read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError):verify_shadow(out)


def test_clock_uncertainty_counts_against_freshness_budget():
    p=packet();p['pose']['captured_ns']-=190_000_000
    prepared=ObservationBridge(profile(clock_uncertainty_s=.02)).prepare(p)
    assert 'pose_stale' in prepared.blocked_reasons
    assert prepared.observation[12]==0


def test_file_parse_failure_marks_shadow_incomplete(tmp_path):
    from flydrone.shadow.recording import verify_shadow
    with pytest.raises(ValueError):
        with ShadowSession(profile(), PlumbingPolicy(), tmp_path/'shadow') as session:
            session.step(packet())
            raise ValueError('input truncated')
    with pytest.raises(ValueError):verify_shadow(tmp_path/'shadow')



def test_empty_sessions_are_not_complete(tmp_path):
    with Recorder(tmp_path/'record', profile()): pass
    with ShadowSession(profile(), PlumbingPolicy(), tmp_path/'shadow'): pass
    assert not json.loads((tmp_path/'record/footer.json').read_text())['completed']
    assert not json.loads((tmp_path/'shadow/result.json').read_text())['completed']


@pytest.mark.parametrize('failure', ['clock', 'mask'])
def test_bad_inference_is_latched(tmp_path, failure):
    class BadPolicy(PlumbingPolicy):
        def infer(self, prepared):
            self.brain_tick += 3 if failure == 'clock' else 4
            return 8, {'selected_action': 8}, 'plumbing-only'
    with ShadowSession(profile(), BadPolicy(), tmp_path/'shadow') as session:
        with pytest.raises(ValueError): session.step(packet())
    assert session.failed
    assert not json.loads((tmp_path/'shadow/result.json').read_text())['completed']
