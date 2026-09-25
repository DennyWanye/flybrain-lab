#!/usr/bin/env python3
"""Validate bundled observation contracts and explicit synthetic fixtures.

Not a viewer or trainer. Requires jsonschema. Rejects invalid JSON and performs
cross-field checks in addition to JSON Schema. No network or hardware access.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit('Install jsonschema in the separate viewer venv, then retry.') from exc

ROOT = Path(__file__).resolve().parents[2] / 'contracts' / 'vis'
ACTION_NAMES = ['HOLD', 'POS_X', 'NEG_X', 'POS_Y', 'NEG_Y']
OFFSETS = [(0., 0.), (.2, 0.), (-.2, 0.), (0., .2), (0., -.2)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reject_constant(value: str) -> Any:
    raise ValueError(f'Non-finite JSON constant: {value}')


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def finite_tree(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), 'Non-finite number')
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def loads(text: str) -> Any:
    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    finite_tree(value)
    return value


def load(path: Path) -> Any:
    return loads(path.read_text(encoding='utf-8'))


def validator(filename: str) -> Draft202012Validator:
    schema = load(ROOT / filename)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def validate_event(event: dict[str, Any], schema: Draft202012Validator | None = None) -> None:
    finite_tree(event)
    (schema or validator('event.schema.json')).validate(event)
    try:
        parsed_time = datetime.fromisoformat(event['emitted_at_utc'].replace('Z', '+00:00'))
        require(parsed_time.tzinfo is not None, 'emitted_at_utc must include timezone')
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Invalid RFC3339 emitted_at_utc') from exc
    p = event['payload']
    kind = event['kind']
    if event['source_kind'] != 'synthetic_demo' and kind == 'transition':
        require(p['action_selection'] != 'scripted_fixture', 'Fixture action in real source')
    if kind == 'transition':
        require(p['action_name'] == ACTION_NAMES[p['action_id']], 'Action name/id mismatch')
        require(0 < p['sim_tick_after'] - p['sim_tick_before'] <= 50, 'Invalid physics tick interval')
        require(not (p['terminated'] and p['truncated']), 'Both terminal and truncated')
        reason = p['end_reason']
        require((reason is None) == (not p['terminated'] and not p['truncated']), 'End reason/flags mismatch')
        if p['truncated']:
            require(reason == 'external_truncation', 'Truncation reason mismatch')
        if p['terminated']:
            require(reason in ['success', 'boundary', 'deadline'], 'Terminal reason mismatch')
        if p['action_probabilities'] is not None:
            require(approx(sum(p['action_probabilities']), 1.), 'Probability sum != 1')
        reward = p['reward_parts']
        require(approx(reward['total'], sum(v for k, v in reward.items() if k != 'total')), 'Reward sum mismatch')
        b, a = p['state_before'], p['state_after']
        require(b['goal_xy_m'] == a['goal_xy_m'], 'Goal changes within transition')
        progress = 2. * (math.dist(b['position_xy_m'], b['goal_xy_m']) - math.dist(a['position_xy_m'], a['goal_xy_m']))
        require(approx(progress, reward['progress']), 'P1 progress reward mismatch')
        expected = {'step_cost': -.02, 'success_bonus': 10. if reason == 'success' else 0.,
                    'boundary_cost': -10. if reason == 'boundary' else 0.,
                    'deadline_cost': -2. if reason == 'deadline' else 0.}
        for key, value in expected.items():
            require(approx(reward[key], value), f'P1 {key} mismatch')
        q = [b['position_xy_m'][i] + OFFSETS[p['action_id']][i] for i in range(2)]
        require(all(approx(x, y) for x, y in zip(q, p['local_waypoint_xy_m'])), 'Local waypoint mismatch')
        dt = (p['sim_tick_after'] - p['sim_tick_before']) * p['physics_dt_s']
        require(approx(a['remaining_s'], max(0., b['remaining_s'] - dt)), 'Remaining time mismatch')
        expected_obs = [(b['goal_xy_m'][0] - b['position_xy_m'][0]) / 4.,
                        (b['goal_xy_m'][1] - b['position_xy_m'][1]) / 4.,
                        b['velocity_xy_mps'][0] / .3, b['velocity_xy_mps'][1] / .3,
                        b['position_xy_m'][0] / 2., b['position_xy_m'][1] / 2.,
                        b['dwell_s'] / 2., b['remaining_s'] / 60.]
        expected_obs = [max(-1., min(1., x)) for x in expected_obs]
        require(all(approx(x, y) for x, y in zip(expected_obs, p['observation'])), 'Observation/state mismatch')
        enc = [z for x in p['observation'][:6] for z in (max(x, 0.), max(-x, 0.))] + p['observation'][6:]
        require(all(approx(x, y) for x, y in zip(enc, p['encoded_observation'])), 'Encoded observation mismatch')
        path = p['physics_path']
        if path is not None:
            require(len(path) == p['sim_tick_after'] - p['sim_tick_before'] + 1, 'Physical path not complete')
            require([x['physics_tick'] for x in path] == list(range(p['sim_tick_before'], p['sim_tick_after'] + 1)), 'Physical ticks not contiguous')
            for point, state in [(path[0], b), (path[-1], a)]:
                for key in ['position_xy_m', 'velocity_xy_mps']:
                    require(all(approx(x, y) for x, y in zip(point[key], state[key])), 'Physical endpoints mismatch')
        snap = p['readout_snapshot']
        if snap:
            require(len({n['neuron_id'] for n in snap['neurons']}) == len(snap['neurons']), 'Duplicate sampled neuron')
    elif kind == 'neural_sample':
        require(len({n['neuron_id'] for n in p['neurons']}) == len(p['neurons']), 'Duplicate sampled neuron')
    elif kind == 'training_metric':
        n, k = p['window_episodes'], p['successes']
        require(k <= n, 'Success numerator exceeds denominator')
        if n == 0:
            require(p['success_rate'] is None and p['mean_return'] is None, 'Metric with zero denominator must be null')
        else:
            require(p['success_rate'] is not None and approx(p['success_rate'], k / n), 'Success rate mismatch')
        if p['split'] != 'training_window':
            require(p['evaluation_manifest_sha256'] is not None, 'Evaluation needs provenance hash')
            require(p['checkpoint_sha256'] is not None, 'Evaluation needs checkpoint hash')


def validate_fixture(directory: Path) -> dict[str, int]:
    manifest = load(directory / 'view_manifest.json')
    validator('view_manifest.schema.json').validate(manifest)
    if manifest['source_kind'] == 'synthetic_demo':
        require(bool(manifest['synthetic_warning']), 'Synthetic label missing')
        require(manifest['policy_learning_status'] == 'fixture_only', 'Fixture claims learning success')
    else:
        require(manifest['p1_mode'] != 'fixture', 'Real source uses fixture mode')
        require(manifest['policy_learning_status'] != 'fixture_only', 'Real source declares fixture-only')
    catalog = load(directory / 'neurons.json')
    validator('neuron_catalog.schema.json').validate(catalog)
    for field in ['dataset_id', 'graph_sha256', 'mapping_sha256']:
        require(catalog[field] == manifest[field], f'Catalog {field} mismatch')
    nodes = catalog['nodes']
    require(len({n['neuron_id'] for n in nodes}) == len(nodes), 'Duplicate catalog neuron ID')
    require(len({n['neuron_index'] for n in nodes}) == len(nodes), 'Duplicate catalog neuron index')
    lookup = {n['neuron_id']: n for n in nodes}
    for n in nodes:
        require(n['dataset_id'] == manifest['dataset_id'], 'Node dataset mismatch')
        require(n['neuron_index'] < manifest['node_count'], 'Node index out of bounds')
    require(set(manifest['recorded_neuron_ids']) <= set(lookup), 'Recorded IDs absent in catalog')
    for file in manifest['files']:
        path = directory / file['path']
        require(path.resolve().parent == directory.resolve(), 'File escapes view directory')
        b = path.read_bytes()
        require(len(b) == file['bytes'], 'File size mismatch')
        require(hashlib.sha256(b).hexdigest() == file['sha256'], 'File hash mismatch')
    if manifest['topology_file']:
        topology = load(directory / 'topology.json')
        validator('topology.schema.json').validate(topology)
        require(topology['graph_sha256'] == manifest['graph_sha256'], 'Topology graph mismatch')
        require(topology['dataset_id'] == manifest['dataset_id'], 'Topology dataset mismatch')
        for h in topology['neighborhoods']:
            require(h['center_id'] in h['node_ids'], 'Center not in neighborhood')
            require(set(h['node_ids']) <= set(lookup), 'Topology node absent in catalog')
            for edge in h['edges']:
                require(edge['source_id'] in h['node_ids'] and edge['target_id'] in h['node_ids'], 'Edge endpoint missing')
    lines = (directory / 'events.jsonl').read_text(encoding='utf8').splitlines()
    events = [loads(x) for x in lines if x.strip()]
    require(len(events) == manifest['event_count'], 'Event count mismatch')
    validator_event = validator('event.schema.json')
    last_seq = -1
    transitions: dict[tuple[Any, ...], dict[str, Any]] = {}
    neural_keys: list[tuple[Any, ...]] = []
    for e in events:
        validate_event(e, validator_event)
        for key in ['run_id', 'source_epoch', 'source_kind']:
            require(e[key] == manifest[key], f'Event {key} mismatch')
        require(e['seq'] > last_seq, 'Duplicate/nonmonotonic seq')
        last_seq = e['seq']
        p = e['payload']
        if e['kind'] in ['transition', 'neural_sample']:
            key = tuple(p[k] for k in ['env_index', 'episode_id', 'case_id', 'decision_step', 'policy_update'])
            ns = p['neurons'] if e['kind'] == 'neural_sample' else (p['readout_snapshot'] or {}).get('neurons', [])
            for n in ns:
                require(n['neuron_id'] in lookup, 'Unrecognized dynamic node ID')
                require(n['neuron_index'] == lookup[n['neuron_id']]['neuron_index'], 'ID/index mismatch')
                require(lookup[n['neuron_id']]['dynamic_recorded'], 'Unexpected dynamic data')
            if e['kind'] == 'transition':
                require(key not in transitions, 'Duplicate decision transition')
                transitions[key] = p
            else:
                neural_keys.append(key)
    if manifest['trace_complete']:
        require(set(neural_keys) <= set(transitions), 'Neural sample not joined to decision')
    return {'schema_files': len(list(ROOT.glob('*.schema.json'))), 'events': len(events), 'catalog_nodes': len(nodes), 'transitions': len(transitions)}


def self_test(directory: Path) -> int:
    for path in sorted((ROOT / 'contracts').glob('*.schema.json')):
        validator(path.name)
    result = validate_fixture(directory)
    print('PASS bundled schemas and synthetic fixture:', json.dumps(result))
    events = [loads(x) for x in (directory / 'events.jsonl').read_text().splitlines() if x]
    tr = next(e for e in events if e['kind'] == 'transition')
    ne = next(e for e in events if e['kind'] == 'neural_sample')
    metric = next(e for e in events if e['kind'] == 'training_metric')
    v = validator('event.schema.json')
    tests: list[tuple[str, Any]] = []

    def bad(name: str, event: dict[str, Any], change: Any) -> None:
        data = copy.deepcopy(event)
        change(data)
        tests.append((name, lambda: validate_event(data, v)))

    bad('wrong_action_name', tr, lambda e: e['payload'].__setitem__('action_name', 'NEG_X'))
    bad('unsafe_numeric_neuron_id', tr, lambda e: e['payload']['readout_snapshot']['neurons'][0].__setitem__('neuron_id', 9007199254740993))
    bad('invalid_reward_sum', tr, lambda e: e['payload']['reward_parts'].__setitem__('total', 9.))
    bad('bad_probability_sum', tr, lambda e: e['payload'].__setitem__('action_probabilities', [.5] * 5))
    bad('neural_substep_out_of_range', ne, lambda e: e['payload'].__setitem__('neural_substep', 4))
    bad('unknown_action_id', tr, lambda e: e['payload'].__setitem__('action_id', 7))
    bad('unexpected_control_field', tr, lambda e: e['payload'].__setitem__('takeoff', True))
    bad('nonfinite_state', tr, lambda e: e['payload'].__setitem__('value_estimate', float('inf')))
    bad('zero_denominator_score', metric, lambda e: e['payload'].__setitem__('success_rate', 1.))
    bad('wrong_waypoint', tr, lambda e: e['payload'].__setitem__('local_waypoint_xy_m', [-.2, 0.]))
    bad('terminal_flag_without_reason', tr, lambda e: e['payload'].__setitem__('terminated', True))
    bad('wrong_remaining_time', tr, lambda e: e['payload']['state_after'].__setitem__('remaining_s', 1.))
    bad('reward_in_observation_dimension', tr, lambda e: e['payload']['observation'].append(.3))
    bad('unknown_schema_version', tr, lambda e: e.__setitem__('schema_version', '2.0.0'))
    bad('unknown_real_flight_source', tr, lambda e: e.__setitem__('source_kind', 'real_tello'))
    bad('invalid_datetime', tr, lambda e: e.__setitem__('emitted_at_utc', 'tomorrow'))
    bad('wrong_encoded_sign', tr, lambda e: e['payload']['encoded_observation'].__setitem__(1, .2))
    bad('observation_state_mismatch', tr, lambda e: e['payload']['observation'].__setitem__(0, -.2))
    bad('missing_neuron_phase', tr, lambda e: e['payload']['readout_snapshot'].pop('phase'))
    bad('future_eval_without_provenance', metric, lambda e: e['payload'].__setitem__('split', 'sealed_test'))
    bad('numeric_policy_seq_too_large', tr, lambda e: e.__setitem__('seq', 2**60))
    bad('duplicate_sampled_id', ne, lambda e: e['payload']['neurons'].append(copy.deepcopy(e['payload']['neurons'][0])))
    tests.extend([
        ('json_duplicate_key', lambda: loads('{"x":1,"x":2}')),
        ('json_nan', lambda: loads('{"x":NaN}')),
        ('json_infinity', lambda: loads('{"x":Infinity}')),
        ('json_overflow', lambda: loads('{"x":1e999}')),
    ])
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f'PASS reject {name}: {type(exc).__name__}')
        else:
            raise AssertionError(f'Invalid test accepted: {name}')
    print(f'SELF_TEST_PASS: {len(tests)} negative cases + 1 bundle positive validation')
    return len(tests) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, default=ROOT / 'fixtures')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.fixture_dir)
    else:
        print(json.dumps(validate_fixture(args.fixture_dir), indent=2))


if __name__ == '__main__':
    main()
