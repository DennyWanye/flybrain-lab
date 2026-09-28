"""Append-only, hash-linked sample recording; incomplete logs cannot be replayed."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from .inputs import Profile, ObservationBridge


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(line):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(line, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class Recorder:
    def __init__(self, directory, profile):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.header = {'schema': 'flybrain.shadow.recording/1', 'profile': asdict(profile),
                       'transport': 'absent', 'REAL_FLIGHT_READY': False}
        self.bridge = ObservationBridge(profile)
        self.chain = digest(self.header)
        self.count = 0
        self.closed = False
        (self.directory / 'header.json').write_text(canonical(self.header) + '\n')
        self.stream = (self.directory / 'samples.jsonl').open('x')

    def append(self, packet):
        if self.closed:
            raise RuntimeError('recorder closed')
        self.bridge.prepare(packet)
        body = {'index': self.count, 'previous_sha256': self.chain, 'sample': packet}
        self.chain = digest(body)
        self.stream.write(canonical({**body, 'sha256': self.chain}) + '\n')
        self.stream.flush()
        self.count += 1

    def close(self, completed=True):
        if self.closed:
            return
        self.stream.close()
        self.closed = True
        footer = {'completed': bool(completed and self.count), 'samples': self.count, 'final_sha256': self.chain}
        with (self.directory / 'footer.json').open('x') as stream:
            stream.write(canonical(footer) + '\n')

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(completed=kind is None)


def load_header(directory):
    header = read_json((Path(directory) / 'header.json').read_text())
    if (set(header) != {'schema', 'profile', 'transport', 'REAL_FLIGHT_READY'} or
        header['schema'] != 'flybrain.shadow.recording/1' or header['transport'] != 'absent' or
        header['REAL_FLIGHT_READY'] is not False):
        raise ValueError('invalid recording header')
    Profile(**header['profile'])
    return header


def samples(directory):
    directory = Path(directory)
    header = load_header(directory)
    footer = read_json((directory / 'footer.json').read_text())
    if (set(footer) != {'completed', 'samples', 'final_sha256'} or footer['completed'] is not True or
        type(footer['samples']) is not int or footer['samples'] <= 0):
        raise ValueError('incomplete or empty recording')
    chain, count = digest(header), 0
    validator = ObservationBridge(Profile(**header['profile']))
    with (directory / 'samples.jsonl').open() as stream:
        for line in stream:
            row = read_json(line)
            if set(row) != {'index', 'previous_sha256', 'sample', 'sha256'}:
                raise ValueError('invalid recording row')
            body = {k: row[k] for k in ('index', 'previous_sha256', 'sample')}
            if row['index'] != count or row['previous_sha256'] != chain or row['sha256'] != digest(body):
                raise ValueError('recording hash chain mismatch')
            validator.prepare(row['sample'])
            chain, count = row['sha256'], count + 1
            yield row['sample']
    if count != footer['samples'] or chain != footer['final_sha256']:
        raise ValueError('recording count or final hash mismatch')


def verify(directory):
    count = sum(1 for _ in samples(directory))
    return {'samples': count, 'header': load_header(directory),
            'footer': read_json((Path(directory) / 'footer.json').read_text())}


def verify_shadow(directory):
    """Check a completed decision journal without loading any neural model."""
    directory = Path(directory)
    header = read_json((directory / 'session.json').read_text())
    result = read_json((directory / 'result.json').read_text())
    if (header.get('schema') != 'flybrain.shadow.session/1' or header.get('transport') != 'absent'
            or header.get('REAL_FLIGHT_READY') is not False or not result.get('completed')
            or result.get('schema') != 'flybrain.shadow.result/1'
            or result.get('REAL_FLIGHT_READY') is not False or result.get('hardware_validated') is not False
            or result.get('transmitted_commands') != 0 or result.get('model') != header.get('model')):
        raise ValueError('incomplete or invalid shadow journal')
    chain = digest({'profile': header['profile'], 'model': header['model']})
    if chain != header['initial_sha256']:
        raise ValueError('shadow header hash mismatch')
    count = blocked = 0
    from flydrone.tellosim.training.contracts import ACTIONS
    from flydrone.tellosim.sdk.codec import TelloCommand, encode_command
    with (directory / 'decisions.jsonl').open() as stream:
        for line in stream:
            row = read_json(line)
            claimed = row.pop('sha256')
            if (row['index'] != count or row['previous_sha256'] != chain or claimed != digest(row)
                    or row['transmitted'] is not False or row['brain_tick'] != 4*(count+1)):
                raise ValueError('shadow journal hash or clock mismatch')
            if row['blocked_reasons']:
                if row['proposed_action'] is not None or row['proposed_command'] is not None:
                    raise ValueError('blocked input produced a proposal')
                blocked += 1
            else:
                action = row['proposed_action']
                if type(action) is not int or not 0 <= action < 9 or not row['mask'][action]:
                    raise ValueError('invalid shadow proposal')
                if row['proposed_command'] != encode_command(TelloCommand(*ACTIONS[action])):
                    raise ValueError('command/action mismatch')
            chain, count = claimed, count + 1
    if (not count or result['samples'] != count or result['blocked_samples'] != blocked
            or result['brain_substeps'] != 4*count or result['final_sha256'] != chain):
        raise ValueError('shadow result does not match journal')
    return result
