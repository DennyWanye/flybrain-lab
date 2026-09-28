"""File/stdin ingestion and offline C2W shadow replay. No network commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .inputs import Profile
from .recording import Recorder, read_json, samples, verify, verify_shadow, canonical


def packets(path, idle_timeout_s=None):
    stream = sys.stdin if path == '-' else open(path)
    try:
        lines = stream
        if path == '-' and idle_timeout_s is not None:
            from queue import Queue, Empty
            from threading import Thread
            queue = Queue(maxsize=1)
            def reader():
                try:
                    for line in stream:
                        queue.put(line)
                except Exception as error:
                    queue.put(error)
                finally:
                    queue.put(None)
            Thread(target=reader, daemon=True, name='shadow-stdin-reader').start()
            def timed_lines():
                while True:
                    try:
                        item = queue.get(timeout=idle_timeout_s)
                    except Empty:
                        raise TimeoutError('observation stream stalled; shadow session stopped')
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            lines = timed_lines()
        for index, line in enumerate(lines, 1):
            if not line.strip():
                raise ValueError(f'blank input line {index}')
            yield read_json(line)
    finally:
        if stream is not sys.stdin:
            stream.close()


def fixture():
    """Analytic interface fixture, not a flight trajectory or model success test."""
    from .inputs import SCHEMA
    for seq in range(24):
        phase = ('altitude', 'navigation', 'heading')[seq // 8]
        now = 1_000_000_000 + seq * 100_000_000
        yield {'schema': SCHEMA, 'seq': seq, 'time_ns': now,
               'pose': {'captured_ns': now, 'received_ns': now, 'valid': True,
                        'frame_id': 'fixture_z_up', 'position_m': [0., 0., 1.],
                        'velocity_mps': [0., 0., 0.], 'yaw_rad': .2},
               'state': {'captured_ns': now, 'received_ns': now, 'valid': True,
                         'height_m': 1., 'battery_fraction': .8, 'airborne': True},
               'context': {'phase': phase, 'goal_m': [.6 if phase == 'navigation' else 0., 0., 1.4],
                           'target_yaw_rad': 1., 'previous_action': None,
                           'previous_duration_s': 0., 'remaining_s': 60. - (seq % 8) / 10.,
                           'operation_status': 'idle'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    example = sub.add_parser('example', help='create explicitly synthetic interface input')
    example.add_argument('--output', required=True)
    for name in ('record', 'shadow'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--input', required=True, help='normalized JSONL file, or - for stdin')
        cmd.add_argument('--profile', required=True)
        cmd.add_argument('--output', required=True, help='new output directory; never overwritten')
        if name == 'shadow':
            cmd.add_argument('--idle-timeout-s', type=float, default=1.0)
            cmd.add_argument('--root', required=True)
            cmd.add_argument('--seed', type=int, choices=(11, 22, 33), default=11)
            cmd.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    check = sub.add_parser('verify')
    check.add_argument('--recording', required=True)
    check_shadow = sub.add_parser('verify-shadow')
    check_shadow.add_argument('--session', required=True)
    replay = sub.add_parser('replay')
    replay.add_argument('--recording', required=True)
    replay.add_argument('--output', required=True)
    replay.add_argument('--root', required=True)
    replay.add_argument('--seed', type=int, choices=(11, 22, 33), default=11)
    replay.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'example':
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        profile = Profile('analytic-interface-fixture-v1', 'synthetic_fixture', 'fixture_z_up')
        (out / 'profile.json').write_text(canonical(asdict(profile)) + '\n')
        (out / 'samples.jsonl').write_text(''.join(canonical(row) + '\n' for row in fixture()))
        print(canonical({'output': str(out), 'samples': 24, 'provenance': profile.provenance}))
        return
    if args.command == 'verify-shadow':
        print(canonical(verify_shadow(args.session)))
        return
    if args.command == 'verify':
        print(canonical(verify(args.recording)))
        return
    if args.command == 'record':
        profile = Profile(**read_json(Path(args.profile).read_text()))
        with Recorder(args.output, profile) as recorder:
            for packet in packets(args.input):
                recorder.append(packet)
        print(canonical(verify(args.output)))
        return
    output = Path(args.output)
    # Reserve before model initialization, so an existing run can never be touched.
    output.mkdir(parents=True, exist_ok=False)
    policy = None
    try:
        if args.command == 'replay':
            verified = verify(args.recording)
            profile = Profile(**verified['header']['profile'])
            source = samples(args.recording)
            (output / 'input-verification.json').write_text(canonical(verified) + '\n')
        else:
            profile = Profile(**read_json(Path(args.profile).read_text()))
            if not 0 < args.idle_timeout_s <= 10:
                raise ValueError('idle timeout must be in (0,10] seconds')
            source = packets(args.input, args.idle_timeout_s)
        from .runtime import C2WPolicy, ShadowSession
        policy = C2WPolicy(args.root, args.seed, args.device)
        if args.command == 'shadow' and args.input == '-':
            print(canonical({'event': 'SHADOW_INPUT_READY', 'clock': 'host_monotonic_ns', 'transport': 'absent'}), file=sys.stderr, flush=True)
        with Recorder(output / 'input', profile) as recording, ShadowSession(profile, policy, output / 'shadow') as session:
            for packet in source:
                recording.append(packet)
                if args.command == 'shadow' and args.input == '-':
                    import time
                    session.step(packet, received_now_ns=time.monotonic_ns())
                else:
                    session.step(packet)
            if session.count == 0:
                raise ValueError('empty input cannot complete a shadow session')
        result = verify_shadow(output / 'shadow')
        verify(output / 'input')
        print(canonical(result))
    except Exception as error:
        (output / 'failure.json').write_text(canonical({'error_type': type(error).__name__, 'message': str(error),
                                                       'REAL_FLIGHT_READY': False}) + '\n')
        raise
    finally:
        if policy is not None:
            policy.close()


if __name__ == '__main__':
    main()
