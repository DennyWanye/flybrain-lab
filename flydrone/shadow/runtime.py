"""C2W observation-only inference. No transport, device, or send callback exists."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .inputs import ObservationBridge
from .recording import canonical, digest
from flydrone.tellosim.training.contracts import ACTIONS
from flydrone.tellosim.sdk.codec import TelloCommand, encode_command


class C2WPolicy:
    def __init__(self, root, seed=11, device='cuda'):
        # Local, frozen and trusted project checkpoints only.
        if seed not in (11, 22, 33):
            raise ValueError('expected one of the three frozen C2W seeds')
        from flydrone.tellosim.training.spatial_continuous import SpatialPool, load_spatial
        root = Path(root)
        self.pool = SpatialPool(root, seed, batch=1, device=device)
        checkpoint = root / f'runs/tellosim-sdk9/spatial-continuous-s{seed}/checkpoint.pt'
        load_spatial(checkpoint, self.pool, root)
        self.lane = self.pool.lanes[0]
        self.phase = None
        self.metadata = {'model': 'C2W', 'seed': seed,
                         'checkpoint_sha256': self.pool.checkpoint_sha256,
                         'graph_sha256': self.pool.brain.graph_sha256,
                         'input_source': 'reservoir_v_trace', 'feature_dim': 128,
                         'new_training_actions': 0, 'device': device}

    @property
    def brain_tick(self):
        return self.lane.brain_tick

    def infer(self, prepared):
        if self.phase != prepared.phase:
            self.lane.switch_skill(prepared.phase)
            self.phase = prepared.phase
        features = self.lane.observe(prepared.observation, prepared.sample_id)
        action, _, _, policy = self.lane.decision(features, prepared.mask, True)
        if not np.isfinite(features).all():
            raise ValueError('nonfinite neural features')
        return action, policy, hashlib.sha256(np.asarray(features).tobytes()).hexdigest()

    def close(self):
        self.lane = None
        self.pool = None
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ShadowSession:
    def __init__(self, profile, policy, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.bridge = ObservationBridge(profile)
        self.policy = policy
        self.failed = False
        self.closed = False
        self.count = self.blocked = 0
        self.chain = digest({'profile': profile.__dict__, 'model': policy.metadata})
        self.initial_chain = self.chain
        self.stream = (self.output / 'decisions.jsonl').open('x')
        (self.output / 'session.json').write_text(canonical({
            'schema': 'flybrain.shadow.session/1', 'profile': profile.__dict__,
            'model': policy.metadata, 'initial_sha256': self.chain,
            'phase_source': 'recorded upstream context; proposals never imply completion',
            'transport': 'absent', 'REAL_FLIGHT_READY': False}) + '\n')

    def step(self, packet, received_now_ns=None):
        if self.failed or self.closed:
            raise RuntimeError('shadow session stopped; create a new session')
        try:
            prepared = self.bridge.prepare(packet)
            if received_now_ns is not None:
                age = (received_now_ns - prepared.time_ns) / 1e9
                if age < 0 or age + self.bridge.profile.clock_uncertainty_s > self.bridge.profile.max_age_s:
                    raise ValueError('live observation clock mismatch or wall-clock backlog')
            before = self.policy.brain_tick
            action, policy, features_sha = self.policy.infer(prepared)
            if self.policy.brain_tick - before != 4:
                raise ValueError('neural clock must advance exactly four substeps')
            if type(action) is not int or not 0 <= action < 9 or not prepared.mask[action]:
                raise ValueError('policy returned invalid or masked action')
            blocked = bool(prepared.blocked_reasons)
            row = {'index': self.count, 'sample_sha256': digest(packet),
                   'sample_id': prepared.sample_id, 'time_ns': prepared.time_ns,
                   'phase': prepared.phase, 'observation26': prepared.observation.tolist(),
                   'mask': prepared.mask.tolist(), 'brain_tick': self.policy.brain_tick,
                   'features_sha256': features_sha, 'policy': policy,
                   'proposed_action': None if blocked else action,
                   'proposed_command': None if blocked else encode_command(TelloCommand(*ACTIONS[action])),
                   'blocked_reasons': prepared.blocked_reasons,
                   'transmitted': False, 'previous_sha256': self.chain}
            chain = digest(row)
            self.stream.write(canonical({**row, 'sha256': chain}) + '\n')
            self.stream.flush()
            self.chain = chain
            self.count += 1
            self.blocked += int(blocked)
            return row
        except Exception as error:
            self.failed = True
            (self.output / 'failure.json').write_text(canonical({
                'error_type': type(error).__name__, 'message': str(error),
                'accepted_samples': self.count, 'transmitted': False}) + '\n')
            raise

    def close(self, completed=True):
        if self.closed:
            return
        self.stream.close()
        self.closed = True
        result = {'schema': 'flybrain.shadow.result/1', 'completed': bool(completed and not self.failed and self.count),
                  'samples': self.count, 'blocked_samples': self.blocked,
                  'brain_substeps': self.policy.brain_tick, 'final_sha256': self.chain,
                  'transmitted_commands': 0, 'REAL_FLIGHT_READY': False,
                  'hardware_validated': False, 'model': self.policy.metadata}
        (self.output / 'result.json').write_text(canonical(result) + '\n')
        return result

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(completed=kind is None)
