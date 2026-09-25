import numpy as np
import sys

from flydrone.brain import FrozenReservoir


def test_brain_shapes_and_independent_reset(tmp_path):
    import subprocess
    graph = tmp_path / "g.npz"
    subprocess.run([sys.executable, "flylab.py", "prepare", "--source", "synthetic", "--neurons", "512", "--out", str(graph)], check=True)
    brain = FrozenReservoir(str(graph), 2, inputs_per_channel=2, readout_neurons=16)
    brain.advance(np.ones((2, 14), np.float32))
    before = brain.v[:, 1].clone()
    features = brain.current_features(); assert features.shape == (2, 32)
    count = brain.advance_count; _ = brain.current_features(); assert brain.advance_count == count
    brain.reset(np.array([True, False])); assert float(brain.v[:, 0].abs().sum()) == 0; assert float(brain.v[:, 1].abs().sum()) == float(before.abs().sum())


def test_mapping_contract_is_stable(tmp_path):
    import subprocess
    graph = tmp_path / "g.npz"; subprocess.run([sys.executable, "flylab.py", "prepare", "--source", "synthetic", "--neurons", "512", "--out", str(graph)], check=True)
    a = FrozenReservoir(str(graph), 1, inputs_per_channel=2, readout_neurons=16)
    b = FrozenReservoir(str(graph), 1, inputs_per_channel=2, readout_neurons=16)
    assert a.mapping_sha256 == b.mapping_sha256
