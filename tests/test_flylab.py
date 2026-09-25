import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flylab as f


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "synthetic.npz"
    f.prepare_graph(argparse.Namespace(source="synthetic", neurons=256, seed=64,
                                      out=str(path), overwrite=False))
    return path


def test_id_precision_and_edge_direction(tmp_path, monkeypatch):
    import pandas as pd
    comp, con, out = tmp_path / "comp.csv", tmp_path / "con.parquet", tmp_path / "graph.npz"
    ids = ["720575940660219265", "720575940660219266", "720575940660219267"]
    comp.write_text("root_id,other\n" + "\n".join(x + ",x" for x in ids))
    con.write_bytes(b"Mock parquet; real parquet reader is not tested here.")
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame({
        "Presynaptic_Index": [0, 1], "Postsynaptic_Index": [1, 2],
        "Excitatory x Connectivity": [2., -4.]}))
    f.prepare_graph(argparse.Namespace(source="shiu", complete=str(comp), connectivity=str(con),
                                      out=str(out), overwrite=False))
    w, actual = f.load_graph(out)
    assert actual.tolist() == ids
    assert w[1, 0] == 1 and w[2, 1] == -1 and w[0, 1] == 0


def test_male_converter(tmp_path):
    weights, meta, out = tmp_path / "weights.npz", tmp_path / "brain.npz", tmp_path / "converted.npz"
    w = sp.csr_matrix(np.array([[0, 2.], [-1., 0.]], dtype=np.float32))
    sp.save_npz(weights, w)
    np.savez(meta, ids=np.array(["10", "11"]))
    f.prepare_graph(argparse.Namespace(source="male", weights=str(weights), brain=str(meta),
                                      out=str(out), overwrite=False))
    got, ids = f.load_graph(out)
    assert ids.tolist() == ["10", "11"] and got[0, 1] == 1 and got[1, 0] == -1


def test_graph_normalization(graph):
    w, ids = f.load_graph(graph)
    assert max(np.asarray(abs(w).sum(axis=1)).ravel()) <= 1.00001
    assert w.nnz > 0 and len(ids) == 256
    assert json.loads(Path(str(graph)+".json").read_text())["synthetic"] is True


def test_overwrite_guard(graph):
    with pytest.raises(FileExistsError):
        f.prepare_graph(argparse.Namespace(source="synthetic", neurons=256, seed=64,
                                          out=str(graph), overwrite=False))


def test_reservoir_determinism_and_reset(graph):
    torch.set_num_threads(1)
    b = f.Reservoir(str(graph), 2, readout=32)
    obs = f.Maze(2, 10).observations()
    a = b.features(obs)
    b.reset()
    torch.testing.assert_close(a, b.features(obs))
    b.reset(np.array([True, False]))
    assert b.v[:, 0].abs().sum() == 0
    assert not b.w.requires_grad
    assert not a.requires_grad
    assert set(b.inputs.tolist()).isdisjoint(b.outputs.tolist())


def test_csr_matches_coo(graph):
    obs = f.Maze(2, 10).observations()
    csr = f.Reservoir(str(graph), 2, readout=32, layout="csr")
    coo = f.Reservoir(str(graph), 2, readout=32, layout="coo")
    torch.testing.assert_close(csr.features(obs), coo.features(obs), atol=2e-5, rtol=2e-5)


def test_controls_keep_mapping(graph):
    original = f.Reservoir(str(graph), 2, readout=32)
    shuffled = f.Reservoir(str(graph), 2, mode="shuffled", readout=32)
    none = f.Reservoir(str(graph), 2, mode="no_recurrence", readout=32)
    assert original.mapping_sha256 == shuffled.mapping_sha256 == none.mapping_sha256
    obs1, obs2 = f.Maze(2, 1).observations(), f.Maze(2, 2).observations()
    a = none.features(obs1); none.reset(); b = none.features(obs2)
    torch.testing.assert_close(a, b)
    original.reset(); a = original.features(obs1); original.reset(); b = original.features(obs2)
    assert (a - b).abs().max() > 0


def test_terminal_gae():
    r = torch.tensor([[1.], [2.]])
    v = torch.tensor([[.2], [.3]])
    done = torch.ones_like(r)
    advantage, ret = f.gae(r, v, done, torch.tensor([999.]), .99, .95)
    torch.testing.assert_close(ret, r)
    torch.testing.assert_close(advantage, r-v)


def test_maze_connected_and_deadline():
    maze = f.Maze(8, 2, horizon=4)
    seen, pending = set(), [tuple(maze.free[0])]
    while pending:
        xy = pending.pop()
        if xy in seen:
            continue
        seen.add(xy)
        for move in maze.MOVES:
            p = np.array(xy)+move
            if not maze.walls[p[1], p[0]] and tuple(p) not in seen:
                pending.append(tuple(p))
    assert len(seen) == len(maze.free)
    assert maze.observations().shape == (8, 9)
    for _ in range(4):
        _, done, _ = maze.step(np.zeros(8, dtype=int))
    assert done.all()


def test_policy_backward():
    model = f.Policy(64)
    policy, value = model(torch.randn(8, 64))
    loss = -policy.log_prob(torch.zeros(8, dtype=torch.int64)).mean()+value.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gpu_unavailable_is_error():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="No silent CPU fallback"):
            f.checked_device("cuda")


def test_reference_notebook_parser(tmp_path):
    spec = importlib.util.spec_from_file_location("reference", Path(__file__).resolve().parents[1]/"scripts/reference.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path = tmp_path/"example.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["x = [1, 2, 3]\n"]}]}))
    assert module.notebook_constant(path, "x") == [1, 2, 3]


def test_cuda_device_without_index_is_resolved(monkeypatch):
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert f.checked_device("cuda") == torch.device("cuda:0")
    assert selected == [0]
