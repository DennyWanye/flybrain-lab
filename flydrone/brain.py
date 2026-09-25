from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp

from .encoder import FeatureNormalizer, SensoryEncoder


def graph_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FrozenReservoir:
    """A frozen CSR LIF reservoir with independent state per batch environment."""

    def __init__(self, graph: str, batch: int, device="cpu", mapping_seed=64,
                 inputs_per_channel=16, readout_neurons=128, internal_steps=4,
                 gain=1.0, tonic=.14, input_gain=.8, normalizer=None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")
        with np.load(graph, allow_pickle=False) as z:
            n = int(z["n"]); w = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
            ids = z["ids"].astype(str)
        self.n, self.batch, self.graph = n, batch, graph
        self.graph_sha256 = graph_hash(graph)
        rng = np.random.default_rng(mapping_seed)
        channels = 14
        total_inputs = channels * inputs_per_channel
        if total_inputs > n:
            raise ValueError("graph too small for requested input mapping")
        inputs = rng.choice(n, total_inputs, replace=False)
        indicator = np.zeros(n, np.float32); indicator[inputs] = 1
        candidates = np.flatnonzero(np.asarray(abs(w) @ indicator).ravel() > 0)
        candidates = np.setdiff1d(candidates, inputs, assume_unique=False)
        if len(candidates) < readout_neurons:
            raise ValueError("not enough readout candidates")
        readouts = rng.choice(candidates, readout_neurons, replace=False)
        self.input_indices = inputs.astype(np.int64)
        self.input_channels = np.repeat(np.arange(channels), inputs_per_channel).astype(np.int64)
        self.readout_indices = readouts.astype(np.int64)
        self.input_ids, self.readout_ids = ids[inputs].tolist(), ids[readouts].tolist()
        self.mapping_sha256 = hashlib.sha256(inputs.tobytes() + self.input_channels.tobytes() + readouts.tobytes()).hexdigest()
        self.feature_dim = 2 * readout_neurons
        self.internal_steps, self.gain, self.tonic = internal_steps, gain, tonic
        self.encoder = SensoryEncoder(input_gain)
        self.normalizer = normalizer or FeatureNormalizer.identity(self.feature_dim)
        self.w = torch.sparse_csr_tensor(torch.as_tensor(w.indptr, dtype=torch.int64, device=self.device),
            torch.as_tensor(w.indices, dtype=torch.int64, device=self.device),
            torch.as_tensor(w.data, dtype=torch.float32, device=self.device), size=w.shape, device=self.device)
        self.inputs = torch.as_tensor(self.input_indices, dtype=torch.long, device=self.device)
        self.channels = torch.as_tensor(self.input_channels, dtype=torch.long, device=self.device)
        self.outputs = torch.as_tensor(self.readout_indices, dtype=torch.long, device=self.device)
        self.v = torch.zeros((n, batch), dtype=torch.float32, device=self.device)
        self.s = torch.zeros_like(self.v); self.trace = torch.zeros_like(self.v)
        self.advance_count = 0

    @torch.no_grad()
    def reset(self, mask=None):
        if mask is None:
            self.v.zero_(); self.s.zero_(); self.trace.zero_(); return
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self.v[:, m] = 0; self.s[:, m] = 0; self.trace[:, m] = 0

    @torch.no_grad()
    def advance(self, encoded_obs: np.ndarray, mask=None):
        encoded = torch.as_tensor(encoded_obs, dtype=torch.float32, device=self.device)
        if encoded.shape != (self.batch, 14):
            raise ValueError(f"expected {(self.batch, 14)} encoded observations")
        active = torch.ones(self.batch, dtype=torch.bool, device=self.device) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        active_idx = torch.nonzero(active, as_tuple=False).flatten()
        if active_idx.numel() == 0:
            return self.current_features()
        drive = .8 * encoded[:, self.channels].T
        decay = float(np.exp(-.020 / .100))
        for _ in range(self.internal_steps):
            current = torch.sparse.mm(self.w, self.s[:, active_idx])
            v_active = self.v[:, active_idx] * decay + self.gain * current + self.tonic
            v_active[self.inputs] += drive[:, active_idx]
            fired = v_active >= 1.
            v_active = v_active.masked_fill(fired, 0.)
            self.v[:, active_idx] = v_active
            self.s[:, active_idx] = fired.to(self.s.dtype)
            self.trace[:, active_idx] = self.trace[:, active_idx] * .8 + .2 * fired
        self.advance_count += int(active_idx.numel())
        return self.current_features()

    @torch.no_grad()
    def current_features(self) -> np.ndarray:
        raw = torch.cat((self.v[self.outputs].T, self.trace[self.outputs].T), dim=1)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("nonfinite reservoir feature")
        return self.normalizer(raw.cpu().numpy())

    def state_dict(self):
        return {"v": self.v.cpu(), "s": self.s.cpu(), "trace": self.trace.cpu(), "advance_count": self.advance_count}

    def load_state_dict(self, state):
        self.v.copy_(state["v"].to(self.device)); self.s.copy_(state["s"].to(self.device)); self.trace.copy_(state["trace"].to(self.device)); self.advance_count = int(state.get("advance_count", 0))

    def contract(self):
        return {"graph_sha256": self.graph_sha256, "mapping_sha256": self.mapping_sha256,
                "input_indices": self.input_indices, "input_channels": self.input_channels,
                "readout_indices": self.readout_indices, "input_ids": self.input_ids,
                "readout_ids": self.readout_ids, "feature_dim": self.feature_dim}
