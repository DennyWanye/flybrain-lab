#!/usr/bin/env python3
"""FlyBrain Lab v0.1: engineered frozen-connectome reservoir + PPO.

This is NOT a numerical reproduction of the Shiu et al. biological model.
Only the small CPU policy/value readout is trained; the sparse reservoir is frozen.
Run `python flylab.py --help`. See the accompanying Chinese deployment guide.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist
from torch import nn
from torch.distributions import Categorical

VERSION = "0.1.0"
OBS_DIM = 9


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checked_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. No silent CPU fallback.")
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device.index)
    return device


def load_graph(path: str | Path) -> tuple[sp.csr_matrix, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        n = int(z["n"])
        w = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
        ids = z["ids"].astype(str)
    w.check_format(full_check=True)
    if len(ids) != n or len(np.unique(ids)) != n:
        raise ValueError("Neuron IDs must be unique and match matrix dimensions.")
    if not np.isfinite(w.data).all():
        raise ValueError("Nonfinite edge weights.")
    return w.astype(np.float32), ids


def prepare_graph(a: argparse.Namespace) -> None:
    """Convert data without loading any N-by-N dense matrix."""
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new path.")
    inputs: dict[str, str] = {}
    if a.source == "synthetic":
        if a.neurons < 128:
            raise ValueError("Use at least 128 synthetic neurons.")
        rng = np.random.default_rng(a.seed)
        n = a.neurons
        rows, cols = rng.integers(n, size=(2, n * 16))
        # Dale-like source sign is only for a software fixture, not real fly data.
        signs = rng.choice([-1., 1.], size=n, p=[.2, .8])
        vals = rng.uniform(.2, 1., len(rows)) * signs[cols]
        w = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        ids = np.array([f"SYNTHETIC_{i}" for i in range(n)])
    elif a.source == "shiu":
        import pandas as pd
        if not a.complete or not a.connectivity:
            raise ValueError("shiu requires --complete and --connectivity.")
        # Preserve row order: source parquet endpoints are row indices, not IDs.
        comp = pd.read_csv(a.complete, dtype=str, keep_default_na=False)
        ids = comp.iloc[:, 0].to_numpy(dtype=str)
        if any(not value.isdigit() for value in ids):
            raise ValueError("First completeness column must contain integer ID strings.")
        n = len(ids)
        frame = pd.read_parquet(a.connectivity)
        required = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Unexpected parquet schema; missing {sorted(missing)}")
        endpoints = []
        for key in required[:2]:
            values = frame[key].to_numpy()
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Invalid integer endpoint column {key}")
            values = values.astype(np.int64)
            if values.size and (values.min() < 0 or values.max() >= n):
                raise ValueError(f"Endpoint index out of range: {key}")
            endpoints.append(values)
        vals = frame[required[2]].to_numpy(dtype=np.float32)
        w = sp.coo_matrix((vals, (endpoints[1], endpoints[0])), shape=(n, n)).tocsr()
        inputs = {str(Path(p).name): sha256(p) for p in [a.complete, a.connectivity]}
    elif a.source == "male":
        if not a.weights or not a.brain:
            raise ValueError("male requires --weights and --brain.")
        w = sp.load_npz(a.weights).tocsr()
        with np.load(a.brain, allow_pickle=False) as z:
            ids = z["ids"].astype(str)
        n = len(ids)
        if w.shape != (n, n):
            raise ValueError("MaleCNS metadata/weight shape mismatch.")
        inputs = {str(Path(p).name): sha256(p) for p in [a.weights, a.brain]}
    else:
        raise ValueError(a.source)
    if n == 0 or len(np.unique(ids)) != n or not np.isfinite(w.data).all():
        raise ValueError("Empty graph, duplicate IDs, or nonfinite weights.")
    w.sum_duplicates()
    w.eliminate_zeros()
    # Engineering normalization, not paper-exact synaptic conductance.
    sums = np.asarray(abs(w).sum(axis=1)).ravel()
    w = (sp.diags(1. / np.maximum(sums, 1.)) @ w).tocsr().astype(np.float32)
    w.sort_indices()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        np.savez_compressed(f, n=np.int64(n), indptr=w.indptr.astype(np.int64),
                            indices=w.indices.astype(np.int64), data=w.data, ids=ids)
    meta = {"version": VERSION, "source": a.source, "synthetic": a.source == "synthetic",
            "neurons": n, "nonzero_neuron_pair_edges": int(w.nnz),
            "orientation": "W[postsynaptic, presynaptic]",
            "normalization": "divide each row by max(absolute incoming sum, 1)",
            "file_sha256": sha256(out), "input_sha256": inputs,
            "warning": "Engineered reservoir, not a biologically validated whole-brain digital twin."}
    write_json(str(out) + ".json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


class Reservoir:
    """Sparse, frozen simplified LIF network. Batch state has shape (neurons, envs).

    Equation per internal step:
      v <- exp(-dt/tau)*v + gain*W*s + tonic + engineered sensory drive
      s <- 1[v >= 1]; v[s] <- 0; trace <- .8*trace + .2*s
    dt=.020, tau=.100; no biological synaptic delay/refractory/receptor model.
    """
    def __init__(self, path: str, batch: int, device: str = "cpu", mode: str = "brain",
                 graph_seed: int = 64, readout: int = 128, internal_steps: int = 4,
                 gain: float = 1., tonic: float = .14, layout: str = "csr"):
        if internal_steps < 1 or batch < 1 or readout < 1:
            raise ValueError("Invalid reservoir dimensions.")
        self.device = checked_device(device)
        self.batch, self.mode = batch, mode
        self.internal_steps, self.gain, self.tonic = internal_steps, gain, tonic
        self.n_features = 2 * readout
        w, _ = load_graph(path)
        self.n = w.shape[0]
        rng = np.random.default_rng(graph_seed)
        per_channel = min(16, max(2, self.n // (OBS_DIM * 8)))
        inputs = rng.choice(self.n, OBS_DIM * per_channel, replace=False)
        # Keep identical I/O mappings across brain/shuffled/no_recurrence controls.
        # Read out one-hop targets, excluding the injected input cells themselves.
        indicator = np.zeros(self.n, dtype=np.float32)
        indicator[inputs] = 1
        targets = np.asarray(abs(w) @ indicator).ravel()
        targets[inputs] = 0
        candidates = np.flatnonzero(targets > 0)
        if len(candidates) < readout:
            raise ValueError(f"Only {len(candidates)} non-input one-hop targets; reduce --readout.")
        outputs = rng.choice(candidates, readout, replace=False)
        if mode == "shuffled":
            # Global source-label permutation. Not per-neuron degree-preserving rewiring.
            permutation = rng.permutation(self.n)
            w = sp.csr_matrix((w.data.copy(), permutation[w.indices], w.indptr.copy()),
                              shape=w.shape)
            w.sort_indices()
        self.mapping_sha256 = hashlib.sha256(inputs.tobytes() + outputs.tobytes()).hexdigest()
        self.inputs = torch.as_tensor(inputs, dtype=torch.long, device=self.device)
        self.channels = torch.as_tensor(np.repeat(np.arange(OBS_DIM), per_channel),
                                        dtype=torch.long, device=self.device)
        self.outputs = torch.as_tensor(outputs, dtype=torch.long, device=self.device)
        self.w = torch.sparse_csr_tensor(
            torch.as_tensor(w.indptr.astype(np.int64), device=self.device),
            torch.as_tensor(w.indices.astype(np.int64), device=self.device),
            torch.as_tensor(w.data, device=self.device),
            size=w.shape, dtype=torch.float32, device=self.device)
        if layout == "coo":
            self.w = self.w.to_sparse_coo().coalesce()
        self.v = torch.zeros((self.n, batch), dtype=torch.float32, device=self.device)
        self.s = torch.zeros_like(self.v)
        self.trace = torch.zeros_like(self.v)
        self.last_activity = 0.

    @torch.no_grad()
    def reset(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.v.zero_(); self.s.zero_(); self.trace.zero_()
        elif np.any(mask):
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            self.v[:, m] = 0; self.s[:, m] = 0; self.trace[:, m] = 0

    @torch.no_grad()
    def features(self, observations: np.ndarray) -> torch.Tensor:
        if observations.shape != (self.batch, OBS_DIM):
            raise ValueError(f"Expected observation shape {(self.batch, OBS_DIM)}")
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        drive = .8 * obs[:, self.channels].T
        for _ in range(self.internal_steps):
            if self.mode != "no_recurrence":
                current = torch.sparse.mm(self.w, self.s)
                self.v.mul_(math.exp(-.020 / .100)).add_(current, alpha=self.gain)
            else:
                self.v.mul_(math.exp(-.020 / .100))
            self.v.add_(self.tonic)
            self.v[self.inputs] += drive
            fired = self.v >= 1.
            self.s.copy_(fired)
            self.v.masked_fill_(fired, 0.)
            self.trace.mul_(.8).add_(self.s, alpha=.2)
        self.last_activity = float(self.s.mean().item())
        features = torch.cat((self.v[self.outputs].T, self.trace[self.outputs].T), dim=1)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite reservoir state; inspect parameters.")
        return features.cpu().contiguous()


class Maze:
    """Small fixed map, varying free start/goal, fully observable engineered sensors.

    Reaching the goal OR the visible task deadline is a true finite-horizon terminal.
    This is not Gym's generic TimeLimit truncation. No sensor claims about real flies.
    """
    MOVES = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    def __init__(self, batch: int, seed: int, horizon: int = 64):
        self.batch, self.horizon = batch, horizon
        self.rng = np.random.default_rng(seed)
        self.walls = np.zeros((7, 7), dtype=bool)
        self.walls[0, :] = self.walls[-1, :] = True
        self.walls[:, 0] = self.walls[:, -1] = True
        self.walls[1:4, 3] = True
        self.free = np.argwhere(~self.walls)[:, ::-1].copy()  # x,y
        self.pos = np.zeros((batch, 2), dtype=np.int64)
        self.goal = np.zeros_like(self.pos)
        self.age = np.zeros(batch, dtype=np.int64)
        self.returns = np.zeros(batch, dtype=np.float32)
        self.reset(np.ones(batch, dtype=bool))

    def reset(self, mask: np.ndarray) -> None:
        for i in np.flatnonzero(mask):
            choices = self.rng.choice(len(self.free), 2, replace=False)
            self.pos[i], self.goal[i] = self.free[choices]
            self.age[i] = 0
            self.returns[i] = 0

    def observations(self) -> np.ndarray:
        d = (self.goal - self.pos) / 5.
        relative = np.stack([np.maximum(d[:, 0], 0), np.maximum(-d[:, 0], 0),
                             np.maximum(d[:, 1], 0), np.maximum(-d[:, 1], 0)], axis=1)
        neighbor = self.pos[:, None, :] + self.MOVES[None, :, :]
        blocked = self.walls[neighbor[:, :, 1], neighbor[:, :, 0]]
        remain = (1. - self.age / self.horizon)[:, None]
        return np.concatenate((relative, blocked, remain), axis=1).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        if actions.shape != (self.batch,) or not np.isin(actions, np.arange(4)).all():
            raise ValueError("Invalid action array.")
        nxt = self.pos + self.MOVES[actions]
        blocked = self.walls[nxt[:, 1], nxt[:, 0]]
        self.pos[~blocked] = nxt[~blocked]
        self.age += 1
        success = np.all(self.pos == self.goal, axis=1)
        rewards = (-.01 - .02 * blocked + success).astype(np.float32)
        self.returns += rewards
        done = success | (self.age >= self.horizon)
        episodes = [{"return": float(self.returns[i]), "success": int(success[i]),
                     "length": int(self.age[i])} for i in np.flatnonzero(done)]
        return rewards, done, episodes

    def text(self, i: int = 0) -> str:
        rows = np.where(self.walls, "#", ".")
        x, y = self.goal[i]; rows[y, x] = "G"
        x, y = self.pos[i]; rows[y, x] = "F"
        return "\n".join("".join(row) for row in rows)


class Policy(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        # Fixed scale + no running statistics; same features reused during PPO update.
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh())
        self.actor, self.critic = nn.Linear(128, 4), nn.Linear(128, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2)); nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)

    def forward(self, features: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        hidden = self.net(features)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
        last_value: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        nxt = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 1. - dones[t]
        delta = rewards[t] + gamma * nxt * mask - values[t]
        running = delta + gamma * lam * mask * running
        advantages[t] = running
    return advantages, advantages + values


def init_dist() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo", timeout=timedelta(minutes=10))
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def agree(value: Any, world: int) -> None:
    if world > 1:
        all_values: list[Any] = [None] * world
        dist.all_gather_object(all_values, value)
        if any(item != all_values[0] for item in all_values):
            raise RuntimeError("Workers disagree about code/data/config: " + repr(all_values))


def mean_gradients(model: nn.Module, world: int) -> None:
    if world > 1:
        params = list(model.parameters())
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        offset = 0
        for p in params:
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


def standardize(advantages: torch.Tensor, world: int) -> torch.Tensor:
    stats = torch.tensor([advantages.double().sum(), advantages.double().square().sum(),
                          advantages.numel()], dtype=torch.float64)
    if world > 1:
        dist.all_reduce(stats)
    mean = stats[0] / stats[2]
    var = (stats[1] / stats[2] - mean.square()).clamp_min(0.)
    return (advantages - float(mean)) / (math.sqrt(float(var)) + 1e-8)


def reservoir_from(a: argparse.Namespace, batch: int) -> Reservoir | None:
    if a.mode == "direct":
        return None
    return Reservoir(a.graph, batch, a.device, a.mode, a.graph_seed, a.readout,
                     a.internal_steps, a.gain, a.tonic, a.layout)


def get_features(brain: Reservoir | None, env: Maze) -> torch.Tensor:
    obs = env.observations()
    return brain.features(obs) if brain else torch.from_numpy(obs)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(a: argparse.Namespace) -> None:
    rank, world = init_dist()
    try:
        if a.steps * a.envs % a.minibatch:
            raise ValueError("--steps * --envs must be divisible by --minibatch.")
        checked_device(a.device)  # cuda means an actual GPU must exist, even for controls.
        signature = sha256(a.graph)
        agreement = {k: v for k, v in vars(a).items()
                     if k not in {"graph", "out", "device", "func", "warm_start"}}
        agree({"graph": signature, "code": sha256(__file__), "config": agreement,
               "torch": str(torch.__version__)}, world)
        out = Path(a.out)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            if (out / "metrics.jsonl").exists():
                raise FileExistsError("Use a new --out directory; previous run exists.")
        if world > 1:
            dist.barrier()
        set_seed(a.seed)
        brain = reservoir_from(a, a.envs)
        model = Policy(brain.n_features if brain else OBS_DIM)  # explicitly CPU
        if a.warm_start:
            saved = torch.load(a.warm_start, map_location="cpu", weights_only=True)
            for k in ["mode", "graph_seed", "readout", "internal_steps", "gain", "tonic"]:
                if saved["config"][k] != getattr(a, k):
                    raise ValueError(f"Warm-start configuration differs: {k}")
            if saved["graph_sha256"] != signature:
                raise ValueError("Warm-start graph hash mismatch.")
            model.load_state_dict(saved["policy"])
        state_hash = hashlib.sha256(torch.nn.utils.parameters_to_vector(
            model.parameters()).detach().numpy().tobytes()).hexdigest()
        agree(state_hash, world)
        # Rank-specific environment and action sampling, after identical model init.
        set_seed(a.seed + rank * 100003)
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
        env = Maze(a.envs, a.seed + rank * 100003, a.horizon)
        features = get_features(brain, env)
        config = {k: v for k, v in vars(a).items() if k != "func"}
        config.update(version=VERSION, world_size=world)
        if rank == 0:
            write_json(out / "config.json", {**config, "graph_sha256": signature,
                      "mapping_sha256": brain.mapping_sha256 if brain else None,
                      "trainable_parameters": sum(p.numel() for p in model.parameters()),
                      "python": platform.python_version(), "torch": str(torch.__version__)})
        start = time.perf_counter()
        for update in range(1, a.updates + 1):
            xs, acts, logs, vals, rewards, dones, episodes = [], [], [], [], [], [], []
            for _ in range(a.steps):
                with torch.no_grad():
                    distribution, value = model(features)
                    action = distribution.sample()
                    old_log = distribution.log_prob(action)
                reward, done, ended = env.step(action.numpy())
                xs.append(features); acts.append(action); logs.append(old_log); vals.append(value)
                rewards.append(torch.from_numpy(reward)); dones.append(torch.from_numpy(done).float())
                episodes.extend(ended)
                env.reset(done)
                if brain:
                    brain.reset(done)
                features = get_features(brain, env)
            with torch.no_grad():
                _, last_value = model(features)
                advantage, targets = gae(torch.stack(rewards), torch.stack(vals), torch.stack(dones),
                                          last_value, a.gamma, a.gae_lambda)
            x = torch.stack(xs).flatten(0, 1)
            actions = torch.stack(acts).flatten()
            old_logs = torch.stack(logs).flatten()
            advantages = standardize(advantage.flatten(), world)
            targets = targets.flatten()
            loss_total, kl_total, batches = 0., 0., 0
            for _ in range(a.epochs):
                order = torch.randperm(len(x))
                for idx in order.split(a.minibatch):
                    distribution, value = model(x[idx])
                    new_log = distribution.log_prob(actions[idx])
                    ratio = (new_log - old_logs[idx]).exp()
                    plain = advantages[idx] * ratio
                    clipped = advantages[idx] * ratio.clamp(1 - a.clip, 1 + a.clip)
                    loss = -torch.minimum(plain, clipped).mean()
                    loss = loss + .5 * (value - targets[idx]).square().mean()
                    loss = loss - a.entropy * distribution.entropy().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite PPO loss.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    mean_gradients(model, world)
                    nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        kl_total += float(((ratio - 1.) - (new_log - old_logs[idx])).mean())
                    loss_total += float(loss.detach()); batches += 1
            totals = torch.tensor([len(episodes), sum(e["success"] for e in episodes),
                                    sum(e["return"] for e in episodes), loss_total / batches,
                                    kl_total / batches], dtype=torch.float64)
            if world > 1:
                dist.all_reduce(totals)
            # Catch parameter divergence instead of silently saving rank0 only.
            param_hash = hashlib.sha256(torch.nn.utils.parameters_to_vector(
                model.parameters()).detach().numpy().tobytes()).hexdigest()
            agree(param_hash, world)
            if rank == 0:
                count = float(totals[0])
                metrics = {"update": update, "env_steps": update * a.steps * a.envs * world,
                           "episodes": int(count), "success_rate": float(totals[1] / count) if count else None,
                           "mean_return": float(totals[2] / count) if count else None,
                           "loss": float(totals[3] / world), "approx_kl": float(totals[4] / world),
                           "last_spike_fraction_rank0": brain.last_activity if brain else None,
                           "wall_seconds": time.perf_counter() - start, "policy_sha256": param_hash}
                with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
                if update % a.save_every == 0 or update == a.updates:
                    atomic_checkpoint(out / "policy.pt", {"policy": model.state_dict(),
                        "config": config, "graph_sha256": signature, "updates": update,
                        "note": "Readout warm start only: optimizer and environment are not resumed."})
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate(a: argparse.Namespace) -> None:
    checked_device(a.device)
    set_seed(a.seed)
    checkpoint = None
    if a.checkpoint:
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["graph_sha256"] != sha256(a.graph):
            raise ValueError("Evaluation graph differs from training graph.")
        for key in ["mode", "graph_seed", "readout", "internal_steps", "gain", "tonic", "horizon"]:
            setattr(a, key, checkpoint["config"][key])
    elif not a.random:
        raise ValueError("Supply --checkpoint or --random.")
    # Exactly one complete episode at a time avoids selecting only faster-finishing envs.
    env = Maze(1, a.seed, a.horizon)
    brain = reservoir_from(a, 1) if not a.random else None
    model = Policy(brain.n_features if brain else OBS_DIM)
    if checkpoint and not a.random:
        model.load_state_dict(checkpoint["policy"])
    results = []
    trajectory = []
    for episode in range(a.episodes):
        if episode:
            env.reset(np.array([True]))
        if brain:
            brain.reset()
        for step in range(a.horizon):
            if episode < a.trace_episodes:
                trajectory.append(f"episode={episode} step={step}\n{env.text()}\n")
            with torch.no_grad():
                if a.random:
                    action = torch.randint(4, (1,))
                else:
                    distribution, _ = model(get_features(brain, env))
                    action = distribution.probs.argmax(-1) if a.greedy else distribution.sample()
            _, done, ended = env.step(action.numpy())
            if done[0]:
                results.extend(ended)
                break
    successes = sum(x["success"] for x in results)
    n = len(results)
    p = successes / n
    z = 1.96
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = {"episodes": n, "seed": a.seed, "successes": successes, "success_rate": p,
              "wilson_95_interval_descriptive": [center-half, center+half],
              "mean_return": float(np.mean([x["return"] for x in results])),
              "mean_length": float(np.mean([x["length"] for x in results])),
              "action_selection": "uniform_random" if a.random else "greedy" if a.greedy else "sampled",
              "graph_sha256": sha256(a.graph), "mode": a.mode,
              "warning": "Same fixed map; varying start/goal. Not an unseen-map generalization test."}
    write_json(a.out, report)
    Path(str(a.out) + ".trace.txt").write_text("\n".join(trajectory), encoding="utf-8")
    print(json.dumps(report, indent=2))


def gpu_check(a: argparse.Namespace) -> None:
    device = checked_device(a.device)
    crow = torch.tensor([0, 2, 3], device=device)
    col = torch.tensor([0, 1, 1], device=device)
    val = torch.tensor([2., -1., 3.], device=device)
    w = torch.sparse_csr_tensor(crow, col, val, size=(2, 2), device=device)
    x = torch.tensor([[1., 4.], [2., 5.]], device=device)
    result = torch.sparse.mm(w, x).cpu()
    torch.testing.assert_close(result, torch.tensor([[0., 3.], [6., 15.]]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps({"device": str(device), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "architecture": platform.machine(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "capability": torch.cuda.get_device_capability(device) if device.type == "cuda" else None,
        "csr_spmm": "PASS"}, indent=2))


def dist_check(a: argparse.Namespace) -> None:
    rank, world = init_dist()
    try:
        value = torch.tensor([rank + 1.], dtype=torch.float64)
        if world > 1:
            dist.all_reduce(value)
        assert value.item() == world * (world + 1) / 2
        print(json.dumps({"rank": rank, "world_size": world, "backend": "gloo/CPU",
                          "sum": value.item(), "status": "PASS"}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def bench(a: argparse.Namespace) -> None:
    brain = reservoir_from(a, a.envs)
    if brain is None:
        raise ValueError("Benchmark requires a reservoir mode.")
    obs = Maze(a.envs, a.seed).observations()
    for _ in range(3):
        brain.features(obs)
    brain.reset()
    first = brain.features(obs)
    brain.reset()
    changed = obs.copy(); changed[:, :4] = np.roll(changed[:, :4], 1, axis=1)
    second = brain.features(changed)
    contrast = float((first - second).abs().mean())
    if brain.device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    activity = []
    for _ in range(a.iterations):
        brain.features(obs)
        activity.append(brain.last_activity)
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    report = {"mode": a.mode, "device": a.device, "batch_envs": a.envs,
              "internal_steps_per_env_step": a.internal_steps,
              "batched_calls": a.iterations, "seconds": elapsed,
              "aggregate_env_steps_per_second": a.iterations*a.envs/elapsed,
              "last_spike_fraction": brain.last_activity,
              "mean_spike_fraction": float(np.mean(activity)),
              "saturation_warning": bool(np.mean(activity) > .5),
              "reset_input_contrast_mean_abs": contrast,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if brain.device.type == "cuda" else None,
              "graph_sha256": sha256(a.graph), "mapping_sha256": brain.mapping_sha256,
              "warning": "Reservoir-only benchmark including feature transfer; not PPO/FlyGym throughput."}
    write_json(a.out, report); print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4, help="CPU Torch threads per worker.")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", choices=["synthetic", "shiu", "male"], required=True)
    q.add_argument("--out", required=True)
    for arg in ["complete", "connectivity", "weights", "brain"]:
        q.add_argument("--"+arg)
    q.add_argument("--neurons", type=int, default=512)
    q.add_argument("--seed", type=int, default=64)
    q.add_argument("--overwrite", action="store_true")
    q.set_defaults(func=prepare_graph)
    q = sub.add_parser("inspect"); q.add_argument("--graph", required=True)
    q.set_defaults(func=lambda a: print(json.dumps({"sha256": sha256(a.graph),
        "shape": load_graph(a.graph)[0].shape, "nnz": load_graph(a.graph)[0].nnz,
        "metadata": json.loads(Path(a.graph+".json").read_text())
            if Path(a.graph+".json").exists() else None}, indent=2)))
    q = sub.add_parser("gpu-check"); q.add_argument("--device", default="cuda"); q.set_defaults(func=gpu_check)
    q = sub.add_parser("dist-check"); q.set_defaults(func=dist_check)
    for command, func in [("train", train), ("eval", evaluate), ("bench", bench)]:
        q = sub.add_parser(command)
        q.add_argument("--graph", required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--mode", choices=["brain", "shuffled", "no_recurrence", "direct"], default="brain")
        q.add_argument("--graph-seed", type=int, default=64)
        q.add_argument("--readout", type=int, default=128)
        q.add_argument("--internal-steps", type=int, default=4)
        q.add_argument("--gain", type=float, default=1.)
        q.add_argument("--tonic", type=float, default=.14)
        q.add_argument("--layout", choices=["csr", "coo"], default="csr")
        q.add_argument("--seed", type=int, default=1 if command == "train" else 900001)
        q.add_argument("--out", required=True)
        if command in {"train", "bench"}:
            q.add_argument("--envs", type=int, default=8)
        if command in {"train", "eval"}:
            q.add_argument("--horizon", type=int, default=64)
        if command == "train":
            q.add_argument("--steps", type=int, default=64)
            q.add_argument("--updates", type=int, default=100)
            q.add_argument("--epochs", type=int, default=4)
            q.add_argument("--minibatch", type=int, default=128)
            q.add_argument("--lr", type=float, default=3e-4)
            q.add_argument("--gamma", type=float, default=.99)
            q.add_argument("--gae-lambda", type=float, default=.95)
            q.add_argument("--clip", type=float, default=.2)
            q.add_argument("--entropy", type=float, default=.01)
            q.add_argument("--save-every", type=int, default=10)
            q.add_argument("--warm-start")
        elif command == "eval":
            q.add_argument("--checkpoint")
            q.add_argument("--random", action="store_true")
            q.add_argument("--greedy", action="store_true")
            q.add_argument("--episodes", type=int, default=200)
            q.add_argument("--trace-episodes", type=int, default=2)
        else:
            q.add_argument("--iterations", type=int, default=20)
        q.set_defaults(func=func)
    return p


def main() -> None:
    a = parser().parse_args()
    torch.set_num_threads(a.threads)
    for key in ["threads", "episodes", "horizon", "envs", "updates", "steps", "epochs", "minibatch", "save_every", "iterations"]:
        if hasattr(a, key) and getattr(a, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive.")
    a.func(a)


if __name__ == "__main__":
    main()
