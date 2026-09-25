from __future__ import annotations

import numpy as np


class SensoryEncoder:
    """Encode signed navigation state into fourteen nonnegative channels."""

    def __init__(self, input_gain: float = 0.8):
        self.input_gain = float(input_gain)

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 8:
            raise ValueError("expected [batch, 8] observations")
        signed = obs[:, :6]
        out = np.concatenate((np.maximum(signed, 0), np.maximum(-signed, 0), obs[:, 6:8]), axis=1)
        # Interleave positive/negative pairs to make mapping channels explicit.
        out = out[:, [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 13]]
        return (out * self.input_gain).astype(np.float32)


class FeatureNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, std_floor=1e-3, clip_abs=5.0):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std_floor = float(std_floor)
        self.clip_abs = float(clip_abs)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean/std shape mismatch")

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.clip((features - self.mean) / np.maximum(self.std, self.std_floor),
                       -self.clip_abs, self.clip_abs).astype(np.float32)

    @classmethod
    def identity(cls, dim: int):
        return cls(np.zeros(dim, np.float32), np.ones(dim, np.float32))

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "std_floor": self.std_floor,
                "clip_abs": self.clip_abs}
