from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, feature_dim: int, action_dim=5):
        super().__init__()
        self.feature_dim, self.action_dim = feature_dim, action_dim
        self.body = nn.Sequential(nn.Linear(feature_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, features):
        hidden = self.body(features)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden).squeeze(-1)

    def act(self, features, deterministic=False):
        policy, value = self(features)
        action = policy.probs.argmax(-1) if deterministic else policy.sample()
        return action, policy.log_prob(action), value

    def act_with_probs(self, features, deterministic=False):
        """Sample once and return the already-computed policy probabilities."""
        policy, value = self(features)
        action = policy.probs.argmax(-1) if deterministic else policy.sample()
        return action, policy.log_prob(action), value, policy.probs
