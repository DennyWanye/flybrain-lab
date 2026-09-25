from __future__ import annotations

import numpy as np
import torch


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=.99, gae_lambda=.95):
    rewards, values, next_values = [np.asarray(x, np.float32) for x in (rewards, values, next_values)]
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    adv = np.zeros_like(rewards); running = np.zeros(rewards.shape[1:], np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        bootstrap = 1.0 - terminated[t]
        cont = 1.0 - np.logical_or(terminated[t], truncated[t]).astype(np.float32)
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        running = delta + gamma * gae_lambda * cont * running
        adv[t] = running
    return adv, adv + values


def ppo_update(model, optimizer, batch, epochs=4, minibatch=128, clip_ratio=.2,
               value_coef=.5, entropy_coef=.01, max_grad_norm=.5, target_kl=.03):
    features, actions = batch["features"], batch["actions"]
    old_log_probs, advantages, returns = batch["old_log_probs"], batch["advantages"], batch["returns"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    metrics = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "approx_kl": 0., "clip_fraction": 0., "gradient_norm": 0.}
    count = 0
    for _ in range(epochs):
        for indices in torch.randperm(len(features)).split(minibatch):
            policy, value = model(features[indices])
            logp = policy.log_prob(actions[indices]); ratio = torch.exp(logp - old_log_probs[indices])
            surr1 = ratio * advantages[indices]; surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages[indices]
            policy_loss = -torch.min(surr1, surr2).mean(); value_loss = .5 * (value - returns[indices]).square().mean(); entropy = policy.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(); loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm); optimizer.step()
            approx_kl = ((ratio - 1) - (logp - old_log_probs[indices])).mean().item()
            count += 1
            for key, val in [("policy_loss", policy_loss.item()), ("value_loss", value_loss.item()), ("entropy", entropy.item()), ("approx_kl", approx_kl), ("clip_fraction", (torch.abs(ratio - 1) > clip_ratio).float().mean().item()), ("gradient_norm", float(grad))]: metrics[key] += val
            if approx_kl > target_kl: break
    return {key: value / max(count, 1) for key, value in metrics.items()}
