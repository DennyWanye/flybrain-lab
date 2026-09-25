import numpy as np
import torch

from flydrone.policy import ActorCritic
from flydrone.ppo import compute_gae, ppo_update


def test_gae_terminal_and_truncation_masks():
    adv, ret = compute_gae(np.array([[1.]], np.float32), np.array([[2.]], np.float32), np.array([[5.]], np.float32), np.array([[True]]), np.array([[False]]), .9, .95)
    np.testing.assert_allclose(adv, [[-1.]])
    adv, _ = compute_gae(np.array([[1.]], np.float32), np.array([[2.]], np.float32), np.array([[5.]], np.float32), np.array([[False]]), np.array([[True]]), .9, .95)
    np.testing.assert_allclose(adv, [[3.5]])


def test_ppo_changes_policy_parameters_without_brain():
    model = ActorCritic(8); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {"features": torch.randn(16, 8), "actions": torch.randint(0, 5, (16,)), "old_log_probs": torch.zeros(16), "advantages": torch.ones(16), "returns": torch.ones(16)}
    before = [p.detach().clone() for p in model.parameters()]; ppo_update(model, optimizer, batch, epochs=1, minibatch=16)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
