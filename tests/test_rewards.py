import itertools

import pytest
import torch
import torch.nn.functional as F

from rlcd.env import BanditEnv
from rlcd.rewards import (REWARDS, kl_categorical, policy_gradient_loss, reward_rlcd,
                          reward_rlvr, supervised_loss)
from rlcd.schema import NEG


def test_env_step_reveals_only_correctness():
    env = BanditEnv(torch.tensor([2, 0, 1]))
    idx = torch.tensor([0, 2])
    actions = torch.tensor([[2, 1], [1, 1]])
    out = env.step(idx, actions)
    assert out.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert len(env) == 3
    assert env.reveal(idx).tolist() == [2, 1]


def test_env_step_rejects_misshapen_actions():
    env = BanditEnv(torch.tensor([2, 0, 1]))
    with pytest.raises(AssertionError):
        env.step(torch.tensor([0, 2]), torch.tensor([2, 1]))
    with pytest.raises(AssertionError):
        env.step(torch.tensor([0, 2]), torch.tensor([[2, 1]]))


def test_reward_functions():
    outcomes = torch.tensor([[1.0, 0.0]])
    p_a = torch.tensor([[0.9, 0.9]])
    assert reward_rlvr(outcomes, p_a).tolist() == [[1.0, 0.0]]
    assert torch.allclose(reward_rlcd(outcomes, p_a), torch.tensor([[0.1, -0.9]]))
    assert set(REWARDS) == {"rlvr", "rlcd"}


def _exact_reinforce_grad(theta, y, reward_fn):
    """Enumerate every action: E_a[ r(a) grad log p_a ]."""
    p = torch.softmax(theta, 0)
    logp = torch.log_softmax(theta, 0)
    pd = p.detach()
    surrogate = torch.zeros(())
    for a in range(theta.numel()):
        c = torch.tensor(float(a == y))
        r = reward_fn(c[None, None], pd[a][None, None])[0, 0]
        surrogate = surrogate + pd[a] * r * logp[a]
    (g,) = torch.autograd.grad(surrogate, theta)
    return g


def test_rlcd_reinforce_is_unbiased_for_brier_gradient():
    torch.manual_seed(0)
    theta = torch.randn(5, requires_grad=True)
    y = 2
    p = torch.softmax(theta, 0)
    brier = -((p - F.one_hot(torch.tensor(y), 5).float()) ** 2).sum()
    (g_brier,) = torch.autograd.grad(brier, theta)
    g_reinforce = _exact_reinforce_grad(theta, y, reward_rlcd)
    assert torch.allclose(2 * g_reinforce, g_brier, atol=1e-6)


def test_rlvr_reinforce_is_accuracy_gradient():
    torch.manual_seed(1)
    theta = torch.randn(5, requires_grad=True)
    y = 3
    p = torch.softmax(theta, 0)
    (g_acc,) = torch.autograd.grad(p[y], theta)
    g_reinforce = _exact_reinforce_grad(theta, y, reward_rlvr)
    assert torch.allclose(g_reinforce, g_acc, atol=1e-6)


def test_policy_gradient_loss_leave_one_out_baseline():
    logp = torch.log_softmax(torch.zeros(1, 26), -1).requires_grad_(True)
    actions = torch.tensor([[0, 1, 2]])
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    loss = policy_gradient_loss(logp, actions, rewards)
    # baselines: for a0 mean(0,0)=0 -> adv 1; for a1 mean(1,0)=.5 -> adv -.5; a2 -> -.5
    expected = -(1.0 * logp[0, 0] - 0.5 * logp[0, 1] - 0.5 * logp[0, 2]) / 3
    assert torch.allclose(loss, expected)


def test_policy_gradient_loss_single_sample_has_no_baseline():
    logp = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]), -1)
    loss = policy_gradient_loss(logp, torch.tensor([[0], [1]]), torch.tensor([[1.0], [0.5]]))
    expected = -(1.0 * logp[0, 0] + 0.5 * logp[1, 1]) / 2
    assert torch.allclose(loss, expected)
    assert not torch.allclose(loss, torch.tensor(0.0))


def test_production_loss_with_group_of_two_is_unbiased_for_half_brier():
    """Drive policy_gradient_loss itself: enumerate every ordered action pair of a 3-way
    softmax, weight each pair's loss gradient by its joint probability, and compare with the
    gradient of Brier / 2. The loss is a mean over B*G = 2 terms with a minus sign; the
    leave-one-out baseline doubles the per-term advantage scale, so the two cancel."""
    torch.manual_seed(0)
    theta = torch.randn(3, requires_grad=True)
    y = 1
    env = BanditEnv(torch.tensor([y]))
    p = torch.softmax(theta, 0).detach()
    expected_grad = torch.zeros(3)
    for a0, a1 in itertools.product(range(3), repeat=2):
        logp = torch.log_softmax(theta, 0)[None, :]
        actions = torch.tensor([[a0, a1]])
        outcomes = env.step(torch.tensor([0]), actions)
        rewards = reward_rlcd(outcomes, p[actions])
        loss = policy_gradient_loss(logp, actions, rewards)
        (g,) = torch.autograd.grad(loss, theta)
        expected_grad += p[a0] * p[a1] * g
    brier = ((torch.softmax(theta, 0) - F.one_hot(torch.tensor(y), 3).float()) ** 2).sum()
    (g_brier,) = torch.autograd.grad(brier, theta)
    assert torch.allclose(expected_grad, 0.5 * g_brier, atol=1e-5)
    assert g_brier.abs().max() > 1e-2


def test_kl_is_zero_for_identical_and_finite_with_masking():
    logits = torch.tensor([[1.0, 2.0, NEG, NEG]])
    logp = torch.log_softmax(logits, -1)
    assert torch.allclose(kl_categorical(logp, logp), torch.zeros(1))
    other = torch.log_softmax(torch.tensor([[2.0, 1.0, NEG, NEG]]), -1)
    kl = kl_categorical(logp, other)
    assert torch.isfinite(kl).all() and kl.item() > 0


def test_kl_value_is_pinned():
    logp = torch.log_softmax(torch.tensor([[1.0, 2.0]]), -1)
    ref = torch.log_softmax(torch.tensor([[2.0, 1.0]]), -1)
    # p = [.2689, .7311], log ratio = [-1, 1] -> .7311 - .2689
    assert torch.allclose(kl_categorical(logp, ref), torch.tensor([0.4621]), atol=1e-3)


def test_kl_backward_is_finite_and_zero_at_masked_entries():
    logits = torch.tensor([[1.0, 2.0, NEG, NEG]], requires_grad=True)
    ref = torch.log_softmax(torch.tensor([[2.0, 1.0, NEG, NEG]]), -1)
    kl_categorical(torch.log_softmax(logits, -1), ref).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].item() == 0.0 and logits.grad[0, 3].item() == 0.0
    assert logits.grad[0, :2].abs().min() > 0


def test_kl_sends_no_gradient_to_the_reference():
    logits = torch.tensor([[1.0, 2.0, NEG, NEG]], requires_grad=True)
    ref_logits = torch.tensor([[2.0, 1.0, NEG, NEG]], requires_grad=True)
    kl_categorical(torch.log_softmax(logits, -1), torch.log_softmax(ref_logits, -1)).sum().backward()
    assert logits.grad is not None and ref_logits.grad is None


def test_supervised_loss_is_nll():
    logp = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0]]), -1)
    assert torch.allclose(supervised_loss(logp, torch.tensor([2])), -logp[0, 2])
