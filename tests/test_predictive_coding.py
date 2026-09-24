"""Predictive coding: exact gradients, the Gaussian posterior, and the link to backprop."""

from itertools import pairwise

import pytest
import torch

from neurosush.predictive_coding import PredictiveCodingNetwork

f64 = torch.float64


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def network(sizes, **kwargs):
    return PredictiveCodingNetwork(sizes, dtype=f64, **kwargs)


def autograd_free_energy(net, mu):
    """``sum F`` with every state and parameter as a leaf that records gradients."""
    leaves = {
        "mu": [m.clone().requires_grad_() for m in mu],
        "weights": [w.clone().requires_grad_() for w in net.weights],
        "variances": [v.clone().requires_grad_() for v in net.variances],
        "prior": net.prior_mean.clone().requires_grad_(),
    }
    copy = network(net.sizes, activation="tanh")
    copy.weights, copy.variances, copy.prior_mean = (
        leaves["weights"],
        leaves["variances"],
        leaves["prior"],
    )
    copy.free_energy(leaves["mu"]).sum().backward()
    return leaves


class TestGradients:
    def state(self):
        net = network([4, 3, 2], variances=[0.5, 2.0, 1.5])
        net.prior_mean = torch.tensor([0.3, -0.2], dtype=f64)
        g = gen()
        mu = [torch.randn(5, n, generator=g, dtype=f64) for n in net.sizes]
        return net, mu

    def test_inference_follows_the_free_energy_gradient(self):
        net, mu = self.state()
        leaves = autograd_free_energy(net, mu)
        after = net.infer(mu[0], mu=mu, steps=1, rate=1e-3)
        for layer in (1, 2):
            torch.testing.assert_close((after[layer] - mu[layer]) / 1e-3, -leaves["mu"][layer].grad)

    def test_clamped_top_does_not_move(self):
        net, mu = self.state()
        after = net.infer(mu[0], top=mu[2], mu=mu, steps=5)
        torch.testing.assert_close(after[2], mu[2])
        torch.testing.assert_close(after[0], mu[0])

    def test_learning_follows_the_free_energy_gradient(self):
        net, mu = self.state()
        leaves = autograd_free_energy(net, mu)
        weights = [w.clone() for w in net.weights]
        variances = [v.clone() for v in net.variances]
        prior = net.prior_mean.clone()
        net.learn(mu, 1e-3, prior_rate=1e-3, variance_rate=1e-3)
        rows = mu[0].shape[0]  # updates are batch means; F was summed
        for new, old, leaf in zip(net.weights, weights, leaves["weights"], strict=True):
            torch.testing.assert_close((new - old) / 1e-3, -leaf.grad / rows)
        for new, old, leaf in zip(net.variances, variances, leaves["variances"], strict=True):
            torch.testing.assert_close((new - old) / 1e-3, -leaf.grad / rows)
        torch.testing.assert_close((net.prior_mean - prior) / 1e-3, -leaves["prior"].grad / rows)

    def test_free_energy_decreases_monotonically_to_a_fixed_point(self):
        net = network([6, 5, 4, 3])
        x = torch.randn(8, 6, generator=gen(1), dtype=f64)
        mu = net.infer(x, steps=0)
        energies = []
        for _ in range(400):
            energies.append(net.free_energy(mu).sum().item())
            mu = net.infer(x, mu=mu, steps=1, rate=0.05)
        assert all(b <= a + 1e-12 for a, b in pairwise(energies))
        assert energies[-1] < energies[0]
        assert energies[-2] - energies[-1] < 1e-7
        leaves = autograd_free_energy(net, mu)
        for layer in (1, 2, 3):
            assert leaves["mu"][layer].grad.abs().max().item() < 1e-3


class TestLinearGaussian:
    def test_inference_converges_to_the_exact_posterior_mean(self):
        # x = W z + noise(sx), z ~ N(mp, sp): the posterior mean of z is
        # (W^T W / sx + I / sp)^-1 (W^T x / sx + mp / sp)
        net = network([5, 3], activation="linear", variances=[0.5, 2.0], seed=3)
        net.prior_mean = torch.tensor([0.5, -1.0, 0.2], dtype=f64)
        x = torch.randn(4, 5, generator=gen(), dtype=f64)
        w, sx, sp, mp = net.weights[0], 0.5, 2.0, net.prior_mean
        precision = w.T @ w / sx + torch.eye(3, dtype=f64) / sp
        exact = torch.linalg.solve(precision, (x @ w / sx + mp / sp).T).T
        torch.testing.assert_close(net.infer(x, steps=2000, rate=0.1)[1], exact)

    def test_learning_recovers_the_generating_subspace(self):
        g = gen(5)
        true = torch.randn(10, 3, generator=g, dtype=f64)
        data = torch.randn(2000, 3, generator=g, dtype=f64) @ true.T
        data += 0.1 * torch.randn(2000, 10, generator=g, dtype=f64)
        net = network([10, 3], activation="linear", seed=6)
        for _ in range(30):
            for batch in data.split(100):
                net.learn(net.infer(batch, steps=50, rate=0.1), 0.05)
        basis_true, _ = torch.linalg.qr(true)
        basis_learned, _ = torch.linalg.qr(net.weights[0])
        cosines = torch.linalg.svdvals(basis_true.T @ basis_learned)  # principal angles
        assert cosines.min().item() > 0.99

    def test_prior_and_variance_learn_the_maximum_likelihood_gaussian(self):
        # with a single layer, F is the Gaussian negative log likelihood of the data
        data = 2.0 + 0.7 * torch.randn(5000, 3, generator=gen(7), dtype=f64)
        net = network([3], activation="linear")
        for _ in range(1000):
            net.learn(net.infer(data), 0.0, prior_rate=0.05, variance_rate=0.05)
        torch.testing.assert_close(net.prior_mean, data.mean(0))
        torch.testing.assert_close(net.variances[0], data.var(0, correction=0))


class TestBackpropLimit:
    """Whittington and Bogacz (2017): with the top clamped to an input and a small output
    error, the relaxed prediction errors approach backprop's weight gradients as the
    hidden layers' variances shrink relative to the output's."""

    def deviation(self, ratio):
        net = network([2, 6, 5, 3], variances=[1.0, ratio, ratio, 1.0], seed=8)
        x = torch.randn(3, generator=gen(9), dtype=f64)
        target = net.predict(x)[0] + 1e-3 * torch.randn(2, generator=gen(10), dtype=f64)
        mu = net.infer(target, top=x, steps=4000, rate=0.3 * ratio)
        eps = net._eps(mu)
        updates = [eps[layer].unsqueeze(-1) * net.f(mu[layer + 1]) for layer in range(3)]

        weights = [w.clone().requires_grad_() for w in net.weights]
        output = x
        for weight in reversed(weights):
            output = net.f(output) @ weight.T
        (0.5 * ((target - output) ** 2).sum()).backward()
        return max(
            ((update + w.grad).norm() / w.grad.norm()).item()
            for update, w in zip(updates, weights, strict=True)
        )

    def test_weight_updates_converge_to_backprop(self):
        deviations = [self.deviation(ratio) for ratio in (1e-1, 1e-2, 1e-3)]
        # the relative deviation shrinks in proportion to the variance ratio
        for coarse, fine in pairwise(deviations):
            assert fine == pytest.approx(coarse / 10, rel=0.2)
        assert deviations[-1] < 0.005


class TestValidation:
    @pytest.mark.parametrize(
        ("args", "kwargs", "match"),
        [
            ([[3, 0]], {}, "sizes"),
            ([[3, 2]], {"activation": "relu6"}, "activation"),
            ([[3, 2]], {"variances": [1.0]}, "variance"),
            ([[3, 2]], {"variances": [1.0, -1.0]}, "variance"),
        ],
    )
    def test_invalid_arguments(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PredictiveCodingNetwork(*args, **kwargs)
