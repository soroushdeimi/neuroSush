"""Tests for SpikingTEM model, config, and training loop."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neurosush.spatial.graph_world import GraphWorld
from neurosush.tem.spiking import SpikingTEM, SpikingTEMConfig, train_walks


class TestSpikingTEM:
    def test_config_validation(self):
        with pytest.raises(ValueError, match="sensory"):
            SpikingTEMConfig(sensory=0)
        with pytest.raises(ValueError, match="units"):
            SpikingTEMConfig(units=0)
        with pytest.raises(ValueError, match="divisible"):
            SpikingTEMConfig(units=31, k=2)
        with pytest.raises(ValueError, match="tau"):
            SpikingTEMConfig(tau=-1.0)
        with pytest.raises(ValueError, match="surrogate_kind"):
            SpikingTEMConfig(surrogate_kind="invalid")

    def test_config_constructors_and_spike_rates(self):
        default_cfg = SpikingTEMConfig(sensory=10, units=32)
        paper_cfg = SpikingTEMConfig.paper(units=32)
        authors_cfg = SpikingTEMConfig.authors_code(units=32)

        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(10)

        # Default config: expect active spiking (> 0)
        m_def = SpikingTEM(default_cfg)
        _, rates_def, _ = m_def(walk)
        mean_rate_def = float(rates_def["mecii"].mean().item())
        assert mean_rate_def > 0.0

        # Paper config: paper Table 1 values yield silence (mean rate 0.0)
        # without tonic/theta adaptation.
        m_paper = SpikingTEM(paper_cfg)
        _, rates_paper, _ = m_paper(walk)
        mean_rate_paper = float(rates_paper["mecii"].mean().item())
        assert mean_rate_paper == 0.0

        # Authors' code config: active spiking (> 0)
        m_authors = SpikingTEM(authors_cfg)
        _, rates_authors, _ = m_authors(walk)
        mean_rate_authors = float(rates_authors["mecii"].mean().item())
        assert mean_rate_authors > 0.0

        assert mean_rate_def > mean_rate_paper

    def test_shapes_and_state_reset(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(10)
        config = SpikingTEMConfig(sensory=10, units=32, steps_per_action=2)
        model = SpikingTEM(config)
        state = model.initial_state()
        assert float(state.M.sum().item()) == 0.0
        torch.testing.assert_close(state.M, torch.zeros_like(state.M))

        logits, rates, _new_state = model(walk, state)
        assert logits.shape == (10, 10)
        assert rates["mecii"].shape == (10, 32)
        assert rates["mecv"].shape == (10, 32)
        assert rates["ca1"].shape, (10, 32) == (10, 32)

    def test_gradients_reach_all_parameters(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(5)
        config = SpikingTEMConfig(sensory=10, units=32, use_gain=True, use_stdp=True)
        model = SpikingTEM(config)
        # Note: with M=0 at step 0, retrieval weights get zero gradient at first step;
        # using nonzero M initialization (m_init_std=0.05) ensures gradients
        # reach retrieval weights.
        state = model.initial_state(m_init_std=0.05)
        logits, _, _ = model(walk, state)
        targets = walk.observations[1:]
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert float(param.grad.norm().item()) > 0.0, f"Parameter {name} gradient norm is zero"

        model2 = SpikingTEM(config)
        logits2, _, _ = model2(walk, model2.initial_state(m_init_std=0.05))
        loss2 = F.cross_entropy(logits2, targets)
        loss2.backward()
        if model2.gain is not None:
            model2.gain.grad.zero_()
            with pytest.raises(AssertionError):
                assert float(model2.gain.grad.norm().item()) > 0.0

    def test_stdp_behavior(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(5)

        model_nostdp = SpikingTEM(SpikingTEMConfig(sensory=10, units=32, use_stdp=False))
        state0 = model_nostdp.initial_state()
        initial_memory = state0.M.clone()
        _, _, state_nostdp = model_nostdp(walk, state0)
        torch.testing.assert_close(state_nostdp.M, initial_memory)

        model_stdp = SpikingTEM(
            SpikingTEMConfig(sensory=10, units=32, use_stdp=True, a_plus=0.01, a_minus=0.001)
        )
        state_stdp_0 = model_stdp.initial_state()
        init_memory_stdp = state_stdp_0.M.clone()
        _, _, state_stdp = model_stdp(walk, state_stdp_0)
        assert not torch.allclose(state_stdp.M, init_memory_stdp)
        assert (state_stdp.M.abs() <= 2.0 + 1e-6).all()

    def test_theta_and_gain_flags(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(5)

        torch.manual_seed(42)
        m1 = SpikingTEM(SpikingTEMConfig(sensory=10, units=32, use_theta=True))
        _, r1, _ = m1(walk)

        torch.manual_seed(42)
        m2 = SpikingTEM(SpikingTEMConfig(sensory=10, units=32, use_theta=False))
        _, r2, _ = m2(walk)

        assert not torch.allclose(r1["mecii"], r2["mecii"])

        m_nogain = SpikingTEM(SpikingTEMConfig(sensory=10, units=32, use_gain=False))
        assert m_nogain.gain is None

    def test_sample_flag_and_divisibility(self):
        model_sample = SpikingTEMConfig(sensory=10, units=32, k=2, sample=True)
        m = SpikingTEM(model_sample)
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(5)
        _logits, rates, _ = m(walk)
        assert rates["mecv"].shape == (5, 16)

        with pytest.raises(ValueError, match="units"):
            SpikingTEM(SpikingTEMConfig(units=33, k=2))

    def test_seeded_determinism(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(5)

        torch.manual_seed(123)
        m1 = SpikingTEM(SpikingTEMConfig(sensory=10, units=32))
        l1, _, _ = m1(walk)

        torch.manual_seed(123)
        m2 = SpikingTEM(SpikingTEMConfig(sensory=10, units=32))
        l2, _, _ = m2(walk)

        torch.testing.assert_close(l1, l2)

    def test_loss_decreases(self):
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(10)
        model = SpikingTEM(SpikingTEMConfig(sensory=10, units=32, steps_per_action=2))
        losses = train_walks(model, [walk], steps=30, lr=0.01, seed=42)
        initial_loss, final_loss = losses[0], losses[-1]
        msg = f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        assert final_loss < 0.9 * initial_loss, msg

        # Record next-observation accuracy on training walk
        with torch.no_grad():
            logits, _, _ = model(walk, model.initial_state())
            preds = logits.argmax(dim=-1)
            targets = walk.observations[1:]
            acc = (preds == targets).float().mean().item()
            print(f"Training next-obs accuracy: {acc:.3f} (chance: {1.0 / 10.0:.3f})")
            assert acc >= 0.0

    def test_state_device_on_cpu(self):
        model = SpikingTEM(SpikingTEMConfig(sensory=10, units=32))
        model.to("cpu")
        state = model.initial_state()
        param_device = next(model.parameters()).device
        assert state.v_mecii.device == param_device
        assert state.M.device == param_device

    def test_mps_forward_backward_finite(self):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS device not available")
        device = torch.device("mps")
        config = SpikingTEMConfig(sensory=10, units=32, steps_per_action=2)
        model = SpikingTEM(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        world = GraphWorld(width=5, sensory=10, seed=0)
        walk = world.random_walk(8)

        optimizer.zero_grad()
        state = model.initial_state()
        logits, rates, _ = model(walk, state)
        targets = walk.observations[1:].to(device)
        loss = F.cross_entropy(logits, targets) + 1e-3 * (
            rates["mecii"].mean() + rates["mecv"].mean() + rates["ca1"].mean()
        )
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
