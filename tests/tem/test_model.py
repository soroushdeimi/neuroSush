import pytest
import torch

from neurosush.memory import hebbian_update
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.loss import total_loss
from neurosush.tem.model import bind, precision_mean


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def small_model(seed=0):
    return TEM(TEMConfig(sensory=4, compressed=4, modules=(3, 2), hidden=5), generator=gen(seed))


def test_precision_hand_value_and_invalid_sigma():
    # Means 2 and 10 with variances 1 and 4: (2 + 10/4)/(1 + 1/4) = 3.6.
    result = precision_mean(
        torch.tensor([2.0]), torch.tensor([10.0]), torch.tensor([1.0]), torch.tensor([2.0])
    )
    assert result.item() == pytest.approx(3.6)
    for invalid in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match=str(invalid)):
            precision_mean(torch.ones(1), torch.ones(1), torch.tensor([invalid]), torch.ones(1))


def test_binding_hand_value():
    assert torch.equal(
        bind(torch.tensor([[1.0, 0.5]]), torch.tensor([[0.2, 0.4]])),
        torch.tensor([[0.2, 0.4, 0.1, 0.2]]),
    )


def test_shapes_unique_codes_and_identity_transition():
    model = small_model()
    state = model.reset(5, generator=gen(1))
    assert torch.equal(model.codes.sum(-1), torch.full((4,), 2.0))
    assert model.codes.unique(dim=0).shape[0] == 4
    for f, g in enumerate(state.g):
        assert torch.equal(model.transition(g, torch.arange(5), f), g)
    output, after = model(torch.tensor([0, 1, 2, 3, 0]), torch.arange(5), state)
    assert output.logits_gen.shape == output.logits_g.shape == output.logits_p.shape == (5, 4)
    assert output.g_prior.shape == output.g_inf.shape == (5, 5)
    assert output.p_gen.shape == output.p_inf.shape == (5, 20)
    assert after.gen[0].shape == after.inf[0].shape == (5, 12, 12)
    assert after.gen[1].shape == (5, 8, 8)
    assert after.x[0].shape == (5, 4)


def test_both_memory_updates_hand_formula_and_no_mutation():
    model = small_model()
    state = model.reset(2, generator=gen(1))
    obs, action = torch.tensor([0, 1]), torch.tensor([1, 3])
    output, after = model(obs, action, state, rate=0.2, decay=0.9)
    p = output.p_inf[:, :12]
    pred = output.p_gen[:, :12]
    assert torch.equal(after.gen[0], hebbian_update(state.gen[0], p, pred, rate=0.2, decay=0.9))
    # With empty inference memory, its attractor only retains the sensory cue.
    x = after.x[0]
    sensory = torch.nn.functional.normalize(torch.relu(x - x.mean(-1, keepdim=True)), dim=-1)
    sensory_pred = sensory.repeat(1, 3) * model.config.retention**model.config.iterations
    assert torch.allclose(
        after.inf[0], hebbian_update(state.inf[0], p, sensory_pred, rate=0.2, decay=0.9)
    )
    assert all(torch.count_nonzero(m) == 0 for m in (*state.gen, *state.inf))
    assert not torch.equal(after.gen[0], after.inf[0])
    assert all(m.abs().max() <= 1 for m in (*after.gen, *after.inf))


def test_reset_and_detach_memory_not_parameters():
    model = small_model()
    state = model.reset(2, generator=gen(1))
    _, after = model(torch.tensor([0, 1]), torch.tensor([4, 4]), state)
    assert after.gen[0].grad_fn is not None
    detached = after.detach()
    assert detached.gen[0].grad_fn is None
    assert torch.equal(after.gen[0], detached.gen[0])
    reset = model.reset(2, generator=gen(2))
    assert all(torch.count_nonzero(m) == 0 for m in (*reset.gen, *reset.inf, *reset.x))
    assert all(not isinstance(m, torch.nn.Parameter) for m in (*after.gen, *after.inf))
    assert not any("gen" in name or "inf" in name for name, _ in model.named_parameters())


def test_generative_prediction_cannot_read_target():
    model = small_model()
    state = model.reset(2, generator=gen(1))
    _, state = model(torch.tensor([0, 1]), torch.tensor([4, 4]), state)
    a, _ = model(torch.tensor([0, 0]), torch.tensor([1, 2]), state)
    b, _ = model(torch.tensor([2, 3]), torch.tensor([1, 2]), state)
    assert torch.equal(a.logits_gen, b.logits_gen)
    assert not torch.equal(a.p_inf, b.p_inf)


def test_every_parameter_group_receives_gradient_through_chunk():
    model = small_model()
    state = model.reset(2, generator=gen(1))
    losses = []
    for t in range(4):
        target = torch.tensor([t % 4, (t + 1) % 4])
        output, state = model(target, torch.tensor([t % 5, (t + 2) % 5]), state)
        loss, _ = total_loss(output, target, model.config, iteration=2000)
        losses.append(loss)
    torch.stack(losses).mean().backward()
    for group in (model.transitions, model.memory_means, model.sigmas, model.decoder, model.g_init):
        gradients = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0
    assert model.alpha.grad is not None
    assert model.alpha.grad.abs().sum() > 0
    # Zero output weights block hidden transition gradients on the first backward.
    assert model.transitions[0].first.weight.grad.count_nonzero() == 0
    assert model.transitions[0].last.weight.grad.abs().sum() > 0


def test_seeded_determinism_and_global_rng_unchanged():
    global_before = torch.random.get_rng_state().clone()
    a, b = small_model(7), small_model(7)
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    sa, sb = a.reset(2, generator=gen(3)), b.reset(2, generator=gen(3))
    for _ in range(3):
        oa, sa = a(torch.tensor([0, 1]), torch.tensor([1, 2]), sa)
        ob, sb = b(torch.tensor([0, 1]), torch.tensor([1, 2]), sb)
        assert torch.equal(oa.logits_gen, ob.logits_gen)
        assert all(torch.equal(x, y) for x, y in zip(sa.gen, sb.gen, strict=True))
    assert torch.equal(global_before, torch.random.get_rng_state())


def test_invalid_observations_actions_batch():
    model = small_model()
    state = model.reset(1, generator=gen())
    with pytest.raises(ValueError, match="-1"):
        model(torch.tensor([-1]), torch.tensor([0]), state)
    with pytest.raises(ValueError, match="5"):
        model(torch.tensor([0]), torch.tensor([5]), state)
    with pytest.raises(ValueError, match="0"):
        model.reset(0, generator=gen())


def test_filter_hand_values_and_bptt_through_memory():
    model = small_model()
    with torch.no_grad():
        model.alpha.zero_()
    state = model.reset(1, generator=gen(2))
    _, first = model(torch.tensor([0]), torch.tensor([4]), state)
    assert torch.equal(first.x[0], 0.5 * model.codes[0:1])
    first.gen[0].retain_grad()
    output, second = model(torch.tensor([1]), torch.tensor([1]), first)
    assert torch.equal(second.x[0], 0.25 * model.codes[0:1] + 0.5 * model.codes[1:2])
    output.logits_gen.square().sum().backward()
    assert first.gen[0].grad is not None
    assert first.gen[0].grad.abs().sum() > 0


def test_reset_uses_shared_learned_anchor_without_randomness():
    model = small_model()
    generator = gen(9)
    before = generator.get_state().clone()
    first = model.reset(3, generator=generator)
    second = model.reset(3, generator=gen(22))
    assert torch.equal(before, generator.get_state())
    for f, g in enumerate(first.g):
        assert torch.equal(g, second.g[f])
        assert torch.equal(g, model.g_init[f].clamp(-1, 1).expand(3, -1))
        assert torch.equal(g[0], g[1])
        assert torch.equal(g[1], g[2])
    sum(g.sum() for g in first.g).backward()
    for parameter in model.g_init:
        expected = ((parameter >= -1) & (parameter <= 1)).float() * 3
        assert torch.equal(parameter.grad, expected)


def test_sigma_floor_is_preserved_for_extreme_negative_logits(monkeypatch):
    import neurosush.tem.model as model_module

    config = TEMConfig(sensory=4, compressed=4, modules=(3,), hidden=5)
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        model.sigmas[0].last.weight.zero_()
        model.sigmas[0].last.bias.fill_(-1000)
    original = model_module._precision_mean
    captured = []

    def capture(prior, memory, sigma_prior, sigma_memory):
        captured.extend((sigma_prior.detach(), sigma_memory.detach()))
        return original(prior, memory, sigma_prior, sigma_memory)

    monkeypatch.setattr(model_module, "_precision_mean", capture)
    output, _ = model(torch.tensor([0]), torch.tensor([4]), model.reset(1))
    assert all(torch.equal(sigma, torch.full_like(sigma, 0.001)) for sigma in captured)
    assert torch.isfinite(output.g_inf).all()


def test_sensory_cues_already_nonnegative(monkeypatch):
    import neurosush.tem.model as model_module

    original = model_module.attractor
    captured = []

    def capture(cue, memory, *args, **kwargs):
        captured.append(cue.detach())
        return original(cue, memory, *args, **kwargs)

    monkeypatch.setattr(model_module, "attractor", capture)
    config = TEMConfig(sensory=4, compressed=4, modules=(3,), hidden=5)
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        model.g_init[0].copy_(torch.tensor([-0.5, 0.5, 0.25]))
    model(torch.tensor([0]), torch.tensor([4]), model.reset(1))
    assert (captured[0] < 0).any()
    assert (captured[1] >= 0).all()
