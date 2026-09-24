"""Active dendrites: the gating equations and context-dependent routing (Iyer et al. 2022)."""

import pytest
import torch
from torch import nn

from neurosush.htm.active_dendrites import ActiveDendrites, kwta


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


class TestKWTA:
    def test_keeps_exactly_the_k_largest(self):
        x = torch.tensor([[0.3, -1.0, 2.0, 0.5], [4.0, 3.0, -2.0, 1.0]])
        assert kwta(x, 2).tolist() == [[0.0, 0.0, 2.0, 0.5], [4.0, 3.0, 0.0, 0.0]]

    def test_gradient_reaches_only_the_winners(self):
        x = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
        kwta(x, 1).sum().backward()
        assert x.grad.tolist() == [0.0, 1.0, 0.0]

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be"):
            kwta(torch.zeros(3), 4)


class TestGating:
    def layer(self, **kwargs):
        torch.manual_seed(0)
        return ActiveDendrites(5, 4, 3, segments=6, **kwargs)

    def test_matches_the_definition(self):
        layer = self.layer()
        x, c = torch.randn(7, 5, generator=gen()), torch.randn(7, 3, generator=gen(1))
        with torch.no_grad():
            out = layer(x, c)
        weight, bias, segments = (
            p.detach() for p in (layer.linear.weight, layer.linear.bias, layer.segments)
        )
        for b in range(7):
            for unit in range(4):
                d = max(float(segments[unit, j] @ c[b]) for j in range(6))
                y = float(weight[unit] @ x[b] + bias[unit])
                assert out[b, unit].item() == pytest.approx(
                    y / (1 + torch.exp(torch.tensor(-d)).item()), rel=1e-5
                )

    def test_absolute_gating_can_silence_a_unit(self):
        layer = self.layer(absolute=True)
        with torch.no_grad():
            layer.segments.zero_()
            layer.segments[0, 2] = torch.tensor([-50.0, 0.0, 0.0])
            layer.segments[0, 3] = torch.tensor([1.0, 0.0, 0.0])
        d = layer.dendritic_activation(torch.tensor([1.0, 0.0, 0.0]))
        assert d[0].item() == -50.0
        gated = layer(torch.ones(5), torch.tensor([1.0, 0.0, 0.0]))
        assert abs(gated[0].item()) < 1e-15

    def test_only_the_selected_segment_learns(self):
        layer = self.layer()
        c = torch.randn(3, generator=gen(2))
        layer(torch.randn(5, generator=gen(3)), c).sum().backward()
        chosen = (layer.segments @ c).argmax(-1)
        for unit in range(4):
            for j in range(6):
                has_grad = bool(layer.segments.grad[unit, j].abs().sum() > 0)
                assert has_grad == (j == int(chosen[unit]))

    def test_kwta_output_sparsity(self):
        layer = self.layer(k=2)
        out = layer(torch.randn(10, 5, generator=gen()), torch.randn(10, 3, generator=gen(1)))
        assert ((out != 0).sum(-1) <= 2).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"), [({"segments": 0}, "segments"), ({"segments": 2, "k": 9}, "k must")]
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ActiveDendrites(5, 4, 3, **kwargs)


class TestMultiTask:
    """Two tasks give every input opposite labels, so no function of the input alone
    can beat 50% on both; a context-gated network can solve both."""

    def data(self, n=256):
        g = gen(4)
        x = torch.randn(n, 10, generator=g)
        direction = torch.randn(10, generator=g)
        label = (x @ direction > 0).long()
        contexts = torch.eye(2)
        inputs = torch.cat([x, x])
        context = torch.cat([contexts[0].expand(n, 2), contexts[1].expand(n, 2)])
        targets = torch.cat([label, 1 - label])
        return inputs, context, targets

    def train(self, forward, parameters, inputs, context, targets):
        optimizer = torch.optim.Adam(parameters, lr=0.01)
        for _ in range(250):
            optimizer.zero_grad()
            nn.functional.cross_entropy(forward(inputs, context), targets).backward()
            optimizer.step()
        with torch.no_grad():
            return (forward(inputs, context).argmax(-1) == targets).float().mean().item()

    def test_context_gating_solves_contradictory_tasks(self):
        inputs, context, targets = self.data()
        torch.manual_seed(0)
        hidden = ActiveDendrites(10, 64, 2, segments=2, k=16)
        readout = nn.Linear(64, 2)
        accuracy = self.train(
            lambda x, c: readout(torch.relu(hidden(x, c))),
            [*hidden.parameters(), *readout.parameters()],
            inputs,
            context,
            targets,
        )
        assert accuracy > 0.95

    def test_an_ungated_network_is_stuck_at_chance(self):
        inputs, context, targets = self.data()
        torch.manual_seed(0)
        mlp = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        accuracy = self.train(lambda x, c: mlp(x), mlp.parameters(), inputs, context, targets)
        # each input appears once with each label: exactly one of the two copies is right
        assert accuracy == 0.5
