import pytest
import torch

from neurosush.encoding import intensity_to_latency, interval_poisson, rate_poisson


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


class TestRatePoisson:
    def test_shape_and_dtype(self):
        spikes = rate_poisson(torch.full((2, 3), 0.5), 4, generator=gen())
        assert spikes.shape == (4, 2, 3)
        assert spikes.dtype == torch.bool

    def test_zero_never_spikes_and_one_always_spikes(self):
        spikes = rate_poisson(torch.tensor([0.0, 1.0]), 50, generator=gen())
        assert spikes[:, 0].sum().item() == 0
        assert spikes[:, 1].sum().item() == 50

    def test_rate_matches_probability(self):
        spikes = rate_poisson(torch.full((1000,), 0.2), 50, generator=gen())
        assert spikes.float().mean().item() == pytest.approx(0.2, abs=0.01)

    def test_ratio_scales_probability(self):
        spikes = rate_poisson(torch.full((1000,), 0.8), 50, ratio=0.5, generator=gen())
        assert spikes.float().mean().item() == pytest.approx(0.4, abs=0.01)

    def test_seeded_generator_is_reproducible(self):
        x = torch.rand(10, generator=gen(1))
        assert torch.equal(
            rate_poisson(x, 5, generator=gen(3)), rate_poisson(x, 5, generator=gen(3))
        )

    def test_input_is_not_modified(self):
        x = torch.tensor([0.3, 0.6])
        rate_poisson(x, 3, ratio=0.5, generator=gen())
        assert x.tolist() == pytest.approx([0.3, 0.6])

    @pytest.mark.parametrize(("steps", "ratio", "match"), [(0, 1.0, "steps"), (3, -0.1, "ratio")])
    def test_invalid_arguments(self, steps, ratio, match):
        with pytest.raises(ValueError, match=match):
            rate_poisson(torch.zeros(2), steps, ratio=ratio)

    def test_negative_input_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            rate_poisson(torch.tensor([-0.1]), 3)


class TestIntervalPoisson:
    def test_shape_and_dtype(self):
        spikes = interval_poisson(torch.full((2, 2), 0.5), 6, generator=gen())
        assert spikes.shape == (6, 2, 2)
        assert spikes.dtype == torch.bool

    def test_zero_never_spikes(self):
        spikes = interval_poisson(torch.tensor([0.0, 0.9]), 20, generator=gen())
        assert spikes[:, 0].sum().item() == 0

    def test_intensity_one_spikes_every_step(self):
        # mean interval 1 / (value * ratio) = 1; zero intervals are raised to 1
        spikes = interval_poisson(torch.ones(200), 10, generator=gen())
        assert spikes.any(0).all()
        assert spikes.float().mean().item() > 0.5

    def test_stronger_input_spikes_more(self):
        x = torch.cat([torch.full((500,), 0.1), torch.full((500,), 0.5)])
        spikes = interval_poisson(x, 100, generator=gen())
        weak, strong = spikes[:, :500].sum().item(), spikes[:, 500:].sum().item()
        assert strong > 2 * weak

    def test_input_is_not_modified(self):
        x = torch.tensor([0.25, 0.5])
        interval_poisson(x, 5, ratio=2.0, generator=gen())
        assert x.tolist() == [0.25, 0.5]

    def test_reproducible(self):
        x = torch.rand(20, generator=gen(2))
        a = interval_poisson(x, 30, generator=gen(5))
        b = interval_poisson(x, 30, generator=gen(5))
        assert torch.equal(a, b)


class TestIntensityToLatency:
    def test_linear_times_without_trimming(self):
        x = torch.tensor([1.0, 0.5, 0.0])
        spikes = intensity_to_latency(x, 5, threshold=0.25, trim_low=False, trim_high=False)
        assert spikes.shape == (5, 3)
        assert spikes.sum(0).tolist() == [1, 1, 0]
        assert spikes[:, 0].nonzero().item() == 0
        assert spikes[:, 1].nonzero().item() == 2

    def test_trimming_spreads_active_values_over_the_window(self):
        x = torch.tensor([1.0, 0.5, 0.0])
        spikes = intensity_to_latency(x, 5, threshold=0.25)
        assert spikes[:, 0].nonzero().item() == 0
        assert spikes[:, 1].nonzero().item() == 4
        assert spikes[:, 2].sum().item() == 0

    def test_trim_low_only(self):
        x = torch.tensor([0.6, 0.2])
        # (0.6 - 0.2) / (1.0 - 0.2) = 0.5 -> time round(0.5 * 4) = 2; weakest -> last step
        spikes = intensity_to_latency(x, 5, trim_high=False)
        assert spikes[:, 0].nonzero().item() == 2
        assert spikes[:, 1].nonzero().item() == 4

    def test_every_active_value_spikes_exactly_once(self):
        x = torch.rand(4, 4, generator=gen(0))
        spikes = intensity_to_latency(x, 7)
        assert spikes.shape == (7, 4, 4)
        assert torch.equal(spikes.sum(0), torch.ones(4, 4, dtype=torch.long))

    def test_sparsity_keeps_the_strongest_fraction(self):
        x = torch.arange(10, dtype=torch.float32) / 10
        spikes = intensity_to_latency(x, 4, sparsity=0.3)
        assert spikes.any(0).tolist() == [False] * 7 + [True] * 3

    def test_value_range(self):
        x = torch.tensor([10.0, 5.0])
        spikes = intensity_to_latency(
            x, 3, value_range=(0.0, 10.0), trim_low=False, trim_high=False
        )
        assert spikes[:, 0].nonzero().item() == 0
        assert spikes[:, 1].nonzero().item() == 1

    def test_no_active_value_gives_no_spikes(self):
        spikes = intensity_to_latency(torch.tensor([0.1, 0.2]), 3, threshold=0.5)
        assert not spikes.any()

    def test_input_is_not_modified(self):
        x = torch.tensor([0.9, 0.4, 0.1])
        intensity_to_latency(x, 5, value_range=(0.1, 0.9))
        assert x.tolist() == pytest.approx([0.9, 0.4, 0.1])

    def test_repeated_calls_do_not_share_state(self):
        a = intensity_to_latency(torch.arange(10.0) / 10, 4, sparsity=0.5)
        intensity_to_latency(torch.arange(10.0) / 100, 4, sparsity=0.1)
        b = intensity_to_latency(torch.arange(10.0) / 10, 4, sparsity=0.5)
        assert torch.equal(a, b)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"threshold": 0.5, "sparsity": 0.5}, "threshold"),
            ({"sparsity": 0.0}, "sparsity"),
            ({"sparsity": 1.5}, "sparsity"),
            ({"value_range": (1.0, 1.0)}, "value_range"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            intensity_to_latency(torch.rand(3), 4, **kwargs)

    def test_invalid_steps(self):
        with pytest.raises(ValueError, match="steps"):
            intensity_to_latency(torch.rand(3), 0)
