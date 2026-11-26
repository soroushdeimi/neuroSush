"""Convolutional winner-take-all selection and simplified STDP against hand-computed values.

``select_conv_winners`` walks candidates sorted by (time, -potential, c, y, x) and keeps at most
``k`` winners, one per feature map, no two within Chebyshev distance ``radius``. ``simplified_stdp``
is Kheradpisheh Eq. 3: LTP when pre <= post, LTD otherwise, scaled by ``w (1 - w)``.
"""

import math

import pytest
import torch

from neurosush.hmax.wta import select_conv_winners, simplified_stdp, stdp_convergence


def fixture_f():
    """Fixture F: C=2, H=1, W=3 (see task SPEC)."""
    first_spike_time = torch.tensor(
        [[[2.0, 1.0, math.inf]], [[1.0, math.inf, 3.0]]], dtype=torch.float64
    )
    potential = torch.tensor([[[0.0, 5.0, 0.0]], [[4.0, 0.0, 0.0]]], dtype=torch.float64)
    return first_spike_time, potential


class TestSelectConvWinners:
    def test_k2_radius0_accepts_two_maps(self):
        fst, pot = fixture_f()
        winners = select_conv_winners(fst, pot, k=2, radius=0)
        assert winners.tolist() == [[0, 0, 1], [1, 0, 0]]

    def test_k2_radius1_accepts_only_first(self):
        fst, pot = fixture_f()
        winners = select_conv_winners(fst, pot, k=2, radius=1)
        assert winners.tolist() == [[0, 0, 1]]

    def test_equal_time_and_potential_lowest_c_wins(self):
        fst = torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float64)
        pot = torch.tensor([[[0.0]], [[0.0]]], dtype=torch.float64)
        winners = select_conv_winners(fst, pot, k=1, radius=0)
        assert winners.tolist() == [[0, 0, 0]]

    def test_k1_returns_only_first_winner(self):
        fst, pot = fixture_f()
        winners = select_conv_winners(fst, pot, k=1, radius=0)
        assert winners.tolist() == [[0, 0, 1]]

    def test_all_inf_returns_empty(self):
        fst = torch.full((2, 1, 3), math.inf, dtype=torch.float64)
        pot = torch.zeros(2, 1, 3, dtype=torch.float64)
        winners = select_conv_winners(fst, pot, k=2, radius=0)
        assert winners.shape == (0, 3)

    def test_does_not_mutate_inputs(self):
        fst, pot = fixture_f()
        fst_copy, pot_copy = fst.clone(), pot.clone()
        select_conv_winners(fst, pot, k=2, radius=1)
        assert torch.equal(fst, fst_copy)
        assert torch.equal(pot, pot_copy)

    def test_k0_raises(self):
        fst, pot = fixture_f()
        with pytest.raises(ValueError, match="k must be at least 1"):
            select_conv_winners(fst, pot, k=0, radius=0)

    def test_negative_radius_raises(self):
        fst, pot = fixture_f()
        with pytest.raises(ValueError, match="radius must be non-negative"):
            select_conv_winners(fst, pot, k=1, radius=-1)

    def test_shape_mismatch_raises(self):
        fst = torch.zeros(2, 1, 3, dtype=torch.float64)
        pot = torch.zeros(2, 1, 2, dtype=torch.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            select_conv_winners(fst, pot, k=1, radius=0)

    def test_2d_input_raises(self):
        fst = torch.zeros(2, 3, dtype=torch.float64)
        pot = torch.zeros(2, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="must be 3-D"):
            select_conv_winners(fst, pot, k=1, radius=0)


class TestSimplifiedStdp:
    def test_ltp_branch(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w.item() == pytest.approx(0.501, abs=1e-9)

    def test_ltd_branch(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([3.0], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w.item() == pytest.approx(0.49925, abs=1e-9)

    def test_inf_pre_time_takes_ltd_branch(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([math.inf], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w.item() == pytest.approx(0.49925, abs=1e-9)

    def test_equal_times_take_ltp_branch(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([2.0], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w.item() == pytest.approx(0.501, abs=1e-9)

    def test_w_at_extremes_unchanged(self):
        w = torch.tensor([0.0, 1.0], dtype=torch.float64)
        pre = torch.tensor([1.0, 3.0], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w.tolist() == pytest.approx([0.0, 1.0], abs=1e-9)

    def test_returns_new_tensor(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        new_w = simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)
        assert new_w is not w
        assert w.item() == 0.5

    def test_w_shape_mismatch_raises(self):
        w = torch.tensor([0.5, 0.5], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="shape mismatch"):
            simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)

    def test_w_outside_unit_interval_raises(self):
        w = torch.tensor([1.2], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match=r"1\.2"):
            simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)

    def test_negative_w_raises(self):
        w = torch.tensor([-0.1], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match=r"-0\.1"):
            simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=0.003)

    def test_negative_a_plus_raises(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="a_plus must be non-negative"):
            simplified_stdp(w, pre, post_time=2.0, a_plus=-0.004, a_minus=0.003)

    def test_negative_a_minus_raises(self):
        w = torch.tensor([0.5], dtype=torch.float64)
        pre = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="a_minus must be non-negative"):
            simplified_stdp(w, pre, post_time=2.0, a_plus=0.004, a_minus=-0.003)


class TestStdpConvergence:
    def test_mean_w_one_minus_w(self):
        assert stdp_convergence(torch.tensor([0.0, 1.0, 0.5, 0.5])).item() == pytest.approx(
            0.125, abs=1e-9
        )

    def test_w_outside_unit_interval_raises(self):
        with pytest.raises(ValueError, match=r"1\.2"):
            stdp_convergence(torch.tensor([1.2]))
