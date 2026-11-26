"""Spatial lateral inhibition against hand-computed values.

At each pixel, if the pixel already fired every spike is suppressed; otherwise only the map with
the highest potential survives (ties broken by lowest c). ``new_fired`` is the union of the input
``fired`` mask and any pixel that still has a surviving spike.
"""

import pytest
import torch

from neurosush.hmax.inhibition import spatial_lateral_inhibition


class TestSpatialLateralInhibition:
    def test_highest_potential_survives(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[False]], [[True]]]]
        assert new_fired.tolist() == [[[True]]]

    def test_equal_potential_lowest_c_survives(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[4.0]], [[4.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[True]], [[False]]]]
        assert new_fired.tolist() == [[[True]]]

    def test_nonspiking_map_cannot_survive_with_neg_inf_potentials(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[float("-inf")]], [[float("-inf")]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[False]], [[True]]]]
        assert new_fired.tolist() == [[[True]]]

    def test_both_neg_inf_lowest_c_survives(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[float("-inf")]], [[float("-inf")]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[True]], [[False]]]]
        assert new_fired.tolist() == [[[True]]]

    def test_survivor_must_be_spiking_map(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[0.0]], [[-1.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[False]], [[True]]]]
        assert new_fired.tolist() == [[[True]]]

    def test_already_fired_pixel_suppresses_all(self):
        spikes = torch.zeros(1, 2, 1, 2, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        spikes[0, 0, 0, 1] = True
        spikes[0, 1, 0, 1] = True
        potential = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 2, dtype=torch.bool)
        fired[0, 0, 0] = True
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        # pixel 0 already fired: both maps suppressed
        assert new_spikes[0, :, 0, 0].tolist() == [False, False]
        # pixel 1 unaffected: highest potential (map 1) survives
        assert new_spikes[0, :, 0, 1].tolist() == [False, True]
        assert new_fired.tolist() == [[[True, True]]]

    def test_second_call_suppresses_already_fired(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        _new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        # second call: feed the returned fired mask back
        new_spikes2, new_fired2 = spatial_lateral_inhibition(spikes, potential, new_fired)
        assert new_spikes2.tolist() == [[[[False]], [[False]]]]
        assert new_fired2.tolist() == [[[True]]]

    def test_bit_identical_across_calls(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        a_spikes, a_fired = spatial_lateral_inhibition(spikes, potential, fired)
        b_spikes, b_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert torch.equal(a_spikes, b_spikes)
        assert torch.equal(a_fired, b_fired)

    def test_no_spike_pixel_stays_unfired(self):
        spikes = torch.zeros(1, 2, 1, 2, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        potential = torch.tensor([[[[3.0, 0.0]], [[0.0, 0.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 2, dtype=torch.bool)
        new_spikes, new_fired = spatial_lateral_inhibition(spikes, potential, fired)
        assert new_spikes.tolist() == [[[[True, False]], [[False, False]]]]
        assert new_fired.tolist() == [[[True, False]]]

    def test_does_not_mutate_inputs(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        spikes[0, 0, 0, 0] = True
        spikes[0, 1, 0, 0] = True
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        spikes_copy, pot_copy, fired_copy = spikes.clone(), potential.clone(), fired.clone()
        spatial_lateral_inhibition(spikes, potential, fired)
        assert torch.equal(spikes, spikes_copy)
        assert torch.equal(potential, pot_copy)
        assert torch.equal(fired, fired_copy)

    def test_spikes_float_dtype_raises(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="spikes must be bool"):
            spatial_lateral_inhibition(spikes, potential, fired)

    def test_spikes_3d_raises(self):
        spikes = torch.zeros(2, 1, 1, dtype=torch.bool)
        potential = torch.zeros(2, 1, 1, dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="spikes must be 4-D"):
            spatial_lateral_inhibition(spikes, potential, fired)

    def test_potential_shape_mismatch_raises(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        potential = torch.zeros(1, 2, 1, 2, dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="shape mismatch"):
            spatial_lateral_inhibition(spikes, potential, fired)

    def test_fired_shape_mismatch_raises(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 2, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="shape mismatch"):
            spatial_lateral_inhibition(spikes, potential, fired)

    def test_fired_not_bool_raises(self):
        spikes = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
        potential = torch.tensor([[[[3.0]], [[5.0]]]], dtype=torch.float64)
        fired = torch.zeros(1, 1, 1, dtype=torch.float64)
        with pytest.raises(ValueError, match="fired must be bool"):
            spatial_lateral_inhibition(spikes, potential, fired)
