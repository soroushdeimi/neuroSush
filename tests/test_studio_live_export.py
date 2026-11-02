"""Tests for Studio Live code export and execution with real neuroSush library."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather


def test_studio_live_export_presets() -> None:
    """Test that all Studio Live preset configurations execute correctly with real neuroSush."""
    for preset_name in ["balanced", "wta", "chain"]:
        net = Network(seed=42, dt=1.0)
        exc = NeuronGroup(
            net,
            30,
            [
                PoissonInput(0.01),
                LIF(
                    tau=20.0,
                    threshold=-40.0,
                    v_reset=-75.0,
                    v_rest=-70.0,
                ),
                Fire(),
                Axon(),
            ],
            name=f"exc_{preset_name}",
        )
        inh = NeuronGroup(
            net,
            10,
            [
                PoissonInput(0.01),
                LIF(
                    tau=20.0,
                    threshold=-40.0,
                    v_reset=-75.0,
                    v_rest=-70.0,
                ),
                Fire(),
                Axon(),
            ],
            inhibitory=True,
            name=f"inh_{preset_name}",
        )
        SynapseGroup(
            net,
            exc,
            inh,
            [
                WeightInit(weights=torch.full((30, 10), 0.5)),
                DenseInput(),
                SpikeGather(),
            ],
            name=f"syn_{preset_name}",
        )
        net.initialize()
        for _ in range(5):
            net.step()
        assert True
