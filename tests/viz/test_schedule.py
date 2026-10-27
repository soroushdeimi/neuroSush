"""Tests for neurosush.viz.schedule."""

from __future__ import annotations

import itertools

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.data import spike_frames
from neurosush.encoding import rate_poisson
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import ELIF, LIF, Fire
from neurosush.synapses.constraints import WeightClip
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.viz.schedule import schedule_lanes, schedule_table


def build_quickstart_network() -> Network:
    net = Network(seed=0)
    train = rate_poisson(torch.full((20,), 0.3), 100, generator=net.generator)
    frames = itertools.cycle(spike_frames([(train, None)]))
    source = NeuronGroup(net, 20, [SpikeInput(frames), Axon()], name="source")
    target = NeuronGroup(
        net,
        2,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            Fire(),
            Axon(),
        ],
        name="target",
    )
    SynapseGroup(
        net,
        source,
        target,
        [
            WeightInit(mode="uniform", scale=0.5),
            DenseInput(coef=5.0),
            SpikeGather(),
            Traces(tau_pre=10.0),
            STDP(a_plus=0.02, a_minus=0.01, bound="soft"),
            WeightClip(w_min=0.0, w_max=1.0),
        ],
        name="syn",
    )
    return net


def test_schedule_lanes() -> None:
    lanes = schedule_lanes()
    assert isinstance(lanes, list)
    assert len(lanes) == len(Order)
    for i, lane in enumerate(lanes):
        assert set(lane.keys()) == {"name", "order"}
        assert isinstance(lane["name"], str)
        assert isinstance(lane["order"], int)
        if i > 0:
            assert lanes[i]["order"] >= lanes[i - 1]["order"]

    order_map = {m.name: int(m.value) for m in Order}
    for lane in lanes:
        assert order_map[lane["name"]] == lane["order"]


def test_schedule_table_rows_match_network_order() -> None:
    net = build_quickstart_network()
    # Uninitialized net should be automatically initialized
    assert not net.initialized
    table = schedule_table(net)
    assert net.initialized

    assert len(table) == len(net.schedule)
    for i, (host, behavior) in enumerate(net.schedule):
        row = table[i]
        assert row["index"] == i
        assert row["order"] == behavior.order
        assert row["lane"] == Order(behavior.order).name
        assert row["host"] == host.name
        if isinstance(host, NeuronGroup):
            assert row["host_kind"] == "neurons"
        elif isinstance(host, SynapseGroup):
            assert row["host_kind"] == "synapses"
        elif isinstance(host, Network):
            assert row["host_kind"] == "network"
        assert row["behavior"] == type(behavior).__name__
        assert row["enabled"] is True
        assert isinstance(row["summary"], str)
        assert len(row["summary"]) > 0
        assert isinstance(row["reads"], list)
        assert isinstance(row["writes"], list)


def test_schedule_table_disabled_behavior_and_custom_order() -> None:
    class CustomBehavior(Behavior):
        """Custom test behavior with arbitrary order.

        Extra details.
        """

        order = 999

    net = Network()
    custom = CustomBehavior()
    custom.enabled = False
    net.behaviors = net._attach(net, [custom])

    table = schedule_table(net)
    assert len(table) == 1
    row = table[0]
    assert row["order"] == 999
    assert row["lane"] == "999"
    assert row["enabled"] is False
    assert row["host"] == "network"
    assert row["host_kind"] == "network"
    assert row["summary"] == "Custom test behavior with arbitrary order."


def test_reads_writes_honesty_part_c() -> None:
    # A network that exercises all 11 specified behaviors.
    net = Network(seed=0)
    train = rate_poisson(torch.full((4,), 0.5), 20, generator=net.generator)
    frames = itertools.cycle(spike_frames([(train, None)]))
    src = NeuronGroup(net, 4, [SpikeInput(frames), Axon()], name="src")
    dst1 = NeuronGroup(
        net,
        4,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=2),
            Fire(),
            Axon(),
        ],
        name="dst1",
    )
    dst2 = NeuronGroup(
        net,
        4,
        [
            DendriteStructure(),
            DendriteIntegration(),
            ELIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0, delta=2.0, theta_rh=-50.0),
            Fire(),
            Axon(),
        ],
        name="dst2",
    )
    SynapseGroup(
        net,
        src,
        dst1,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=2.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.02, a_minus=0.01),
        ],
        name="syn1",
    )
    SynapseGroup(
        net,
        src,
        dst2,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=2.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.02, a_minus=0.01),
        ],
        name="syn2",
    )

    net.run(3)

    target_classes = {
        LIF,
        ELIF,
        Fire,
        Axon,
        DendriteStructure,
        DendriteIntegration,
        DenseInput,
        SpikeGather,
        Traces,
        STDP,
        KWTA,
    }
    found_classes = set()

    for host, behavior in net.schedule:
        b_cls = type(behavior)
        if b_cls not in target_classes:
            continue
        found_classes.add(b_cls)

        assert isinstance(behavior.reads, tuple)
        assert isinstance(behavior.writes, tuple)

        for name in behavior.writes:
            msg = f"{b_cls.__name__} writes {name!r} which is missing on {host.name}"
            assert hasattr(host, name), msg

        for name in behavior.reads:
            if isinstance(host, SynapseGroup):
                # May live on the synapse group or on connected groups (e.g. spike_history)
                lives_on_syn = hasattr(host, name)
                lives_on_group = hasattr(host.src, name) or hasattr(host.dst, name)
                msg = f"{b_cls.__name__} reads {name!r} which is missing on {host.name}"
                assert lives_on_syn or lives_on_group, msg
            else:
                msg = f"{b_cls.__name__} reads {name!r} which is missing on {host.name}"
                assert hasattr(host, name), msg

    assert found_classes == target_classes
