"""Small driver behaviors shared by the validation tests."""

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.order import Order


class ConstantCurrent(Behavior):
    """Holds ``group.I`` at a fixed value (in place of synaptic input)."""

    order = Order.DENDRITE_INTEGRATION

    def __init__(self, value):
        self.value = value

    def initialize(self, group):
        group.I = group.state(self.value)

    def forward(self, group):
        group.I = group.state(self.value)


class ScriptedSpikes(Behavior):
    """Fires the neurons listed for each step (1-based): ``{step: [neuron, ...]}``."""

    order = Order.FIRE

    def __init__(self, schedule):
        self.schedule = schedule

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        spikes = group.state(False, dtype=torch.bool)
        spikes[..., self.schedule.get(group.net.iteration, [])] = True
        group.spikes = spikes


class BernoulliSpikes(Behavior):
    """Every neuron fires independently with probability ``p`` (one value, or one per neuron)."""

    order = Order.FIRE

    def __init__(self, p):
        self.p = p

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        group.spikes = group.rand() < self.p
