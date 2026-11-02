import re

import neurosush
from neurosush import Network, NeuronGroup, SynapseGroup


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", neurosush.__version__)


def test_top_level_exports():
    assert Network is not None
    assert NeuronGroup is not None
    assert SynapseGroup is not None
    assert hasattr(neurosush, "Network")
    assert hasattr(neurosush, "NeuronGroup")
    assert hasattr(neurosush, "SynapseGroup")
