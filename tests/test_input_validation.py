import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.init import WeightInit
from neurosush.viz import TraceRecorder, render_plate


def lif() -> LIF:
    return LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)


def test_render_plate_without_trace_needs_live_source() -> None:
    with pytest.raises(ValueError, match="needs a trace_dict"):
        render_plate("demo", None)
    with pytest.raises(ValueError, match="source='sweep' needs a trace_dict"):
        render_plate("N01", source="sweep")


def test_render_live_plate_without_trace() -> None:
    assert 'data-live="true"' in render_plate("N01", source="live")


def test_trace_recorder_rejects_duplicate_attributes() -> None:
    with pytest.raises(ValueError, match=r"unique, got \['v', 'v'\]"):
        TraceRecorder("v", "v", max_steps=5)


def test_trace_before_the_recorder_ran_explains_itself() -> None:
    with pytest.raises(RuntimeError, match="run the network before calling trace"):
        TraceRecorder("v", max_steps=5).trace()


def test_trace_after_run_still_works() -> None:
    net = Network(seed=0)
    recorder = TraceRecorder("v", max_steps=5)
    NeuronGroup(net, 2, [lif(), Fire(), recorder])
    net.run(3)
    assert recorder.trace().steps == [1, 2, 3]


@pytest.mark.parametrize("steps", [1.5, "3", None, True])
def test_run_rejects_non_integer_steps(steps: object) -> None:
    with pytest.raises(TypeError, match="integer number of time steps"):
        Network(seed=0).run(steps)  # type: ignore[arg-type]


def test_weight_init_tensor_mode_points_to_weights() -> None:
    with pytest.raises(ValueError, match="weights="):
        WeightInit(mode=torch.ones(2, 2))  # type: ignore[arg-type]


def test_poisson_rate_above_one_spike_per_step_is_rejected() -> None:
    from neurosush.neurons.inputs import PoissonInput

    net = Network(seed=0, dt=1.0)
    NeuronGroup(net, 2, [PoissonInput(rates=40.0)])
    with pytest.raises(ValueError, match=r"rates \* dt must be at most 1 .*got 40"):
        net.initialize()
