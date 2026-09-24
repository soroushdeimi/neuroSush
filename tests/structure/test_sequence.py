import pytest

from neurosush.structure.sequence import Neuron, sequence_timing


def test_timing_for_the_default_neuron():
    timing = sequence_timing(60, 25, Neuron())
    assert (timing.primed, timing.unprimed) == (8, 18)
    assert timing.coincidence == 11
    # the plateau primes the next element and ends before the one after it
    assert 60 + 8 - 18 <= timing.plateau <= 2 * 60 - 8
    # the previous element is 50 to 70 steps old: inside; the current one (<= 10) and the one
    # before (>= 110) are outside
    low, high = timing.context
    assert 10 < low <= 50
    assert 70 <= high < 110


@pytest.mark.parametrize(
    ("period", "window", "neuron", "match"),
    [
        (60, 15, Neuron(), "inside the window"),
        (60, 25, Neuron(gain=1.2), "gain"),
        (20, 25, Neuron(), "period"),
        (60, 25, Neuron(drive=5.0), "never brings"),
    ],
)
def test_unsound_timing_is_refused(period, window, neuron, match):
    with pytest.raises(ValueError, match=match):
        sequence_timing(period, window, neuron)
