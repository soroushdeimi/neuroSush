import pytest
import torch

from neurosush.core.buffers import ArrivalBuffer, HistoryBuffer


class TestHistoryBuffer:
    def test_starts_empty(self):
        buf = HistoryBuffer(depth=3, size=2)
        assert buf.read(0).tolist() == [False, False]
        assert buf.read(torch.tensor([2, 1])).tolist() == [False, False]

    def test_read_zero_is_newest(self):
        buf = HistoryBuffer(depth=3, size=2)
        buf.push(torch.tensor([True, False]))
        buf.push(torch.tensor([False, True]))
        assert buf.read(0).tolist() == [False, True]
        assert buf.read(1).tolist() == [True, False]
        assert buf.read(2).tolist() == [False, False]

    def test_per_neuron_delay(self):
        buf = HistoryBuffer(depth=2, size=3)
        buf.push(torch.tensor([True, True, False]))
        buf.push(torch.tensor([False, False, True]))
        # neuron 0 looks one step back, neuron 1 and 2 see the newest value
        assert buf.read(torch.tensor([1, 0, 0])).tolist() == [True, False, True]

    def test_oldest_value_is_dropped(self):
        buf = HistoryBuffer(depth=2, size=1)
        for value in (True, False, False):
            buf.push(torch.tensor([value]))
        assert buf.read(torch.tensor([1])).tolist() == [False]

    def test_float_values(self):
        buf = HistoryBuffer(depth=2, size=2, dtype=torch.float32)
        buf.push(torch.tensor([0.5, -1.0]))
        assert buf.read(0).tolist() == [0.5, -1.0]

    def test_push_copies_input(self):
        buf = HistoryBuffer(depth=2, size=2)
        value = torch.tensor([True, False])
        buf.push(value)
        value[1] = True
        assert buf.read(0).tolist() == [True, False]

    def test_read_does_not_alias_storage(self):
        buf = HistoryBuffer(depth=1, size=2)
        buf.push(torch.tensor([True, False]))
        out = buf.read(0)
        out[1] = True
        assert buf.read(0).tolist() == [True, False]

    def test_reset(self):
        buf = HistoryBuffer(depth=2, size=1)
        buf.push(torch.tensor([True]))
        buf.reset()
        assert buf.read(0).tolist() == [False]

    @pytest.mark.parametrize("delay", [3, -1])
    def test_delay_out_of_range(self, delay):
        buf = HistoryBuffer(depth=3, size=2)
        with pytest.raises(ValueError, match="delay"):
            buf.read(torch.tensor([0, delay]))

    def test_push_wrong_shape(self):
        buf = HistoryBuffer(depth=2, size=3)
        with pytest.raises(ValueError, match="shape"):
            buf.push(torch.tensor([True, False]))

    @pytest.mark.parametrize(("depth", "size"), [(0, 1), (1, 0), (-2, 3)])
    def test_invalid_dimensions(self, depth, size):
        with pytest.raises(ValueError, match="positive"):
            HistoryBuffer(depth=depth, size=size)

    def test_properties(self):
        buf = HistoryBuffer(depth=4, size=5)
        assert (buf.depth, buf.size) == (4, 5)


class TestArrivalBuffer:
    def test_zero_delay_arrives_now(self):
        buf = ArrivalBuffer(depth=2, size=2)
        buf.add(torch.tensor([1.0, 2.0]), 0)
        assert buf.current().tolist() == [1.0, 2.0]

    def test_delayed_value_arrives_after_advances(self):
        buf = ArrivalBuffer(depth=3, size=1)
        buf.add(torch.tensor([5.0]), torch.tensor([2]))
        seen = []
        for _ in range(3):
            seen.append(buf.current().item())
            buf.advance()
        assert seen == [0.0, 0.0, 5.0]

    def test_contributions_accumulate(self):
        buf = ArrivalBuffer(depth=2, size=2)
        buf.add(torch.tensor([1.0, 1.0]), torch.tensor([0, 1]))
        buf.add(torch.tensor([0.5, 3.0]), torch.tensor([0, 1]))
        assert buf.current().tolist() == [1.5, 0.0]
        buf.advance()
        assert buf.current().tolist() == [0.0, 4.0]

    def test_advance_clears_the_new_last_slot(self):
        buf = ArrivalBuffer(depth=2, size=1)
        buf.add(torch.tensor([1.0]), 0)
        buf.advance()
        buf.advance()
        assert buf.current().tolist() == [0.0]

    def test_current_does_not_alias_storage(self):
        buf = ArrivalBuffer(depth=1, size=1)
        buf.add(torch.tensor([1.0]), 0)
        buf.current()[0] = 9.0
        assert buf.current().tolist() == [1.0]

    def test_add_does_not_mutate_input(self):
        buf = ArrivalBuffer(depth=2, size=2)
        value = torch.tensor([1.0, 2.0])
        buf.add(value, torch.tensor([1, 0]))
        assert value.tolist() == [1.0, 2.0]

    def test_delay_out_of_range(self):
        buf = ArrivalBuffer(depth=2, size=1)
        with pytest.raises(ValueError, match="delay"):
            buf.add(torch.tensor([1.0]), torch.tensor([2]))

    def test_add_wrong_shape(self):
        buf = ArrivalBuffer(depth=2, size=2)
        with pytest.raises(ValueError, match="shape"):
            buf.add(torch.tensor([1.0]), 0)

    def test_reset(self):
        buf = ArrivalBuffer(depth=2, size=1)
        buf.add(torch.tensor([1.0]), torch.tensor([1]))
        buf.reset()
        buf.advance()
        assert buf.current().tolist() == [0.0]

    def test_default_dtype_is_float32(self):
        assert ArrivalBuffer(depth=1, size=1).current().dtype == torch.float32
