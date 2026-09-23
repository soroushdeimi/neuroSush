import pytest
import torch

from neurosush.data import LocationDataset, spike_frames


class TinyDataset:
    def __init__(self):
        self.items = [(torch.tensor([1.0, 2.0]), 3), (torch.tensor([4.0, 5.0]), 7)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TestLocationDataset:
    def test_without_transforms(self):
        ds = LocationDataset(TinyDataset())
        assert len(ds) == 2
        image, location, label = ds[1]
        assert image.tolist() == [4.0, 5.0]
        assert location is None
        assert label == 7

    def test_transform_order(self):
        ds = LocationDataset(
            TinyDataset(),
            pre_transform=lambda img: (img * 10, img.sum()),
            post_transform=lambda img: img + 1,
            location_transform=lambda loc: loc * 2,
            target_transform=lambda y: y - 1,
        )
        image, location, label = ds[0]
        assert image.tolist() == [11.0, 21.0]
        assert location.item() == 6.0
        assert label == 2

    def test_location_transform_needs_a_location(self):
        ds = LocationDataset(TinyDataset(), location_transform=lambda loc: loc)
        with pytest.raises(ValueError, match="pre_transform"):
            ds[0]


def train(rows):
    return torch.tensor(rows, dtype=torch.bool)


class TestSpikeFrames:
    def test_frames_are_flattened_and_labelled(self):
        samples = [(train([[True, False], [False, True]]), "a")]
        frames = list(spike_frames(samples))
        assert [(f.tolist(), y) for f, y in frames] == [([True, False], "a"), ([False, True], "a")]

    def test_multi_dimensional_frames(self):
        sample = torch.zeros(3, 2, 2, dtype=torch.bool)
        frames = list(spike_frames([(sample, 0)]))
        assert len(frames) == 3
        assert frames[0][0].shape == (4,)

    def test_silence_between_samples(self):
        samples = [(train([[True]]), 1), (train([[True]]), 2)]
        frames = list(spike_frames(samples, silence=2))
        assert [(f.tolist(), y) for f, y in frames] == [
            ([True], 1),
            ([False], None),
            ([False], None),
            ([True], 2),
            ([False], None),
            ([False], None),
        ]

    def test_is_lazy(self):
        def samples():
            yield train([[True]]), 0
            raise AssertionError("second sample must not be read yet")

        first = next(spike_frames(samples()))
        assert first[0].tolist() == [True]

    def test_non_bool_trains_are_converted(self):
        frames = list(spike_frames([(torch.tensor([[0.0, 1.0]]), 0)]))
        assert frames[0][0].dtype == torch.bool
        assert frames[0][0].tolist() == [False, True]

    def test_invalid_silence(self):
        with pytest.raises(ValueError, match="silence"):
            list(spike_frames([], silence=-1))
