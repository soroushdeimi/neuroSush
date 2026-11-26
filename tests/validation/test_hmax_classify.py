"""The hmax_c2_classify example: dataset, C2 pipeline, and end-to-end accuracy."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "hmax_c2_classify.py"


def load():
    spec = importlib.util.spec_from_file_location("hmax_c2_classify", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hmax_c2_classify"] = module
    spec.loader.exec_module(module)
    return module


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestMakeBarDataset:
    def test_shapes_and_sorted_labels(self):
        ex = load()
        images, labels = ex.make_bar_dataset(5, 64, gen(0))
        assert images.shape == (10, 1, 64, 64)
        assert images.dtype == torch.float32
        assert labels.shape == (10,)
        assert labels.dtype == torch.long
        assert labels.tolist() == [0] * 5 + [1] * 5

    def test_same_seed_identical(self):
        ex = load()
        a = ex.make_bar_dataset(4, 64, gen(7))
        b = ex.make_bar_dataset(4, 64, gen(7))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_different_seed_differs(self):
        ex = load()
        a = ex.make_bar_dataset(4, 64, gen(1))
        b = ex.make_bar_dataset(4, 64, gen(2))
        assert not torch.equal(a[0], b[0])

    def test_horizontal_bars_vary_more_across_rows(self):
        ex = load()
        images, labels = ex.make_bar_dataset(10, 64, gen(0))
        row_sums = images.sum(dim=-1)  # (N, 1, H): sum over columns
        var0 = row_sums[labels == 0].var(dim=-1).mean().item()
        var1 = row_sums[labels == 1].var(dim=-1).mean().item()
        assert var0 > var1

    @pytest.mark.parametrize("n_per_class", [0, -1])
    def test_n_per_class_must_be_positive(self, n_per_class):
        ex = load()
        with pytest.raises(ValueError, match=str(n_per_class)):
            ex.make_bar_dataset(n_per_class, 64, gen(0))

    @pytest.mark.parametrize("size", [39, 0])
    def test_size_must_be_at_least_40(self, size):
        ex = load()
        with pytest.raises(ValueError, match=str(size)):
            ex.make_bar_dataset(2, size, gen(0))


class TestHmaxC2:
    def test_deterministic_and_shape(self):
        ex = load()
        images, _ = ex.make_bar_dataset(6, 64, gen(0))
        protos = ex.learn_prototypes(images, sizes=(4, 8), per_size=5, generator=gen(1))
        a = ex.hmax_c2(images, protos)
        b = ex.hmax_c2(images, protos)
        assert a.shape == (12, 10)  # 2 * 6 images; 5 + 5 prototypes across the two sizes
        assert torch.equal(a, b)

    def test_does_not_mutate_inputs(self):
        ex = load()
        images, _ = ex.make_bar_dataset(4, 64, gen(0))
        protos = ex.learn_prototypes(images, sizes=(4,), per_size=3, generator=gen(1))
        img_ref = images.clone()
        proto_ref = {n: p.clone() for n, p in protos.items()}
        ex.hmax_c2(images, protos)
        assert torch.equal(images, img_ref)
        for n, p in proto_ref.items():
            assert torch.equal(protos[n], p)


class TestEndToEnd:
    def test_accuracy_at_least_0_9(self):
        ex = load()
        for seed in (0, 1):
            g = gen(seed)
            train = ex.make_bar_dataset(30, 64, g)
            test = ex.make_bar_dataset(30, 64, gen(seed + 1000))
            protos = ex.learn_prototypes(train[0], sizes=(4, 8), per_size=20, generator=g)
            model = ex.fit_linear(ex.hmax_c2(train[0], protos), train[1])
            acc = ex.accuracy(model, ex.hmax_c2(test[0], protos), test[1])
            assert acc >= 0.9

    def test_shuffled_train_labels_drop_accuracy(self):
        ex = load()
        g = gen(0)
        train = ex.make_bar_dataset(30, 64, g)
        test = ex.make_bar_dataset(30, 64, gen(1000))
        protos = ex.learn_prototypes(train[0], sizes=(4, 8), per_size=20, generator=g)
        perm = torch.randperm(60, generator=gen(123))
        model = ex.fit_linear(ex.hmax_c2(train[0], protos), train[1][perm])
        acc = ex.accuracy(model, ex.hmax_c2(test[0], protos), test[1])
        assert acc < 0.8


class TestLearnPrototypes:
    def test_empty_sizes_raises(self):
        ex = load()
        images, _ = ex.make_bar_dataset(2, 64, gen(0))
        with pytest.raises(ValueError, match="empty"):
            ex.learn_prototypes(images, sizes=(), per_size=5, generator=gen(0))
