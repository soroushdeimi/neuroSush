"""STDP feature learning on C1 spikes (examples/hmax_stdp_features.py) against measured values.

The reference solution trains F = 4 feature neurons with 3x3 kernels over the 4 C1
orientation channels by simplified STDP (Kheradpisheh et al. 2018, Eq. 3-4) on
horizontal and vertical bars. Measured with seed 0 (30 bars per class, 3 epochs,
THRESH = 2.0): the convergence index falls from 0.24916988611221313 to
0.08415135741233826 and the dominant orientation channels are [2, 2, 0, 0]
(channel 0 = horizontal axis, channel 2 = vertical axis of the Serre bank).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "hmax_stdp_features", Path(__file__).resolve().parents[2] / "examples" / "hmax_stdp_features.py"
)
assert SPEC is not None
assert SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

SEED = 0
N_PER_CLASS = 30
EPOCHS = 3
Trained = tuple[torch.Tensor, list[float], torch.Tensor]


@pytest.fixture(scope="module")
def trained() -> Trained:
    """Train once per module: (final W, convergence history, initial W)."""
    images = example.make_images(N_PER_CLASS, torch.Generator().manual_seed(SEED))
    w0 = 0.45 + 0.1 * torch.rand(
        example.N_FEATURES, 4, 3, 3, generator=torch.Generator().manual_seed(SEED)
    )
    w, history = example.train(images, torch.Generator().manual_seed(SEED), epochs=EPOCHS)
    return w, history, w0


class TestTrain:
    def test_determinism_same_seed_identical_weights(self) -> None:
        images = example.make_images(2, torch.Generator().manual_seed(SEED))
        w1, _ = example.train(images, torch.Generator().manual_seed(SEED), epochs=1)
        w2, _ = example.train(images, torch.Generator().manual_seed(SEED), epochs=1)
        assert torch.equal(w1, w2)

    def test_weights_stay_in_unit_interval(self, trained: Trained) -> None:
        w = trained[0]
        assert float(w.min()) >= 0.0
        assert float(w.max()) <= 1.0

    def test_weights_changed_from_initial(self, trained: Trained) -> None:
        w, _, w0 = trained
        assert not torch.equal(w, w0)

    def test_convergence_history(self, trained: Trained) -> None:
        _, history, _ = trained
        assert len(history) == EPOCHS + 1
        assert history[0] == pytest.approx(0.24916988611221313, abs=1e-6)
        assert history[-1] == pytest.approx(0.08415135741233826, abs=1e-6)

    def test_convergence_decreases(self, trained: Trained) -> None:
        _, history, _ = trained
        assert history[-1] < history[0]

    def test_dominant_orientations(self, trained: Trained) -> None:
        w = trained[0]
        dom = example.dominant_orientations(w)
        assert dom == [2, 2, 0, 0]
        assert len(set(dom)) >= 2
        assert 0 in dom
        assert 2 in dom

    def test_control_zero_learning_rates_unchanged(self) -> None:
        images = example.make_images(2, torch.Generator().manual_seed(SEED))
        w0 = 0.45 + 0.1 * torch.rand(
            example.N_FEATURES, 4, 3, 3, generator=torch.Generator().manual_seed(SEED)
        )
        img = images[0]
        t_in, spikes = example.encode(img)
        w = example.train_step(w0, t_in, spikes, a_plus=0.0, a_minus=0.0)
        assert torch.equal(w, w0)
        assert float(example.stdp_convergence(w)) == pytest.approx(
            float(example.stdp_convergence(w0)), abs=1e-9
        )


class TestNoMutation:
    def test_encode_does_not_mutate_image(self) -> None:
        img = example.make_images(1, torch.Generator().manual_seed(SEED))[0]
        img_copy = img.clone()
        example.encode(img)
        assert torch.equal(img, img_copy)

    def test_layer_forward_does_not_mutate_inputs(self) -> None:
        img = example.make_images(1, torch.Generator().manual_seed(SEED))[0]
        _, spikes = example.encode(img)
        w = 0.45 + 0.1 * torch.rand(
            example.N_FEATURES, 4, 3, 3, generator=torch.Generator().manual_seed(SEED)
        )
        spikes_copy, w_copy = spikes.clone(), w.clone()
        example.layer_forward(spikes, w, example.THRESH)
        assert torch.equal(spikes, spikes_copy)
        assert torch.equal(w, w_copy)

    def test_train_step_does_not_mutate_inputs(self) -> None:
        img = example.make_images(1, torch.Generator().manual_seed(SEED))[0]
        t_in, spikes = example.encode(img)
        w = 0.45 + 0.1 * torch.rand(
            example.N_FEATURES, 4, 3, 3, generator=torch.Generator().manual_seed(SEED)
        )
        t_in_copy, spikes_copy, w_copy = t_in.clone(), spikes.clone(), w.clone()
        example.train_step(w, t_in, spikes)
        assert torch.equal(t_in, t_in_copy)
        assert torch.equal(spikes, spikes_copy)
        assert torch.equal(w, w_copy)

    def test_train_does_not_mutate_images(self) -> None:
        images = example.make_images(2, torch.Generator().manual_seed(SEED))
        images_copy = images.clone()
        example.train(images, torch.Generator().manual_seed(SEED), epochs=1)
        assert torch.equal(images, images_copy)

    def test_dominant_orientations_does_not_mutate_weights(self) -> None:
        w = 0.45 + 0.1 * torch.rand(
            example.N_FEATURES, 4, 3, 3, generator=torch.Generator().manual_seed(SEED)
        )
        w_copy = w.clone()
        example.dominant_orientations(w)
        assert torch.equal(w, w_copy)


class TestFailureCases:
    def test_encode_steps_zero_raises(self) -> None:
        img = example.make_images(1, torch.Generator().manual_seed(SEED))[0]
        with pytest.raises(ValueError, match="got 0"):
            example.encode(img, steps=0)

    def test_make_images_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="got 0"):
            example.make_images(0, torch.Generator().manual_seed(SEED))
