"""End-to-end HMAX pipeline: Gabor -> S1 -> C1 -> S2 -> C2, plus the package exports."""

import re

import torch

import neurosush.hmax as hmax
from neurosush.hmax import (
    SERRE2007_BANDS,
    bar_image,
    c1_bands_from_spec,
    c1_pool,
    c2_global_max,
    normalized_correlation,
    s1_response,
    s2_rbf,
    sample_c1_patches,
    serre2007_gabor_bank,
    shift_image,
)

BANK = serre2007_gabor_bank()
BANDS = c1_bands_from_spec(SERRE2007_BANDS)


def c1_of(img: torch.Tensor) -> list[torch.Tensor]:
    """S1 then C1 for a (H, W) image, using the Serre 2007 bank and band table."""
    s1 = s1_response(img[None, None], BANK)
    return c1_pool(s1, BANDS)


def test_c1_shapes():
    img = bar_image(128, 41, 5, "v", (64, 64))
    c1 = c1_of(img)
    assert [tuple(t.shape) for t in c1] == [
        (1, 4, 31, 31),
        (1, 4, 24, 24),
        (1, 4, 20, 20),
        (1, 4, 17, 17),
        (1, 4, 15, 15),
        (1, 4, 13, 13),
        (1, 4, 11, 11),
        (1, 4, 10, 10),
    ]


def _c2_for_bands(
    c1: list[torch.Tensor], n: int, beta: float, bands: range, seed: int
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    protos = sample_c1_patches([t[0] for t in c1[: bands.stop]], n, 4, gen)
    s2 = [s2_rbf(c1[k], protos, beta) for k in bands]
    return c2_global_max(s2)


def test_pipeline_is_deterministic():
    # CPU only: the whole chain (bank -> S1 -> C1 -> sample -> S2 -> C2) is bit-identical
    # across two runs with the same seed.
    img = bar_image(128, 41, 5, "v", (64, 64))
    c1 = c1_of(img)
    a = _c2_for_bands(c1, 4, 1.0, range(3), 0)
    b = _c2_for_bands(c1, 4, 1.0, range(3), 0)
    assert torch.equal(a, b)


def test_c1_is_more_shift_tolerant_than_s1():
    img = bar_image(128, 41, 5, "v", (64, 64))
    shifted = shift_image(img, 0, 10)
    s1 = s1_response(img[None, None], BANK)
    s1s = s1_response(shifted[None, None], BANK)
    c1 = c1_of(img)
    c1s = c1_of(shifted)
    for k in (0, 2, 4):
        s1_corr = normalized_correlation(s1[2 * k], s1s[2 * k])
        c1_corr = normalized_correlation(c1[k], c1s[k])
        assert c1_corr > s1_corr


def test_c2_self_match_is_one():
    img = bar_image(128, 41, 5, "v", (64, 64))
    c1 = c1_of(img)
    c2 = _c2_for_bands(c1, 4, 1.0, range(3), 0)
    assert c2.shape == (1, 4)
    assert torch.allclose(c2, torch.ones_like(c2), atol=1e-6)


def test_c2_on_different_image_is_below_one():
    img = bar_image(128, 41, 5, "v", (64, 64))
    c1 = c1_of(img)
    gen = torch.Generator().manual_seed(0)
    protos = sample_c1_patches([t[0] for t in c1[:4]], 10, 4, gen)
    rand_img = torch.rand(1, 1, 128, 128, generator=torch.Generator().manual_seed(1))
    c1r = c1_of(rand_img[0, 0])
    s2 = [s2_rbf(c1r[k], protos, 1.0) for k in range(4)]
    c2 = c2_global_max(s2)
    assert c2.shape == (1, 10)
    assert bool((c2 > 0).all() and (c2 < 1).all())
    # a wider radial basis (smaller beta) matches the random image far better
    s2_wide = [s2_rbf(c1r[k], protos, 0.01) for k in range(4)]
    c2_wide = c2_global_max(s2_wide)
    assert bool((c2_wide > 0.5).all())


def _isort_key(name: str):
    # ruff RUF022 orders __all__ by casing (SCREAMING_SNAKE, CamelCase, other) then
    # natural sort within each group; mirror that so the check matches the linter.
    if len(name) > 1 and name.isupper() and any(c.isalpha() for c in name):
        category = 0
    elif name[0].isupper():
        category = 1
    else:
        category = 2
    return (category, [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", name)])


def test_all_is_sorted_and_importable():
    assert hmax.__all__ == sorted(hmax.__all__, key=_isort_key)
    for name in hmax.__all__:
        assert hasattr(hmax, name)
