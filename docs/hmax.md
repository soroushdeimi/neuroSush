# HMAX building blocks

`neurosush.hmax` provides the building blocks of the HMAX model of Serre,
Wolf, Bileschi, Riesenhuber and Poggio (2007, TPAMI 29(3), Section 2,
Table 1), plus helpers for the convolutional STDP method of Kheradpisheh
et al. (2018, Neural Networks 99, Equations 3–4).
S1, C1, S2 and C2 operate on analog responses. The functions in `wta.py`
operate on first spike times and weights; they do not simulate a network.
`inhibition.py` operates on spike masks and potentials.

## Pipeline

`B` is the image count, `T` the orientation count, `P` the prototype count,
and `n` the prototype side length in C1 cells. Spatial tensor dimensions
are rows then columns; spatial image and kernel sizes are in pixels.

```text
image (B, 1, H, W)
  -> S1: list of (B, T, H, W), one per scale with padding="same"
  -> C1: list of (B, T, H_b, W_b), one per band
  -> S2: list of (B, P, H_b-n+1, W_b-n+1), using prototypes (P, T, n, n)
  -> C2: (B, P), max over supplied bands and spatial positions
```

Use only C1 bands large enough for the prototypes. This runnable CPU example
samples from the same image for brevity; it does not evaluate generalization.

```python
import torch
import neurosush.hmax as h

g = torch.Generator().manual_seed(1)
image = torch.rand(1, 1, 128, 128, generator=g)
s1 = h.s1_response(image, h.serre2007_gabor_bank())
c1 = h.c1_pool(s1, h.c1_bands_from_spec(h.SERRE2007_BANDS))
prototypes = h.sample_c1_patches([band[0] for band in c1[:4]], 10, 4, g)
s2 = [h.s2_rbf(band, prototypes, beta=0.05) for band in c1[:4]]
c2 = h.c2_global_max(s2)
print('image:', tuple(image.shape))
print('S1:', len(s1), 'scales, each', tuple(s1[0].shape))
print('C1:', [tuple(band.shape) for band in c1])
print('prototypes:', tuple(prototypes.shape))
print('S2:', [tuple(band.shape) for band in s2])
print('C2:', tuple(c2.shape))
```

Printed with the task's Python environment (`PYTHONPATH=src`):

```text
image: (1, 1, 128, 128)
S1: 16 scales, each (1, 4, 128, 128)
C1: [(1, 4, 31, 31), (1, 4, 24, 24), (1, 4, 20, 20), (1, 4, 17, 17), (1, 4, 15, 15), (1, 4, 13, 13), (1, 4, 11, 11), (1, 4, 10, 10)]
prototypes: (10, 4, 4, 4)
S2: [(1, 10, 28, 28), (1, 10, 21, 21), (1, 10, 17, 17), (1, 10, 14, 14)]
C2: (1, 10)
```

## Conventions

| Item | Convention |
|---|---|
| Filter axes | In the Gabor formula, `x` runs along rows and `y` along columns. |
| Orientation | `theta=0` radians gives horizontal wave fronts: stripes run along columns and the carrier varies along rows. |
| S1 operation | Absolute cross-correlation using `conv2d`, without flipping the kernel; not mathematical convolution. |
| Gabor normalization | Subtract the kernel mean, then divide by its L2 norm in float64 before casting to the requested dtype. |
| S1 padding | Default `"same"`: zero padding of `s // 2` pixels for odd kernels; `"valid"` is also supported. |
| S1 normalization | Default `normalize=False`; when enabled, divide by `sqrt(sum(input_patch**2) + eps)`. |
| C1 pooling | Elementwise max over the paired scales, then spatial max pooling with no padding; `H_b = floor((H-grid)/stride)+1`, likewise for width. Orientations stay separate. |
| S2 windows | Valid windows only, at every C1 position; output spatial size is `(H_b-n+1, W_b-n+1)`. |
| S2 distance | Compute `d2 = sum(X**2) - 2*sum(X*P) + sum(P**2)` in float64; clamp at zero before `exp(-beta*d2)`, then cast to the C1 dtype. |

## Public API

All names below are exported from `neurosush.hmax`; module paths are relative
to that package. `C` denotes feature maps, and `S` the number of shifts.
Spike times use caller-chosen, consistent time units. Analog responses,
weights and correlations have no physical unit; beta has inverse squared
C1-response units.

| Name | Module | Inputs and result |
|---|---|---|
| `SERRE2007_BANDS` | `gabor` | Tuple of band specifications; parameters below. |
| `SERRE2007_GAMMA` | `gabor` | Scalar envelope aspect ratio, dimensionless, `0.3`. |
| `SERRE2007_THETAS` | `gabor` | Orientation tuple in radians: `(0, pi/4, pi/2, 3*pi/4)`. |
| `GaborBandSpec` | `gabor` | Frozen record: `band`, `sizes`, `sigmas`, `wavelengths`, `c1_grid`, `c1_stride`. |
| `gabor_bank` | `gabor` | Scale parameters and angles → list of kernels `(T, 1, s, s)`. |
| `serre2007_gabor_bank` | `gabor` | Optional dtype → 16 scale tensors `(4, 1, s, s)`. |
| `s1_response` | `s1` | Images `(B, 1, H, W)` and bank → scale list `(B, T, H, W)` with same padding, or `(B, T, H-s+1, W-s+1)` with valid padding. |
| `c1_bands_from_spec` | `c1` | Specs → `(2*i, 2*i+1, grid, stride)` tuples, with `i` starting at zero. |
| `c1_pool` | `c1` | S1 list and band tuples → list `(B, T, H_b, W_b)`. |
| `sample_c1_patches` | `s2` | Single-image bands `(T, H_b, W_b)`, count `P`, size `n`, generator → copies `(P, T, n, n)`, sampling uniformly over eligible bands, then valid positions. |
| `s2_rbf` | `s2` | C1 `(B, T, H_b, W_b)`, prototypes `(P, T, n, n)`, positive beta → `(B, P, H_b-n+1, W_b-n+1)`. |
| `c2_global_max` | `c2` | S2 band list with matching `B, P` → `(B, P)`. |
| `select_conv_winners` | `wta` | Times and potentials `(C, H, W)`, `k`, radius in cells → long `(N, 3)` rows `(c, y, x)`, `N <= k`; finite times only, ordered by time, descending potential, then indices, with at most one winner per map and Chebyshev exclusion across maps. |
| `simplified_stdp` | `wta` | Weights and pre-times of equal arbitrary shape, scalar post-time and nonnegative learning rates → new same-shape weights; `+a_plus*w*(1-w)` when pre ≤ post, otherwise `-a_minus*w*(1-w)` (including never-spiked pre-times for finite post-time). |
| `stdp_convergence` | `wta` | Weights of arbitrary shape → scalar `mean(w*(1-w))`. |
| `spatial_lateral_inhibition` | `inhibition` | Bool spikes and potentials `(B, C, H, W)`, bool fired mask `(B, H, W)` → new spikes and fired mask of those shapes; highest potential survives per pixel, lowest map index breaks ties, already-fired pixels suppress all spikes. |
| `bar_image` | `probe` | Image size, odd bar length/width in pixels, `"h"`/`"v"`, center `(row, col)` and value → `(size, size)` float image. |
| `shift_image` | `probe` | Tensor with spatial trailing dimensions, integer `(dy, dx)` in pixels → same shape, zero-filled shift; positive means down/right. |
| `normalized_correlation` | `probe` | Equal-shaped tensors with nonzero variance → scalar Pearson correlation over all elements. |
| `invariance_curve` | `probe` | Callable, image and `S` shift pairs → correlations `(S,)` of the base response with each shifted response. |

## Band parameters

The eight rows below were generated by iterating over the printed
`SERRE2007_BANDS`, rather than transcribing the paper. Band labels follow
the specification; Python list indices start at zero. Each pair lists the
two S1 scales. C1 grid and stride are measured on the S1 pixel lattice.

| Band | Sizes (pixels) | Sigma (pixels) | Lambda (pixels) | C1 grid (pixels) | Stride (pixels) |
|---|---|---|---|---|---|
| 1 | (7, 9) | (2.8, 3.6) | (3.5, 4.6) | 8 | 4 |
| 2 | (11, 13) | (4.5, 5.4) | (5.6, 6.8) | 10 | 5 |
| 3 | (15, 17) | (6.3, 7.3) | (7.9, 9.1) | 12 | 6 |
| 4 | (19, 21) | (8.2, 9.2) | (10.3, 11.5) | 14 | 7 |
| 5 | (23, 25) | (10.2, 11.3) | (12.7, 14.1) | 16 | 8 |
| 6 | (27, 29) | (12.3, 13.4) | (15.4, 16.8) | 18 | 9 |
| 7 | (31, 33) | (14.6, 15.8) | (18.2, 19.7) | 20 | 10 |
| 8 | (35, 37) | (17.0, 18.2) | (21.2, 22.8) | 22 | 11 |

## Choosing beta

S2 uses `exp(-beta * ||X-P||**2)` without dividing the distance by patch
size. Beta must shrink as `n` grows to retain useful response magnitudes.
With `beta=1`, large prototypes underflow towards zero; the measured values
below are still nonzero, so this run shows decay rather than actual rounding
to zero.

Measured on CPU with torch 2.14.1 and float32 inputs/outputs, using a single
`torch.Generator().manual_seed(1)`: draw two independent uniform random
images of shape `(1, 1, 128, 128)` consecutively, compute unnormalized S1
and C1, and keep Python band indices 0–3. For each `n` in ascending order,
sample 10 prototypes from the first image using `sample_c1_patches` and
the same generator, then apply those prototypes to all four bands of the
second image. Reuse the prototypes for both beta values. Min/max are over
the resulting 10 C2 values, which are dimensionless. `n` is in C1 cells;
beta is in inverse squared C1-response units.

| n (C1 cells) | beta | Min C2 | Max C2 |
|---|---|---|---|
| 4 | 1.0 | 3.36314701e-02 | 4.15275037e-01 |
| 4 | 0.05 | 8.43989968e-01 | 9.57010686e-01 |
| 8 | 1.0 | 3.07483560e-06 | 3.10260849e-03 |
| 8 | 0.05 | 5.30140638e-01 | 7.49180317e-01 |
| 12 | 1.0 | 3.68175596e-13 | 2.23817210e-07 |
| 12 | 0.05 | 2.38947645e-01 | 4.65044677e-01 |
| 16 | 1.0 | 2.64588724e-26 | 2.65853657e-14 |
| 16 | 0.05 | 5.26173003e-02 | 2.09523201e-01 |

These values describe this random-image experiment, not a recommended beta
for a dataset. Input scaling and S1 normalization also change the distances.

## Limits

The package has no classifier, dataset loader or spiking network object.
Serre's original circular aperture and exact S1 normalization are not
implemented; the optional local L2 normalization is the formula above.
Reproducibility claims are limited to CPU: the validation pipeline checks
repeatability with fixed seeds there. GPU behavior and agreement across
devices or torch versions are not verified.

## References

- Thomas Serre, Lior Wolf, Stanley Bileschi, Maximilian Riesenhuber and
  Tomaso Poggio. *Robust Object Recognition with Cortex-Like Mechanisms*.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(3),
  2007, Section 2 and Table 1. DOI NOT VERIFIED.
- Saeed Reza Kheradpisheh, Mohammad Ganjtabesh, Simon J. Thorpe and
  Timothée Masquelier. *STDP-based spiking deep convolutional neural
  networks for object recognition*. Neural Networks, 99, 2018,
  Equations 3–4. DOI: `10.1016/j.neunet.2017.12.005` (verified in the
  local paper text).
