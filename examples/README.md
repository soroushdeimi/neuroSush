# Examples

Scripts that run on a CPU unless noted. Each module docstring has the details, the measured
numbers and the deviations from the paper.

| example | what it shows | paper | runtime |
|---|---|---|---|
| `two_patterns.py` | two LIF output neurons learn to tell two Poisson input patterns apart with STDP, k-winners-take-all and homeostasis | none | seconds |
| `sequence_prediction.py` | spatial pooler, temporal memory and classifier learn `ABCDE` and `XBCDY`; anomaly drops to 0 after the first symbol and both endings are predicted | none | seconds |
| `object_recognition.py` | columns that vote recognize random objects in fewer touches; mean touches for 1 to 8 columns | Lewis et al. 2019, Fig. 5 | seconds |
| `diehl_cook_mnist.py` | unsupervised MNIST with STDP, adaptive threshold and lateral inhibition; `pair` and `triplet` rules | Diehl and Cook 2015 | quick check: CPU, about 2 samples/s; full setting about an hour on a GPU |
| `stdp_frequency.py` | pair versus triplet STDP as a function of pairing frequency | Sjostrom et al. 2001; Pfister and Gerstner 2006 | a few seconds |
| `balanced_network.py` | inhibitory STDP drives a postsynaptic rate to a 5 Hz target | Vogels et al. 2011 (reduced) | about 15 s |
| `predictive_coding_mnist.py` | supervised predictive coding network on MNIST, local weight updates | Whittington and Bogacz 2017 | about 30 s |
| `competitive_stdp.py` | additive STDP with slight depression bias: bimodal weights, output rate nearly independent of input rate, correlated inputs win | Song, Miller and Abbott 2000 | quick defaults about 2.5 min; paper-scale about 7 min per input rate |
| `spike_pattern_detection.py` | a LIF neuron with nearest-spike STDP becomes selective to a repeating 50 ms pattern in noise | Masquelier, Guyonneau and Thorpe 2008 | about 50 s |
| `brunel_network.py` | sparse excitatory/inhibitory LIF network in four regimes (SR, AI, SI fast, SI slow) | Brunel 2000 | about 40 s to 4 min |
| `conv_stdp_mnist.py` | STDP-trained spiking convolutional network with latency coding on MNIST | Kheradpisheh et al. 2018 | about 1 minute on a GPU |
| `intrinsic_timing_ramps.py` | ramping activity and a population code for elapsed time from relaxation, with no prediction (illustration) | Huang et al. 2026 (qualitative) | about 1.5 min |

Run from the repository root with the package installed (`pip install -e .`):

```bash
python examples/two_patterns.py
python examples/sequence_prediction.py
python examples/object_recognition.py
python examples/diehl_cook_mnist.py --data path/to/MNIST/raw
python examples/stdp_frequency.py
python examples/balanced_network.py
python examples/predictive_coding_mnist.py --data path/to/MNIST/raw
python examples/competitive_stdp.py
python examples/spike_pattern_detection.py
python examples/brunel_network.py
python examples/conv_stdp_mnist.py --data path/to/MNIST/raw --device cuda
python examples/intrinsic_timing_ramps.py
```

## Results and deviations from the paper

- **stdp_frequency**: weight change over 60 pairings (initial weight 0.5, no bounds):

  | freq (Hz) | pair +10 | pair -10 | triplet +10 | triplet -10 |
  |---|---|---|---|---|
  | 0.1 | +0.1624 | -0.3152 | +0.0000 | -0.3152 |
  | 10 | +0.1335 | -0.3300 | +0.1162 | -0.3301 |
  | 20 | +0.0096 | -0.3765 | +0.2207 | -0.3426 |
  | 40 | -0.2902 | -0.4378 | +0.5037 | +0.1482 |
  | 50 | -0.4427 | -0.4582 | +0.7184 | +0.7052 |

  The triplet rule turns depression into potentiation at 40-50 Hz for both lags; the pair
  rule never potentiates more at high frequency. Triplet parameters are the minimal
  all-to-all visual cortex values of Pfister and Gerstner (2006), Table 3 (A2+ = 0,
  A3+ = 6.5e-3, A2- = 7.1e-3, A3- = 0, tau_y = 114 ms, tau_plus = 16.8, tau_minus = 33.7),
  checked against the open-access text (PMC6674434). Deviations: exact spike times instead
  of noisy real synapses, and absolute weight changes of a weight of 0.5 instead of the
  paper's relative changes, so only the shape is comparable.
- **balanced_network** (60 s, 5 s windows, seed 0): the mean excitatory rate falls from
  44 Hz in the first window through 16, 9.8 and 7.6 Hz to between 4.4 and 7.8 Hz (mean about
  5.7 Hz) from 25 s on; the inhibitory weights grow from 0.02 to a mean of 0.224 (std
  0.027). Deviations: 20 neurons with independent Poisson inputs and all-to-all connectivity
  instead of recurrent networks of 8000 excitatory and 2000 inhibitory neurons, and only
  convergence to the target rate is shown.
- **predictive_coding_mnist** (CPU, defaults, seed 0, about 30 s): test accuracy on the
  10000 test images 0.838 after the first epoch, 0.907 after five and 0.915 after ten (a
  fully trained MLP reaches about 98 %). Deviations: a plain tanh MLP with one hidden layer,
  mini-batches, a fixed number of inference steps, a lowered hidden variance, no biases and
  no hyper-parameter search.
- **competitive_stdp** (paper-scale run, 1000 s, seed 0, last 100 s, uniform initial
  weights): output rate 9.3 Hz at 10 Hz input and 10.3 Hz at 40 Hz input (11.8 and 10.3 Hz
  from all weights at g_max); weights below 0.1 g_max / above 0.9 g_max: 26 % / 34 % at 10 Hz
  and 78 % / 8 % at 40 Hz. Quick defaults (100 s, A+ 5 times larger, 11.6, 19.1 and 38.5 Hz
  at 10, 20 and 40 Hz input) are far from equilibrium and the rate is not yet independent of
  the input. With correlated inputs (500 of 1000 afferents, c = 0.2, 10 Hz) the correlated
  group ends at a mean weight of 0.95 g_max, the others at 0.36 g_max. Deviations: the
  parameters were quoted from memory and not checked against the paper, dt = 1 ms, no
  refractory period, 5 times larger A+ in the quick defaults, and my own correlation scheme.
- **spike_pattern_detection** (150 s, seeds 0, 1, 2, last 50 s): hit rate 93.4 %, 93.7 % and
  85.9 %, one false alarm each, latency first fifth -> last fifth 7.5 -> 3.8, 11.4 -> 3.5 and
  11.5 -> 3.2 ms; by the example's criterion (hit rate above 90 %) 2 of 3 seeds are
  selective. The paper reports 96 % of 100 runs, 99.1 % hits and a latency of about 4 ms.
  Deviations: a LIF neuron with exponential current and a tuned threshold instead of the
  spike response model, a local slow after-spike current, 150 s instead of 450 s, and my own
  rate-change process, gap distribution and jitter.
- **brunel_network** (2000 excitatory neurons, 2 s after 0.5 s, seed 0, excitatory
  neurons):

  | regime | g | eta | rate Hz | CV | chi | peak Hz |
  |---|---|---|---|---|---|---|
  | SR | 3 | 2 | 328.2 | 0.07 | 0.44 | 166 |
  | AI | 5 | 2 | 43.4 | 1.10 | 0.14 | 0.5 |
  | SI fast | 6 | 4 | 60.6 | 1.45 | 0.14 | 170 |
  | SI slow | 4.5 | 0.9 | 16.4 | 0.85 | 0.20 | 42 |

  Deviations: 2000 instead of 10000 excitatory neurons with J scaled to 0.5 mV, dt = 0.5 ms
  instead of 0.1 ms, a short run from one seed, and the paper's parameters were taken from a
  reproduction, not checked against its text. All numbers are qualitative.
- **conv_stdp_mnist** (RTX 3060 laptop, about 1 minute, 3000 STDP images per layer, 10000
  classifier and 10000 test images): 89.8 % with global pooling (100 features, seed 0) and
  97.2 % with `--grid 2` (400 features, seed 1); the paper reports 98.4 % after training on
  the whole set. Single runs. Deviations: logistic regression instead of a linear SVM, a few
  thousand training images per layer, STDP rates 10 times the paper's, closed-form IF
  dynamics instead of a stepped `Network`, and some values (DoG size, input threshold,
  inhibition radius) are my choices.
- **intrinsic_timing_ramps** (defaults, 8 sessions of 150 s, seed 0, about 1.5 min on CPU):
  100/100 `a` cells activated and 91/100 `b` cells inhibited; fitted time constants have a
  median of 0.77 s for both groups (many slow cells sit at the fit limit); elapsed-time
  decoding (chance 17 %) 74.6 % for activated cells, 65.7 % for inhibited and 73.7 % for
  both (no gain from combining in this run); the deviant response is 5.24 Hz in the fixed
  context and 5.01 Hz in the jittered one. Deviations: this is an illustration,
  not a reproduction (the paper has no network model); the log-uniform time constants are put
  in by hand, so the spread of tau is an assumption, and the first trial after a block
  switch is not yet at steady state.
- **diehl_cook_mnist**: the quick check (`--train-samples 500 --label-samples 1000
  --test-samples 1000`, CPU) only shows that the code runs; accuracy is low for the first
  few thousand training samples. A full-epoch reproduction
  (100 neurons, 60000 samples) on an RTX 3090 is in progress; its results will be added
  here. Deviations: the 150 ms rest between samples is replaced by an in-place state reset
  plus the relaxation of theta (`--presentation` sets the sample length in ms and `--dt` the
  step; winner diagnostics are printed), and a sample with fewer than 5 excitatory spikes is shown
  again with higher input intensity.
- **object_recognition**, **sequence_prediction**, **two_patterns**: no numbers are stored
  in the docstrings; the scripts print them. `object_recognition` follows the voting idea of
  Lewis et al. 2019 on random objects, not their full model.
