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

Run from the repository root with the package installed (`pip install -e .`):

```bash
python examples/two_patterns.py
python examples/sequence_prediction.py
python examples/object_recognition.py
python examples/diehl_cook_mnist.py --data path/to/MNIST/raw
python examples/stdp_frequency.py
python examples/balanced_network.py
python examples/predictive_coding_mnist.py --data path/to/MNIST/raw
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
- **diehl_cook_mnist**: the quick check (`--train-samples 500 --label-samples 1000
  --test-samples 1000`, CPU) only shows that the code runs; accuracy is low for the first
  few thousand training samples. A full-epoch reproduction
  (100 neurons, 60000 samples) on an RTX 3090 is in progress; its results will be added
  here. Deviations: the 150 ms rest between samples is replaced by an in-place state reset
  plus the relaxation of theta, and a sample with fewer than 5 excitatory spikes is shown
  again with higher input intensity.
- **object_recognition**, **sequence_prediction**, **two_patterns**: no numbers are stored
  in the docstrings; the scripts print them. `object_recognition` follows the voting idea of
  Lewis et al. 2019 on random objects, not their full model.
