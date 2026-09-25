# Changelog

All notable changes to neuroSush are documented here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

## [0.4.0] - 2026-09-25

### Added
- `GraphStepper` (`neurosush.core.graph`) captures a network's step as a CUDA graph and
  replays it, for networks whose behaviors are all graph-ready. On an RTX 3090 a 784-input,
  400-neuron STDP network runs 5.5 times faster than eager stepping, with bit-for-bit the
  same results.
- A behavior protocol for it: `Behavior.graph_safe`, `graph_ready(host)`, `graph_key(host)`
  and `prepare(host)`, which `Network.step` calls before the schedule for the behaviors that
  override it.
- `benchmarks/scaling.py` (steps per second across devices, threads, batch sizes and CUDA
  graphs), `benchmarks/plot_scaling.py`, and `docs/BENCHMARKS.md` with results and figures
  from an RTX 3090 machine.

### Changed
- Fewer operations per simulation step: dendrites of depth one pass their input through,
  synaptic buffers without delays skip their ring, traces and homeostasis update in place,
  KWTA sorts once, and STDP on a GPU no longer synchronizes with the host. Results are
  unchanged bit for bit.
- `SpatialPooler` learning updates only the rows of learning columns and caches the
  connected synapses (8.7 times faster on the benchmark suite), and `TemporalMemory`'s work
  per step no longer grows with its total number of cells and segments (1.6 times faster).
  Results are unchanged bit for bit.
- `ActivityHomeostasis.rate` is a 0-d float64 tensor on the network's device (its
  `state_dict` still holds a float), and `SpikeInput` reads its next frame in `prepare`.

## [0.3.2] - 2026-09-25

### Fixed
- `SegmentLearning` failed on torch before 2.6 (a boolean mask followed by an index), and
  `SensorColumn.candidates` needed torch 2.2; both now run on torch 2.1, the oldest version
  `pyproject.toml` allows. CI tests that version too.

## [0.3.1] - 2026-09-24

### Added
- Package metadata for PyPI (summary, keywords, project links, classifiers) and
  `CITATION.cff`.

### Changed
- The README introduces the whole project, links absolutely (so it also works on PyPI), shows
  the spiking sequence memory learning, and lists the limitations.

## [0.3.0] - 2026-09-24

### Added
- Spiking temporal memory: `ActiveSegments` (distal dendritic segments with NMDA-like
  plateaus and a coincidence window) and `MinicolumnInhibition`; a layer built from them
  activates exactly the cells `TemporalMemory` activates, checked by
  `tests/validation/test_segment_math.py` and `test_sequence_math.py`.
- `SegmentLearning`: the temporal memory's learning in spike time, and `sequence_memory`,
  which builds a spiking sequence memory layer with derived and checked timing. It learns
  the same curves as `TemporalMemory` (`tests/validation/test_sequence_learning.py`).
- `experiments/sequence_learning.py`: the full learning curves, with parameters and seeds.
- `SpatialPooler.state_dict()`/`load_state_dict()` and the same for `TemporalMemory`
  (including its random generator), so both resume exactly.
- More of `tests/validation`: conv2d, local2d, lateral and pooling currents and the conv and
  local STDP rules against the dense synapse-by-synapse definition; reward-modulated STDP
  against its closed form, and the distal reward problem (Izhikevich 2007).
- `benchmarks/suite.py` and `compare.py`, run weekly by the Benchmarks workflow: throughput
  normalized by a calibration workload, compared with a baseline recorded on a GitHub
  runner (a case that halves fails).

### Fixed
- STDP computed in torch's default dtype instead of the traces' dtype, so a float64
  network learned in float32 precision (and conv and local synapses got float32 updates).

## [0.2.0] - 2026-09-24

### Added
- `neurosush.htm`: SDR operations with exact match probabilities, scalar, random
  distributed and category encoders, `SDRClassifier`, `SpatialPooler`, `TemporalMemory`,
  grid cell modules, `ActiveDendrites`, and voting columns for object recognition.
- `neurosush.predictive_coding.PredictiveCodingNetwork`: hierarchical predictive coding with
  learned weights, priors and variances.
- `examples/sequence_prediction.py` and `examples/object_recognition.py`.
- `Recorder`: records attributes of a network, neuron group or synapse group over time.
- `neurosush.checkpoint`: save and load the complete state of a network and resume it
  exactly, batched or not.
- `Behavior.state_dict()`/`load_state_dict()`, and the same for delay buffers.
- Strict mypy type checking of the package (in `scripts/check.sh` and CI).
- `tests/validation`: the spiking core checked against the closed-form solutions of its
  equations (LIF, exponential and adaptive LIF, traces, STDP, inhibitory STDP, homeostasis,
  dopamine, Poisson encoders and delays).
- Tests marked `gpu` that compare CUDA and CPU runs, pre-commit hooks (also run by the CI
  lint job), and Dependabot updates for pip and GitHub Actions.

### Changed
- A synaptic input without `SpikeGather` is now an error; before, the synapse silently
  delivered no current.
- `SpikeGather` needs an `Axon` only on the source; `syn.post_spike` is gathered when the
  destination has one too (traces and plasticity need it and say so).
- `Axon` checks that the `dst_delay` of incoming synapses fits its history.
- A group shape in a JSON spec that is neither an int nor three ints raises `ValueError`.

## [0.1.0] - 2026-09-24

### Added
- Simulation core: `Network`, `NeuronGroup`, `SynapseGroup`, `Behavior` with an explicit
  execution `Order`, delay buffers, and seeded randomness.
- Neurons: `LIF`, `ELIF`, `AdaptiveELIF` with pure dynamics, `Fire`, `KWTA`, `InherentNoise`,
  `Axon` delays, `DendriteStructure` and `DendriteIntegration` (proximal, distal, apical),
  `ActivityHomeostasis`, `VoltageHomeostasis`, and `SpikeInput`.
- Synapses: `WeightInit` (dense or sparse), `DelayInit`, dense, one-to-one, sparse, conv2d,
  local2d, lateral and average-pooling inputs, `SpikeGather`, `Traces`, `STDP`, `RSTDP`,
  `ISTDP`, `WeightClip`, `WeightNormalization`, `CurrentNormalization`.
- `Payoff` and `Dopamine` reward modulation.
- Poisson and latency encoders, DoG and Gabor kernels, grid masks, `LocationDataset` and
  `spike_frames`.
- Cortical structures: layers with ports, `connect`, `CorticalColumn`, and JSON specs built
  through a class registry.
- `examples/two_patterns.py`, which learns two input patterns with STDP.
- Fast paths: ring-buffer delays validated once, and event-driven in-place STDP for dense
  synapses (5-6x faster on CPU); `benchmarks/dense_stdp.py` measures it.
- Batched simulation: `Network(batch_size=B)` and `spike_frames(..., batch_size=B)` run `B`
  samples side by side with shared weights (about 23,000 sample-steps/s on a laptop GPU for
  the 784 -> 400 benchmark).
- CI (lint, tests on Python 3.10 to 3.13, coverage, build) and a tag-driven release workflow
  with PyPI trusted publishing.
