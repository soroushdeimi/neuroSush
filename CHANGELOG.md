# Changelog

All notable changes to neuroSush are documented here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

### Added
- `neurosush.htm`: SDR operations with exact match probabilities, scalar, random
  distributed and category encoders, `SDRClassifier`, `SpatialPooler`, `TemporalMemory`,
  grid cell modules, `ActiveDendrites`, and voting columns for object recognition.
- `neurosush.predictive_coding.PredictiveCodingNetwork`: hierarchical predictive coding with
  learned weights, priors and variances.
- `examples/sequence_prediction.py` and `examples/object_recognition.py`.

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
