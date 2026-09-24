# Changelog

All notable changes to neuroSush are documented here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

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
- CI (lint, tests on Python 3.10 to 3.13, coverage, build) and a tag-driven release workflow
  with PyPI trusted publishing.
