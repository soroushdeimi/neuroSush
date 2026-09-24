# neuroSush architecture

neuroSush simulates spiking cortical networks on PyTorch: LIF-family neurons, dendritic
compartments, axonal delays, STDP-family plasticity, dopamine modulation, spike encoders and
cortical structures.

## Principles

1. **Math is pure.** Every equation lives in a small function that takes tensors and returns
   tensors, never mutates its inputs, and has unit tests with hand-computed values.
2. **Behaviors are thin.** A `Behavior` validates its arguments in `__init__`, allocates
   state in `initialize`, and in `forward` calls the pure functions and stores the result.
3. **Explicit order.** Each behavior class declares `order` (see `neurosush.core.order`).
   Initialization and every step run in that order, ties broken by registration order.
   No global name-to-priority table, no `initialize_last` special cases.
4. **Fail early.** Bad arguments raise `ValueError`/`TypeError` in `__init__`; missing
   collaborators (for example an STDP synapse whose source has no `Axon`) raise in
   `initialize` with a message that names the object.
5. **Reproducible.** Randomness uses the network's `torch.Generator` (`Network(seed=...)`).
6. **One dependency.** Only `torch` at runtime.
7. **No eval.** Serialization uses a registry of classes, never `exec`/`eval`.

## Package layout (`src/neurosush`)

| module | contents |
|---|---|
| `core/order.py` | `Order` constants: when each kind of behavior runs |
| `core/behavior.py` | `Behavior` base class |
| `core/buffers.py` | `HistoryBuffer` (past values, read with per-neuron delay), `ArrivalBuffer` (future accumulation for dendritic delays) |
| `core/network.py` | `Network`, `NeuronGroup`, `SynapseGroup`, `Compartment` |
| `neurons/dynamics.py` | pure LIF / ELIF / AdEx equations and threshold crossing |
| `neurons/models.py` | `LIF`, `ELIF`, `AdaptiveELIF`, `Fire` |
| `neurons/competition.py` | `KWTA`, `InherentNoise` |
| `neurons/axon.py` | `Axon` (spike history for delays) |
| `neurons/dendrite.py` | `DendriteStructure`, `DendriteIntegration` |
| `neurons/homeostasis.py` | `ActivityHomeostasis`, `VoltageHomeostasis` |
| `neurons/inputs.py` | `SpikeInput` (drives a group from a stream of spike frames) |
| `synapses/init.py` | `WeightInit`, `DelayInit`, `sparse_random` |
| `synapses/currents.py` | pure current functions + `DenseInput`, `OneToOneInput`, `SparseInput`, `Conv2dInput`, `Local2dInput`, `LateralInput`, `AvgPool2dInput` |
| `synapses/traces.py` | `SpikeGather`, `Traces`, `trace_step` |
| `synapses/bounds.py` | `soft_bound`, `hard_bound`, `no_bound` (directional learning gates) |
| `synapses/plasticity.py` | pure STDP / iSTDP kernels per connectivity + `STDP`, `RSTDP`, `ISTDP` |
| `synapses/constraints.py` | `WeightClip`, `WeightNormalization`, `CurrentNormalization` |
| `modulation.py` | `Payoff`, `Dopamine` |
| `encoding.py` | `rate_poisson`, `interval_poisson`, `intensity_to_latency` |
| `filters.py` | `dog_kernel`, `gabor_kernel` |
| `transforms.py` | `grid_boxes`, `GridErase`, `GridKeep`, `GridCrop`, `split_polarity`, `FilterBank` |
| `data.py` | `LocationDataset`, `spike_frames` |
| `structure/layer.py` | `Layer`, `CorticalLayer` (named groups and ports) |
| `structure/connect.py` | `connect` (one synapse group per source/destination pair) |
| `structure/column.py` | `CorticalColumn` |
| `structure/spec.py` | `ColumnSpec` and friends, `build_column`, `to_json`/`from_json`, `register` |
| `recording.py` | `Recorder` (runs last, at `Order.RECORD`) |
| `checkpoint.py` | `state_dict`, `load_state_dict`, `save`, `load` |
| `htm/sdr.py` | SDR operations, `match_probability`, union capacity |
| `htm/encoders.py` | `ScalarEncoder`, `RandomDistributedScalarEncoder`, `CategoryEncoder` |
| `htm/classifier.py` | `SDRClassifier` (softmax regression) |
| `htm/spatial_pooler.py` | `SpatialPooler` (topology, global or local inhibition, boosting, bumping) |
| `htm/temporal_memory.py` | `TemporalMemory` (segments stored as tensors) |
| `htm/grid_cells.py` | `GridCellModule`, `GridCellModules`, `hexagonal_rate` |
| `htm/active_dendrites.py` | `ActiveDendrites` (an `nn.Module`), `kwta` |
| `htm/objects.py` | `ObjectLibrary`, `SensorColumn`, `ColumnEnsemble`, `vote` |
| `predictive_coding.py` | `PredictiveCodingNetwork` |

## Simulation model

A step of `Network.step()` runs every enabled behavior's `forward` once, sorted by
`(order, registration index)`. The canonical order within a step:

| order | behaviors | reads | writes |
|---|---|---|---|
| 100 | `Payoff` | user state | `net.payoff` |
| 120 | `Dopamine` | `net.payoff` | `net.dopamine` |
| 180 | synaptic inputs | `syn.pre_spike`, `syn.weights` | `syn.I` |
| 200 | `CurrentNormalization` | `syn.weights` | `syn.I` |
| 220 | `DendriteStructure` | afferent `syn.I`, `syn.dst_delay` | `ng.I_proximal/distal/apical` |
| 240 | `DendriteIntegration` | compartment currents | `ng.I` |
| 260 | `LIF`/`ELIF`/`AdaptiveELIF` | `ng.I` | `ng.v` |
| 280 | `InherentNoise` | | `ng.v` |
| 300 | `KWTA` | `ng.v` | `ng.v` |
| 310 | `VoltageHomeostasis` | `ng.v` | `ng.v` |
| 340 | `Fire`, `SpikeInput` | `ng.v` | `ng.spikes`, `ng.v` |
| 350 | `ActivityHomeostasis` | `ng.spikes` | `ng.threshold` |
| 380 | `Axon` | `ng.spikes` | `ng.spike_history` |
| 420 | `SpikeGather` | `spike_history`, delays | `syn.pre_spike`, `syn.post_spike` |
| 460 | `Traces` | gathered spikes | `syn.pre_trace`, `syn.post_trace` |
| 500 | `STDP`/`RSTDP`/`ISTDP` | spikes, traces, `net.dopamine` | `syn.weights` |
| 520 | `WeightNormalization` | | `syn.weights` |
| 540 | `WeightClip` | | `syn.weights` |

Synaptic input at step t uses spikes gathered at step t-1 (one step of transmission).

## Conventions

- Dense weights are `(n_src, n_dst)`. Current into `dst` is `pre_spike.float() @ W`.
- A `NeuronGroup` has `shape = (depth, height, width)`; an `int` size `n` means `(1, 1, n)`.
- Every tensor lives on `net.device` with float dtype `net.dtype`; spikes are `torch.bool`.
- Per-sample state (`v`, `spikes`, `I`, traces, delay buffers) has shape `group.state_shape`:
  `(size,)`, or `(batch_size, size)` with `Network(batch_size=...)`. Parameters (`weights`,
  `threshold`) are shared by the batch; learning rules and activity homeostasis use the batch
  mean, so rates do not depend on the batch size.

## Performance

- Delay buffers are rings: a step moves a head index instead of copying `depth` rows, and a
  delay tensor is validated once (remembered by identity and version), which removes a
  host-device synchronization from every read.
- Dense STDP on a single sample is event-driven and in place: potentiation touches only the
  columns of spiking postsynaptic neurons, depression only the rows of spiking presynaptic
  neurons.
- On a GPU one sample is bound by kernel-launch latency, so batches are the main lever:
  `benchmarks/dense_stdp.py` (784 -> 400, dense STDP) runs about 90 steps/s at batch 1 and at
  batch 256 alike, i.e. about 23,000 sample-steps/s on a laptop RTX 3060.
- Time constants and `dt` share one unit (ms by convention). Every decay uses `dt / tau`.
- Inhibitory source groups (`NeuronGroup(..., inhibitory=True)`) make currents negative.

## Design decisions

- Time constants share the unit of `dt`; decays use `dt / tau`; STDP updates are per spike
  pair; reward modulation is a rate, `dw/dt = dopamine * eligibility`.
- Learning bounds only gate potentiation (by `w_max`) and depression (by `w_min`); clipping
  is the separate `WeightClip`.
- Weights are magnitudes; the source group's `inhibitory` flag gives the current its sign.
- Sparse weights are a values vector with `src_idx`/`dst_idx`, not torch sparse tensors, so
  currents are one `index_add` and learning updates the values directly.
- Specs describe construction only; learned tensors are saved separately.

## Checkpoints

State lives in two places: on hosts (tensors and buffers that behaviors put on the network,
neuron groups and synapse groups) and, rarely, on a behavior itself. A checkpoint walks the
hosts in a fixed order and saves every tensor, every delay buffer (`state_dict()` of its
slots and head) and the network's scalars; behaviors that keep state of their own
(`ActivityHomeostasis`: its activity counter and decayed rate) expose it through
`Behavior.state_dict()`, keyed by host, class and position. Anything new that a behavior
stores on its host is therefore saved without extra code; state kept on a behavior needs a
`state_dict`/`load_state_dict` pair. The host classes declare the state attributes that the
library's behaviors set (as annotations without values, so `hasattr` still tells whether a
behavior is present), which is what lets strict mypy check behavior code.

## Thousand Brains models

The `htm` package and `predictive_coding` are plain tensor algorithms, not behaviors: they
step in discrete time on binary codes (or rates), so the spiking scheduler would add
nothing. They share the conventions above: seeded generators, validated arguments with the
offending value in the message, and a batch dimension where the algorithm allows one.

- **Spatial pooler.** Overlaps for a batch are one matrix product; local inhibition
  compares every column with its neighborhood through `unfold`, so no Python loop runs over
  columns. Learning goes sample by sample because every sample changes the duty cycles.
- **Temporal memory.** Segments live in fixed-width tensors (`segment_cell`, `presynaptic`,
  `permanence`) that double when full. Segment activity for all segments is one gather
  and sum; only the few segments that learn in a step are touched one by one.
- **Grid cells.** A module is its lattice basis `A`: phases are `A^-1 x mod 1`, so path
  integration is exact and independent of the path taken.
- **Voting columns.** A column's hypotheses are a boolean `(object, y, x)` tensor; sensing
  is an `&` with the feature map and moving is a `roll`.
- **Predictive coding.** All updates are the closed-form gradients of the free energy;
  the tests compare them with autograd.
