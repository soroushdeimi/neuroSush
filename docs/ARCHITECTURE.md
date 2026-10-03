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
| `core/compiled.py` | `CompiledStepper`: the step compiled with `torch.compile`, optionally replayed as a CUDA graph |
| `core/buffers.py` | `HistoryBuffer` (past values, read with per-neuron delay), `ArrivalBuffer` (future accumulation for dendritic delays) |
| `core/network.py` | `Network`, `NeuronGroup`, `SynapseGroup`, `Compartment` |
| `neurons/dynamics.py` | pure LIF / ELIF / AdEx equations and threshold crossing |
| `neurons/models.py` | `LIF`, `ELIF`, `AdaptiveELIF`, `Izhikevich` (Izhikevich 2003), `Refractory`, `Fire` |
| `neurons/adaptation.py` | `SpikeTriggeredCurrent` (after-spike current that decays with `tau`) |
| `neurons/params.py` | `positive`, `at_least`, `per_neuron`, `state_like`: validate a parameter given as a number or per neuron |
| `neurons/competition.py` | `KWTA`, `MinicolumnInhibition`, `InherentNoise` |
| `neurons/axon.py` | `Axon` (spike history for delays) |
| `neurons/dendrite.py` | `DendriteStructure`, `DendriteIntegration`, `ConductanceIntegration`, `conductance_step` |
| `neurons/homeostasis.py` | `ActivityHomeostasis`, `VoltageHomeostasis`, `AdaptiveThreshold` |
| `neurons/inputs.py` | `SpikeInput` (drives a group from a stream of spike frames), `PoissonInput` (random spikes at per-neuron rates), `PoissonDrive` (Poisson count of delta synapses added to the membrane, Brunel 2000), `CorrelatedPoissonInput` (trains with pairwise correlation `c`, multiple interaction process of Kuhn, Aertsen and Rotter 2003) |
| `synapses/init.py` | `WeightInit` (`sparse=True`, `in_degree=k` for a fixed in-degree), `DelayInit`, `sparse_random`, `fixed_in_degree` |
| `synapses/currents.py` | pure current functions + `DenseInput`, `OneToOneInput`, `SparseInput`, `Conv2dInput`, `Local2dInput`, `LateralInput`, `AvgPool2dInput`, `MaxPool2dInput`, `delta_coef` (coefficient for a delta-current synapse) |
| `synapses/traces.py` | `SpikeGather`, `Traces` (`interaction="all"` or `"nearest"`), `trace_step` |
| `synapses/segments.py` | `ActiveSegments`, `segment_counts`, `plateau_step` |
| `synapses/segment_learning.py` | `SegmentLearning` |
| `synapses/bounds.py` | `soft_bound`, `hard_bound`, `no_bound` (directional learning gates) |
| `synapses/plasticity.py` | pure STDP / iSTDP kernels per connectivity + `STDP`, `RSTDP`, `ISTDP` (`pairing="all"` or `"nearest"`) |
| `synapses/triplet.py` | `TripletSTDP` (Pfister and Gerstner 2006; the Diehl and Cook 2015 rule is a special case) |
| `synapses/constraints.py` | `WeightClip`, `WeightNormalization`, `CurrentNormalization` |
| `modulation.py` | `Payoff`, `Dopamine` |
| `encoding.py` | `rate_poisson`, `interval_poisson`, `intensity_to_latency` |
| `filters.py` | `dog_kernel` (zero mean by default), `gabor_kernel` |
| `transforms.py` | `grid_boxes`, `GridErase`, `GridKeep`, `GridCrop`, `split_polarity`, `FilterBank` |
| `data.py` | `LocationDataset`, `spike_frames`, `read_idx`, `load_mnist` |
| `structure/layer.py` | `Layer`, `CorticalLayer` (named groups and ports) |
| `structure/connect.py` | `connect` (one synapse group per source/destination pair) |
| `structure/column.py` | `CorticalColumn` |
| `structure/sequence.py` | `sequence_memory`, `sequence_timing`, `Neuron` |
| `structure/spec.py` | `ColumnSpec` and friends, `build_column`, `to_json`/`from_json`, `register` |
| `recording.py` | `Recorder` (runs last, at `Order.RECORD`), `SpikeCounter` (spike counts per neuron, graph-safe) |
| `readout.py` | `assign_labels`, `classify`, `accuracy`: labels for an unsupervised layer |
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
| 240 | `DendriteIntegration`, `ConductanceIntegration` | compartment currents; `syn.I`, `ng.v` | `ng.I` (`ng.g_exc`, `ng.g_inh`) |
| 250 | `SpikeTriggeredCurrent` | `ng.spikes` | `ng.I` (`ng.I_adapt`) |
| 260 | `LIF`/`ELIF`/`AdaptiveELIF`/`Izhikevich` | `ng.I` | `ng.v` (`ng.u` for `Izhikevich`) |
| 280 | `InherentNoise`, `PoissonDrive` | | `ng.v` |
| 300 | `KWTA` | `ng.v` | `ng.v` |
| 310 | `VoltageHomeostasis` | `ng.v` | `ng.v` |
| 330 | `Refractory` | `ng.spikes`, `ng.v` | `ng.v`, `ng.refractory` |
| 340 | `Fire`, `SpikeInput`, `PoissonInput`, `CorrelatedPoissonInput` | `ng.v`, `ng.rates` | `ng.spikes`, `ng.v` |
| 350 | `ActivityHomeostasis`, `AdaptiveThreshold` | `ng.spikes` | `ng.threshold` (`ng.theta`) |
| 380 | `Axon` | `ng.spikes` | `ng.spike_history` |
| 420 | `SpikeGather` | `spike_history`, delays | `syn.pre_spike`, `syn.post_spike` |
| 460 | `Traces` | gathered spikes | `syn.pre_trace`, `syn.post_trace` |
| 500 | `STDP`/`RSTDP`/`ISTDP`/`TripletSTDP` | spikes, traces, `net.dopamine` | `syn.weights` |
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
- **Independent batches.** `Network(batch_size=B, independent=True)` makes the `B` batch
  members `B` separate networks that run in one set of kernels. Per-member tensors get a
  leading `B`: neuron parameters and state `(B, size)` (`group.vector()` returns
  `(B, size)` instead of `(size,)`), dense weights `(B, n_src, n_dst)`, one-to-one weights
  `(B, n)`. Dense currents are `torch.bmm`; STDP, `TripletSTDP`, `ISTDP`, `WeightClip`,
  `WeightNormalization`, `CurrentNormalization`, `AdaptiveThreshold`, `ActivityHomeostasis`
  and `KWTA` act on each member with no mean over the batch, so member `b` evolves exactly as
  an unbatched network with the same weights and inputs (checked to 1e-12 in float64 by
  `tests/core/test_independent.py`). Initial weights, Poisson spikes and noise are drawn from
  the one network generator, so members differ; `WeightInit(weights=...)` accepts `(B, ...)`
  weights and `LIF(threshold=...)` a `(B, size)` threshold for per-member values. Scalar
  hyperparameters (time constants, learning rates, `dt`), delays and the
  `ActivityHomeostasis` rate schedule are shared by all members. A behavior opts in with
  `independent_ok = True`; `Network.initialize` raises `NotImplementedError` naming every other
  behavior, so sparse, conv2d, local2d, lateral and pooling connectivity, `RSTDP`,
  `ActiveSegments`, `SegmentLearning`, `VoltageHomeostasis`, `MinicolumnInhibition`, modulation
  and `Recorder` are refused rather than computed wrongly. `GraphStepper`, checkpoints and
  `reset_state` work unchanged (a checkpoint needs the same `B`). Dense learning touches the
  whole weight tensor each step, so cost grows with `B` once the GPU is memory-bound:
  `benchmarks/independent.py`.

## Performance

- Delay buffers are rings: a step moves a head index instead of copying `depth` rows, and a
  delay tensor is validated once (remembered by identity and version), which removes a
  host-device synchronization from every read.
- Nearest-spike plasticity (`Traces(interaction="nearest")` resets a trace to `scale` at a
  spike instead of adding to it; `STDP(pairing="nearest")` keeps the steps since each neuron's last
  spike, so a presynaptic spike depresses only if the neuron fired since that afferent's
  previous spike)
  works with dense and one-to-one synapses and also applies to `RSTDP`. `TripletSTDP` has its
  own `interaction`.
- Dense STDP on a single sample is event-driven and in place: potentiation touches only the
  columns of spiking postsynaptic neurons, depression only the rows of spiking presynaptic
  neurons.
- On a GPU one sample is bound by kernel-launch latency, so batches are the main lever:
  `benchmarks/dense_stdp.py` (784 -> 400, dense STDP) measured on 2026-10-02 on an RTX 3060
  Laptop GPU (Linux, torch 2.14) at 1,746 steps/s eager and unbatched, 1,310 steps/s
  (335,301 sample-steps/s) at batch 256 and 3,887 steps/s with a CUDA graph. See
  [BENCHMARKS.md](BENCHMARKS.md).
- Independent networks in one batch (`independent=True`) fill the GPU the same way:
  `benchmarks/independent.py` (Diehl and Cook, 100 neurons, CUDA graph, RTX 3060 Laptop GPU)
  measured 6,029 steps/s for one network and 27,720 member-steps/s with 16 members.
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

## Resetting between samples

`Network.reset_state()` calls `Behavior.reset_state(host)` of every behavior (enabled or not,
in schedule order). A behavior clears the per-sample dynamic state it keeps on its host in
place (`fill_`, `zero_`, buffer `reset()`): voltages go back to `v_init` or `v_rest`, and
currents, conductances, spike histories, traces, eligibility, dendritic delay buffers,
refractory and plateau countdowns and the recent spike times of `SegmentLearning` are
cleared; `AdaptiveELIF` returns `omega` to `omega_init` and `Payoff`/`Dopamine` to their
initial values. Learned and parameter state is kept: weights, permanences, thresholds,
`theta`, the `ActivityHomeostasis` counter and `VoltageHomeostasis` exhaustion (both adapt
over many samples). `SpikeInput` keeps its place in the frame stream and `PoissonInput` its
rates; a `Recorder` keeps its recordings. Everything is in place, so a captured CUDA graph
stays valid. `HistoryBuffer.reset()` also sets a Python head index, which is harmless for
the depth-1 buffers a graph can capture but valid for deeper ones only between eager steps.

## CUDA graphs

`neurosush.core.graph.GraphStepper` replaces `Network.step` with a captured-and-replayed
version of the same step, for behaviors below `Order.RECORD` (`net.schedule` split into
`captured` and `after`; `after` runs eagerly, since it only reads state). Three things make
this possible:

- **Fixed addresses.** A behavior assigns a new tensor to a host attribute every step
  (`group.v = ...`); a captured graph instead has to keep reading and writing the same
  memory. Before capturing, the stepper remembers the identity of every tensor attribute of
  the network, its neuron and synapse groups and the captured behaviors. It runs the step's
  `forward` calls inside `torch.cuda.graph(graph)`, and for every attribute whose identity
  changed, copies the new tensor into the original one (a captured `copy_`) before restoring
  the attribute to that original tensor. A later replay therefore recomputes into the same
  "new" address every time and copies into the same "old" address every time; the Python
  reassignment itself happens only once, during capture.
- **Keys.** A step's Python-side decisions (a homeostasis window ending, a behavior's
  `enabled` flag) are frozen into whichever graph is captured for them. The stepper computes
  a key, `tuple((behavior.enabled, behavior.graph_key(host)) for host, behavior in captured)`,
  and captures a new graph the first time a key is seen, in its own memory pool (the
  default for `torch.cuda.graph`), since graphs for different keys may then replay in either
  order. `prepare(host)` runs before every step, eager or replayed, for behaviors that
  override it, so host-side work that must happen every step (`SpikeInput` staging the next
  frame into a fixed tensor) still does.
- **Warm-up.** The first few steps (`warmup`, default 3) run the captured behaviors' `forward`
  eagerly on a side stream (`s.wait_stream(current)` before, `current.wait_stream(s)` after),
  as the CUDA graphs documentation recommends, so that one-time library initialization
  (cuDNN, cuBLAS workspaces) happens outside any capture.

Randomness draws from `net.generator`, a CUDA generator; when
`torch.cuda.CUDAGraph.register_generator_state` exists, the stepper registers it with each
graph before capturing, which is what makes `InherentNoise` graph-ready (its own
`graph_ready` checks for a CUDA generator and that method).

### Compiled stepper

`neurosush.core.compiled.CompiledStepper` takes the same `captured`/`after` split but traces
all captured `forward` calls into one function with `torch.compile(fullgraph=True)`, so
Inductor fuses the element-wise work (154 kernels per Diehl and Cook step become about 12 to
14). It is opt-in and tolerance-equivalent to eager; `GraphStepper` and `Network.step` stay
the bit-exact references. Four pieces differ from the graph stepper:

- **Write-back inside the function.** The compiled function reads every tensor attribute the
  snapshot lists, runs the forwards, and for each reassigned attribute does `old.copy_(new)`
  and `setattr(obj, name, old)`, so state keeps fixed addresses. Without the in-place
  write-back a gathered spike tensor (`SpikeGather.read`) could alias the mutated history
  storage, so one population would see another's spikes a step early; the tests compare
  spikes step by step.
- **Pre-drawn randomness.** Dynamo cannot trace draws from a custom `torch.Generator`.
  `Behavior.draw(host)` returns the named tensors a behavior's next `forward` needs
  (`PoissonInput`, `InherentNoise`, `PoissonDrive`, `CorrelatedPoissonInput`); the stepper calls it eagerly, in schedule order (the
  order eager stepping draws in), and sets `behavior.drawn` around the compiled call.
  `forward` uses `self.drawn.get(name)` and otherwise draws from the group, so eager and
  graph stepping are unchanged. Under a CUDA graph the draws are captured with it.
- **Readiness.** `Behavior.compile_ready` defaults to `graph_ready`; `InherentNoise` (draws
  come from `draw`) and `STDP` (ready on the CPU unless its event-driven `nonzero()` path
  applies) override it. For depth-1 history buffers `_Buffer._no_delay` is decided by the
  depth alone, with no data-dependent branch, which is what lets `SpikeGather` trace.
- **Keys and warm-up.** The first step of a key runs through the compiled function on a side
  stream (this is where compilation happens, never inside a capture); the next step of that
  key is captured and replayed. Python decisions that vary (a homeostasis window ending, an
  `enabled` flag) make Dynamo guard and compile a variant per value, matching the per-key
  graphs. The stepper raises Dynamo's recompile limit to 64 so networks of several shapes
  can coexist in one process.

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

## Spiking temporal memory

A minicolumn layer computes the temporal memory with spikes. Every element is shown for a
window of `W` steps, every `P` steps.

- **Prediction** is a plateau: `ActiveSegments` (on a distal synapse group) counts the
  connected synapses whose presynaptic cell fired within the coincidence window, and a
  segment over threshold holds `amplitude` on the distal compartment for the plateau.
  `DendriteIntegration(distal_gain=g)` turns it into priming towards
  `v_rest + g (theta - v_rest)`, below threshold.
- **Activation** is a race: under the same proximal drive a primed cell reaches threshold
  after `a` steps and an unprimed one after `b > a`; `MinicolumnInhibition` holds the rest of
  a minicolumn at `v_reset` once some of its cells fire, for the rest of the window.
- **Equivalence** with `TemporalMemory` needs `a < b <= W`, a plateau that covers the next
  element's crossing (`b + plateau >= P + a`) and ends before the element after it
  (`a + plateau <= 2 P`), and a coincidence window of at least `b - a + 1` steps when an
  element mixes predicted and bursting minicolumns. `tests/validation/test_sequence_math.py`
  derives `a` and `b` from the exact Euler map and checks all of it.
- **Learning** (`SegmentLearning`) applies the temporal memory's rules when cells fire. The
  previous element is the set of cells that fired (or won) within the learning context,
  a range of ages around one period; the current element's own spikes are younger. Spikes,
  wins and plateau starts keep their last two times, so a cell whose minicolumn is active
  in two consecutive elements still counts as context. `sequence_memory` derives the
  context, plateau and coincidence window from the neuron (`sequence_timing`).
