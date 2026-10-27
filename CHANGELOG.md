# Changelog

All notable changes to neuroSush are documented here, following the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

## [0.6.0] - 2026-10-28

### Fixed
- Atlas plates `N03`, `N07`, `N01`, `N08` and `Y02`: remaining label overlaps fixed, N07 shows a faint idle crowd when both conductances are zero, and the phone layouts keep every control reachable.
- Atlas plate `N02`: missing spaces after commas in the predict-then-reveal text.
- Clearer input errors: `render_plate` takes an optional trace for live plates and explains what a replay needs, `TraceRecorder.trace()` before a run and duplicate recorded attributes raise explicit errors, `Network.run` rejects non-integer steps, `WeightInit` points to `weights=` for tensors, and `PoissonInput` rejects `rates * dt > 1` instead of firing every step.
- Atlas plates `N01`, `N02`, `N05`, `N08`, `M01` and `Y02`: label overlaps and clipping fixed (channel and thermometer labels, caliper marker, legend and ramp readout, tilt control and phone layout, margin styling, weight axis labels).
- Atlas replay decodes recorded step-index lists, preserves indices and uses sample positions
  for playback and spike events; the demo reads neuron zero with the recorded neuron stride.
- Atlas shell displays `meta.library` without a fallback version; HTML export rejects
  unsupported themes, and recording validates sample shape and dtype before advancing.
- Schedule inspection reports Izhikevich voltage/recovery dependencies and the attributes
  selected by each TraceRecorder. README includes an executable visualization example.

### Fixed
- Atlas plate `N05` rate functions now use the library's Taylor expansion near the removable singularities instead of a constant.

### Added
- Golden vectors now cover the Hodgkin-Huxley gate rates, steady states, time constants and `lif_rate`, and `tests/viz/js/golden.test.mjs` checks the plate cores against them with the same per-case tolerances.
- Atlas plate `N03`: AdEx adaptation in the phase plane, a live figure with nullclines, spike-triggered adaptation and an inter-spike-interval ladder.
- Atlas plate `N07`: conductance synapse tug of war, a live rope-and-teams figure with conductance-dependent forces and the effective time constant.
- Atlas plate `Y02`: triplet STDP and frequency dependence, a live figure with fast and slow traces, pair versus triplet weight curves and the frequency sweep.
- Atlas plate `N04`: Izhikevich zoo of firing patterns, eight live cells with a computed pattern fingerprint, draggable parameter maps and the theory ghost.
- Graph readiness inventory for every behavior, with CPU AST guards and seeded 200-step delay-buffer equivalence tests.
- Atlas plate `N02`: ELIF potential-well plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `N08`: f-I curve and bifurcation plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- README example: a standalone float-only LIF simulation reports its measured five-second firing rate next to `dynamics.lif_rate`.
- `dynamics.lif_rate`: the closed-form LIF firing rate for a constant current, now the single source used by the validation test.
- `dynamics.fire` accepts plain floats (returns a `(bool, float)` pair); `Recorder.get(name, last=N)` returns only the last N samples; the README states the units convention (mV, ms, MOhm, nA).
- Atlas plate `M01`: Euler steps, convergence and instability plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas shell: live plates (`source="live"`) hide the timeline, an empty margin column collapses, the header carries the neuroSush mark, and the layout scales at phone width.
- Atlas plate `H01`: sparse distributed representation plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `M03`: Nernst equation plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `M02`: Poisson spike rain plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `Y01`: STDP traces and spike-pair plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `N01`: leaky integrate-and-fire bucket plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- Atlas plate `N05`: Hodgkin-Huxley membrane plate, a live interactive figure with chapters, predict-then-reveal and the theory ghost.
- `neurosush.viz.export`: `render_plate`, `write_html` and `available_plates` produce a single
  self-contained HTML plate (inline CSS, JavaScript and packed trace data, light and dark
  themes) from the Atlas web shell in `neurosush/viz/web`; JavaScript logic is tested with
  `node --test`.
- `neurosush.viz.trace`: compact JSON-serializable `Trace` with base64 little-endian array
  packing and `TraceRecorder` behavior preallocating device-side buffers.
- `neurosush.viz.schedule`: `schedule_table` and `schedule_lanes` for inspecting simulation
  step execution order and `Behavior.reads`/`Behavior.writes` dataflow metadata.
- `neurosush/viz/i18n/glossary_en.json`: an English glossary of 78 core neuroscience terms
  (`term`, `short` and `long`, the plate ids it appears on and the library `symbol`) for the
  Atlas plate hover callouts, validated in `tests/viz/test_glossary.py`.
- `neurosush.viz.catalog`: the machine-readable plate catalog (`PlateSpec`, `PLATES` for all
  43 planned plates with their English titles, waves and flagship flags, `PLANNED`)
  and the coverage introspector (`public_models`, `EXEMPT`, `uncovered`) that a test keeps
  honest: every public model must be covered by a plate or explicitly exempted.
- `benchmarks/profile_step.py`: per-step wall time, operator, CUDA-kernel and launch-bound
  profiling for the dense STDP and Diehl and Cook workloads across eager, graph and compiled
  steppers.
- `lif_terms`, `elif_terms`, `adex_terms`, `izhikevich_terms` (`neurosush.neurons.dynamics`):
  pure functions decomposing one membrane update step into its named terms (leak, input,
  spike initiation, adaptation, quadratic, linear, recovery), verified to sum to the step
  update; deterministic golden vectors in `tests/golden/equations.json`.
- `HodgkinHuxley` (`neurosush.neurons.hodgkin_huxley`, Hodgkin and Huxley 1952):
  squid-axon neuron model with visible ion-channel state (gates `m`, `h`, `n` and
  `i_na`, `i_k`, `i_leak` currents), forward Euler on `v`, exact analytic gating steps, and
  upward-crossing spike detection without reset. Runs at `Order.NEURON_DYNAMICS`.
- `Refractory(period, *, block=None)`: with `block` a neuron also cannot fire for
  `ceil(block / dt)` steps after a spike while its voltage integrates freely once the `period`
  hold ends (the timer of the original Diehl and Cook code); `None` keeps the plain refractory
  period. `dynamics.fire` and `izhikevich_reset` take an optional `blocked` mask.
- `examples/diehl_cook_mnist.py --exc-block MS` (50 reproduces the original code's firing
  timer; the paper's text does not mention it), recorded in the JSON as `exc_block`.

## [0.5.0] - 2026-10-03

### Added
- `Izhikevich` (`neurosush.neurons.models`, Izhikevich 2003): two-variable neuron with
  per-neuron `a`, `b`, `c`, `d` and `substeps` Euler sub-steps of `v`; the spike test and reset
  are in `fire`, so it works with `Fire` and with dendritic input. Runs at
  `Order.NEURON_DYNAMICS`.
- `SpikeTriggeredCurrent` (`neurosush.neurons.adaptation`, new `Order.ADAPTATION = 250`): a
  current that jumps by `amplitude` at each spike and decays with `tau`, added to `group.I`
  before the neuron model; a negative amplitude gives spike-frequency adaptation for a `LIF`.
- `PoissonDrive` (`neurosush.neurons.inputs`, `Order.NOISE`): background drive of `count`
  Poisson sources of rate `rate`, each spike adding `jump` to the membrane (delta synapses,
  Brunel 2000). `CorrelatedPoissonInput` (`Order.FIRE`): spike trains with pairwise
  correlation `c` from a mother train (multiple interaction process, Kuhn, Aertsen and Rotter
  2003). Both implement `draw` for the compiled stepper.
- `Traces(interaction="nearest")` and `STDP(pairing="nearest")` (also `RSTDP`): nearest-spike
  and nearest-pair plasticity (Masquelier et al. 2008) for dense and one-to-one synapses.
- `WeightInit(sparse=True, in_degree=k)` and `synapses.init.fixed_in_degree`: sparse
  connections with exactly `k` distinct sources per destination.
- `synapses.currents.delta_coef` (input coefficient that makes a spike jump a `LIF` membrane by
  a given voltage) and `MaxPool2dInput` (a destination spikes if any source in its window did).
- `neurosush.neurons.params`: helpers that validate a parameter given as a number or one
  value per neuron.
- Examples `competitive_stdp.py` (Song, Miller and Abbott 2000), `spike_pattern_detection.py`
  (Masquelier et al. 2008), `brunel_network.py` (Brunel 2000), `conv_stdp_mnist.py`
  (Kheradpisheh et al. 2018) and `intrinsic_timing_ramps.py` (illustrating Huang et al. 2026),
  with rows in `examples/README.md`.
- `diehl_cook_mnist.py` gets `--dt`, `--presentation` (ms) and winner diagnostics;
  `spike_pattern_detection.py` and `brunel_network.py` use the library nearest-spike traces
  and pairing, fixed in-degree, `delta_coef` and `PoissonDrive`.
- `CompiledStepper` (`neurosush.core.compiled`): opt-in, same `step()` and `run(steps)` as
  `GraphStepper`; compiles the whole step with `torch.compile` and, on CUDA, replays it as a CUDA
  graph per key. Needs torch >= 2.3 (`RuntimeError` on older versions). CPU and CUDA, shared batch and `independent=True`. Equal to eager to a
  tolerance, not bit for bit (`GraphStepper` stays exact). New hooks `Behavior.compile_ready`
  and `Behavior.draw` (random numbers drawn eagerly in the eager order; `PoissonInput` and
  `InherentNoise` use them); a depth-1 `HistoryBuffer` read no longer inspects its delays.
  `benchmarks/compiled.py` compares eager, graph and compiled; `diehl_cook_mnist.py` gets
  `--stepper {graph,compiled}`.
- `Network(batch_size=B, independent=True)`: the `B` batch members are `B` independent
  networks with their own weights, thresholds, `theta`, homeostasis and learning, with no
  averaging over the batch (dense `(B, n_src, n_dst)` and one-to-one `(B, n)` weights,
  `group.vector()` returns `(B, size)`). Supports the behaviors of the Diehl and Cook network;
  `Network.initialize` raises `NotImplementedError` for the others (a behavior opts in with
  `independent_ok = True`). `diehl_cook_mnist.py --members B` trains `B` networks at once and
  reports per-member and mean accuracy; `benchmarks/independent.py` measures throughput.
- Examples `stdp_frequency.py` (pair versus triplet STDP across pairing frequencies),
  `balanced_network.py` (inhibitory STDP, reduced Vogels et al. 2011),
  `predictive_coding_mnist.py` (predictive coding on MNIST) and `diehl_cook_mnist.py`
  (Diehl and Cook 2015), with a table in `examples/README.md`.
- `TripletSTDP` (`neurosush.synapses.triplet`): the triplet rule of Pfister and Gerstner
  (2006) with all-to-all or nearest-spike interaction, soft or hard bounds, batch-mean
  updates and checkpointed traces; the Diehl and Cook (2015) rule is the special case
  `a2_plus=0, a3_minus=0` with nearest interaction.
- `PoissonInput` (`neurosush.neurons.inputs`): every step each neuron spikes with probability
  `rates * dt`, drawn on the network's device. The rates live on the group as `group.rates`
  and can be changed in place, also while a CUDA graph replays.
- `Refractory(period)` (`neurosush.neurons.models`, at the new `Order.REFRACTORY = 330`):
  holds the membrane at `v_reset` for `ceil(period / dt)` steps after a spike.
- `AdaptiveThreshold(increment, tau=None)` (`neurosush.neurons.homeostasis`): raises a
  neuron's threshold by `increment` per spike and relaxes it with `tau`; in a batch the
  threshold rises by the batch mean of the spikes. Set `enabled = False` to freeze it.
- `ConductanceIntegration` and `conductance_step` (`neurosush.neurons.dendrite`):
  conductance-based synapses where excitatory and inhibitory input adds to `g_exc` and
  `g_inh`, which decay with their own time constants and pull the voltage towards the
  reversal potentials. The step is the exact solution of the equation over one step, so it
  stays stable for large conductances. All four behaviors are graph-ready.
- `Network.reset_state()` and `Behavior.reset_state(host)`: clear the per-sample dynamic
  state (voltages, currents, conductances, traces, spike histories, delay buffers,
  countdowns, eligibility, plateaus) in place between samples, keeping weights, thresholds,
  `theta` and homeostasis counters. Every behavior that keeps per-sample state implements
  it; in-place clearing keeps CUDA graphs valid.
- `SpikeCounter` (`neurosush.recording`): counts each neuron's spikes in
  `group.spike_count`; graph-safe, zeroed by `reset_state`.
- `neurosush.readout`: `assign_labels`, `classify` and `accuracy` to label an unsupervised
  layer from its spike counts, as in Diehl and Cook (2015).
- `neurosush.data.load_mnist` and `read_idx`: read the MNIST IDX files (plain or `.gz`) with
  pure torch, optionally downloading missing files.

### Changed
- `AdaptiveELIF` (`alpha`, `beta`, `tau_w`) and `ConductanceIntegration` (`tau_exc`,
  `tau_inh`) accept a tensor of shape `(size,)` for one value per neuron.
- `filters.dog_kernel` defaults to `zero_mean=True`; callers that relied on the old output pass
  `zero_mean=False`.

## [0.4.0] - 2026-09-25

### Added
- `GraphStepper` (`neurosush.core.graph`) captures a network's step as a CUDA graph and
  replays it, for networks whose behaviors are all graph-ready. On an RTX 3090 a 784-input,
  400-neuron STDP network runs 5.5 times faster than eager stepping, with bit-for-bit the
  same results. (Measured on one Windows machine; on an RTX 3060 Laptop GPU on Linux the
  gain was 2.2 times. It depends on the platform's kernel-launch overhead.)
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
  (Clarification: the curves are identical in repetitions 0-3 and 8-11 of the 12 in
  `experiments/sequence_learning.py` and differ in 4-7; the end states agree in burst counts,
  not in synapses.)
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
  the 784 -> 400 benchmark, measured on a throttled CPU; see docs/BENCHMARKS.md for current
  numbers).
- CI (lint, tests on Python 3.10 to 3.13, coverage, build) and a tag-driven release workflow
  with PyPI trusted publishing.
