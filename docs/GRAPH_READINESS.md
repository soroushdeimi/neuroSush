# Graph and compile readiness audit

This inventory covers all 49 Behavior subclasses under `src/neurosush`, including the
private abstract input base, plus `Behavior` itself. The CPU inventory test discovers
inheritance and requires every row. Values below include inherited `graph_safe` flags.
“CompiledStepper” describes its admission policy, not a claim of successful CUDA execution
for every configuration. The cited forwards and their immediate compute/current helpers
were inspected; an AST guard alone cannot establish graph safety transitively.

`core/compiled.py:83–115` initializes state, checks `compile_ready` for orders below RECORD,
and snapshots direct tensor attributes; `:162–171` draws eagerly and invokes the compiled
function; `:189–206` copies reassigned tensors back. `core/graph.py:89–106` similarly checks
`graph_ready`; `:120–125` selects variants with `graph_key`; `:185–220` preserves direct tensor
addresses. Nested buffers are not recursively copied back. `core/behavior.py:25–29,40–69`
defines the default false flag, independent-batch contract and compile fallback to graph
readiness. `independent_ok` is a separate initialization gate, not graph readiness.
Record-order behaviors execute eagerly after both steppers. Python configuration branches
are acceptable when static; changing per-step branches need `graph_key` or device state.

| Behavior | graph_safe | CompiledStepper | Forward evidence and blocker/condition | Smallest change toward graph safety |
| --- | --- | --- | --- | --- |
| `ActiveSegments` | true (from _SynapticInput) | Yes | [synapses/segments.py:160](../src/neurosush/synapses/segments.py#L160): Fixed-width segment_counts and plateau_step; no segment growth in this forward. | None identified in forward; GPU execution remains to be checked. |
| `ActivityHomeostasis` | true | Yes | [neurons/homeostasis.py:62](../src/neurosush/neurons/homeostasis.py#L62): Fixed-shape activity; window decision is exposed by graph_key (homeostasis.py:77). | None identified in forward; GPU execution remains to be checked. |
| `AdaptiveELIF` | true (from LIF) | Yes | [neurons/models.py:141](../src/neurosush/neurons/models.py#L141): Inherits LIF.forward; derivative at models.py:242 and fire at :246 update fixed-shape adaptation. | None identified in forward; GPU execution remains to be checked. |
| `AdaptiveThreshold` | true | Yes | [neurons/homeostasis.py:134](../src/neurosush/neurons/homeostasis.py#L134): Shape/independent/tau branches are static configuration; fixed-shape theta update. | None identified in forward; GPU execution remains to be checked. |
| `AvgPool2dInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:560](../src/neurosush/synapses/currents.py#L560). | None identified in forward; GPU execution remains to be checked. |
| `Axon` | false (from Behavior) | Conditional: max_delay = 1 | [neurons/axon.py:64](../src/neurosush/neurons/axon.py#L64): Calls HistoryBuffer.push; depth > 1 changes a Python head. | Device-side ring head and fixed-shape indexing; preserve reset/checkpoint semantics. |
| `Behavior` | false | No: default readiness is false | [core/behavior.py:37](../src/neurosush/core/behavior.py#L37): Default flag is false; empty forward. | No concrete computation; subclasses declare readiness. |
| `ConductanceIntegration` | true | Yes | [neurons/dendrite.py:304](../src/neurosush/neurons/dendrite.py#L304): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `Conv2dInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:435](../src/neurosush/synapses/currents.py#L435). | None identified in forward; GPU execution remains to be checked. |
| `CorrelatedPoissonInput` | true | Yes | [neurons/inputs.py:254](../src/neurosush/neurons/inputs.py#L254): Fixed-shape mother/copy draws; CompiledStepper supplies draw dictionary. | None identified in forward; GPU execution remains to be checked. |
| `CurrentNormalization` | true | Yes | [synapses/constraints.py:125](../src/neurosush/synapses/constraints.py#L125): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `DelayInit` | true | Yes | [core/behavior.py:37](../src/neurosush/core/behavior.py#L37): Inherits empty Behavior.forward; validation/randomness occur only in initialize. | None identified in forward; GPU execution remains to be checked. |
| `DendriteIntegration` | true | Yes | [neurons/dendrite.py:178](../src/neurosush/neurons/dendrite.py#L178): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `DendriteStructure` | false (from Behavior) | Conditional: occupied compartments have depth 1 | [neurons/dendrite.py:82](../src/neurosush/neurons/dendrite.py#L82): Delayed path advances ArrivalBuffer Python head and may validate a new delay version. | Device-side arrival head plus validation outside capture. |
| `DenseInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:324](../src/neurosush/synapses/currents.py#L324). | None identified in forward; GPU execution remains to be checked. |
| `Dopamine` | false (from Behavior) | No | [modulation.py:69](../src/neurosush/modulation.py#L69): Updates Python float net.dopamine; no tensor sync in its own arithmetic, but captured Python state would freeze. | Device scalar recurrence with specified dtype and public-state/checkpoint compatibility. |
| `ELIF` | true (from LIF) | Yes | [neurons/models.py:141](../src/neurosush/neurons/models.py#L141): Inherits LIF.forward; derivative override at models.py:185 adds a fixed-shape exponential. | None identified in forward; GPU execution remains to be checked. |
| `Fire` | true | Yes | [neurons/models.py:391](../src/neurosush/neurons/models.py#L391): Delegates to initialized group.model.fire (LIF :147, AdaptiveELIF :246, Izhikevich :365, HodgkinHuxley :441). | None identified in forward; GPU execution remains to be checked. |
| `HodgkinHuxley` | true | Yes | [neurons/hodgkin_huxley.py:389](../src/neurosush/neurons/hodgkin_huxley.py#L389): Static substeps loop; fixed-shape gate/membrane arithmetic and spike masks. | None identified in forward; GPU execution remains to be checked. |
| `ISTDP` | true | Yes | [synapses/plasticity.py:617](../src/neurosush/synapses/plasticity.py#L617): Branches on fixed connectivity; fixed-shape updates, including indexed sparse edges. | None identified in forward; GPU execution remains to be checked. |
| `InherentNoise` | false (from Behavior) | Yes via draw; graph conditional | [neurons/competition.py:212](../src/neurosush/neurons/competition.py#L212): Random samples drawn eagerly by CompiledStepper; GraphStepper checks CUDA generator and random_numbers_supported(). | No compile change; graph RNG compatibility must be checked on GPU. |
| `Izhikevich` | true | Yes | [neurons/models.py:353](../src/neurosush/neurons/models.py#L353): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `KWTA` | true | Yes | [neurons/competition.py:84](../src/neurosush/neurons/competition.py#L84): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `LIF` | true | Yes | [neurons/models.py:141](../src/neurosush/neurons/models.py#L141): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `LateralInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:528](../src/neurosush/synapses/currents.py#L528). | None identified in forward; GPU execution remains to be checked. |
| `Local2dInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:484](../src/neurosush/synapses/currents.py#L484). | None identified in forward; GPU execution remains to be checked. |
| `MaxPool2dInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:620](../src/neurosush/synapses/currents.py#L620). | None identified in forward; GPU execution remains to be checked. |
| `MinicolumnInhibition` | true | Yes | [neurons/competition.py:177](../src/neurosush/neurons/competition.py#L177): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `OneToOneInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:356](../src/neurosush/synapses/currents.py#L356). | None identified in forward; GPU execution remains to be checked. |
| `Payoff` | false (from Behavior) | No | [modulation.py:34](../src/neurosush/modulation.py#L34): Arbitrary callback and float(self.fn(net)); Python payoff reassignment, and host sync if callback returns a device tensor. | Stage callback outside capture, publish a stable device scalar under an explicit callback contract. |
| `PoissonDrive` | false (from Behavior) | Yes via draw; graph conditional | [neurons/inputs.py:167](../src/neurosush/neurons/inputs.py#L167): Random counts drawn eagerly by CompiledStepper; GraphStepper needs a CUDA generator and random_numbers_supported(). | No compile change; graph RNG compatibility must be checked on GPU. |
| `PoissonInput` | true | Yes | [neurons/inputs.py:116](../src/neurosush/neurons/inputs.py#L116): Fixed-shape draw/threshold; CompiledStepper supplies uniform through draw. | None identified in forward; GPU execution remains to be checked. |
| `RSTDP` | false (from Behavior) | No | [synapses/plasticity.py:558](../src/neurosush/synapses/plasticity.py#L558): Reads changing Python float net.dopamine; readiness overrides both return false. Unlike STDP.forward, calls compute directly, with no event-path nonzero. | Explicit device scalar reward contract, with exact recurrence and promotion tests; see reward decision below. |
| `Recorder` | false (from Behavior) | Yes, eager after step | [recording.py:50](../src/neurosush/recording.py#L50): Order.RECORD bypasses readiness checks. Appends Python lists and may copy device tensors to CPU. | For capture: fixed-capacity device storage and device cursor; keep export outside capture. |
| `Refractory` | true | Yes | [neurons/models.py:447](../src/neurosush/neurons/models.py#L447): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `STDP` | false (from Behavior) | Conditional: not CPU unbatched dense all-pairing | [synapses/plasticity.py:482](../src/neurosush/synapses/plasticity.py#L482): CPU event path calls apply_stdp_dense_ with nonzero-driven variable-size indices; compile_ready rejects it. CUDA uses compute. | Use fixed-shape compute on CPU too, after proving exact arithmetic equivalence. |
| `SegmentLearning` | false (from Behavior) | No | [synapses/segment_learning.py:158](../src/neurosush/synapses/segment_learning.py#L158): bool(fired.any()), nonzero().tolist(), masked variable-size indexing; helpers grow synapses and select random variable-size candidates using Python iteration/time. | Fixed-width masked updates, device time, deterministic fixed-shape selection/growth; requires a larger equivalence study. |
| `SparseInput` | true (from _SynapticInput) | Yes | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Inherits tensor-only _SynapticInput.forward; concrete fixed-shape current at [synapses/currents.py:385](../src/neurosush/synapses/currents.py#L385). | None identified in forward; GPU execution remains to be checked. |
| `SpikeCounter` | true | Yes | [recording.py:108](../src/neurosush/recording.py#L108): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `SpikeGather` | false (from Behavior) | Conditional: all histories read have depth 1 | [synapses/traces.py:60](../src/neurosush/synapses/traces.py#L60): HistoryBuffer.read uses Python head and delay-validation cache at depth > 1. | Device-side history indexing plus validation outside capture. |
| `SpikeInput` | true | Yes | [neurons/inputs.py:68](../src/neurosush/neurons/inputs.py#L68): Publishes staged tensor; prepare stages input outside captured/compiled forward. | None identified in forward; GPU execution remains to be checked. |
| `SpikeTriggeredCurrent` | true | Yes | [neurons/adaptation.py:66](../src/neurosush/neurons/adaptation.py#L66): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `TraceRecorder` | false (from Behavior) | Yes, eager after step | [viz/trace.py:297](../src/neurosush/viz/trace.py#L297): Order.RECORD bypasses readiness checks. Appends steps and advances Python _count; capacity/interval are Python decisions. | For capture: device cursor and overflow/interval contract; preallocation alone is insufficient. |
| `Traces` | true | Yes | [synapses/traces.py:122](../src/neurosush/synapses/traces.py#L122): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `TripletSTDP` | true | Yes | [synapses/triplet.py:250](../src/neurosush/synapses/triplet.py#L250): Delegates to fixed-shape compute and tensor weight addition. | None identified in forward; GPU execution remains to be checked. |
| `VoltageHomeostasis` | true | Yes | [neurons/homeostasis.py:196](../src/neurosush/neurons/homeostasis.py#L196): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `WeightClip` | true | Yes | [synapses/constraints.py:76](../src/neurosush/synapses/constraints.py#L76): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `WeightInit` | true | Yes | [core/behavior.py:37](../src/neurosush/core/behavior.py#L37): Inherits empty Behavior.forward; allocation/randomness occur only in initialize. | None identified in forward; GPU execution remains to be checked. |
| `WeightNormalization` | true | Yes | [synapses/constraints.py:102](../src/neurosush/synapses/constraints.py#L102): Fixed-shape tensor arithmetic/masking; no host scalar extraction in forward. | None identified in forward; GPU execution remains to be checked. |
| `_SynapticInput` | true | Abstract base | [synapses/currents.py:295](../src/neurosush/synapses/currents.py#L295): Tensor scaling delegates to abstract current; cannot instantiate. | Concrete subclasses provide current. |

## Reward decision: preserve the eager API

No changes to RSTDP, Payoff or Dopamine are made here. Their relevant forwards are
`modulation.py:34–36,69–71` and `synapses/plasticity.py:558–573`.

* Payoff promises `Callable[[Network], float]` and calls it each step. Arbitrary callbacks
  may observe Python state or have side effects; blindly tracing them would freeze those
  effects. `float(...)` can synchronize if the callback returns a tensor, but replacing it
  also changes the public Python-float contract. A staging API needs a separate design.
* Dopamine uses Python double-precision scalar arithmetic. The seeded 200-step CPU test
  `test_float32_dopamine_replacement_changes_eager_bits` demonstrates that a direct float32
  device-scalar substitute does not preserve even the rounded dopamine bits. A float64
  design would still need dtype/promotion, callback, reset, checkpoint and CUDA evidence.
* RSTDP already computes fixed-shape eligibility and weights without host extraction. Its
  blocker is the changing Python dopamine dependency, not a local `.item()` to remove.
  Promoting its readiness alone would not make the scheduled reward path capturable.

## Delayed buffers: one bounded improvement

`core/buffers.py:80–103` caches validation by delay identity and version. On an uncached valid
version, two separate `bool(tensor)` extractions are replaced with one two-flag `tolist()`
transfer. Cached reads still extract nothing; invalid versions retain the original error
message. No speedup or GPU synchronization timing is claimed from CPU evidence. An invalid
version still transfers the offending delays to format its error.

`tests/test_graph_readiness.py` retains the original validation implementation as a reference
and compares **every serialized tensor** with `torch.equal` after each of 200 seeded CPU
steps (also checks head and cache state). It covers arrival and history buffers, depths
1/2/7, unbatched/batched operation, float32/float64 arrivals, boolean histories, zero/mixed/
integer delays, repeated accumulation, in-place delay edits, reset and ring wrap. Failure
cases include wrong shape, negative and out-of-range delays. A separate extraction test
rejects separate `bool(tensor)` calls and counts one flag transfer per changed version.

This does **not** make delays > 1 graph-ready: `HistoryBuffer.push` (`buffers.py:121`) and
`ArrivalBuffer.advance` (`:161`) still move Python `_head`. Capture would freeze those
indices. Moving the head to a device tensor also requires fixed-shape gather/scatter,
reset/load semantics, validation staging, and treatment of nested buffer storage in the
steppers. That larger change is deferred; existing depth-1 gates remain in place.

## Guard scope and CPU evidence

The AST guard resolves inherited flags and inherited forwards across every source module.
For every effective true flag it rejects attribute calls `.item()` and `.tolist()`, and
builtin `bool(...)` calls. It conservatively treats every such call in forward as a possible
tensor extraction. There are currently **no exceptions**. Tensor `.bool()` is a device cast
and is allowed. Helpers, implicit tensor truth tests, `float(tensor)`, dynamic boolean
indexing and runtime callbacks are outside this deliberately simple guard; table review
covers the known cases rather than claiming the scan proves capture safety. Scanner tests
exercise all three forbidden spellings and the allowed tensor cast.

## NEEDS 3090

* Run the existing GPU graph, compiled, batching, reset and checkpoint tests on CUDA;
  check graph/eager bit equality and compiled/eager tolerances separately.
* Profile cold versus cached delay validation on CUDA to verify transfer/synchronization
  counts and measure whether combining flags helps; there is no steady-state speed claim.
* Replay depth-1 networks across graph-key transitions and resets; validate RNG draw order
  for each stochastic behavior against the installed torch generator-capture support.
* Any later device reward or deeper-ring implementation needs 200-step CUDA state comparisons,
  repeated capture/replay, buffer address checks, and reset/checkpoint continuation. Current
  deeper-delay and reward readiness rejection must remain until that implementation exists.

No CUDA capture, replay, GPU timing or 3090 result was obtained in this CPU-only worktree.
