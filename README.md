# neuroSush

Brain-inspired computing on PyTorch, validated against its mathematics.

neuroSush simulates **spiking neural networks** (leaky, exponential and adaptive
integrate-and-fire neurons with dendritic compartments and delays, STDP-family plasticity
with dopamine modulation) on a CPU or GPU, in batches. It implements the **Thousand Brains
and HTM models** (sparse distributed representations, spatial pooler, temporal memory, grid
cells, active dendrites, voting columns) and **hierarchical predictive coding**, and it joins
the two worlds: a **spiking layer that provably computes the temporal memory** and learns
the same sequences. Every model is tested against what it claims: exact solutions,
closed-form probabilities and expectations, and the results of its papers.

| | |
|---|---|
| [Spiking networks](#quickstart) | neurons, dendrites, delays, STDP, reward, homeostasis, batching, recording and checkpoints |
| [Thousand Brains models](#thousand-brains-models) | SDRs, encoders, spatial pooler, temporal memory, grid cells, active dendrites, voting columns, predictive coding |
| [Spiking temporal memory](#spiking-temporal-memory) | dendritic segments with NMDA-like plateaus, minicolumn inhibition, sequence learning in spike time |
| [Validation](#validation) | the mathematics every part is checked against |

## Design

- **Pure math, thin behaviors.** Every equation is a small function that returns new tensors
  and has tests with hand-computed values; behaviors only hold state and call them.
- **Explicit execution order.** Each behavior class declares when it runs in a step; there is
  no global priority table and no special initialization order.
- **Reproducible.** All randomness comes from the network's seeded generator.
- **One dependency.** `torch` only at runtime.
- **Safe serialization.** Structure specs load through a registry of classes, never `eval`.
- **Validated.** Every equation has tests with hand-computed values, and every model is
  checked against its mathematics (see [Validation](#validation)); examples that learn run
  in CI.

## Install

```bash
pip install neurosush
```

Every [GitHub release](https://github.com/soroushdeimi/neuroSush/releases) also carries
the wheel and the source archive. From source, with CPU PyTorch (on macOS use
`pip install torch` instead):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Python 3.10 or newer.

## Quickstart

Twenty Poisson inputs drive two LIF neurons through plastic synapses:

```python
import itertools

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.data import spike_frames
from neurosush.encoding import rate_poisson
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.constraints import WeightClip
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

net = Network(seed=0)
train = rate_poisson(torch.full((20,), 0.3), 100, generator=net.generator)
source = NeuronGroup(net, 20, [SpikeInput(itertools.cycle(spike_frames([(train, None)]))), Axon()])
target = NeuronGroup(
    net,
    2,
    [
        DendriteStructure(),  # routes synaptic currents to the soma
        DendriteIntegration(),
        LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
        Fire(),
        Axon(),
    ],
)
synapse = SynapseGroup(
    net,
    source,
    target,
    [
        WeightInit(mode="uniform", scale=0.5),
        DenseInput(coef=5.0),
        SpikeGather(),
        Traces(tau_pre=10.0),
        STDP(a_plus=0.02, a_minus=0.01, bound="soft"),
        WeightClip(w_min=0.0, w_max=1.0),
    ],
)

spikes = 0
for _ in range(200):
    net.step()
    spikes += int(target.spikes.sum())
print(f"output spikes: {spikes}, mean weight: {synapse.weights.mean():.3f}")
```

[`examples/two_patterns.py`](https://github.com/soroushdeimi/neuroSush/blob/main/examples/two_patterns.py) goes further: two output neurons with
k-winners-take-all and homeostasis learn to respond to one input pattern each.

## Concepts

- A **`Network`** owns neuron groups, synapse groups, the time step `dt`, the dtype, the
  device and a seeded random generator.
- A **`NeuronGroup`** has a shape `(depth, height, width)` (an int `n` means `(1, 1, n)`), and a
  **`SynapseGroup`** connects two groups and targets one dendritic compartment of the
  destination: `proximal` (drives the soma), `distal` or `apical` (prime it).
- A **`Behavior`** holds state and dynamics. Behaviors run in this order in every step (and
  are initialized in the same order):

| order | behaviors | order | behaviors |
|---|---|---|---|
| 0 | `WeightInit`, `DelayInit` | 300 | `KWTA`, `MinicolumnInhibition` |
| 100 | `Payoff` | 310 | `VoltageHomeostasis` |
| 120 | `Dopamine` | 340 | `Fire`, `SpikeInput` |
| 180 | synaptic inputs, `ActiveSegments` | 350 | `ActivityHomeostasis` |
| 200 | `CurrentNormalization` | 380 | `Axon` |
| 220 | `DendriteStructure` | 420 | `SpikeGather` |
| 240 | `DendriteIntegration` | 460 | `Traces` |
| 260 | `LIF`, `ELIF`, `AdaptiveELIF` | 500 | `STDP`, `RSTDP`, `ISTDP`, `SegmentLearning` |
| 280 | `InherentNoise` | 520, 540 | `WeightNormalization`, `WeightClip` |
|  |  | 1000 | `Recorder` |

- **Spikes travel through `SpikeGather`**: a synaptic input reads `syn.pre_spike`, which
  `SpikeGather` fills from the source's `Axon`, so every synapse group with an input needs
  both; building without them is an error, not a silent network.
- **Delays** are integers in steps: `src_delay` is axonal (read from the source's `Axon`
  history), `dst_delay` is dendritic (arrival in the destination's `DendriteStructure`).
- **Units**: every time constant shares the unit of `dt` (milliseconds by convention).
- **Signs**: weights are magnitudes; `NeuronGroup(..., inhibitory=True)` makes its outgoing
  currents negative.
- **Batches**: `Network(batch_size=B)` simulates `B` samples side by side. State tensors get
  shape `(B, size)`; weights and thresholds stay shared, and learning uses the batch mean.
  Feed it with `spike_frames(samples, batch_size=B)`. On a GPU a batch costs about as much
  as one sample, so throughput grows almost linearly with `B`.

More in [docs/ARCHITECTURE.md](https://github.com/soroushdeimi/neuroSush/blob/main/docs/ARCHITECTURE.md).

## Recording and checkpoints

A `Recorder` copies attributes of its host at the end of every `interval`-th step. A
checkpoint saves the complete state of an initialized network (iteration, random
generator, every tensor and delay buffer, and behavior state such as homeostasis), and
loading it into a network built the same way continues the run exactly:

```python
import tempfile
from pathlib import Path

import torch

from neurosush import checkpoint
from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.competition import InherentNoise
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire
from neurosush.recording import Recorder


def build():
    net = Network(seed=0)
    NeuronGroup(
        net,
        3,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            InherentNoise(scale=8.0),  # random drive from the network generator
            Fire(),
            Recorder("v", "spikes", interval=5),
        ],
    )
    return net


net = build()
net.run(50)
recorder = net.groups[0].behaviors[-1]
print(recorder.get("v").shape, recorder.steps[:3])  # torch.Size([10, 3]) [5, 10, 15]

path = Path(tempfile.mkdtemp()) / "net.pt"
checkpoint.save(net, path)
net.run(20)

resumed = build()
checkpoint.load(resumed, path)
resumed.run(20)
assert torch.equal(resumed.groups[0].v, net.groups[0].v)  # continues exactly
```

Loading uses `torch.load(weights_only=True)`, so a checkpoint cannot run code. Input
streams (`SpikeInput`) are not part of it: resume them yourself.

## Modules

| module | contents |
|---|---|
| `neurosush.core` | `Network`, `NeuronGroup`, `SynapseGroup`, `Behavior`, `Order`, delay buffers |
| `neurosush.neurons.models` | `LIF`, `ELIF`, `AdaptiveELIF`, `Fire` (equations in `dynamics`) |
| `neurosush.neurons.competition` | `KWTA`, `MinicolumnInhibition`, `InherentNoise` |
| `neurosush.neurons.axon`, `.dendrite` | `Axon`; `DendriteStructure`, `DendriteIntegration` |
| `neurosush.neurons.homeostasis` | `ActivityHomeostasis`, `VoltageHomeostasis` |
| `neurosush.neurons.inputs` | `SpikeInput` |
| `neurosush.synapses.init` | `WeightInit` (dense or sparse), `DelayInit` |
| `neurosush.synapses.currents` | `DenseInput`, `OneToOneInput`, `SparseInput`, `Conv2dInput`, `Local2dInput`, `LateralInput`, `AvgPool2dInput` |
| `neurosush.synapses.traces` | `SpikeGather`, `Traces` |
| `neurosush.synapses.segments` | `ActiveSegments` (dendritic segments with NMDA-like plateaus) |
| `neurosush.synapses.segment_learning` | `SegmentLearning` (temporal memory learning in spike time) |
| `neurosush.synapses.plasticity` | `STDP`, `RSTDP`, `ISTDP` (bounds in `bounds`) |
| `neurosush.synapses.constraints` | `WeightClip`, `WeightNormalization`, `CurrentNormalization` |
| `neurosush.modulation` | `Payoff`, `Dopamine` |
| `neurosush.encoding` | `rate_poisson`, `interval_poisson`, `intensity_to_latency` |
| `neurosush.filters`, `.transforms` | DoG and Gabor kernels; grid masks, polarity split, filter bank |
| `neurosush.data` | `LocationDataset`, `spike_frames` |
| `neurosush.structure` | layers, ports, `connect`, `CorticalColumn`, JSON specs, `sequence_memory` |
| `neurosush.recording` | `Recorder` |
| `neurosush.checkpoint` | `state_dict`, `load_state_dict`, `save`, `load` |
| `neurosush.htm.sdr`, `.encoders`, `.classifier` | SDR operations and match probabilities; scalar, RDSE and category encoders; `SDRClassifier` |
| `neurosush.htm.spatial_pooler`, `.temporal_memory` | `SpatialPooler`, `TemporalMemory` |
| `neurosush.htm.grid_cells` | `GridCellModule`, `GridCellModules`, `hexagonal_rate` |
| `neurosush.htm.active_dendrites` | `ActiveDendrites`, `kwta` |
| `neurosush.htm.objects` | `ObjectLibrary`, `SensorColumn`, `ColumnEnsemble`, `vote` |
| `neurosush.predictive_coding` | `PredictiveCodingNetwork` |

## Structures and specs

A column is plain data. Building it twice gives two independent columns; JSON only names
registered classes, so loading it never runs code.

```python
from neurosush.core.network import Network
from neurosush.structure.spec import (
    BehaviorSpec,
    ColumnSpec,
    GroupSpec,
    LayerSpec,
    SynapseSpec,
    build_column,
    from_json,
    to_json,
)

lif = BehaviorSpec("LIF", {"tau": 10.0, "threshold": -55.0, "v_reset": -70.0, "v_rest": -65.0})
neurons = (
    BehaviorSpec("DendriteStructure"),
    BehaviorSpec("DendriteIntegration"),
    lif,
    BehaviorSpec("Fire"),
    BehaviorSpec("Axon"),
)
dense = (
    BehaviorSpec("WeightInit", {"mode": "uniform"}),
    BehaviorSpec("DenseInput"),
    BehaviorSpec("SpikeGather"),
)
layer = LayerSpec(
    groups={"exc": GroupSpec(8, neurons), "inh": GroupSpec(2, neurons, inhibitory=True)},
    synapses=(SynapseSpec("exc", "inh", dense), SynapseSpec("inh", "exc", dense)),
    outputs={"out": ("exc",)},
)
spec = ColumnSpec(
    layers={"L4": layer, "L23": layer},
    synapses=(SynapseSpec("L4.exc", "L23.exc", dense),),  # "layer.group" inside a column
    inputs={"in": "L4.in"},
    outputs={"out": "L23.out"},
)

assert from_json(to_json(spec)) == spec
net = Network(seed=0)
column = build_column(net, "C1", spec)
net.run(10)
print([group.name for group in column.output_port("out")])  # ['C1.L23.exc']
```

## Thousand Brains models

`neurosush.htm` implements the algorithms of hierarchical temporal memory and the Thousand
Brains theory as tensor code, each checked against the mathematics of its paper:

| model | reference | validated by the tests |
|---|---|---|
| SDRs | Ahmad and Hawkins (2016) | exact false-match probabilities, confirmed by Monte Carlo |
| encoders | Numenta encoders | overlap of scalar codes is `max(0, w - distance)` |
| spatial pooler | Cui et al. (2017) | exact update rules; learned codes stay stable under 20% input noise; boosting spreads activity |
| temporal memory | Hawkins and Ahmad (2016) | first- and high-order sequences, branching unions, punishment of wrong predictions |
| grid cells | Hawkins et al. (2019) | exact path integration on the torus; six-fold symmetric fields; several modules locate far beyond one scale |
| active dendrites | Iyer et al. (2022) | gating equations; context solves two tasks that give every input opposite labels |
| voting columns | Lewis et al. (2019) | the true object is never lost; elimination rates match closed-form expectations; more columns need fewer touches |
| predictive coding | Rao and Ballard (1999), Bogacz (2017) | inference and learning follow the free-energy gradient; the exact Gaussian posterior; convergence to backprop (Whittington and Bogacz 2017) |

```python
import torch

from neurosush.htm.encoders import CategoryEncoder
from neurosush.htm.sdr import match_probability
from neurosush.htm.temporal_memory import TemporalMemory

# chance that a random 40-of-2048 SDR shares 20 or more bits with a stored one
print(f"{match_probability(2048, 40, 40, 20):.1e}")

a, b, c, d = CategoryEncoder(256, 12, 4, seed=0).encode(torch.arange(4))
tm = TemporalMemory(
    256,
    8,
    activation_threshold=8,
    min_threshold=6,
    initial_permanence=0.51,
    max_new_synapses=12,
    max_synapses_per_segment=16,
)
for _ in range(3):
    tm.reset()
    for symbol in (a, b, c, d):
        tm.compute(symbol)
tm.reset()
tm.compute(a, learn=False)
tm.compute(b, learn=False)
assert torch.equal(tm.predicted_columns(), c)  # after A B it expects C
```

[`examples/sequence_prediction.py`](https://github.com/soroushdeimi/neuroSush/blob/main/examples/sequence_prediction.py) chains an encoder, the
spatial pooler, temporal memory and a classifier on sequences that differ only in their
first symbol; [`examples/object_recognition.py`](https://github.com/soroushdeimi/neuroSush/blob/main/examples/object_recognition.py) shows voting
columns recognizing objects in fewer touches.

## Spiking temporal memory

The same sequence memory also runs as a spiking network. `ActiveSegments` gives every cell
distal segments that fire a dendritic spike when enough of their synapses see input within a
coincidence window, and hold a plateau that primes the cell below threshold.
`MinicolumnInhibition` lets the first cells of a minicolumn to reach threshold silence the
rest. A primed (predicted) cell therefore fires alone, and a minicolumn without one fires
all at once (a burst). `tests/validation/test_sequence_math.py` checks that such a layer
activates exactly the cells `TemporalMemory` activates, element by element, under timing
conditions it also checks.

`SegmentLearning` adds the temporal memory's learning in spike time (reinforcement, growth
towards the previous element's winners, new segments in bursting minicolumns, punishment
of wrong predictions), and `sequence_memory` builds the whole layer. It derives the
plateau, coincidence window and learning context from the neuron's exact race times and
refuses parameters under which the equivalence would fail. On sequences that share their
middle it learns the same curves as `TemporalMemory` and ends in the same state
(`tests/validation/test_sequence_learning.py`, and `experiments/sequence_learning.py` for
the full curves).

A four-element sequence, learned in one repetition (`initial_permanence=0.51` makes new
synapses connected at once):

```python
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.htm.sdr import random_sdr
from neurosush.neurons.axon import Axon
from neurosush.structure.sequence import sequence_memory

COLUMNS, PERIOD, WINDOW = 32, 60, 25
sequence = random_sdr(COLUMNS, 6, batch=(4,), generator=torch.Generator().manual_seed(0))


class Present(Behavior):
    """Drive the columns of element t // PERIOD for the first WINDOW steps of its period."""

    order = Order.FIRE

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        t = group.net.iteration - 1
        element, offset = divmod(t % (PERIOD * 6), PERIOD)  # 4 elements, then 2 silent
        on = element < len(sequence) and offset < WINDOW
        group.spikes = sequence[element].clone() if on else group.state(False, dtype=torch.bool)


net = Network(seed=0)
columns = NeuronGroup(net, COLUMNS, [Present(), Axon()])
layer, _ = sequence_memory(
    net, columns, cells_per_column=4, period=PERIOD, window=WINDOW, initial_permanence=0.51
)
for repetition in range(2):
    bursts = []
    for element in range(6):
        fired = torch.zeros(layer.size, dtype=torch.bool)
        for _ in range(PERIOD):
            net.step()
            fired |= layer.spikes
        if element < len(sequence):
            active = fired.view(COLUMNS, 4)[sequence[element]]
            bursts.append(int(active.all(-1).sum()))  # columns that were not predicted
    print(repetition, bursts)  # [6, 6, 6, 6], then [6, 0, 0, 0]: only the first element surprises
```

## Validation

Besides unit tests, `tests/validation` checks the spiking core against the mathematics it
implements. The LIF reproduces the exact solution of its Euler scheme and converges to the
continuous solution at first order in `dt`; interspike intervals match the closed form, and
the exponential LIF starts firing exactly at its rheobase `R I = theta_rh - v_rest - delta`.
Traces respond to a spike with exactly `(1 - dt/tau)^k`; the STDP window is the exponential
`a_plus c^k` or `-a_minus c^k`, and uncorrelated spike trains drift the weights by the
expected amount. Inhibitory STDP settles the firing rate at its target, homeostasis settles
the spike count, Poisson spike counts are binomial with geometric intervals, and every spike
arrives exactly `src_delay + dst_delay + 1` steps after it was fired. Conv, local, lateral and
pooling synapses equal their dense synapse-by-synapse definition, for both currents and
STDP; reward-modulated STDP matches the closed form of the three-factor rule and solves the
distal reward problem of Izhikevich (2007).

## Limitations

- **Alpha.** The API may still change between minor versions.
- **Speed.** A simulation step is a sequence of small tensor operations driven from
  Python: a few milliseconds per step on a CPU, whatever the network size up to thousands
  of neurons. Batches spread that cost over many samples, especially on a GPU. The spatial
  pooler learns one sample at a time.
- **CPU-only HTM models.** The `neurosush.htm` models and `SegmentLearning` run unbatched
  on the CPU.
- **Checkpoints** save a network's state but not the input streams feeding `SpikeInput`.
- **The spiking temporal memory** equals the algorithm only under the timing conditions
  that `sequence_timing` checks; `sequence_memory` refuses other parameters.

## Citing

If you use neuroSush in research, please cite it with the metadata in
[`CITATION.cff`](https://github.com/soroushdeimi/neuroSush/blob/main/CITATION.cff) (GitHub shows it under "Cite this repository"), and
cite the papers of the models you use; each module names them.

## Development

`bash scripts/check.sh` runs ruff (lint and format check), strict mypy and the full test
suite. `python benchmarks/suite.py` measures throughput (the Benchmarks workflow runs it
every week), and the scripts in [`experiments/`](https://github.com/soroushdeimi/neuroSush/blob/main/experiments) reproduce the numbers
behind the validation claims. Tests marked `gpu` compare CUDA with CPU results and run only
where a CUDA device exists. See [CONTRIBUTING.md](https://github.com/soroushdeimi/neuroSush/blob/main/CONTRIBUTING.md) for the workflow and
code standards.

## Releases

Releases are automatic. A commit on `main` that sets a new release version with its
changelog section triggers the Release workflow: it checks the version and the changelog,
runs the full CI, creates the tag and a GitHub release with the wheel and the sdist, and
publishes them to PyPI. The steps are in [CONTRIBUTING.md](https://github.com/soroushdeimi/neuroSush/blob/main/CONTRIBUTING.md); the history
is in [CHANGELOG.md](https://github.com/soroushdeimi/neuroSush/blob/main/CHANGELOG.md).

## License

MIT, see [LICENSE](https://github.com/soroushdeimi/neuroSush/blob/main/LICENSE).
