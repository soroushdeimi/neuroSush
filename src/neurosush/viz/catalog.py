"""Machine-readable catalog of the atlas plates and the coverage check for the library models."""

# ruff: noqa: RUF001 - the Persian titles are Unicode text by design; the letter lints target
# ASCII-lookalike mistyping in English strings.
from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass

import neurosush
from neurosush.core.behavior import Behavior

# Modules whose public classes count as "models" even though they are no behaviors, plus the
# named engine classes; the plates cover behavior classes wherever they live.
_NON_BEHAVIOR_MODULES = (
    "neurosush.htm.",
    "neurosush.predictive_coding",
    "neurosush.structure.",
    "neurosush.encoding",
    "neurosush.filters",
    "neurosush.transforms",
)
_NAMED_CLASSES = frozenset(
    {
        "neurosush.core.network.Network",
        "neurosush.core.network.NeuronGroup",
        "neurosush.core.network.SynapseGroup",
        "neurosush.core.graph.GraphStepper",
        "neurosush.core.compiled.CompiledStepper",
    }
)


@dataclass(frozen=True)
class PlateSpec:
    """One plate of the atlas: a stable id, its titles, and the library names it will teach.

    Args:
        id: Plate id from the plan, such as ``"N01"``; stable across waves.
        title_en: Honest English title of the plate.
        title_fa: Persian title of the plate.
        wave: Planned build wave, ``1`` through ``4``.
        flagship: ``True`` when the plan marks the plate as a flagship plate.
        covers: Fully qualified public class names (``"neurosush.neurons.models.LIF"``)
            that the plate will teach.
    """

    id: str
    title_en: str
    title_fa: str
    wave: int
    flagship: bool
    covers: tuple[str, ...]


PLATES: dict[str, PlateSpec] = {
    "N01": PlateSpec(
        id="N01",
        title_en="LIF: the leaky bucket",
        title_fa="LIF: سطل آب نشتی",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.models.LIF", "neurosush.neurons.models.Fire"),
    ),
    "N02": PlateSpec(
        id="N02",
        title_en="ELIF at rheobase: a ball in a potential well",
        title_fa="ELIF و رئوباز: توپ در چاه پتانسیل",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.models.ELIF",),
    ),
    "N03": PlateSpec(
        id="N03",
        title_en="AdEx: adaptation in the phase plane",
        title_fa="AdEx: سازش در صفحه فاز",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.models.AdaptiveELIF",),
    ),
    "N04": PlateSpec(
        id="N04",
        title_en="Izhikevich: a zoo of firing patterns",
        title_fa="ایزایکویچ: باغ‌وحش الگوهای تکانش",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.models.Izhikevich",),
    ),
    "N05": PlateSpec(
        id="N05",
        title_en="Hodgkin-Huxley: membrane cross-section, Na and K channels, gates m, h, n",
        title_fa="هادگکین–هاکسلی: مقطع غشا، کانال‌های یونی سدیم و پتاسیم، دروازه‌های m و h و n",
        wave=2,
        flagship=True,
        covers=(),
    ),
    "N06": PlateSpec(
        id="N06",
        title_en="Refractory and the spike-triggered current",
        title_fa="دوره ناتوانی و جریان برانگیخته از اسپایک",
        wave=2,
        flagship=False,
        covers=(
            "neurosush.neurons.models.Refractory",
            "neurosush.neurons.adaptation.SpikeTriggeredCurrent",
        ),
    ),
    "N07": PlateSpec(
        id="N07",
        title_en="Conductance synapse: tug of war between reversal potentials",
        title_fa="سیناپس رسانایی: طناب‌کشی پتانسیل‌های وارونگی",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.dendrite.ConductanceIntegration",),
    ),
    "N08": PlateSpec(
        id="N08",
        title_en="f-I curve and bifurcation",
        title_fa="منحنی f-I و دوشاخه‌شدگی",
        wave=2,
        flagship=False,
        covers=("neurosush.neurons.models.LIF",),
    ),
    "N09": PlateSpec(
        id="N09",
        title_en="Noise and stochastic channels",
        title_fa="نویز و کانال‌های تصادفی",
        wave=3,
        flagship=False,
        covers=("neurosush.neurons.competition.InherentNoise",),
    ),
    "D01": PlateSpec(
        id="D01",
        title_en="Compartments and the NMDA-like plateau",
        title_fa="کامپارتمان‌ها و صفحه NMDA",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.neurons.dendrite.DendriteStructure",
            "neurosush.neurons.dendrite.DendriteIntegration",
            "neurosush.synapses.segments.ActiveSegments",
        ),
    ),
    "D02": PlateSpec(
        id="D02",
        title_en="Axonal and dendritic delay: spikes travelling, ring buffer inside",
        title_fa="تأخیر آکسونی و دندریتی: اسپایک‌های مسافر و بافر حلقوی",
        wave=3,
        flagship=False,
        covers=("neurosush.neurons.axon.Axon", "neurosush.synapses.init.DelayInit"),
    ),
    "D03": PlateSpec(
        id="D03",
        title_en="Connectivity kernels: dense, sparse, conv, local, lateral, pool",
        title_fa="هسته‌های اتصال: چگال، پراکنده، کانولوشنی، موضعی، جانبی و تجمیع",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.synapses.init.WeightInit",
            "neurosush.synapses.currents.DenseInput",
            "neurosush.synapses.currents.OneToOneInput",
            "neurosush.synapses.currents.SparseInput",
            "neurosush.synapses.currents.Conv2dInput",
            "neurosush.synapses.currents.Local2dInput",
            "neurosush.synapses.currents.LateralInput",
            "neurosush.synapses.currents.AvgPool2dInput",
            "neurosush.synapses.currents.MaxPool2dInput",
        ),
    ),
    "Y01": PlateSpec(
        id="Y01",
        title_en="Traces and the STDP window, with a spike-pair builder",
        title_fa="رد اسپایک و پنجره STDP با سازنده جفت اسپایک",
        wave=2,
        flagship=False,
        covers=(
            "neurosush.synapses.traces.Traces",
            "neurosush.synapses.plasticity.STDP",
        ),
    ),
    "Y02": PlateSpec(
        id="Y02",
        title_en="Triplet rule and frequency dependence",
        title_fa="قاعده سه‌گانه و وابستگی به فرکانس",
        wave=3,
        flagship=False,
        covers=("neurosush.synapses.triplet.TripletSTDP",),
    ),
    "Y03": PlateSpec(
        id="Y03",
        title_en="RSTDP: dopamine and the eligibility trace",
        title_fa="RSTDP: دوپامین و رد شایستگی",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.synapses.plasticity.RSTDP",
            "neurosush.modulation.Payoff",
            "neurosush.modulation.Dopamine",
        ),
    ),
    "Y04": PlateSpec(
        id="Y04",
        title_en="iSTDP: the inhibitory balance",
        title_fa="iSTDP: تعادل مهاری",
        wave=3,
        flagship=False,
        covers=("neurosush.synapses.plasticity.ISTDP",),
    ),
    "Y05": PlateSpec(
        id="Y05",
        title_en="Weights as images: receptive fields forming (Diehl and Cook)",
        title_fa="وزن‌ها به‌سان تصویر: شکل‌گیری میدان‌های گیرشی (Diehl و Cook)",
        wave=3,
        flagship=True,
        covers=(),
    ),
    "Y06": PlateSpec(
        id="Y06",
        title_en="Bounds and normalization",
        title_fa="کران‌ها و نرمال‌سازی",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.synapses.constraints.WeightClip",
            "neurosush.synapses.constraints.WeightNormalization",
            "neurosush.synapses.constraints.CurrentNormalization",
        ),
    ),
    "C01": PlateSpec(
        id="C01",
        title_en="KWTA: the winners-take-all race",
        title_fa="مسابقه KWTA",
        wave=3,
        flagship=False,
        covers=("neurosush.neurons.competition.KWTA",),
    ),
    "C02": PlateSpec(
        id="C02",
        title_en="Homeostasis and the adaptive threshold",
        title_fa="هموستازی و آستانه سازگار",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.neurons.homeostasis.ActivityHomeostasis",
            "neurosush.neurons.homeostasis.VoltageHomeostasis",
            "neurosush.neurons.homeostasis.AdaptiveThreshold",
        ),
    ),
    "C03": PlateSpec(
        id="C03",
        title_en="Minicolumn inhibition",
        title_fa="مهار ریزستونی",
        wave=3,
        flagship=False,
        covers=("neurosush.neurons.competition.MinicolumnInhibition",),
    ),
    "W01": PlateSpec(
        id="W01",
        title_en="Raster, rates and the Brunel regimes",
        title_fa="رستر و نرخ‌ها: رژیم‌های برونل",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.core.network.Network",
            "neurosush.core.network.NeuronGroup",
            "neurosush.core.network.SynapseGroup",
        ),
    ),
    "W02": PlateSpec(
        id="W02",
        title_en="The balanced network",
        title_fa="شبکه متوازن",
        wave=3,
        flagship=False,
        covers=(),
    ),
    "W03": PlateSpec(
        id="W03",
        title_en="Columns, layers and ports: the spec graph",
        title_fa="ستون‌ها، لایه‌ها و پورت‌ها: نمودار spec",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.structure.column.CorticalColumn",
            "neurosush.structure.layer.CorticalLayer",
            "neurosush.structure.layer.Layer",
            "neurosush.structure.spec.BehaviorSpec",
            "neurosush.structure.spec.ColumnSpec",
            "neurosush.structure.spec.GroupSpec",
            "neurosush.structure.spec.LayerSpec",
            "neurosush.structure.spec.SynapseSpec",
        ),
    ),
    "W04": PlateSpec(
        id="W04",
        title_en="Batch versus independent: the multiverse",
        title_fa="دسته در برابر مستقل: چندجهان",
        wave=4,
        flagship=False,
        covers=(),
    ),
    "W05": PlateSpec(
        id="W05",
        title_en="Checkpoints: fork and resume on a timeline",
        title_fa="چک‌پوینت: خط زمانی انشعاب و ازسرگیری",
        wave=4,
        flagship=False,
        covers=(),
    ),
    "E01": PlateSpec(
        id="E01",
        title_en="The Score: every behavior a note",
        title_fa="پارتیتور: هر رفتار یک نت",
        wave=2,
        flagship=False,
        covers=(),
    ),
    "E02": PlateSpec(
        id="E02",
        title_en="The tensor lens",
        title_fa="عدسی تانسور",
        wave=4,
        flagship=False,
        covers=(),
    ),
    "E03": PlateSpec(
        id="E03",
        title_en="Eager versus CUDA graph versus compiled: the kernel timeline",
        title_fa="اجرای مشتاق در برابر CUDA Graph و کامپایل: خط زمانی هسته‌ها",
        wave=4,
        flagship=False,
        covers=("neurosush.core.graph.GraphStepper", "neurosush.core.compiled.CompiledStepper"),
    ),
    "E04": PlateSpec(
        id="E04",
        title_en="Reproducibility: same seed, bitwise diff",
        title_fa="بازتولیدپذیری: بذر یکسان، مقایسه بیتی",
        wave=4,
        flagship=False,
        covers=(),
    ),
    "H01": PlateSpec(
        id="H01",
        title_en="SDR overlap, union and the false match",
        title_fa="همپوشانی و اجتماع SDR و تطبیق نادرست",
        wave=3,
        flagship=False,
        covers=(),
    ),
    "H02": PlateSpec(
        id="H02",
        title_en="The encoders",
        title_fa="رمزگذارها",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.htm.encoders.ScalarEncoder",
            "neurosush.htm.encoders.RandomDistributedScalarEncoder",
            "neurosush.htm.encoders.CategoryEncoder",
        ),
    ),
    "H03": PlateSpec(
        id="H03",
        title_en="The spatial pooler",
        title_fa="Spatial Pooler: تجمیع مکانی رقابتی",
        wave=4,
        flagship=False,
        covers=("neurosush.htm.spatial_pooler.SpatialPooler",),
    ),
    "H04": PlateSpec(
        id="H04",
        title_en="Temporal memory: cells, segments, burst versus predict",
        title_fa="حافظه زمانی (Temporal Memory): سلول‌ها و سگمنت‌ها، بورست در برابر پیش‌بینی",
        wave=4,
        flagship=True,
        covers=(
            "neurosush.htm.temporal_memory.TemporalMemory",
            "neurosush.htm.classifier.SDRClassifier",
        ),
    ),
    "H05": PlateSpec(
        id="H05",
        title_en="Grid cells on a torus: path integration",
        title_fa="سلول‌های شبکه‌ای روی چنبره: انتگرال‌گیری مسیر",
        wave=4,
        flagship=False,
        covers=(
            "neurosush.htm.grid_cells.GridCellModule",
            "neurosush.htm.grid_cells.GridCellModules",
        ),
    ),
    "H06": PlateSpec(
        id="H06",
        title_en="Active dendrites: context gating",
        title_fa="دندریت‌های فعال و دروازه‌زنی زمینه",
        wave=4,
        flagship=False,
        covers=("neurosush.htm.active_dendrites.ActiveDendrites",),
    ),
    "H07": PlateSpec(
        id="H07",
        title_en="The voting columns",
        title_fa="ستون‌های رأی‌دهنده",
        wave=4,
        flagship=False,
        covers=(
            "neurosush.htm.objects.ObjectLibrary",
            "neurosush.htm.objects.SensorColumn",
            "neurosush.htm.objects.ColumnEnsemble",
        ),
    ),
    "H08": PlateSpec(
        id="H08",
        title_en="The spiking temporal memory: plateau and race",
        title_fa="حافظه زمانی اسپایکی: صفحه و مسابقه",
        wave=4,
        flagship=False,
        covers=(
            "neurosush.synapses.segment_learning.SegmentLearning",
            "neurosush.structure.sequence.Neuron",
            "neurosush.structure.sequence.SequenceTiming",
        ),
    ),
    "R01": PlateSpec(
        id="R01",
        title_en="Predictive coding: prediction down, error up, free energy",
        title_fa="کدگذاری پیش‌بین: پیش‌بینی به پایین، خطا به بالا، انرژی آزاد",
        wave=4,
        flagship=False,
        covers=("neurosush.predictive_coding.PredictiveCodingNetwork",),
    ),
    "R02": PlateSpec(
        id="R02",
        title_en="Predictive coding converges to backpropagation",
        title_fa="همگرایی کدگذاری پیش‌بین به پس‌انتشار",
        wave=4,
        flagship=False,
        covers=(),
    ),
    "X01": PlateSpec(
        id="X01",
        title_en="Poisson and latency encoders",
        title_fa="رمزگذار پواسون و رمزگذار تأخیری",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.neurons.inputs.SpikeInput",
            "neurosush.neurons.inputs.PoissonInput",
            "neurosush.neurons.inputs.PoissonDrive",
            "neurosush.neurons.inputs.CorrelatedPoissonInput",
        ),
    ),
    "X02": PlateSpec(
        id="X02",
        title_en="DoG and Gabor filters",
        title_fa="پالایه‌های DoG و گابور",
        wave=3,
        flagship=False,
        covers=(
            "neurosush.transforms.FilterBank",
            "neurosush.transforms.GridCrop",
            "neurosush.transforms.GridKeep",
            "neurosush.transforms.GridErase",
        ),
    ),
    "X03": PlateSpec(
        id="X03",
        title_en="Spike frames",
        title_fa="قاب‌های اسپایک",
        wave=4,
        flagship=False,
        covers=(),
    ),
}

# Classes the plan adds to the library for the plates; they do not exist yet, so they are
# checked only for syntax, not importability.
PLANNED: tuple[str, ...] = (
    "neurosush.neurons.hodgkin_huxley.HodgkinHuxley",
    "neurosush.neurons.hodgkin_huxley.ChannelNoise",
)

# Public classes no plate covers, each with a one-sentence justification.
EXEMPT: dict[str, str] = {
    "neurosush.core.behavior.Behavior": (
        "The base class of all behaviors; the plates teach its subclasses, not the base."
    ),
    "neurosush.recording.Recorder": (
        "Pure plumbing: mirrors attributes of its host around steps; no plate is about it."
    ),
    "neurosush.recording.SpikeCounter": (
        "Pure plumbing that counts spikes per neuron; every data plate uses it indirectly."
    ),
    "neurosush.synapses.traces.SpikeGather": (
        "Pure plumbing that copies source spikes into pre_spike for every synapse group."
    ),
}


def public_models() -> list[str]:
    """Return the fully qualified names of the public classes the plates must cover.

    Walks every submodule of ``neurosush`` with :func:`pkgutil.walk_packages`, skipping
    ``neurosush.viz``, and keeps the classes defined in each module (never re-exports) that
    either subclass :class:`~neurosush.core.behavior.Behavior` or live in the HTM,
    predictive-coding, structure, encoding, filter and transform modules or are one of the
    named engine classes. Underscore-prefixed names are skipped, and names in
    :data:`EXEMPT` are left out.

    Returns:
        Sorted fully qualified public class names.
    """
    names: list[str] = []
    for module_info in pkgutil.walk_packages(neurosush.__path__, "neurosush."):
        module_name = module_info.name
        if module_name == "neurosush.viz" or module_name.startswith("neurosush.viz."):
            continue
        module = importlib.import_module(module_name)
        non_behavior_scope = module_name.startswith(_NON_BEHAVIOR_MODULES)
        for attr_name, obj in sorted(vars(module).items()):
            if attr_name.startswith("_") or not inspect.isclass(obj):
                continue
            if obj.__module__ != module.__name__:
                continue
            qualified = f"{module.__name__}.{attr_name}"
            is_behavior = issubclass(obj, Behavior)
            if not (is_behavior or non_behavior_scope or qualified in _NAMED_CLASSES):
                continue
            if qualified not in EXEMPT:
                names.append(qualified)
    return sorted(names)


def uncovered() -> list[str]:
    """Return the public models no plate covers and no exemption justifies.

    Returns:
        Sorted names from :func:`public_models` that appear in no plate's ``covers``.
    """
    covered = {name for plate in PLATES.values() for name in plate.covers}
    return [name for name in public_models() if name not in covered and name not in EXEMPT]
