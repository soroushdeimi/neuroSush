"""Tests for neurosush.viz.trace."""

from __future__ import annotations

import base64
import json
import os
import struct
import subprocess
import sys

import pytest
import torch

from neurosush import checkpoint
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire
from neurosush.recording import Recorder
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather
from neurosush.viz.trace import Trace, TraceRecorder, pack_array, unpack_array


class TestPackUnpack:
    @pytest.mark.parametrize(
        ("source_dtype", "expected_dtype"),
        [
            (torch.float16, torch.float32),
            (torch.float32, torch.float32),
            (torch.float64, torch.float32),
            (torch.bool, torch.bool),
            (torch.int8, torch.int32),
            (torch.int16, torch.int32),
            (torch.int32, torch.int32),
            (torch.int64, torch.int32),
        ],
    )
    def test_every_allowed_dtype_branch(
        self, source_dtype: torch.dtype, expected_dtype: torch.dtype
    ) -> None:
        if source_dtype == torch.bool:
            tensor = torch.tensor([[True, False], [False, True]], dtype=source_dtype)
        else:
            tensor = torch.tensor([[1, 2], [3, 4]], dtype=source_dtype)

        packed = pack_array(tensor)
        assert isinstance(packed, dict)
        assert set(packed.keys()) == {"dtype", "shape", "data"}
        assert packed["shape"] == [2, 2]
        assert isinstance(packed["data"], str)

        unpacked = unpack_array(packed)
        assert unpacked.dtype == expected_dtype
        assert unpacked.shape == tensor.shape
        if source_dtype == torch.float64:
            assert torch.allclose(unpacked, tensor.to(torch.float32))
        else:
            assert torch.equal(unpacked, tensor.to(expected_dtype))

    def test_bool_exact_round_trip(self) -> None:
        t = torch.tensor([True, False, True, True], dtype=torch.bool)
        packed = pack_array(t)
        assert packed["dtype"] == "uint8"
        unpacked = unpack_array(packed)
        assert unpacked.dtype == torch.bool
        assert torch.equal(unpacked, t)

    def test_empty_tensors_round_trip(self) -> None:
        for shape in [(0,), (0, 5), (3, 0), (2, 0, 4)]:
            t = torch.zeros(shape, dtype=torch.float32)
            packed = pack_array(t)
            assert packed["shape"] == list(shape)
            unpacked = unpack_array(packed)
            assert unpacked.shape == torch.Size(shape)
            assert unpacked.dtype == torch.float32
            assert torch.equal(unpacked, t)

    def test_zero_dim_tensor_round_trip(self) -> None:
        t = torch.tensor(42.5, dtype=torch.float32)
        packed = pack_array(t)
        assert packed["shape"] == []
        unpacked = unpack_array(packed)
        assert unpacked.shape == torch.Size([])
        assert unpacked.dtype == torch.float32
        assert torch.equal(unpacked, t)

    def test_little_endian_independence(self) -> None:
        t_int = torch.tensor([1, 256, 65536], dtype=torch.int32)
        packed_int = pack_array(t_int)
        raw_int = base64.b64decode(packed_int["data"])
        expected_int = struct.pack("<iii", 1, 256, 65536)
        assert raw_int == expected_int

        t_float = torch.tensor([1.0, -2.5], dtype=torch.float32)
        packed_float = pack_array(t_float)
        raw_float = base64.b64decode(packed_float["data"])
        expected_float = struct.pack("<ff", 1.0, -2.5)
        assert raw_float == expected_float

    def test_int64_overflow_raises_value_error(self) -> None:
        too_high = torch.tensor([2**31], dtype=torch.int64)
        with pytest.raises(ValueError, match=r"int64.*range"):
            pack_array(too_high)

        too_low = torch.tensor([-(2**31) - 1], dtype=torch.int64)
        with pytest.raises(ValueError, match=r"int64.*range"):
            pack_array(too_low)

    def test_unsupported_dtype_raises_type_error(self) -> None:
        for bad_t in [
            torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
            torch.tensor([1.0], dtype=torch.bfloat16),
            torch.tensor([1], dtype=torch.uint8),
        ]:
            with pytest.raises(TypeError, match=str(bad_t.dtype)):
                pack_array(bad_t)

    def test_unpack_unknown_dtype_raises(self) -> None:
        bad_packed = {"dtype": "float128", "shape": [1], "data": ""}
        with pytest.raises(TypeError, match="float128"):
            unpack_array(bad_packed)


class TestTrace:
    def test_length_mismatch_raises_in_post_init(self) -> None:
        with pytest.raises(ValueError, match=r"series 'v'.*length 3.*steps.*2"):
            Trace(
                meta={},
                steps=[1, 2],
                series={"v": torch.zeros((3, 4), dtype=torch.float32)},
            )

    def test_non_cpu_series_raises(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        with pytest.raises(ValueError, match=r"series 'v'.*CPU"):
            Trace(
                meta={},
                steps=[1],
                series={"v": torch.zeros((1, 2), device="cuda")},
            )

    def test_to_dict_from_dict_round_trip(self) -> None:
        trace = Trace(
            meta={"dt": 0.5, "experiment": "test"},
            steps=[1, 2, 3],
            series={
                "v": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
                "spikes": torch.tensor([[True, False], [False, True], [False, False]]),
            },
        )
        d = trace.to_dict()
        assert d["meta"] == {"dt": 0.5, "experiment": "test"}
        assert d["steps"] == [1, 2, 3]
        assert "v" in d["series"]
        assert "spikes" in d["series"]

        restored = Trace.from_dict(d)
        assert restored.meta == trace.meta
        assert restored.steps == trace.steps
        assert torch.equal(restored.series["v"], trace.series["v"])
        assert torch.equal(restored.series["spikes"], trace.series["spikes"])

    def test_save_load_round_trip_is_valid_json_no_pickle(self, tmp_path) -> None:
        trace = Trace(
            meta={"dt": 1.0, "net": "simple"},
            steps=[0, 10],
            series={"v": torch.tensor([[-65.0], [-55.0]], dtype=torch.float32)},
        )
        file_path = tmp_path / "trace.json"
        trace.save(file_path)

        with open(file_path, encoding="utf-8") as f:
            raw_json = json.load(f)
        assert raw_json["meta"] == {"dt": 1.0, "net": "simple"}
        assert raw_json["steps"] == [0, 10]

        loaded = Trace.load(file_path)
        assert loaded.meta == trace.meta
        assert loaded.steps == trace.steps
        assert torch.equal(loaded.series["v"], trace.series["v"])

    def test_window(self) -> None:
        steps = [10, 20, 30, 40, 50]
        v = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
        trace = Trace(meta={"dt": 1.0}, steps=steps, series={"v": v})

        sub = trace.window(20, 45)
        assert sub.steps == [20, 30, 40]
        assert torch.equal(sub.series["v"], torch.tensor([[1.0], [2.0], [3.0]]))
        assert sub.meta == {"dt": 1.0}

        empty = trace.window(100, 200)
        assert empty.steps == []
        assert empty.series["v"].shape == (0, 1)

    def test_diff(self) -> None:
        t1 = Trace(
            meta={},
            steps=[1, 2],
            series={"v": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        )
        t2 = Trace(
            meta={},
            steps=[1, 2],
            series={"v": torch.tensor([[1.0, 2.5], [3.0, 3.2]])},
        )
        assert pytest.approx(t1.diff(t2, "v")) == 0.8

        with pytest.raises(KeyError, match=r"'missing'"):
            t1.diff(t2, "missing")

        t3 = Trace(
            meta={},
            steps=[1, 2],
            series={"other": torch.tensor([[1.0], [2.0]])},
        )
        with pytest.raises(KeyError, match=r"'v'"):
            t1.diff(t3, "v")

        t_shape_mismatch = Trace(
            meta={},
            steps=[1, 2],
            series={"v": torch.tensor([[1.0], [2.0]])},
        )
        with pytest.raises(ValueError, match=r"\(2, 2\).*\(2, 1\)"):
            t1.diff(t_shape_mismatch, "v")

    def test_events(self) -> None:
        trace = Trace(
            meta={},
            steps=[10, 20, 30, 40],
            series={
                "spikes": torch.tensor(
                    [
                        [False, False],
                        [True, False],
                        [False, False],
                        [False, True],
                    ]
                ),
                "intensity": torch.tensor(
                    [
                        [0.1, 0.2],
                        [0.6, 0.1],
                        [0.4, 0.3],
                        [0.0, 0.9],
                    ]
                ),
            },
        )
        assert trace.events("spikes") == [20, 40]
        assert trace.events("intensity", threshold=0.5) == [20, 40]
        assert trace.events("intensity", threshold=0.8) == [40]

        with pytest.raises(KeyError, match=r"'absent'"):
            trace.events("absent")

    def test_make_meta(self) -> None:
        net = Network(dt=0.5, seed=42)
        meta = Trace.make_meta(net, run_name="test_run")
        import neurosush

        assert meta["library"] == neurosush.__version__
        assert meta["torch"] == torch.__version__
        assert meta["dt"] == 0.5
        assert meta["seed"] is None  # Network does not store seed as an attribute
        assert meta["device"] == "cpu"
        assert meta["run_name"] == "test_run"


class TestTraceRecorder:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="give at least one attribute to record"):
            TraceRecorder(max_steps=10)

        with pytest.raises(ValueError, match="interval must be positive"):
            TraceRecorder("v", interval=0, max_steps=10)

        with pytest.raises(ValueError, match="max_steps must be positive"):
            TraceRecorder("v", max_steps=0)

    def test_missing_attribute_at_initialize_raises(self) -> None:
        net = Network()
        group = NeuronGroup(net, 5, [LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)])
        group.behaviors = net._attach(group, [TraceRecorder("missing_attr", max_steps=10)])
        msg = r"TraceRecorder on ng0: no attribute\(s\) \['missing_attr'\]"
        with pytest.raises(RuntimeError, match=msg):
            net.initialize()

    def test_values_equal_recorder_output_on_same_run(self) -> None:
        def build_net(rec_cls, **rec_kwargs):
            net = Network(seed=123)
            group = NeuronGroup(
                net,
                4,
                [
                    PoissonInput(rates=torch.tensor([0.2, 0.5, 0.1, 0.8])),
                    LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                    Fire(),
                    Axon(),
                    rec_cls("v", "spikes", **rec_kwargs),
                ],
                name="group",
            )
            return net, group

        ref_net, ref_group = build_net(Recorder)
        ref_rec = ref_group.behaviors[-1]
        ref_net.run(20)

        trace_net, trace_group = build_net(TraceRecorder, max_steps=20)
        trace_rec = trace_group.behaviors[-1]
        trace_net.run(20)

        trace = trace_rec.trace(tag="run1")
        assert trace.steps == ref_rec.steps
        assert torch.equal(trace.series["v"], ref_rec.get("v"))
        assert torch.equal(trace.series["spikes"], ref_rec.get("spikes"))
        assert trace.meta == {"tag": "run1"}

    def test_interval(self) -> None:
        net = Network(seed=0)
        group = NeuronGroup(
            net,
            2,
            [
                LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                TraceRecorder("v", interval=3, max_steps=5),
            ],
        )
        rec = group.behaviors[-1]
        net.run(9)
        trace = rec.trace()
        assert trace.steps == [3, 6, 9]
        assert trace.series["v"].shape == (3, 2)

    def test_overflow_raises_runtime_error(self) -> None:
        net = Network()
        NeuronGroup(
            net,
            2,
            [
                LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                TraceRecorder("v", max_steps=2),
            ],
        )
        net.run(2)
        with pytest.raises(RuntimeError, match=r"max_steps=2"):
            net.step()

    def test_reset_and_repr(self) -> None:
        rec = TraceRecorder("v", "spikes", interval=2, max_steps=50)
        assert repr(rec) == "TraceRecorder('v', 'spikes', interval=2, max_steps=50)"
        rec.steps = [1, 2]
        rec._count = 2
        rec.reset()
        assert rec.steps == []
        assert rec._count == 0

    def test_checkpoint_and_resume(self, tmp_path) -> None:
        def build_net():
            net = Network(seed=42)
            src = NeuronGroup(
                net,
                3,
                [PoissonInput(rates=torch.tensor([0.4, 0.4, 0.4])), Axon()],
                name="src",
            )
            dst = NeuronGroup(
                net,
                2,
                [
                    DendriteStructure(),
                    DendriteIntegration(),
                    LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                    Fire(),
                    Axon(),
                    TraceRecorder("v", max_steps=30),
                ],
                name="dst",
            )
            SynapseGroup(
                net,
                src,
                dst,
                [WeightInit(mode="uniform"), DenseInput(coef=2.0), SpikeGather()],
                name="syn",
            )
            return net

        net1 = build_net()
        net1.run(10)
        ckpt_path = tmp_path / "checkpoint.pt"
        checkpoint.save(net1, ckpt_path)

        net2 = build_net()
        checkpoint.load(net2, ckpt_path)
        net2.run(10)

        assert net2.iteration == 20
        rec2 = net2.groups[1].behaviors[-1]
        trace2 = rec2.trace()
        assert len(trace2.steps) == 10
        assert trace2.steps == list(range(11, 21))

    @pytest.mark.gpu
    def test_graph_stepper_matches_eager_on_cuda(self) -> None:
        def build_net():
            net = Network(device="cuda", seed=99)
            src = NeuronGroup(
                net,
                4,
                [PoissonInput(rates=torch.tensor([0.5, 0.5, 0.5, 0.5])), Axon()],
                name="src",
            )
            dst = NeuronGroup(
                net,
                2,
                [
                    DendriteStructure(),
                    DendriteIntegration(),
                    LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                    Fire(),
                    Axon(),
                    TraceRecorder("v", "spikes", max_steps=20),
                ],
                name="dst",
            )
            SynapseGroup(
                net,
                src,
                dst,
                [WeightInit(mode="uniform"), DenseInput(coef=2.0), SpikeGather()],
                name="syn",
            )
            return net

        eager_net = build_net()
        eager_net.run(20)
        eager_trace = eager_net.groups[1].behaviors[-1].trace()

        graphed_net = build_net()
        stepper = GraphStepper(graphed_net)
        stepper.run(20)
        graphed_trace = graphed_net.groups[1].behaviors[-1].trace()

        assert eager_trace.steps == graphed_trace.steps
        assert torch.equal(eager_trace.series["v"], graphed_trace.series["v"])
        assert torch.equal(eager_trace.series["spikes"], graphed_trace.series["spikes"])


def test_big_endian_byteorder_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "byteorder", "big")
    # On a big-endian host, float32 1.0 in memory has bytes 3f 80 00 00
    t = torch.tensor([0x3F, 0x80, 0x00, 0x00], dtype=torch.uint8).view(torch.float32)
    packed = pack_array(t)
    raw_bytes = base64.b64decode(packed["data"])
    assert raw_bytes == struct.pack("<f", 1.0)
    unpacked = unpack_array(packed)
    assert torch.equal(unpacked, t)


def test_numpy_unimportable_subprocess() -> None:
    code = (
        "import sys; "
        "sys.modules['numpy'] = None; "
        "from neurosush.viz.trace import pack_array, unpack_array; "
        "import torch; "
        "t = torch.tensor([1.0, 2.5], dtype=torch.float32); "
        "packed = pack_array(t); "
        "unpacked = unpack_array(packed); "
        "assert torch.equal(t, unpacked)"
    )
    env = dict(os.environ, PYTHONPATH="src")
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert proc.returncode == 0
