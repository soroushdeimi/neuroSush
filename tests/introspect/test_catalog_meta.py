from __future__ import annotations

import json
import subprocess
import sys

from neurosush._catalog_meta import META
from neurosush.introspect import catalog
from neurosush.structure.spec import registered


def test_meta_covers_registry():
    """Every registered behavior has metadata in META."""
    assert set(META) >= set(registered())


def test_params_match_descriptor():
    """Every parameter in META.params is a real descriptor parameter."""
    cat = catalog()
    for behavior in cat["behaviors"]:
        name = behavior["name"]
        if name in META:
            meta_params = META[name]["params"]
            desc_param_names = {p["name"] for p in behavior["params"]}
            assert set(meta_params) == desc_param_names, (
                f"{name}: META params {set(meta_params)} != desc params {desc_param_names}"
            )


def test_meta_json_serializable():
    """META can be serialized to JSON without error."""
    serialized = json.dumps(META)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert len(deserialized) == len(META)


def test_fingerprint_stable_across_subprocesses():
    """fingerprint() produces identical output across separate python subprocesses."""
    code = "import neurosush.introspect as i; print(i.fingerprint())"
    cmd = [sys.executable, "-c", code]

    res1 = subprocess.run(
        cmd, capture_output=True, text=True, check=True, env={"PYTHONPATH": "src"}
    )
    res2 = subprocess.run(
        cmd, capture_output=True, text=True, check=True, env={"PYTHONPATH": "src"}
    )

    fp1 = res1.stdout.strip()
    fp2 = res2.stdout.strip()
    assert len(fp1) == 64
    assert fp1 == fp2


def test_categories_valid():
    """All categories are within the allowed set."""
    allowed_categories = {
        "neuron_model",
        "input",
        "current",
        "connectivity",
        "plasticity",
        "constraint",
        "competition",
        "homeostasis",
        "dendrite",
        "axon",
        "modulation",
        "recording",
    }
    for name, meta in META.items():
        assert meta["category"] in allowed_categories, (
            f"{name} has invalid category {meta['category']}"
        )


def test_hosts_valid():
    """All hosts are within the allowed set."""
    allowed_hosts = {"neurons", "synapse", "network"}
    for name, meta in META.items():
        assert meta["host"] in allowed_hosts, f"{name} has invalid host {meta['host']}"
