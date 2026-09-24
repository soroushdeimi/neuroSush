import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``gpu`` when no CUDA device is available."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
