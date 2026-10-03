import pytest
import torch

from neurosush.core.compiled import MIN_TORCH_VERSION, torch_supports_compile


def pytest_collection_modifyitems(config, items):
    """Skip ``gpu`` tests without CUDA and ``needs_compile`` tests on a too-old torch."""
    no_cuda = pytest.mark.skip(reason="needs a CUDA device")
    old_torch = pytest.mark.skip(
        reason=f"CompiledStepper needs torch >= {MIN_TORCH_VERSION} (found {torch.__version__})"
    )
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(no_cuda)
        if "needs_compile" in item.keywords and not torch_supports_compile():
            item.add_marker(old_torch)
