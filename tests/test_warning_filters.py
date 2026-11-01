import warnings

import pytest


def test_metal_skipping_warning_is_ignored():
    # Under pyproject.toml filterwarnings = ["error", ...],
    # this specific warning should be ignored and NOT raise an error once ignored.
    warnings.warn(
        "Skipping device Apple Paravirtual device that does not support "
        "Metal 2.0 (MPSDevice.mm:28)",
        UserWarning,
        stacklevel=2,
    )


def test_other_user_warning_still_errors():
    # Other UserWarnings should still be treated as errors by default ("error" filter).
    with pytest.raises(UserWarning, match="Some other unexpected warning"):
        warnings.warn("Some other unexpected warning", UserWarning, stacklevel=2)
