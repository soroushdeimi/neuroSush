"""Every Python block in README.md runs against the installed package."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.DOTALL)


def test_readme_has_code_examples():
    assert len(BLOCKS) >= 2


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_block_runs(index):
    code = compile(BLOCKS[index], f"README.md python block {index}", "exec")
    exec(code, {"__name__": "readme_example"})
