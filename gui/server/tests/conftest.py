"""Pytest configuration and autouse fixtures for server tests."""

from __future__ import annotations

import pytest
from studio.config import Config
import studio.config as config_module
import studio.projects.db as projects_db


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Automatically isolate data_dir to a temporary directory for every test."""
    isolated_config = Config(data_dir=tmp_path)
    monkeypatch.setattr(config_module, "config", isolated_config)
    monkeypatch.setattr(projects_db, "config", isolated_config)
