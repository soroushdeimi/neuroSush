"""Configuration settings for neuroSush Studio server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Server configuration settings."""

    data_dir: Path = Path(
        os.environ.get(
            "NS_DATA",
            "/data" if Path("/.dockerenv").exists() else ".studio-data",
        )
    )
    max_jobs: int = int(os.environ.get("NS_MAX_JOBS", "4"))
    max_steps: int = int(os.environ.get("NS_MAX_STEPS", "50000"))
    auth_token: str | None = os.environ.get("NS_AUTH_TOKEN", None)
    allowed_origins: str = os.environ.get("NS_ALLOWED_ORIGINS", "*")
    web_root: Path = Path(os.environ.get("NS_WEB_ROOT", Path(__file__).parents[2] / "web" / "dist"))


config = Config()
