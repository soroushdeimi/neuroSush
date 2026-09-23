"""Base class for all behaviors."""

from __future__ import annotations

from typing import Any, ClassVar


class Behavior:
    """Base class for all behaviors."""

    order: ClassVar[int]
    enabled: bool = True

    def initialize(self, host: Any) -> None:
        """Initialize the behavior."""

    def forward(self, host: Any) -> None:
        """Execute the behavior's logic."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
