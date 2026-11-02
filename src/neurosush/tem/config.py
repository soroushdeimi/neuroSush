"""Validated laptop-scale TEM settings and iteration schedules."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite
from typing import Literal

from neurosush.schedule import linear_ramp, step_decay


@dataclass(frozen=True)
class TEMConfig:
    """TEM dimensions and schedules; ramp lengths count optimizer updates.

    Deviations:
        Modules have independent memories without cross-module hierarchy. All g units
        participate in binding. The decoder is linear and shared across modules.
        Defaults reduce population sizes and sensory labels for laptop experiments.
    """

    sensory: int = 20
    compressed: int = 10
    modules: tuple[int, ...] = (12, 12, 8)
    hidden: int = 20
    batch: int = 16
    bptt: int = 20
    walk_len: int = 200
    widths: tuple[int, ...] = (4, 5)
    lr_max: float = 9.4e-4
    lr_min: float = 8e-5
    lr_decay: float = 0.5
    lr_every: int = 4000
    memory_decay: float = 0.9999
    memory_rate: float = 0.5
    decay_ramp: int = 200
    rate_ramp: int = 16000
    latent_ramp: int = 2000
    iterations: int = 5
    retention: float = 0.8
    clip: float = 1.0
    initial_std: float = 0.5
    weight_x_gen: float = 1.0
    weight_x_g: float = 1.0
    weight_x_p: float = 1.0
    weight_p: float = 1.0
    weight_g: float = 1.0
    reg_g: float = 0.01
    reg_p: float = 0.02
    transition: Literal["matrix", "rotation"] = "matrix"
    inference_memory_loss: bool = False
    calibrated_reductions: bool = False
    bounded_attractor_activation: bool = False
    reliable_fusion: bool = False
    correction_ramp: int = 2000
    reg_p_horizon: int = 4000
    reg_g_horizon: int = 40000000
    sigma_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.transition not in ("matrix", "rotation"):
            raise ValueError(f"transition must be matrix or rotation, got {self.transition}")
        for name in (
            "sensory",
            "compressed",
            "hidden",
            "batch",
            "bptt",
            "walk_len",
            "lr_every",
            "decay_ramp",
            "rate_ramp",
            "latent_ramp",
            "iterations",
            "correction_ramp",
            "reg_p_horizon",
            "reg_g_horizon",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value}")
        for name in ("modules", "widths"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values
            ):
                raise ValueError(f"{name} must contain positive integers, got {values}")
        if self.transition == "rotation" and any(n % 2 for n in self.modules):
            raise ValueError(f"rotation modules must be even, got {self.modules}")
        if any(width < 2 or width > 8 for width in self.widths):
            raise ValueError(f"widths must be in [2, 8], got {self.widths}")
        if self.compressed < 2 or comb(self.compressed, 2) < self.sensory:
            raise ValueError(
                f"compressed={self.compressed} cannot encode "
                f"sensory={self.sensory} uniquely two-hot"
            )
        for name in (
            "lr_max",
            "lr_min",
            "lr_decay",
            "memory_decay",
            "memory_rate",
            "retention",
            "clip",
            "initial_std",
            "weight_x_gen",
            "weight_x_g",
            "weight_x_p",
            "weight_p",
            "weight_g",
            "reg_g",
            "reg_p",
            "sigma_floor",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")
        for name in ("lr_max", "lr_min", "lr_decay", "clip", "initial_std", "sigma_floor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("lr_decay", "memory_decay", "retention"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1, got {getattr(self, name)}")
        if self.lr_min > self.lr_max:
            raise ValueError(f"lr_min={self.lr_min} exceeds lr_max={self.lr_max}")

    def schedules(self, iteration: int) -> tuple[float, float, float, float]:
        """Return learning rate, memory decay, memory rate, and latent loss scale."""
        if iteration < 0:
            raise ValueError(f"iteration must be nonnegative, got {iteration}")
        return (
            max(self.lr_min, step_decay(iteration, self.lr_max, self.lr_decay, self.lr_every)),
            linear_ramp(iteration, self.decay_ramp, 0.0, self.memory_decay),
            linear_ramp(iteration, self.rate_ramp, 0.0, self.memory_rate),
            linear_ramp(iteration, self.latent_ramp, 0.0, 1.0),
        )
