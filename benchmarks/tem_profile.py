"""Profiling throughput and operator breakdown for TEM training updates."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Protocol

import torch
from torch.profiler import ProfilerActivity, profile

from neurosush.spatial.batch import batched_random_walks
from neurosush.spatial.controls import seen_targets
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.loss import total_loss
from neurosush.tem.train import _row, _worlds


class _Event(Protocol):
    key: str
    count: int


def _event_counts(events: Iterable[_Event]) -> tuple[int, int, int]:
    aten = autograd = optimizer = 0
    for event in events:
        if event.key.startswith("aten::"):
            aten += event.count
        elif event.key.startswith(("autograd::", "torch::autograd::")):
            autograd += event.count
        elif event.key.startswith("Optimizer."):
            optimizer += event.count
    return aten, autograd, optimizer


def profile_device(device: str) -> None:
    """Profile 3 training updates of TEM on the given device."""
    print(f"\n--- Profiling TEM on {device} ---")
    config = TEMConfig(batch=16, bptt=20)
    generator = torch.Generator(device=device).manual_seed(42)
    model = TEM(config, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)

    worlds = _worlds(16, config, seed=42)
    walk = batched_random_walks(worlds, 61, stay=False, valid_only=True)
    mask = torch.stack([seen_targets(_row(walk, b)) for b in range(16)]).to(device)

    t_fwd_total = 0.0
    t_bwd_total = 0.0
    t_opt_total = 0.0

    activities = [ProfilerActivity.CPU]

    # Warmup 1 update
    model.train()
    state = model.reset(16)
    optimizer.zero_grad()
    losses = []
    for t in range(1, 21):
        obs = walk.observations[:, t].to(device)
        action = walk.executed_actions[:, t - 1].to(device)
        output, state = model(obs, action, state, decay=0.9, rate=0.2)
        loss, _ = total_loss(output, obs, config, iteration=0, mask=mask[:, t - 1])
        losses.append(loss)
    if losses:
        loss_val = torch.stack(losses).mean()
        loss_val.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    # Profile 3 updates
    start_profile_wall = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        for _update in range(3):
            state = model.reset(16)
            optimizer.zero_grad()
            losses = []

            t0 = time.perf_counter()
            for t in range(1, 21):
                obs = walk.observations[:, t].to(device)
                action = walk.executed_actions[:, t - 1].to(device)
                output, state = model(obs, action, state, decay=0.9, rate=0.2)
                loss, _ = total_loss(output, obs, config, iteration=0, mask=mask[:, t - 1])
                losses.append(loss)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elif device.startswith("mps"):
                torch.mps.synchronize()
            t1 = time.perf_counter()
            t_fwd_total += t1 - t0

            loss_val = torch.stack(losses).mean()
            t2 = time.perf_counter()
            loss_val.backward()  # type: ignore[no-untyped-call]
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elif device.startswith("mps"):
                torch.mps.synchronize()
            t3 = time.perf_counter()
            t_bwd_total += t3 - t2

            t4 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            elif device.startswith("mps"):
                torch.mps.synchronize()
            t5 = time.perf_counter()
            t_opt_total += t5 - t4

    total_wall = time.perf_counter() - start_profile_wall

    events = prof.key_averages()
    total_aten_ops, autograd_events, optimizer_events = _event_counts(events)
    ops_per_update = total_aten_ops / 3.0
    print(f"ATen: {total_aten_ops}; autograd: {autograd_events}; optimizer: {optimizer_events}")

    print("\nTop 12 Ops by Self CPU Time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))

    print("\nTop 12 Ops by Call Count:")
    print(prof.key_averages().table(sort_by="count", row_limit=12))

    print("\nWall time split (3 updates):")
    print(f"  Forward:  {t_fwd_total:.4f}s")
    print(f"  Backward: {t_bwd_total:.4f}s")
    print(f"  Optimizer: {t_opt_total:.4f}s")
    print(f"  Total wall: {total_wall:.4f}s")

    hint_str = (
        f"python-loop overhead hint: {ops_per_update / max(total_wall / 3.0, 1e-6):,.2f} "
        f"aten ops/update-second (total ops per update={ops_per_update:,.1f})"
    )
    print(hint_str)


def main() -> None:
    """Run profiler on CPU and MPS (if available)."""
    profile_device("cpu")
    if torch.backends.mps.is_available():
        profile_device("mps")


if __name__ == "__main__":
    main()
