"""Device support check and speed benchmark for TEM and SpikingTEM models.

Times training updates across available devices (cpu, mps, cuda).
Writes JSON and prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from time import monotonic

import torch
import torch.nn.functional as F

from neurosush.spatial import GraphWorld, Walk
from neurosush.spatial.batch import batched_random_walks
from neurosush.spatial.controls import seen_targets
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.loss import total_loss
from neurosush.tem.spiking import SpikingTEM, SpikingTEMConfig
from neurosush.tem.train import _row, _worlds


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _get_peak_memory(device: str) -> int | None:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device)
    if device.startswith("mps"):
        try:
            return torch.mps.current_allocated_memory()
        except Exception:
            return None
    return None


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _benchmark_tem(device: str, batch: int, bptt: int, steps: int, warmup: int, rep: int) -> float:
    config = TEMConfig(batch=batch, bptt=bptt)
    generator = torch.Generator(device=device).manual_seed(42 + rep)
    model = TEM(config, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)

    total_steps = (warmup + steps) * bptt
    worlds = _worlds(batch, config, 42 + rep)
    walk = batched_random_walks(worlds, total_steps + 1, stay=False, valid_only=True)
    mask = torch.stack([seen_targets(_row(walk, b)) for b in range(batch)]).to(device)

    model.train()
    state = model.reset(batch)
    for update in range(warmup):
        begin = 1 + update * bptt
        optimizer.zero_grad()
        losses: list[torch.Tensor] = []
        for t in range(begin, begin + bptt):
            if t >= walk.observations.shape[1]:
                break
            obs = walk.observations[:, t].to(device)
            action = walk.executed_actions[:, t - 1].to(device)
            output, state = model(obs, action, state, decay=0.9, rate=0.2)
            loss, _ = total_loss(output, obs, config, iteration=0, mask=mask[:, t - 1])
            losses.append(loss)
        if losses:
            loss_val = torch.stack(losses).mean()
            loss_val.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            state = state.detach()

    _sync(device)
    start_time = monotonic()

    for update in range(steps):
        begin = 1 + (warmup + update) * bptt
        optimizer.zero_grad()
        losses = []
        for t in range(begin, begin + bptt):
            if t >= walk.observations.shape[1]:
                break
            obs = walk.observations[:, t].to(device)
            action = walk.executed_actions[:, t - 1].to(device)
            output, state = model(obs, action, state, decay=0.9, rate=0.2)
            loss, _ = total_loss(output, obs, config, iteration=0, mask=mask[:, t - 1])
            losses.append(loss)
        if losses:
            loss_val = torch.stack(losses).mean()
            loss_val.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            state = state.detach()

    _sync(device)
    return steps / (monotonic() - start_time)


def _benchmark_spiking(device: str, bptt: int, steps: int, warmup: int, rep: int) -> float:
    config = SpikingTEMConfig(sensory=20, units=64)
    model = SpikingTEM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    world = GraphWorld(width=5, sensory=config.sensory, seed=42 + rep)
    cpu_walk = world.random_walk((warmup + steps) * bptt + 1)
    walk = Walk(
        nodes=cpu_walk.nodes.to(device),
        observations=cpu_walk.observations.to(device),
        actions=cpu_walk.actions.to(device),
        novel_edge_to_seen=cpu_walk.novel_edge_to_seen.to(device),
        fraction_seen=cpu_walk.fraction_seen.to(device),
        executed_actions=cpu_walk.executed_actions.to(device)
        if cpu_walk.executed_actions is not None
        else None,
    )

    model.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        state = model.initial_state(m_init_std=0.05)
        sub_walk = Walk(
            nodes=walk.nodes[: bptt + 1],
            observations=walk.observations[: bptt + 1],
            actions=walk.actions[:bptt],
            novel_edge_to_seen=walk.novel_edge_to_seen[:bptt],
            fraction_seen=walk.fraction_seen[:bptt],
            executed_actions=walk.executed_actions[:bptt]
            if walk.executed_actions is not None
            else None,
        )
        logits, rates_dict, _ = model(sub_walk, state)
        targets = walk.observations[1 : bptt + 1]
        loss = F.cross_entropy(logits, targets) + 1e-3 * (
            rates_dict["mecii"].mean() + rates_dict["mecv"].mean() + rates_dict["ca1"].mean()
        )
        loss.backward()
        optimizer.step()

    _sync(device)
    start_time = monotonic()

    for s in range(steps):
        optimizer.zero_grad()
        state = model.initial_state(m_init_std=0.05)
        chunk_start = s * bptt
        chunk_end = (s + 1) * bptt
        sub_walk = Walk(
            nodes=walk.nodes[chunk_start : chunk_end + 1],
            observations=walk.observations[chunk_start : chunk_end + 1],
            actions=walk.actions[chunk_start:chunk_end],
            novel_edge_to_seen=walk.novel_edge_to_seen[chunk_start:chunk_end],
            fraction_seen=walk.fraction_seen[chunk_start:chunk_end],
            executed_actions=walk.executed_actions[chunk_start:chunk_end]
            if walk.executed_actions is not None
            else None,
        )
        logits, rates_dict, _ = model(sub_walk, state)
        targets = walk.observations[chunk_start + 1 : chunk_end + 1]
        loss = F.cross_entropy(logits, targets) + 1e-3 * (
            rates_dict["mecii"].mean() + rates_dict["mecv"].mean() + rates_dict["ca1"].mean()
        )
        loss.backward()
        optimizer.step()

    _sync(device)
    return steps / (monotonic() - start_time)


def benchmark_model(
    model_type: str,
    device: str,
    batch: int,
    bptt: int,
    steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Benchmark a model on a device for given warmup and timed steps."""
    rates = []
    peak_memories = []

    for rep in range(repeats):
        _reset_peak_memory(device)
        if model_type == "tem":
            updates_per_s = _benchmark_tem(device, batch, bptt, steps, warmup, rep)
        elif model_type == "spiking":
            updates_per_s = _benchmark_spiking(device, bptt, steps, warmup, rep)
        else:
            raise ValueError(f"unknown model_type={model_type}")

        rates.append(updates_per_s)
        mem = _get_peak_memory(device)
        if mem is not None:
            peak_memories.append(mem)

    median_rate = statistics.median(rates)
    spread = (max(rates) - min(rates)) / 2.0 if len(rates) > 1 else 0.0
    median_mem = statistics.median(peak_memories) if peak_memories else None

    return {
        "updates_per_s": median_rate,
        "spread": spread,
        "runs": rates,
        "peak_memory_bytes": median_mem,
    }


def main() -> None:
    """Run speed benchmarks across devices and models, write JSON, and print table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", default="cpu,mps,cuda")
    parser.add_argument("--models", default="both", choices=["tem", "spiking", "both"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--bptt", type=int, default=20)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="tem_speed.json")
    args = parser.parse_args()

    device_list = [d.strip() for d in args.devices.split(",")]
    models_to_run = ["tem", "spiking"] if args.models == "both" else [args.models]

    rows = []
    for device_name in device_list:
        available = not (
            (device_name.startswith("cuda") and not torch.cuda.is_available())
            or (device_name.startswith("mps") and not torch.backends.mps.is_available())
        )

        if not available:
            for model_type in models_to_run:
                rows.append(
                    {
                        "device": device_name,
                        "model": model_type,
                        "error": f"Device {device_name} not available",
                    }
                )
            continue

        for model_type in models_to_run:
            try:
                result = benchmark_model(
                    model_type,
                    device_name,
                    args.batch,
                    args.bptt,
                    args.steps,
                    args.warmup,
                    args.repeats,
                )
                rows.append(
                    {
                        "device": device_name,
                        "model": model_type,
                        **result,
                    }
                )
            except Exception as error:
                if device_name.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rows.append(
                    {
                        "device": device_name,
                        "model": model_type,
                        "error": str(error)[:300],
                    }
                )

    mps_avail = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    machine = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_avail,
    }

    output_data = {
        "machine": machine,
        "batch": args.batch,
        "bptt": args.bptt,
        "steps": args.steps,
        "repeats": args.repeats,
        "results": rows,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")

    print("| device | model | updates/s | spread | peak memory (MB) |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        head = f"| {row['device']} | {row['model']} |"
        if "error" in row:
            print(f"{head} error: {row['error']} | | |")
        else:
            mem = (
                f"{row['peak_memory_bytes'] / 1024 / 1024:.1f}"
                if row["peak_memory_bytes"] is not None
                else "N/A"
            )
            print(f"{head} {row['updates_per_s']:,.2f} | ±{row['spread']:,.2f} | {mem} |")


if __name__ == "__main__":
    main()
