"""Spiking Tolman-Eichenbaum Machine (TEM) library module.

Deviations:
    No DG/CA3 inference circuit, no LECII/III, no Algorithm 1 buffering.
    MECV pathway reduced. Theta imposed so precession is not emergent.
    Populations use 64 units (default) or configurable units. STDP memory M is environment-local.
    Parameter values differ between paper(), authors_code() and default config (default uses
    lower threshold=0.5, higher tonic=0.5, and theta_scale=0.1 to maintain healthy spiking
    activity at initialization; paper() uses strict paper Table 1 values; authors_code() uses
    released code parameters).
    Theta scaling and tonic drive are named config fields (theta_scale, tonic).
    STDP memory resets per walk in training although the authors' code never resets.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from neurosush.neurons.oscillation import theta_current
from neurosush.neurons.sampling import bernoulli_group_means
from neurosush.neurons.surrogate import lif_step
from neurosush.spatial.graph_world import Walk
from neurosush.synapses.stdp_memory import stdp_memory_update


@dataclass(frozen=True)
class SpikingTEMConfig:
    """Configuration for SpikingTEM model and training.

    Shapes/Units:
        tau and dt in ms; theta_freq in Hz.
    """

    sensory: int = 20
    units: int = 64
    steps_per_action: int = 3
    use_theta: bool = True
    use_stdp: bool = True
    use_gain: bool = True
    tau: float = 10.0
    dt: float = 1.0
    v_rest: float = 0.0
    threshold: float = 0.5
    reset: float = 0.0
    theta_alpha: float = 2.0
    theta_freq: float = 8.0
    k: int = 2
    stdp_clip: float = 2.0
    a_plus: float = 0.05
    a_minus: float = 0.005
    sample: bool = False
    theta_scale: float = 0.1
    tonic: float = 0.5
    surrogate_width: float = 2.0
    surrogate_kind: str = "arctan"
    gain_init: float = 1.2

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sensory < 1:
            raise ValueError(f"sensory must be positive, got {self.sensory}")
        if self.units < 1:
            raise ValueError(f"units must be positive, got {self.units}")
        if self.units % self.k != 0:
            raise ValueError(f"units ({self.units}) must be divisible by k ({self.k})")
        if self.steps_per_action < 1:
            raise ValueError(f"steps_per_action must be positive, got {self.steps_per_action}")
        if self.tau <= 0 or not isfinite(self.tau):
            raise ValueError(f"tau must be positive and finite, got {self.tau}")
        if self.dt <= 0 or not isfinite(self.dt):
            raise ValueError(f"dt must be positive and finite, got {self.dt}")
        if not isfinite(self.v_rest):
            raise ValueError(f"v_rest must be finite, got {self.v_rest}")
        if self.threshold <= 0 or not isfinite(self.threshold):
            raise ValueError(f"threshold must be positive and finite, got {self.threshold}")
        if not isfinite(self.reset):
            raise ValueError(f"reset must be finite, got {self.reset}")
        if not isfinite(self.theta_alpha):
            raise ValueError(f"theta_alpha must be finite, got {self.theta_alpha}")
        if self.theta_freq <= 0 or not isfinite(self.theta_freq):
            raise ValueError(f"theta_freq must be positive and finite, got {self.theta_freq}")
        if self.k < 1:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.stdp_clip < 0 or not isfinite(self.stdp_clip):
            raise ValueError(f"stdp_clip must be non-negative and finite, got {self.stdp_clip}")
        if not isfinite(self.a_plus) or self.a_plus < 0:
            raise ValueError(f"a_plus must be non-negative and finite, got {self.a_plus}")
        if not isfinite(self.a_minus) or self.a_minus < 0:
            raise ValueError(f"a_minus must be non-negative and finite, got {self.a_minus}")
        if not isfinite(self.theta_scale):
            raise ValueError(f"theta_scale must be finite, got {self.theta_scale}")
        if not isfinite(self.tonic):
            raise ValueError(f"tonic must be finite, got {self.tonic}")
        if self.surrogate_width <= 0 or not isfinite(self.surrogate_width):
            raise ValueError(
                f"surrogate_width must be positive and finite, got {self.surrogate_width}"
            )
        if self.surrogate_kind not in ("arctan", "triangle"):
            raise ValueError(
                f"surrogate_kind must be 'arctan' or 'triangle', got {self.surrogate_kind}"
            )
        if not isfinite(self.gain_init) or self.gain_init <= 0:
            raise ValueError(f"gain_init must be positive and finite, got {self.gain_init}")

    @classmethod
    def paper(cls, **kwargs: Any) -> SpikingTEMConfig:
        """Return parameters close to paper Table 1 values."""
        return cls(
            threshold=1.0,
            tonic=0.0,
            theta_scale=1.0,
            gain_init=1.0,
            **kwargs,
        )

    @classmethod
    def authors_code(cls, **kwargs: Any) -> SpikingTEMConfig:
        """Return parameters matching the authors' reference Python codebase."""
        return cls(
            threshold=0.5,
            tonic=0.5,
            theta_scale=0.1,
            gain_init=1.2,
            **kwargs,
        )


@dataclass(frozen=True)
class SpikingState:
    """State tensors for SpikingTEM recurrent and synaptic dynamics."""

    v_mecii: torch.Tensor
    v_meciii: torch.Tensor
    v_ca1: torch.Tensor
    v_mecv: torch.Tensor
    trace_pre: torch.Tensor
    trace_post: torch.Tensor
    M: torch.Tensor
    prev_mecii_spikes: torch.Tensor
    prev_meciii_spikes: torch.Tensor
    step_count: int

    def detached(self) -> SpikingState:
        """Return a detached copy of numerical states for truncated BPTT."""
        return SpikingState(
            v_mecii=self.v_mecii.detach(),
            v_meciii=self.v_meciii.detach(),
            v_ca1=self.v_ca1.detach(),
            v_mecv=self.v_mecv.detach(),
            trace_pre=self.trace_pre.detach(),
            trace_post=self.trace_post.detach(),
            M=self.M.detach(),
            prev_mecii_spikes=self.prev_mecii_spikes.detach(),
            prev_meciii_spikes=self.prev_meciii_spikes.detach(),
            step_count=self.step_count,
        )


class SpikingTEM(nn.Module):
    """Reduced spiking Tolman-Eichenbaum Machine (TEM) model."""

    def __init__(self, config: SpikingTEMConfig | None = None) -> None:
        """Initialize SpikingTEM layers and weights."""
        super().__init__()
        self.config: SpikingTEMConfig = config if config is not None else SpikingTEMConfig()
        u = self.config.units
        s = self.config.sensory

        self.w_sens_mecii = nn.Linear(s, u)
        self.w_meciii_mecii = nn.Linear(u, u)
        self.w_mecii_meciii = nn.Linear(u, u)
        self.w_action_meciii = nn.Linear(5, u)
        self.w_meciii_ca1 = nn.Linear(u, u)
        self.w_retrieved_mecv = nn.Linear(u, u)

        mecv_dim = u // self.config.k if self.config.sample else u
        self.w_mecv_sensory = nn.Linear(mecv_dim, s)

        if self.config.use_gain:
            self.gain: nn.Parameter | None = nn.Parameter(torch.tensor(self.config.gain_init))
        else:
            self.register_parameter("gain", None)
            self.gain = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights and biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.2)

    def initial_state(
        self,
        batch_size: int = 1,
        m_init_std: float = 0.0,
        device: torch.device | str | None = None,
    ) -> SpikingState:
        """Return initial environment-local state on the model's device and dtype.

        Shapes/Units:
            Voltages v_* of shape (batch_size, units); memory M of shape (units, units);
            traces and spikes of shape (batch_size, units).
        """
        dev = device if device is not None else next(self.parameters()).device
        dt = next(self.parameters()).dtype
        u = self.config.units
        memory = (
            torch.randn(u, u, device=dev, dtype=dt) * m_init_std
            if m_init_std > 0.0
            else torch.zeros(u, u, device=dev, dtype=dt)
        )
        return SpikingState(
            v_mecii=torch.full((batch_size, u), self.config.v_rest, device=dev, dtype=dt),
            v_meciii=torch.full((batch_size, u), self.config.v_rest, device=dev, dtype=dt),
            v_ca1=torch.full((batch_size, u), self.config.v_rest, device=dev, dtype=dt),
            v_mecv=torch.full((batch_size, u), self.config.v_rest, device=dev, dtype=dt),
            trace_pre=torch.zeros(batch_size, u, device=dev, dtype=dt),
            trace_post=torch.zeros(batch_size, u, device=dev, dtype=dt),
            M=memory,
            prev_mecii_spikes=torch.zeros(batch_size, u, device=dev, dtype=dt),
            prev_meciii_spikes=torch.zeros(batch_size, u, device=dev, dtype=dt),
            step_count=0,
        )

    def _lif_layer(
        self, v: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform one LIF update step."""
        return lif_step(
            v,
            current,
            tau=self.config.tau,
            dt=self.config.dt,
            v_rest=self.config.v_rest,
            threshold=self.config.threshold,
            reset=self.config.reset,
            width=self.config.surrogate_width,
            kind=self.config.surrogate_kind,
        )

    def forward(
        self, walk_chunk: Walk, state: SpikingState | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], SpikingState]:
        """Forward pass over a walk chunk."""
        dev = next(self.parameters()).device
        dt = next(self.parameters()).dtype
        batch_size = 1
        curr_state = state if state is not None else self.initial_state(batch_size, device=dev)

        actions = walk_chunk.actions
        num_steps = len(actions)
        observations = walk_chunk.observations

        logits_list: list[torch.Tensor] = []
        mecii_rates_list: list[torch.Tensor] = []
        meciii_rates_list: list[torch.Tensor] = []
        mecv_rates_list: list[torch.Tensor] = []
        ca1_rates_list: list[torch.Tensor] = []

        v_mecii = curr_state.v_mecii
        v_meciii = curr_state.v_meciii
        v_ca1 = curr_state.v_ca1
        v_mecv = curr_state.v_mecv
        trace_pre = curr_state.trace_pre
        trace_post = curr_state.trace_post
        memory = curr_state.M
        prev_mecii_spikes = curr_state.prev_mecii_spikes
        prev_meciii_spikes = curr_state.prev_meciii_spikes
        step_count = curr_state.step_count

        for t in range(num_steps):
            action_idx = actions[t].long().to(dev)
            action_onehot = (
                F.one_hot(action_idx, num_classes=5).to(dtype=dt, device=dev).unsqueeze(0)
            )

            sens_idx = observations[t].long().to(dev)
            sens_onehot = (
                F.one_hot(sens_idx, num_classes=self.config.sensory)
                .to(dtype=dt, device=dev)
                .unsqueeze(0)
            )

            mecii_spike_accum = torch.zeros_like(prev_mecii_spikes, device=dev, dtype=dt)
            meciii_spike_accum = torch.zeros_like(prev_mecii_spikes, device=dev, dtype=dt)
            mecv_spike_accum = torch.zeros_like(prev_mecii_spikes, device=dev, dtype=dt)
            ca1_spike_accum = torch.zeros_like(prev_mecii_spikes, device=dev, dtype=dt)

            for s in range(self.config.steps_per_action):
                global_step = step_count + s
                t_sec = global_step * self.config.dt / 1000.0
                raw_theta = (
                    theta_current(t_sec, alpha=self.config.theta_alpha, freq=self.config.theta_freq)
                    if self.config.use_theta
                    else 0.0
                )
                theta_drv = raw_theta * self.config.theta_scale

                i_mecii = (
                    self.w_sens_mecii(sens_onehot)
                    + self.w_meciii_mecii(prev_meciii_spikes)
                    + self.config.tonic
                )
                if self.config.use_gain and self.gain is not None:
                    i_mecii = i_mecii * self.gain
                if self.config.use_theta:
                    i_mecii = i_mecii + theta_drv

                v_mecii, mecii_spikes = self._lif_layer(v_mecii, i_mecii)

                i_meciii = (
                    self.w_mecii_meciii(mecii_spikes)
                    + self.w_action_meciii(action_onehot)
                    + self.config.tonic
                )
                if self.config.use_gain and self.gain is not None:
                    i_meciii = i_meciii * self.gain
                if self.config.use_theta:
                    i_meciii = i_meciii + theta_drv

                v_meciii, meciii_spikes = self._lif_layer(v_meciii, i_meciii)

                i_ca1 = self.w_meciii_ca1(meciii_spikes) + self.config.tonic
                v_ca1, ca1_spikes = self._lif_layer(v_ca1, i_ca1)

                retrieved = ca1_spikes @ memory
                i_mecv = self.w_retrieved_mecv(retrieved) + self.config.tonic
                v_mecv, mecv_spikes = self._lif_layer(v_mecv, i_mecv)

                if self.config.use_stdp:
                    memory, trace_pre, trace_post = stdp_memory_update(
                        memory,
                        pre=ca1_spikes,
                        post=mecv_spikes,
                        a_plus=self.config.a_plus,
                        a_minus=self.config.a_minus,
                        trace_pre=trace_pre,
                        trace_post=trace_post,
                        tau_pre=self.config.tau,
                        tau_post=self.config.tau,
                        dt=self.config.dt,
                        clip=self.config.stdp_clip,
                    )

                mecii_spike_accum = mecii_spike_accum + mecii_spikes
                meciii_spike_accum = meciii_spike_accum + meciii_spikes
                mecv_spike_accum = mecv_spike_accum + mecv_spikes
                ca1_spike_accum = ca1_spike_accum + ca1_spikes

                prev_mecii_spikes = mecii_spikes
                prev_meciii_spikes = meciii_spikes

            step_count += self.config.steps_per_action

            mecii_rate = mecii_spike_accum / self.config.steps_per_action
            meciii_rate = meciii_spike_accum / self.config.steps_per_action
            mecv_rate = mecv_spike_accum / self.config.steps_per_action
            ca1_rate = ca1_spike_accum / self.config.steps_per_action

            if self.config.sample:
                mecv_rate = bernoulli_group_means(mecv_rate.clamp(0, 1), group_size=self.config.k)

            logits = self.w_mecv_sensory(mecv_rate)
            logits_list.append(logits)
            mecii_rates_list.append(mecii_rate)
            meciii_rates_list.append(meciii_rate)
            mecv_rates_list.append(mecv_rate)
            ca1_rates_list.append(ca1_rate)

        logits_tensor = torch.cat(logits_list, dim=0)
        rates_dict = {
            "mecii": torch.cat(mecii_rates_list, dim=0),
            "meciii": torch.cat(meciii_rates_list, dim=0),
            "ca1": torch.cat(ca1_rates_list, dim=0),
            "mecv": torch.cat(mecv_rates_list, dim=0),
        }
        new_state = SpikingState(
            v_mecii=v_mecii,
            v_meciii=v_meciii,
            v_ca1=v_ca1,
            v_mecv=v_mecv,
            trace_pre=trace_pre,
            trace_post=trace_post,
            M=memory,
            prev_mecii_spikes=prev_mecii_spikes,
            prev_meciii_spikes=prev_meciii_spikes,
            step_count=step_count,
        )
        return logits_tensor, rates_dict, new_state


def train_walks(
    model: SpikingTEM,
    walks: list[Walk],
    steps: int,
    *,
    lr: float = 3e-3,
    seed: int = 0,
) -> list[float]:
    """Train SpikingTEM on walks with truncated BPTT."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        walk = walks[step % len(walks)]
        optimizer.zero_grad()
        state = model.initial_state(m_init_std=0.05)
        total_loss = 0.0

        num_steps = len(walk.actions)
        chunk_size = 20
        for start in range(0, num_steps, chunk_size):
            end = min(start + chunk_size, num_steps)
            sub_walk = Walk(
                nodes=walk.nodes[start : end + 1],
                observations=walk.observations[start : end + 1],
                actions=walk.actions[start:end],
                novel_edge_to_seen=walk.novel_edge_to_seen[start:end],
                fraction_seen=walk.fraction_seen[start:end],
                executed_actions=walk.executed_actions[start:end]
                if walk.executed_actions is not None
                else None,
            )
            logits, rates, state = model(sub_walk, state)
            state = state.detached()

            targets = walk.observations[start + 1 : end + 1]
            ce_loss = F.cross_entropy(logits, targets)
            activity_penalty = 1e-3 * (
                rates["mecii"].mean() + rates["mecv"].mean() + rates["ca1"].mean()
            )
            loss = ce_loss + activity_penalty
            loss.backward()
            total_loss += float(loss.item())

        optimizer.step()
        losses.append(total_loss)

    return losses
