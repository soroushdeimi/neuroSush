"""Initial weights and delays."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, SynapseGroup
from neurosush.core.order import Order


def _check_density(density: float) -> None:
    if not 0 < density <= 1:
        raise ValueError(f"density must be in (0, 1], got {density}")


def sparse_random(
    n_src: int,
    n_dst: int,
    density: float,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted source and destination indices for unique random connections.

    Args:
        n_src: Number of source neurons.
        n_dst: Number of destination neurons.
        density: Fraction of possible connections to sample.
        generator: Random generator used to sample connections.
        device: Device for the returned indices.
    """
    _check_density(density)
    count = round(n_src * n_dst * density)
    flat = torch.randperm(n_src * n_dst, generator=generator, device=device)[:count]
    flat = flat.sort().values
    return flat // n_dst, flat % n_dst


def fixed_in_degree(
    n_src: int,
    n_dst: int,
    in_degree: int,
    *,
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source and destination indices with exactly ``in_degree`` sources per destination.

    The sources of a destination are distinct and uniformly random; edges are listed
    in rounds (the first source of every destination, then the second, ...).

    Args:
        n_src: Number of source neurons.
        n_dst: Number of destination neurons.
        in_degree: Number of distinct sources of every destination.
        generator: Random generator used to sample connections.
        device: Device for the returned indices.
    """
    if not 1 <= in_degree <= n_src:
        raise ValueError(f"in_degree must be in [1, {n_src}], got {in_degree}")
    chunk = max(1, (1 << 24) // n_src)  # bounds the (n_src, chunk) random draw
    sources = []
    for start in range(0, n_dst, chunk):
        width = min(chunk, n_dst - start)
        draw = torch.rand(n_src, width, generator=generator, device=device)
        sources.append(draw.argsort(0)[:in_degree])  # (in_degree, width)
    dst = torch.arange(n_dst, device=device).repeat(in_degree)
    return torch.cat(sources, dim=1).flatten(), dst


class WeightInit(Behavior):
    """Initialize copied weights or sampled dense or sparse weights.

    Args:
        mode: Sampling distribution or constant weight value.
        weights: Explicit weights to copy instead of sampling. In an independent network, either
            the geometry's shape (copied to every member) or ``(B, *shape)`` (one set per member).
        scale: Multiplier applied after sampling and transformation.
        offset: Offset added after scaling.
        fn: Optional transformation of sampled weights.
        density: Fraction of connections to retain.
        sparse: Store sampled connections as indices and a weight vector.
        in_degree: With ``sparse=True``, connect every destination to exactly this many distinct
            random sources (fixed in-degree, as in Brunel 2000) instead of a ``density``;
            mutually exclusive with ``density < 1``.
        shape: Weight geometry; defaults to source by destination size.
    """

    order = Order.INITIALIZATION
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        mode: str | float | complex | None = None,
        weights: torch.Tensor | None = None,
        scale: float = 1.0,
        offset: float = 0.0,
        fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        density: float = 1.0,
        sparse: bool = False,
        in_degree: int | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if (mode is None) == (weights is None):
            raise ValueError(
                "exactly one of mode or weights must be given, "
                f"got mode={mode!r}, weights={weights}"
            )
        if mode is not None and (
            (isinstance(mode, str) and mode not in ("uniform", "normal", "zeros", "ones"))
            or not isinstance(mode, (str, Number))
        ):
            raise ValueError(f"mode must be uniform, normal, zeros, ones or a number, got {mode!r}")
        if weights is not None and (scale != 1.0 or offset != 0.0 or fn is not None):
            raise ValueError(
                "explicit weights cannot be combined with scale, offset or fn, "
                f"got scale={scale}, offset={offset}, fn={fn!r}"
            )
        _check_density(density)
        if in_degree is not None:
            if not sparse:
                raise ValueError("in_degree needs sparse=True")
            if density != 1.0:
                raise ValueError(f"in_degree and density are exclusive, got density={density}")
            if isinstance(in_degree, bool) or not isinstance(in_degree, int) or in_degree < 1:
                raise ValueError(f"in_degree must be a positive int, got {in_degree!r}")
            if weights is not None:
                raise ValueError("in_degree samples connections, so it cannot be used with weights")
        if sparse and shape is not None and len(shape) != 2:
            raise ValueError(f"sparse weights require a 2-D shape, got {shape}")
        self.mode = mode
        self.weights = weights
        self.scale = scale
        self.offset = offset
        self.fn = fn
        self.density = density
        self.sparse = sparse
        self.in_degree = in_degree
        self.shape = shape

    def initialize(self, syn: SynapseGroup) -> None:
        """Allocate weights and, for sparse storage, connection indices.

        Args:
            syn: Synapse group receiving the weights.
        """
        net = syn.net
        shape = self.shape or (syn.src.size, syn.dst.size)
        members = net.batch_size or 1
        if net.independent and self.sparse:
            raise NotImplementedError(f"sparse weights ({syn.name}) need independent=False")
        if self.weights is not None:
            if net.independent and self.weights.shape == (members, *shape):
                pass  # one set of weights per member
            elif self.weights.shape != shape:
                raise ValueError(
                    f"weights shape for {syn.name!r} must be {shape}"
                    + (f" or {(members, *shape)}" if net.independent else "")
                    + f", got {tuple(self.weights.shape)}"
                )
            syn.weights = self.weights.to(dtype=net.dtype, device=net.device, copy=True)
            if net.independent and syn.weights.shape == shape:  # same weights for every member
                syn.weights = syn.weights.expand(members, *shape).clone()
            return
        if net.independent:
            shape = (members, *shape)  # every member draws its own weights
        if self.sparse:
            if self.in_degree is not None:
                syn.src_idx, syn.dst_idx = fixed_in_degree(
                    shape[0], shape[1], self.in_degree, generator=net.generator, device=net.device
                )
            else:
                syn.src_idx, syn.dst_idx = sparse_random(
                    shape[0], shape[1], self.density, generator=net.generator, device=net.device
                )
            values = self._sample((syn.src_idx.numel(),), net)
        else:
            values = self._sample(shape, net)
            if self.density < 1:
                values = values * (
                    torch.rand(shape, generator=net.generator, device=net.device) < self.density
                )
        if self.fn is not None:
            values = self.fn(values)
        syn.weights = values * self.scale + self.offset

    def _sample(self, shape: tuple[int, ...], net: Network) -> torch.Tensor:
        if self.mode in ("uniform", "normal"):
            sample = torch.rand if self.mode == "uniform" else torch.randn
            return sample(shape, generator=net.generator, dtype=net.dtype, device=net.device)
        value = {"zeros": 0.0, "ones": 1.0}[self.mode] if isinstance(self.mode, str) else self.mode
        assert value is not None  # Sampling is used only when mode, rather than weights, is set.
        return torch.full(shape, value, dtype=net.dtype, device=net.device)


class DelayInit(Behavior):
    """Initialize source or destination delays in simulation steps.

    Args:
        delays: Non-negative fixed delay or per-neuron delay tensor.
        max_delay: Exclusive upper bound for sampled delays.
        side: Source (src) or destination (dst) receiving delays.
    """

    order = Order.INITIALIZATION
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        delays: int | torch.Tensor | None = None,
        max_delay: int | None = None,
        side: str = "src",
    ) -> None:
        if (delays is None) == (max_delay is None):
            raise ValueError(
                "exactly one of delays or max_delay must be given, "
                f"got delays={delays}, max_delay={max_delay}"
            )
        if delays is not None and bool((torch.as_tensor(delays) < 0).any()):
            raise ValueError(f"delays must be non-negative, got {delays}")
        if max_delay is not None and max_delay < 1:
            raise ValueError(f"max_delay must be at least 1, got {max_delay}")
        if side not in ("src", "dst"):
            raise ValueError(f"side must be src or dst, got {side!r}")
        self.delays = delays
        self.max_delay = max_delay
        self.side = side

    def initialize(self, syn: SynapseGroup) -> None:
        """Allocate a long delay vector for the selected side.

        Args:
            syn: Synapse group receiving the delays.
        """
        net = syn.net
        size = syn.src.size if self.side == "src" else syn.dst.size
        if self.max_delay is not None:
            delays = torch.randint(
                self.max_delay, (size,), generator=net.generator, device=net.device
            )
        else:
            delays = torch.as_tensor(self.delays, dtype=torch.long, device=net.device)
            if delays.ndim == 0:
                delays = delays.expand(size)
            elif delays.shape != (size,):
                raise ValueError(
                    f"delays shape for {syn.name!r} must be ({size},), got {tuple(delays.shape)}"
                )
        setattr(syn, f"{self.side}_delay", delays.to(dtype=torch.long, copy=True))
