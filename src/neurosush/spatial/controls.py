"""Online behavioral and sensory control baselines for navigation."""

from __future__ import annotations

import torch

from neurosush.spatial.graph_world import Walk


def seen_targets(walk: Walk) -> torch.Tensor:
    """Build the loss mask using visited metadata; IDs never enter model features.

    Warning: Never uses future information or targets at the same step.
    """
    seen = {int(walk.nodes[0])}
    result = []
    for node in walk.nodes[1:].tolist():
        result.append(node in seen)
        seen.add(node)
    return torch.tensor(result, dtype=torch.bool)


def sensory_count_predictions(walk: Walk, sensory: int) -> dict[str, torch.Tensor]:
    """Online sensory-only controls with the same past observations as TEM.

    Unigram predicts the most frequent identity observed so far. Conditional
    counts additionally use the current sensory identity and supplied action,
    falling back to unigram when that pair is unseen. Neither uses node IDs,
    geometry or future observations. Ties select the lowest identity index.

    Warning: Never uses future information or targets at the same step.
    """
    if sensory < 1:
        raise ValueError(f"sensory must be positive, got {sensory}")
    action_dim = max(int(walk.actions.max()) + 1 if walk.actions.numel() > 0 else 5, 5)
    unigram = torch.zeros(sensory)
    conditional = torch.zeros(sensory, action_dim, sensory)
    frequency, transition = [], []
    for t, action in enumerate(walk.actions):
        observation = walk.observations[t]
        unigram[observation] += 1
        if t:
            conditional[walk.observations[t - 1], walk.actions[t - 1], observation] += 1
        frequency.append(unigram.argmax())
        row = conditional[observation, action]
        transition.append(row.argmax() if row.sum() else unigram.argmax())
    return {"frequency": torch.stack(frequency), "conditional": torch.stack(transition)}
