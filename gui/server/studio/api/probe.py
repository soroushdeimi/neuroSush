"""Library-backed single-neuron probes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from studio.api.validate import _Behavior, _behaviors
from studio.errors import ApiError
from studio.pool.runtime import get_pool

router = APIRouter()


class ProbeModel(BaseModel):
    """One enabled library neuron model."""

    model_config = ConfigDict(extra="forbid")
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class ProbeRequest(BaseModel):
    """Bounded constant-current simulation."""

    model_config = ConfigDict(extra="forbid")
    model: ProbeModel
    current: float = Field(allow_inf_nan=False)
    dt: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    steps: int = Field(default=500, ge=1, strict=True)


class ProbeResponse(BaseModel):
    """Arrays include initial state: t[k]=k*dt and v[k] follows k completed steps.

    Both arrays have steps+1 samples. Spikes are completed-step indices k in
    1..steps, and their times are t[k].
    """

    t: list[float]
    v: list[float]
    spikes: list[int]
    provenance: dict[str, Any]


@router.post("/probe", response_model=ProbeResponse)
def probe(body: ProbeRequest) -> dict[str, Any]:
    """Run a neuron in the shared isolated worker."""
    if body.steps > 5000:
        raise ApiError("too_many_steps", "Probe steps must be at most 5000", 422)
    issues = _behaviors([_Behavior(kind=body.model.kind, params=body.model.params)], ())
    for issue in issues:
        issue["path"] = issue["path"].replace("/behaviors/0", "/model", 1)
    if issues:
        raise ApiError("invalid_spec", "Invalid neuron model", 422, issues)
    result = get_pool().call("probe", body.model_dump())
    if result.get("valid") is False:
        raise ApiError("invalid_spec", "Invalid neuron model", 422, result["errors"])
    return result
