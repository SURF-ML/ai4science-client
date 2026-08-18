"""Pydantic models for the three data shapes that cross the ai4science
HTTP boundary: the submit request, the submit response, and the results
response. Keeping these separate from client.py means the request/response
contract can be read, imported, or reused (e.g. for type hints elsewhere)
without pulling in requests or any HTTP logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SlurmResourceConfig(BaseModel):
    """Optional overrides for Slurm resource parameters, sent to the
    ai4science API. Field names must stay in sync with the server's
    SlurmResourceConfig (app/common/schemas.py in ai4science-poc) -- this
    is a plain duplicate, not a shared import, since this is a separate
    package. Any field left as None means "use the server's default for
    this job type."
    """

    partition: str | None = None
    nodes: int | str | None = None
    cpus_per_task: int | None = Field(default=None, ge=1)
    tasks_per_node: int | None = Field(default=None, ge=1)
    memory_mb: int | None = Field(
        default=None, ge=1, description="Memory per node, in MB"
    )
    time_limit_minutes: int | None = Field(
        default=None, ge=1, description="Wall-clock time limit, in minutes"
    )
    tres_per_node: str | None = Field(
        default=None, description="Generic resource request, e.g. 'gres:gpu:1'"
    )

    model_config = {"extra": "forbid"}


class JobSubmitRequest(BaseModel):
    """Body sent to POST /ephemeral-job.

    tier/cluster opt into auto-tier-routing: "auto" (or a real tier id)
    estimates resource needs from dependencies/python_script and routes
    to the smallest-fitting cluster; cluster pins a specific one
    directly. Both default to None -- omitting them preserves the
    server's original behavior (submit to its single configured SLURM
    cluster), unchanged.
    """

    dependencies: list[str] = Field(default_factory=list)
    python_script: str
    user: str
    token: str
    hf_token: str | None = None
    resources: SlurmResourceConfig | None = None
    tier: str | None = None
    cluster: str | None = None


class JobSubmitResponse(BaseModel):
    """Response from POST /ephemeral-job.

    The server returns job_id as an int here (it comes straight from
    SlurmJobResult.job_id: int). /results/{job_id}, by contrast, returns
    job_id as a string (a FastAPI path param) -- see JobResult below.
    """

    job_id: int
    status: str | None = None


class JobResult(BaseModel):
    """Response from GET /results/{job_id}."""

    job_id: str
    status: str
    exit_code: int | None = None
    result: Any | None = None