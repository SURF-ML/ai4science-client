"""Pydantic models for the data shapes that cross the ai4science HTTP
boundary: submit requests (ephemeral and ray), the submit response, and
the results response. Keeping these separate from client.py means the
request/response contract can be read, imported, or reused (e.g. for
type hints elsewhere) without pulling in requests or any HTTP logic.
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

    For multi-node jobs (container="ray"), `nodes` controls cluster
    size and the other fields describe resources requested PER NODE --
    no separate multi-node config needed, this same model already
    expresses it, matching the server's own design.
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

    track/experiment_name: MLflow experiment tracking. track=False
    (default) is a no-op -- byte-for-byte identical request shape to
    before these fields existed. Field names must stay in sync with the
    server's EphemeralJobRequest.track/experiment_name
    (app/common/schemas/request_schemas.py in ai4science-poc).
    """

    dependencies: list[str] = Field(default_factory=list)
    python_script: str
    user: str
    token: str
    hf_token: str | None = None
    resources: SlurmResourceConfig | None = None
    tier: str | None = None
    cluster: str | None = None
    track: bool = False
    experiment_name: str | None = None


class RayJobSubmitRequest(BaseModel):
    """Body sent to POST /ray-job.

    Same dependencies+python_script shape as JobSubmitRequest -- the
    driver script runs against a live, multi-node Ray cluster instead
    of a single process. No tier/cluster here: auto-tier-routing is
    only wired up server-side for /ephemeral-job. Node count and
    per-node resources are controlled entirely through `resources`
    (SlurmResourceConfig.nodes/cpus_per_task/memory_mb/tres_per_node).

    track/experiment_name: same as JobSubmitRequest -- see there. Must
    stay in sync with the server's RayJobRequest.track/experiment_name.
    """

    dependencies: list[str] = Field(default_factory=list)
    python_script: str
    user: str
    token: str
    hf_token: str | None = None
    resources: SlurmResourceConfig | None = None
    track: bool = False
    experiment_name: str | None = None


class JobSubmitResponse(BaseModel):
    """Response from POST /ephemeral-job or POST /ray-job -- both
    return the same SlurmJobResult shape server-side.

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


# --- Agentic layer (MCP endpoint at {base_url}/mcp) ---
#
# Response models for what the server's MCP tools return. Plain
# duplicates of the server's schemas (app/common/schemas/agentic_schemas.py
# and community_tool_schemas.py in ai4science-poc), like SlurmResourceConfig
# above. extra="ignore": fields the server adds later won't break older
# clients.


class McpTool(BaseModel):
    """One tool the server offers on its MCP endpoint."""

    model_config = {"extra": "ignore"}

    name: str
    title: str | None = None
    description: str = ""
    read_only: bool = False
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None


class CommunityToolRuntime(BaseModel):
    """How a community tool runs: Apptainer + pip pins, or EESSI modules."""

    model_config = {"extra": "ignore"}

    kind: str
    entrypoint: str
    dependencies: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)


class CommunityToolResources(BaseModel):
    """Resources one run of a community tool needs."""

    model_config = {"extra": "ignore"}

    needs_gpu: bool = False
    cpus: int = 1
    memory_mb: int = 4096
    time_limit_minutes: int = 30


class CommunityTool(BaseModel):
    """A published community-contributed tool (latest version)."""

    model_config = {"extra": "ignore"}

    domain: str
    name: str
    version: str
    title: str
    description: str
    authors: list[str] = Field(default_factory=list)
    license: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    runtime: CommunityToolRuntime | None = None
    resources: CommunityToolResources | None = None

    @property
    def qualified_name(self) -> str:
        """Globally unique name: '<domain>.<name>'."""
        return f"{self.domain}.{self.name}"
