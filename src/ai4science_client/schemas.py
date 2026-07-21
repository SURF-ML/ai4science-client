"""Pydantic models for the three data shapes that cross the ai4science
HTTP boundary: the submit request, the submit response, and the results
response. Keeping these separate from client.py means the request/response
contract can be read, imported, or reused (e.g. for type hints elsewhere)
without pulling in requests or any HTTP logic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobSubmitRequest(BaseModel):
    """Body sent to POST /ephemeral-job."""

    dependencies: list[str] = Field(default_factory=list)
    python_script: str
    user: str
    token: str
    hf_token: str | None = None


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