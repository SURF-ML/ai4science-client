"""Thin Python client for running functions on Snellius via ai4science."""

from .client import Ai4ScienceClient, Ai4ScienceJob
from .config import Ai4ScienceConfig
from .exceptions import (
    Ai4ScienceAPIError,
    Ai4ScienceConnectionError,
    Ai4ScienceError,
    Ai4ScienceJobFailedError,
    Ai4ScienceTimeoutError,
    ArtifactNotFoundError,
)
from .hpc_decorator import job
from .schemas import JobResult, JobSubmitRequest, JobSubmitResponse
from .script_builder import NotSelfContainedError, build_script

__all__ = [
    "Ai4ScienceAPIError",
    "Ai4ScienceClient",
    "Ai4ScienceConfig",
    "Ai4ScienceConnectionError",
    "Ai4ScienceError",
    "Ai4ScienceJob",
    "Ai4ScienceJobFailedError",
    "Ai4ScienceTimeoutError",
    "ArtifactNotFoundError",
    "JobResult",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "NotSelfContainedError",
    "build_script",
    "job",
]

__version__ = "0.1.0"