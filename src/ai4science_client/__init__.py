"""Thin Python client for running functions on Snellius via ai4science."""

from .client import Ai4ScienceClient, Ai4ScienceJob
from .config import Ai4ScienceConfig
from .hpc_decorator import job
from .schemas import JobResult, JobSubmitRequest, JobSubmitResponse
from .script_builder import NotSelfContainedError, build_script

__all__ = [
    "Ai4ScienceClient",
    "Ai4ScienceJob",
    "Ai4ScienceConfig",
    "JobResult",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "job",
    "build_script",
    "NotSelfContainedError",
]

__version__ = "0.1.0"