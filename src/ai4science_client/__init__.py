"""Thin Python client for running functions on Snellius via ai4science."""

from .client import Ai4ScienceClient, Ai4ScienceJob, Container
from .config import Ai4ScienceConfig
from .exceptions import (
    Ai4ScienceAPIError,
    Ai4ScienceConnectionError,
    Ai4ScienceError,
    Ai4ScienceJobFailedError,
    Ai4ScienceTimeoutError,
    Ai4ScienceToolError,
    ArtifactNotFoundError,
)
from .hpc_decorator import job
from .mcp_client import Ai4ScienceMCPClient
from .schemas import (
    CommunityTool,
    JobResult,
    JobSubmitRequest,
    JobSubmitResponse,
    McpTool,
    RayJobSubmitRequest,
    SlurmResourceConfig,
)
from .script_builder import NotSelfContainedError, build_script

__all__ = [
    "Ai4ScienceAPIError",
    "Ai4ScienceClient",
    "Ai4ScienceConfig",
    "Ai4ScienceConnectionError",
    "Ai4ScienceError",
    "Ai4ScienceJob",
    "Ai4ScienceJobFailedError",
    "Ai4ScienceMCPClient",
    "Ai4ScienceTimeoutError",
    "Ai4ScienceToolError",
    "ArtifactNotFoundError",
    "CommunityTool",
    "Container",
    "JobResult",
    "JobSubmitRequest",
    "JobSubmitResponse",
    "McpTool",
    "NotSelfContainedError",
    "RayJobSubmitRequest",
    "SlurmResourceConfig",
    "build_script",
    "job",
]

__version__ = "0.1.0"
