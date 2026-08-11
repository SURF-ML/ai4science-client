"""Exception hierarchy for the ai4science client.

Every error this package can raise is a subclass of Ai4ScienceError, so
callers can do a single `except Ai4ScienceError` if they don't care about
the distinction, or catch specific subclasses when they do -- rather than
needing to know which raw `requests` or builtin exception a given method
happens to leak.
"""

from __future__ import annotations


class Ai4ScienceError(Exception):
    """Base class for all ai4science client errors."""


class Ai4ScienceConnectionError(Ai4ScienceError):
    """Could not reach the ai4science API at all (DNS, connection refused,
    connection reset, or the request timed out before getting a response).
    Retrying later, or checking base_url/network connectivity, is the
    likely fix -- this is not caused by bad request content.
    """


class Ai4ScienceAPIError(Ai4ScienceError):
    """The ai4science API received the request and returned an error
    response (4xx/5xx). Carries the HTTP status code and, where the
    server provided one, its `detail` message -- so callers can see
    *why* the request was rejected instead of just that it failed.
    """

    def __init__(self, message: str, status_code: int, detail: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (status={self.status_code})" if self.status_code else base


class Ai4ScienceTimeoutError(Ai4ScienceError):
    """A job did not reach a terminal status within the requested
    timeout. The job may still be running server-side -- this only means
    the client gave up waiting.
    """


class Ai4ScienceJobFailedError(Ai4ScienceError):
    """A job reached a terminal status other than 'completed'."""

    def __init__(self, message: str, job_id: str, status: str, exit_code: int | None = None):
        super().__init__(message)
        self.job_id = job_id
        self.status = status
        self.exit_code = exit_code


class ArtifactNotFoundError(Ai4ScienceError):
    """A local file path passed via artifacts=... does not exist (or
    isn't a readable file) on this machine.

    Raised entirely client-side, before any HTTP request is made -- this
    is a local usage mistake (bad path, typo, wrong cwd), not something
    the ai4science API was ever asked about, so it's caught and reported
    clearly up front rather than surfacing as a confusing failure deep
    inside job submission.
    """

    def __init__(self, name: str, local_path: str):
        super().__init__(
            f"Artifact '{name}': local file not found at '{local_path}'. "
            "Check the path is correct, exists, and is readable before "
            "calling run() / submit() / the decorator."
        )
        self.name = name
        self.local_path = local_path