"""Thin HTTP client for the ai4science API.

Every method here maps directly to one existing ai4science endpoint:
``submit`` -> POST /ephemeral-job, ``logs`` -> GET /logs/{job_id},
``results`` -> GET /results/{job_id}. No server-side behavior is assumed
beyond what those endpoints already do.

Request/response shapes live in schemas.py -- this file is HTTP
plumbing only.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

from .schemas import JobResult, JobSubmitRequest, JobSubmitResponse
from .script_builder import build_script

TERMINAL_STATUSES = ("completed", "failed")


def _new_chunk(full_text: str | None, seen_len: int) -> tuple[str, int]:
    """Pure helper: given the full current log text and how much of it
    was already seen, return (new_portion, new_seen_len).

    This is what makes client-side "streaming" possible without any
    server changes: /logs/{job_id} always returns the whole log, so we
    just diff against what we already printed.
    """
    if not full_text or len(full_text) <= seen_len:
        return "", seen_len
    return full_text[seen_len:], len(full_text)


class Ai4ScienceJob:
    """A handle to a submitted job. Returned by ``Ai4ScienceClient.submit``."""

    def __init__(self, client: "Ai4ScienceClient", job_id: str):
        self.job_id = job_id
        self._client = client

    def logs(self) -> str | None:
        """Return the current log text, or None if not available yet."""
        return self._client.logs(self.job_id)

    def results(self) -> JobResult:
        """Return the current parsed status/result (may still be 'running')."""
        return self._client.results(self.job_id)

    def wait(
        self,
        interval: int | None = None,
        timeout: int = 3600,
        stream: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> JobResult:
        """Block until the job reaches a terminal status, then return results.

        interval defaults to the client's configured interval (see
        Ai4ScienceClient's interval= / Ai4ScienceConfig.interval) when
        not given explicitly.

        If stream=True, print (or call on_log with) new log output as it
        appears, each time we poll. This is client-side tailing -- it
        polls /logs/{job_id} and diffs against what was already shown,
        not a real server-side streaming connection.
        """
        return self._client.wait(
            self.job_id, interval=interval, timeout=timeout, stream=stream, on_log=on_log
        )

    def __repr__(self) -> str:
        return f"Ai4ScienceJob(job_id={self.job_id!r})"

class Ai4ScienceClient:
    """Client for an ai4science deployment.

    Any of base_url/user/token/interval left as None falls back to the
    environment (AI4SCIENCE_BASE_URL / SLURM_USER / SLURM_TOKEN /
    AI4SCIENCE_POLL_INTERVAL -- see .env.example; a local .env file is
    loaded automatically via config.py). Pass explicit values to skip
    the environment entirely -- no .env file or env vars are required
    if you supply base_url/user/token yourself.

    Parameters
    ----------
    base_url : str | None
        e.g. "https://ai4science.dev.sdp.surf.nl". Falls back to
        AI4SCIENCE_BASE_URL if not given.
    user, token : str | None
        SLURM credentials. Fall back to SLURM_USER / SLURM_TOKEN if not
        given.
    interval : int | None
        Default poll interval in seconds, used by wait()/run() whenever
        a call doesn't specify its own interval. Falls back to
        AI4SCIENCE_POLL_INTERVAL (or 10 if that's also unset).
    """

    def __init__(
        self,
        base_url: str | None = None,
        user: str | None = None,
        token: str | None = None,
        interval: int | None = None,
    ):
        if base_url is None or user is None or token is None or interval is None:
            # Importing here (not at module load) triggers config.py's
            # automatic .env loading only if/when it's actually needed --
            # e.g. never, if the caller passed everything explicitly.
            from . import config as _config  # noqa: F401, PLC0415

            base_url = base_url or os.environ.get("AI4SCIENCE_BASE_URL")
            user = user or os.environ.get("SLURM_USER")
            token = token or os.environ.get("SLURM_TOKEN")
            if interval is None:
                interval = int(os.environ.get("AI4SCIENCE_POLL_INTERVAL", "10"))

        missing = [
            name
            for name, value in [("base_url", base_url), ("user", user), ("token", token)]
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required value(s): {', '.join(missing)}. Pass them "
                "explicitly, or set AI4SCIENCE_BASE_URL / SLURM_USER / "
                "SLURM_TOKEN (env vars or a .env file -- see .env.example)."
            )

        self.base_url = base_url.rstrip("/")
        self.user = user
        self.token = token
        self.default_interval = interval

    def submit(
        self,
        python_script: str,
        dependencies: list[str] | None = None,
        hf_token: str | None = None,
    ) -> Ai4ScienceJob:
        """Submit a raw script string to /ephemeral-job. Returns immediately."""
        request = JobSubmitRequest(
            dependencies=dependencies or [],
            python_script=python_script,
            user=self.user,
            token=self.token,
            hf_token=hf_token,
        )
        resp = requests.post(
            f"{self.base_url}/ephemeral-job",
            json=request.model_dump(exclude_none=True),
            timeout=30,
        )
        resp.raise_for_status()
        submitted = JobSubmitResponse.model_validate(resp.json())
        return Ai4ScienceJob(self, str(submitted.job_id))

    def logs(self, job_id: str) -> str | None:
        """GET /logs/{job_id}. Returns None if the log isn't available yet (404)."""
        resp = requests.get(f"{self.base_url}/logs/{job_id}", timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

    def results(self, job_id: str) -> JobResult:
        """GET /results/{job_id}. Returns a 'status: unknown' placeholder if not ready."""
        resp = requests.get(f"{self.base_url}/results/{job_id}", timeout=15)
        if resp.status_code == 404:
            return JobResult(job_id=job_id, status="unknown")
        resp.raise_for_status()
        return JobResult.model_validate(resp.json())

    def wait(
        self,
        job_id: str,
        interval: int | None = None,
        timeout: int = 3600,
        stream: bool = False,
        on_log: Callable[[str], None] | None = None,
    ) -> JobResult:
        """Poll /results/{job_id} until a terminal status or timeout.

        interval defaults to self.default_interval when not given.

        If stream=True, also poll /logs/{job_id} each round and print
        (or call on_log with) only the new text since the last poll --
        this is client-side tailing, not a real server-sent stream.
        """
        interval = self.default_interval if interval is None else interval
        on_log = on_log or (lambda text: print(text, end="", flush=True))
        seen_len = 0
        elapsed = 0
        while elapsed < timeout:
            if stream:
                chunk, seen_len = _new_chunk(self.logs(job_id), seen_len)
                if chunk:
                    on_log(chunk)

            result = self.results(job_id)
            if result.status in TERMINAL_STATUSES:
                if stream:
                    chunk, seen_len = _new_chunk(self.logs(job_id), seen_len)
                    if chunk:
                        on_log(chunk)
                return result

            time.sleep(interval)
            elapsed += interval
        raise TimeoutError(f"job {job_id} did not finish within {timeout}s")

    def run(
        self,
        func: Callable,
        *args: Any,
        dependencies: list[str] | None = None,
        hf_token: str | None = None,
        interval: int | None = None,
        timeout: int = 3600,
        stream: bool = False,
        on_log: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a script from func, submit it, block until done, return the result.

        This is the single call most users need: it combines
        ``build_script`` + ``submit`` + ``wait`` into one blocking call
        that behaves like a normal (if slow) function call. interval
        defaults to self.default_interval when not given. Pass
        stream=True to print live log output while waiting.
        """
        script = build_script(func, args=args, kwargs=kwargs)
        job = self.submit(script, dependencies=dependencies, hf_token=hf_token)
        result = job.wait(interval=interval, timeout=timeout, stream=stream, on_log=on_log)
        if result.status != "completed":
            raise RuntimeError(
                f"job {job.job_id} finished with status={result.status} "
                f"(exit_code={result.exit_code}). "
                f"Logs: {self.base_url}/logs/{job.job_id}"
            )
        return result.result