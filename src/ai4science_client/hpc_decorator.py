"""The @job decorator: sugar over Ai4ScienceClient.run.

If you don't want a decorator, you don't need this file at all --
Ai4ScienceClient.run(func, *args) or even a plain requests.post using
build_script() works just as well. This exists purely for callers who
prefer decorator ergonomics (e.g. matching an existing HPCFunc-style
codebase).
"""

from __future__ import annotations

import functools
from typing import Callable

from .client import Ai4ScienceClient


def job(
    base_url: str,
    user: str,
    token: str,
    dependencies: list[str] | None = None,
    hf_token: str | None = None,
    interval: int = 10,
    timeout: int = 3600,
    stream: bool = False,
    on_log: Callable[[str], None] | None = None,
):
    """Decorator that runs the wrapped function on Snellius via ai4science.

    The wrapped function must be self-contained (no closures over outer
    variables; imports inside the function body) and its arguments/return
    value must be JSON-serializable.

    Set stream=True to print live log output while the call blocks
    (client-side tailing -- see Ai4ScienceClient.wait).

    Example
    -------
    >>> @job(base_url="https://ai4science.dev.sdp.surf.nl",
    ...      user="juliusa", token=slurm_token)
    ... def custom_sum(x, y):
    ...     return x + y
    >>> custom_sum(3, 4)   # blocks, runs remotely, returns 7
    7
    """
    client = Ai4ScienceClient(base_url=base_url, user=user, token=token)

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return client.run(
                func,
                *args,
                dependencies=dependencies,
                hf_token=hf_token,
                interval=interval,
                timeout=timeout,
                stream=stream,
                on_log=on_log,
                **kwargs,
            )

        wrapper.client = client  # escape hatch: .logs()/.results() access if needed
        return wrapper

    return decorator