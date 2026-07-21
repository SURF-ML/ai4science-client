"""Turn a plain Python function into a standalone script.

The generated script is exactly what the ``/ephemeral-job`` endpoint
expects as ``python_script``: it re-defines the function, calls it with
JSON-serializable args/kwargs, and prints the JSON-encoded return value
between two marker lines that the ai4science server already knows how
to parse (see ``app/utils/log_parser.py`` on the server).

This module has no network dependency and is fully self-testable.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from typing import Any, Callable

RESULT_START = "###JOB RESULT_START###"
RESULT_END = "###JOB RESULT_END###"


class NotSelfContainedError(ValueError):
    """Raised when a function can't be safely run as a standalone script."""


def _strip_decorators(source: str) -> str:
    """Drop any ``@...`` lines above ``def`` (e.g. the ``@job(...)`` line
    itself), since those names won't exist in the standalone script.
    """
    lines = source.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            return "\n".join(lines[i:])
    return source


def _check_self_contained(func: Callable) -> None:
    """Reject functions that close over outer variables.

    A function with free variables (``co_freevars``) depends on state
    that won't exist when the function body is re-defined standalone in
    a remote script -- this would fail confusingly on the cluster rather
    than clearly here, so we catch it up front.
    """
    freevars = func.__code__.co_freevars
    if freevars:
        raise NotSelfContainedError(
            f"Function '{func.__name__}' references outer variable(s) "
            f"{freevars!r}. Functions run via ai4science must be "
            "self-contained: put all needed values in its arguments, "
            "and do any imports inside the function body."
        )


def build_script(
    func: Callable,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Build a standalone Python script that runs ``func(*args, **kwargs)``.

    Parameters
    ----------
    func : callable
        A self-contained function (no closures over outer variables; any
        imports it needs must happen inside its own body).
    args, kwargs :
        Must be JSON-serializable (numbers, strings, lists, dicts).

    Returns
    -------
    str
        A complete script, suitable as the ``python_script`` field of an
        ``/ephemeral-job`` request.
    """
    kwargs = kwargs or {}
    _check_self_contained(func)

    source = _strip_decorators(textwrap.dedent(inspect.getsource(func)))

    try:
        args_json = json.dumps(list(args))
        kwargs_json = json.dumps(kwargs)
    except TypeError as e:
        raise ValueError(
            "args/kwargs must be JSON-serializable (numbers, strings, "
            f"lists, dicts). Original error: {e}"
        ) from e

    return f'''{source}

import base64
import json

_args = json.loads({args_json!r})
_kwargs = json.loads({kwargs_json!r})
_result = {func.__name__}(*_args, **_kwargs)
_encoded = base64.b64encode(json.dumps(_result).encode("utf-8")).decode("utf-8")
print("{RESULT_START}")
print(_encoded)
print("{RESULT_END}")
'''


if __name__ == "__main__":
    def custom_sum(x, y):
        return x + y

    script = build_script(custom_sum, args=(3, 4))
    assert "def custom_sum(x, y):" in script
    assert RESULT_START in script and RESULT_END in script

    # Prove the generated script actually runs standalone and produces
    # the expected marker output.
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == RESULT_START
    assert lines[-1] == RESULT_END
    decoded = json.loads(
        __import__("base64").b64decode(lines[1]).decode("utf-8")
    )
    assert decoded == 7

    # Closure rejection -- must be a real closure (nested function
    # referencing an enclosing local), not a module-level global.
    def _make_closure():
        outer = 10

        def bad_func(x):
            return x + outer

        return bad_func

    try:
        build_script(_make_closure(), args=(1,))
        raise AssertionError("expected NotSelfContainedError")
    except NotSelfContainedError:
        pass

    print("script_builder.py: all sanity checks passed")