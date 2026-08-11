"""Turn a plain Python function into a standalone script.

The generated script is exactly what the ``/ephemeral-job`` endpoint
expects as ``python_script``: it re-defines the function, calls it with
JSON-serializable args/kwargs, and prints the JSON-encoded return value
between two marker lines that the ai4science server already knows how
to parse (see ``app/common/utils/log_parser.py`` on the server).

If ``artifacts`` is given, a small preamble is injected *before* the
function source, defining a module-level ``get_artifact(name)`` helper
and the name->S3-key mapping it uses. Because the preamble and the
function source are both module-level code in the same generated
script, the function body can call ``get_artifact(...)`` even though
it was never imported or defined where the user originally wrote the
function -- Python resolves globals at call time, not definition time.
The user only ever refers to artifacts by logical name; they never see
an S3 key, bucket, or credential. Downloading uses S3 credentials the
server already injects into the job's own environment (S3_ACCESS_KEY,
S3_SECRET_KEY, S3_ENDPOINT, S3_ARTIFACTS_BUCKET) -- this script never
carries any credentials of its own.

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


def _get_source(func: Callable) -> str:
    """Wrap inspect.getsource with a clear, actionable error.

    getsource raises OSError when it can't locate the function's source
    -- typically because func was defined interactively (a REPL/Jupyter
    cell), via exec()/eval(), or its source file was deleted/moved since
    import. That's a real and fairly common way this gets hit (people
    iterating in notebooks before submitting a job), so it gets the same
    NotSelfContainedError treatment as an actual closure, rather than a
    raw OSError with no context.
    """
    try:
        return inspect.getsource(func)
    except OSError as e:
        raise NotSelfContainedError(
            f"Could not read the source of function '{func.__name__}'. "
            "This usually means it was defined interactively (a REPL or "
            "Jupyter cell) or via exec()/eval() -- functions run via "
            "ai4science must be defined in a regular .py file that's "
            "importable/readable at submit time."
        ) from e
    except TypeError as e:
        # getsource also raises TypeError for builtins or other objects
        # that were never valid Python source in the first place.
        raise NotSelfContainedError(
            f"'{func.__name__}' does not appear to be a plain Python "
            "function with retrievable source (e.g. a builtin or "
            "C-implemented callable). Only regular Python functions "
            "defined in a .py file are supported."
        ) from e


def _build_artifact_preamble(artifacts: dict[str, str]) -> str:
    """Build the module-level get_artifact() helper + key mapping,
    injected before the user's function source. Only called when
    artifacts is non-empty.
    """
    artifacts_json = json.dumps(artifacts)
    return f'''import json as _artifact_json
import os as _artifact_os

_ARTIFACT_KEYS = _artifact_json.loads({artifacts_json!r})


def get_artifact(name):
    """Download a job input artifact by its logical name and return the
    local path it was written to. Uses S3 credentials already present
    in this job's environment -- nothing else to configure."""
    import boto3
    from botocore.client import Config as _BotoConfig

    key = _ARTIFACT_KEYS[name]
    local_path = "/tmp/artifact_" + key.replace("/", "_")
    s3 = boto3.client(
        "s3",
        endpoint_url=_artifact_os.environ["S3_ENDPOINT"],
        aws_access_key_id=_artifact_os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=_artifact_os.environ["S3_SECRET_KEY"],
        config=_BotoConfig(signature_version="s3v4", s3={{"addressing_style": "path"}}),
    )
    s3.download_file(_artifact_os.environ["S3_ARTIFACTS_BUCKET"], key, local_path)
    return local_path


'''


def build_script(
    func: Callable,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> str:
    """Build a standalone Python script that runs ``func(*args, **kwargs)``.

    Parameters
    ----------
    func : callable
        A self-contained function (no closures over outer variables; any
        imports it needs must happen inside its own body), defined in a
        regular .py file (not the REPL, a notebook cell, or exec'd code).
    args, kwargs :
        Must be JSON-serializable (numbers, strings, lists, dicts).
    artifacts : dict[str, str] | None
        Logical name -> S3 key, already resolved by the caller (see
        Ai4ScienceClient._upload_artifacts). If given, the generated
        script gains a module-level ``get_artifact(name)`` the function
        body can call to download that file and get back a local path.

    Returns
    -------
    str
        A complete script, suitable as the ``python_script`` field of an
        ``/ephemeral-job`` request.

    Raises
    ------
    NotSelfContainedError
        If func closes over outer variables, or its source can't be
        retrieved (interactive/exec'd definition, builtin, etc.).
    ValueError
        If args/kwargs aren't JSON-serializable.
    """
    kwargs = kwargs or {}
    _check_self_contained(func)
    source = _strip_decorators(textwrap.dedent(_get_source(func)))

    try:
        args_json = json.dumps(list(args))
        kwargs_json = json.dumps(kwargs)
    except TypeError as e:
        raise ValueError(
            "args/kwargs must be JSON-serializable (numbers, strings, "
            f"lists, dicts). Original error: {e}"
        ) from e

    artifact_preamble = _build_artifact_preamble(artifacts) if artifacts else ""

    return f'''{artifact_preamble}{source}
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
    decoded = json.loads(__import__("base64").b64decode(lines[1]).decode("utf-8"))
    assert decoded == 7

    # Artifact preamble: get_artifact must be defined and callable from
    # the function body (we don't actually hit S3 here -- just prove the
    # preamble is syntactically valid and injects before the function).
    def uses_artifact():
        return get_artifact("data")  # noqa: F821 -- injected at script-build time

    art_script = build_script(uses_artifact, artifacts={"data": "artifacts/u/abc/file.csv"})
    assert "def get_artifact(name):" in art_script
    assert art_script.index("def get_artifact") < art_script.index("def uses_artifact")
    assert '"data": "artifacts/u/abc/file.csv"' in art_script

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

    # Unreadable-source rejection -- exec-defined function has no
    # retrievable source, same failure mode as an interactive/notebook def.
    ns: dict[str, Any] = {}
    exec("def exec_defined(x):\n    return x", ns)
    try:
        build_script(ns["exec_defined"], args=(1,))
        raise AssertionError("expected NotSelfContainedError")
    except NotSelfContainedError:
        pass

    print("script_builder.py: all sanity checks passed")