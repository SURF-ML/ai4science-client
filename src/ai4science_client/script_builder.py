"""Turn a plain Python function into a standalone script.

The generated script is exactly what the ``/ephemeral-job`` endpoint
expects as ``python_script``: it re-defines the function, calls it with
JSON-serializable args/kwargs, and prints the JSON-encoded return value
between two marker lines that the ai4science server already knows how
to parse (see ``app/common/utils/log_parser.py`` on the server).

If ``artifacts`` is given, the generated script downloads each file
from S3 *before* calling the function, and passes each one in as a
plain keyword argument -- see build_script's docstring for the exact
contract. The user's function never imports anything artifact-related
or calls anything special; it just receives a local path like any
other argument. Downloading uses S3 credentials the server already
injects into the job's own environment (S3_ACCESS_KEY, S3_SECRET_KEY,
S3_ENDPOINT, S3_ARTIFACTS_BUCKET) -- this script never carries any
credentials of its own.

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


def _build_artifact_download_block(artifacts: dict[str, str]) -> str:
    """Build the glue code that downloads each artifact and binds it
    into _kwargs under its name, run just before the function call.
    Only called when artifacts is non-empty.
    """
    artifacts_json = json.dumps(artifacts)
    return f'''
_artifact_keys = json.loads({artifacts_json!r})
if _artifact_keys:
    import boto3
    from botocore.client import Config as _BotoConfig

    _s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        config=_BotoConfig(signature_version="s3v4", s3={{"addressing_style": "path"}}),
    )
    for _artifact_name, _artifact_key in _artifact_keys.items():
        if _artifact_name in _kwargs:
            raise ValueError(
                f"artifact name '{{_artifact_name}}' collides with an "
                "explicitly passed keyword argument of the same name"
            )
        _local_path = "/tmp/artifact_" + _artifact_key.replace("/", "_")
        _s3.download_file(os.environ["S3_ARTIFACTS_BUCKET"], _artifact_key, _local_path)
        _kwargs[_artifact_name] = _local_path
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
        Maps a parameter name on ``func`` to an S3 key (already
        resolved by the caller -- see Ai4ScienceClient._upload_artifacts,
        which uploads each local file and hands back its key before
        build_script is ever called). For each entry, the generated
        script downloads that file from S3 *before* calling ``func``,
        and passes the local path it was written to as a keyword
        argument under that name -- so ``func`` just needs an ordinary
        parameter with that name, e.g.::

            def read_csv(data_path):
                with open(data_path) as f:
                    ...

            build_script(read_csv, artifacts={"data_path": "artifacts/u/abc/data.csv"})

        There's no special import, no injected global, nothing the
        function needs to know about beyond receiving a path string
        like any other argument. An artifact name that collides with
        an explicitly passed keyword argument raises ValueError inside
        the generated script (caught as a job failure, not silently
        overwritten) -- avoid passing the same name via ``kwargs`` and
        ``artifacts``.

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

    artifact_block = _build_artifact_download_block(artifacts) if artifacts else ""

    return f'''{source}
import base64
import json
import os
_args = json.loads({args_json!r})
_kwargs = json.loads({kwargs_json!r})
{artifact_block}
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

    # Artifact binding: the artifact name becomes a plain kwarg -- no
    # global, no special import in the user's function. We don't hit S3
    # here, just confirm the generated script's shape is correct.
    def uses_artifact(data_path):
        return data_path

    art_script = build_script(
        uses_artifact, artifacts={"data_path": "artifacts/u/abc/file.csv"}
    )
    assert "def uses_artifact(data_path):" in art_script
    assert '"data_path": "artifacts/u/abc/file.csv"' in art_script
    assert "get_artifact" not in art_script  # old mechanism fully gone
    assert art_script.index("_artifact_keys = json.loads") < art_script.index(
        "_result = uses_artifact"
    )

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