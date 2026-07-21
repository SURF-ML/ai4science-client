import base64
import json
import subprocess
import sys

import pytest

from ai4science_client.script_builder import (
    RESULT_END,
    RESULT_START,
    NotSelfContainedError,
    build_script,
)


def custom_sum(x, y):
    return x + y


def uses_stdlib_import():
    import math

    return math.sqrt(16)


def _run_script(script: str):
    """Actually execute a generated script and return its decoded result."""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == RESULT_START
    assert lines[-1] == RESULT_END
    return json.loads(base64.b64decode(lines[1]).decode("utf-8"))


def test_build_script_contains_function_source():
    script = build_script(custom_sum, args=(3, 4))
    assert "def custom_sum(x, y):" in script
    assert RESULT_START in script
    assert RESULT_END in script


def test_generated_script_actually_runs_and_returns_correct_value():
    script = build_script(custom_sum, args=(3, 4))
    assert _run_script(script) == 7


def test_kwargs_supported():
    script = build_script(custom_sum, kwargs={"x": 10, "y": 5})
    assert _run_script(script) == 15


def test_function_with_internal_import_works():
    script = build_script(uses_stdlib_import)
    assert _run_script(script) == 4.0


def test_closure_over_outer_variable_is_rejected():
    def make_closure():
        outer = 10

        def bad_func(x):
            return x + outer

        return bad_func

    with pytest.raises(NotSelfContainedError):
        build_script(make_closure(), args=(1,))


def test_non_json_serializable_args_raise_value_error():
    with pytest.raises(ValueError):
        build_script(custom_sum, args=(object(), 1))