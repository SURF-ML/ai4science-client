"""runtime= submits through /api/v2/jobs (UUID job ids); without it every
call works exactly as before. Logs are read by offset for both kinds."""

import base64
import json
from unittest.mock import MagicMock

import pytest
import requests

from ai4science_client import Ai4ScienceClient
from ai4science_client import client as client_module

UUID = "0123456789abcdef0123456789abcdef"
V2_JOB = {
    "id": UUID,
    "runtime": "apptainer",
    "container": "base",
    "status": "SUBMITTED",
    "slurm_job_id": 4242,
    "created_at": "2026-11-02T18:00:00Z",
}


def _response(status: int, body=None, content: bytes = b"", headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.json.return_value = body
    resp.content = content
    resp.text = content.decode(errors="replace") if content else str(body)
    resp.reason = "x"
    resp.headers = headers or {}
    return resp


class FakeServer:
    """Answers the client's requests; records (method, url, kwargs)."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, str, dict]] = []
        self.log = bytearray()
        self.offsets = True  # False: an older server that ignores ?offset=
        self.result = {"status": "running"}
        monkeypatch.setattr(client_module.requests, "request", self.request)
        monkeypatch.setattr(client_module.requests, "get", self.get)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        path = url.removeprefix("https://srv")
        if method == "POST" and path == "/api/v2/jobs":
            body = kwargs["json"]
            return _response(201, V2_JOB | {"runtime": body["runtime"]})
        if method == "POST" and path == "/ephemeral-job":
            return _response(200, {"job_id": 77, "status": "SUBMITTED"})
        if path == f"/api/v2/jobs/{UUID}":
            return _response(200, V2_JOB)
        if path.endswith("/logs") or path.startswith("/logs/"):
            if not self.log:
                return _response(404, {"detail": "Log not found yet"})
            if not self.offsets:
                return _response(200, content=bytes(self.log))
            offset = kwargs["params"]["offset"]
            return _response(
                200,
                content=bytes(self.log[offset:]),
                headers={"X-Next-Offset": str(max(len(self.log), offset))},
            )
        if path == f"/api/v2/jobs/{UUID}/results":
            return _response(200, {"id": UUID, **self.result})
        if path == "/api/v1/mlflow/4242/run":
            return _response(200, {"run_id": "r"})
        raise AssertionError(f"unexpected {method} {url}")

    def urls(self, method="GET"):
        return [u.removeprefix("https://srv") for m, u, _ in self.calls if m == method]


@pytest.fixture
def server(monkeypatch):
    return FakeServer(monkeypatch)


def _client() -> Ai4ScienceClient:
    return Ai4ScienceClient(base_url="https://srv", user="alice", token="tok", interval=0)


def test_without_runtime_nothing_changes(server):
    job = _client().submit("print(1)")
    assert server.urls("POST") == ["/ephemeral-job"]
    assert (job.job_id, job.runtime, job.slurm_job_id) == ("77", None, 77)


def test_runtime_submits_to_v2(server):
    job = _client().submit("print(1)", dependencies=["numpy"], runtime="apptainer")
    (_, url, kwargs), = server.calls
    assert url == "https://srv/api/v2/jobs"
    assert kwargs["json"] == {
        "runtime": "apptainer",
        "script": "print(1)",
        "dependencies": ["numpy"],
        "modules": [],
        "track": False,
        "user": "alice",
        "token": "tok",
    }
    assert (job.job_id, job.runtime, job.container, job.slurm_job_id) == (
        UUID,
        "apptainer",
        "base",
        4242,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"runtime": "apptainer", "tier": "auto"},
        {"runtime": "ray", "cluster": "snellius"},
        {"modules": ["Python"]},
        {"container": "base"},
    ],
)
def test_combinations_that_cannot_work_fail_before_any_request(server, kwargs):
    with pytest.raises(ValueError):
        _client().submit("print(1)", **kwargs)
    assert server.calls == []


@pytest.mark.parametrize("job_id", [UUID, "77"])
def test_read_log_by_offset(server, job_id):
    c = _client()
    assert c.read_log(job_id) == (b"", 0)
    server.log += b"hello\n"
    assert c.read_log(job_id, 0) == (b"hello\n", 6)
    server.log += b"world\n"
    assert c.read_log(job_id, 6) == (b"world\n", 12)
    prefix = f"/api/v2/jobs/{UUID}/logs" if job_id == UUID else "/logs/77"
    assert set(server.urls()) == {prefix}


def test_read_log_from_a_server_without_offsets(server):
    server.offsets = False
    server.log += b"hello\nworld\n"
    assert _client().read_log("77", 6) == (b"world\n", 12)


def test_wait_streams_only_new_bytes_and_decodes_split_characters(server):
    c = _client()
    shown: list[str] = []
    encoded = base64.b64encode(json.dumps({"answer": 42}).encode()).decode()
    pieces = [
        "step 1 \u2713\n".encode()[:-2],  # ends inside the 3-byte "\u2713"
        "step 1 \u2713\n".encode()[-2:] + b"step 2\n",
        f"###JOB RESULT_START###\n{encoded}\n###JOB RESULT_END###\ndone\n".encode(),
    ]

    def results(job_id):
        if pieces:
            server.log += pieces.pop(0)
            return client_module.JobResult(job_id=job_id, status="running")
        return client_module.JobResult(job_id=job_id, status="completed", result=42)

    c.results = results
    result = c.wait(UUID, stream=True, on_log=shown.append)

    assert result.result == 42
    assert "".join(shown) == (
        "step 1 \u2713\nstep 2\n[result received: {'answer': 42}]\ndone\n"
    )
    asked = [kw["params"]["offset"] for _, _, kw in server.calls]
    assert asked == sorted(asked) and asked[-1] == len(server.log)


def test_v2_results_and_tracking_run(server):
    c = _client()
    server.result = {"status": "completed", "exit_code": 0, "result": [1, 2]}
    result = c.results(UUID)
    assert (result.job_id, result.status, result.result) == (UUID, "completed", [1, 2])

    assert c.tracking_run(UUID) == {"run_id": "r"}  # looks up the Slurm job id
    assert server.urls()[-2:] == [f"/api/v2/jobs/{UUID}", "/api/v1/mlflow/4242/run"]
    assert c.job(UUID).slurm_job_id == 4242


def test_run_with_eessi_sends_modules(server):
    server.result = {"status": "completed", "exit_code": 0, "result": 3}

    def add(x, y):
        return x + y

    assert _client().run(add, 1, 2, runtime="eessi", modules=["SciPy-bundle"]) == 3
    body = server.calls[0][2]["json"]
    assert (body["runtime"], body["modules"], body["dependencies"]) == (
        "eessi",
        ["SciPy-bundle"],
        [],
    )
