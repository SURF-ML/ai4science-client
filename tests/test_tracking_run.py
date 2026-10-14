"""job.tracking_run() returns the MLflow run, and turns server errors into
Ai4ScienceAPIError carrying the server's own explanation."""

from unittest.mock import MagicMock

import pytest
import requests

from ai4science_client import (
    Ai4ScienceAPIError,
    Ai4ScienceClient,
    Ai4ScienceConnectionError,
)
from ai4science_client.client import Ai4ScienceJob

RUN = {"run_id": "abc", "status": "FINISHED", "params": {}, "metrics": {}, "tags": {}}


def _response(status: int, body: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.json.return_value = body
    resp.text = str(body)
    resp.reason = "x"
    return resp


@pytest.fixture
def job(monkeypatch):
    client = Ai4ScienceClient(base_url="https://srv", user="u", token="t")
    return Ai4ScienceJob(client, "27658713")


def test_returns_the_run(job, monkeypatch):
    calls = []

    def fake(method, url, **kwargs):
        calls.append((method, url, kwargs.get("timeout")))
        return _response(200, RUN)

    monkeypatch.setattr(requests, "request", fake)
    assert job.tracking_run() == RUN
    assert calls == [("GET", "https://srv/api/v1/mlflow/27658713/run", 30)]


@pytest.mark.parametrize("status", [404, 503])
def test_server_errors_carry_the_servers_explanation(job, monkeypatch, status):
    detail = "The experiment tracking server (MLflow) is unreachable right now."
    monkeypatch.setattr(
        requests, "request", lambda *a, **k: _response(status, {"detail": detail})
    )
    with pytest.raises(Ai4ScienceAPIError) as info:
        job.tracking_run()
    assert info.value.status_code == status
    assert info.value.detail == detail


def test_unreachable_api_is_a_connection_error(job, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("dropped")

    monkeypatch.setattr(requests, "request", boom)
    with pytest.raises(Ai4ScienceConnectionError):
        job.tracking_run()
