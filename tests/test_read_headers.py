"""Every job read and artifact upload sends the caller's Slurm credentials,
so the server can check the job (or upload prefix) belongs to them
(X-Slurm-User / X-Slurm-Token)."""

from unittest.mock import MagicMock

import requests

from ai4science_client import Ai4ScienceClient
from ai4science_client import client as client_module

EXPECTED = {"X-Slurm-User": "alice", "X-Slurm-Token": "tok"}


def _response(status: int, body) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.json.return_value = body
    resp.text = body if isinstance(body, str) else str(body)
    resp.reason = "x"
    return resp


def _client() -> Ai4ScienceClient:
    return Ai4ScienceClient(base_url="https://srv", user="alice", token="tok")


def test_logs_sends_slurm_headers(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(url=url, headers=kwargs.get("headers"))
        return _response(200, "log text")

    monkeypatch.setattr(client_module.requests, "get", fake_get)
    assert _client().logs("42") == "log text"
    assert seen == {"url": "https://srv/logs/42", "headers": EXPECTED}


def test_results_sends_slurm_headers(monkeypatch):
    seen = {}
    body = {"job_id": "42", "status": "completed", "exit_code": 0, "result": None}

    def fake_get(url, **kwargs):
        seen.update(url=url, headers=kwargs.get("headers"))
        return _response(200, body)

    monkeypatch.setattr(client_module.requests, "get", fake_get)
    _client().results("42")
    assert seen == {"url": "https://srv/results/42", "headers": EXPECTED}


def test_tracking_run_sends_slurm_headers(monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(url=url, headers=kwargs.get("headers"))
        return _response(200, {"run_id": "r"})

    monkeypatch.setattr(client_module.requests, "request", fake_request)
    _client().tracking_run("42")
    assert seen == {"url": "https://srv/api/v1/mlflow/42/run", "headers": EXPECTED}


def test_artifact_upload_sends_slurm_headers(monkeypatch, tmp_path):
    seen = {}
    path = tmp_path / "data.txt"
    path.write_text("x")

    def fake_request(method, url, **kwargs):
        seen.update(
            method=method,
            url=url,
            params=kwargs.get("params"),
            headers=kwargs.get("headers"),
        )
        return _response(201, {"artifact_key": "alice/k", "download_url": "https://dl"})

    monkeypatch.setattr(client_module.requests, "request", fake_request)
    assert _client()._upload_artifacts({"data": str(path)}) == {"data": "https://dl"}
    assert seen == {
        "method": "POST",
        "url": "https://srv/artifacts",
        "params": {"user": "alice"},
        "headers": EXPECTED,
    }
