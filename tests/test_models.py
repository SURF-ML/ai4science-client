"""runs(), list_models() and download_model() read your own runs and models
with your Slurm credentials; a download unpacks into <name>-v<n>/."""

import io
import zipfile
from unittest.mock import MagicMock

import pytest
import requests

from ai4science_client import Ai4ScienceClient
from ai4science_client import client as client_module

HEADERS = {"X-Slurm-User": "alice", "X-Slurm-Token": "tok"}


def _response(body=None, content=b"", headers=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code, resp.ok, resp.reason = 200, True, "OK"
    resp.json.return_value = body
    resp.content = content
    resp.headers = headers or {}
    return resp


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake(method, url, **kwargs):
        recorded.append((method, url, kwargs))
        if url.endswith("/download"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr("MLmodel", "flavors: {}")
            return _response(content=buf.getvalue(), headers={"X-Model-Version": "3"})
        return _response({"items": [{"name": "iris"}]})

    monkeypatch.setattr(client_module.requests, "request", fake)
    return recorded


def _client():
    return Ai4ScienceClient(base_url="https://srv", user="alice", token="tok")


def test_runs_and_models_are_read_as_the_caller(calls):
    c = _client()
    assert c.runs("iris") == [{"name": "iris"}]
    assert c.list_models() == [{"name": "iris"}]
    (_, runs_url, runs_kwargs), (_, models_url, models_kwargs) = calls
    assert runs_url == "https://srv/api/v1/mlflow/by-name/alice/iris/runs"
    assert models_url == "https://srv/api/v1/mlflow/models"
    assert models_kwargs["params"] == {"user": "alice"}
    assert runs_kwargs["headers"] == models_kwargs["headers"] == HEADERS


def test_download_unpacks_the_model_directory(calls, tmp_path):
    target = _client().download_model("iris", dest=tmp_path)
    assert target == tmp_path / "iris-v3"
    assert (target / "MLmodel").read_text() == "flavors: {}"
    assert calls[0][1].endswith("/models/iris/versions/latest/download")
