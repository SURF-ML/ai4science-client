"""With an MLflow login the client reads runs and models from MLflow itself,
with that login set only for the duration of each call."""

import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

import ai4science_client.config  # noqa: F401  (loads .env before the env is patched)
from ai4science_client import Ai4ScienceClient
from ai4science_client.exceptions import Ai4ScienceError
from tests.test_runtime_jobs import V2_JOB, FakeServer, _response

CONFIG = {
    "tracking_uri": "https://mlflow.example.org",
    "ui_base_url": "https://mlflow.example.org",
    "experiment_prefix": "hippocampus/dev/alice/",
    "model_prefix": "hippocampus.dev.alice.",
}
LOGIN_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
)


def _run(run_id="r1", job_id="4242"):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id, experiment_id="7", status="FINISHED", start_time=1
        ),
        data=SimpleNamespace(
            params={"C": "1.0"}, metrics={"accuracy": 0.9}, tags={"job_id": job_id}
        ),
    )


@pytest.fixture
def mlflow(monkeypatch):
    """A stand-in mlflow that records each call and the login it saw."""
    calls: list[tuple] = []

    def seen():
        return {k: os.environ.get(k) for k in LOGIN_VARS if os.environ.get(k)}

    class FakeClient:
        def get_run(self, run_id):
            calls.append(("get_run", run_id, seen()))
            return _run(run_id)

        def get_experiment_by_name(self, name):
            calls.append(("get_experiment_by_name", name))
            return SimpleNamespace(experiment_id="7") if name.endswith("/iris") else None

        def search_runs(self, ids, order_by, max_results):
            calls.append(("search_runs", ids, max_results))
            return [_run()]

        def get_registered_model(self, name):
            calls.append(("get_registered_model", name))
            return SimpleNamespace(
                latest_versions=[SimpleNamespace(version="2"), SimpleNamespace(version="5")]
            )

    def download_artifacts(uri, dst_path):
        calls.append(("download", uri, seen()))
        return dst_path

    module = ModuleType("mlflow")
    module.MlflowClient = FakeClient
    module.artifacts = ModuleType("mlflow.artifacts")
    module.artifacts.download_artifacts = download_artifacts
    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setitem(sys.modules, "mlflow.artifacts", module.artifacts)
    for name in LOGIN_VARS:
        monkeypatch.delenv(name, raising=False)
    return calls


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer(monkeypatch)
    answer = fake.request

    def request(method, url, **kwargs):
        if url.endswith("/api/v1/mlflow/config"):
            fake.calls.append((method, url, kwargs))
            return _response(200, CONFIG)
        if method == "POST" and url.endswith("/api/v2/jobs"):
            fake.calls.append((method, url, kwargs))
            return _response(201, V2_JOB | {"mlflow_run_id": "r1"})
        return answer(method, url, **kwargs)

    monkeypatch.setattr(fake, "request", request)
    monkeypatch.setattr(
        "ai4science_client.client.requests.request", request, raising=True
    )
    return fake


def _client(**login) -> Ai4ScienceClient:
    return Ai4ScienceClient(
        base_url="https://srv", user="alice", token="tok", interval=0, **login
    )


def test_a_tracked_jobs_run_comes_from_mlflow_with_the_users_login(server, mlflow):
    client = _client(mlflow_token="pat", mlflow_username="jane.doe")
    job = client.submit("print(1)", runtime="apptainer", track=True)
    assert job.mlflow_run_id == "r1"
    run = job.tracking_run()
    assert run["metrics"] == {"accuracy": 0.9}
    assert run["ui_url"] == "https://mlflow.example.org/#/experiments/7/runs/r1"
    assert mlflow == [
        (
            "get_run",
            "r1",
            {
                "MLFLOW_TRACKING_URI": "https://mlflow.example.org",
                "MLFLOW_TRACKING_USERNAME": "jane.doe",
                "MLFLOW_TRACKING_PASSWORD": "pat",
            },
        )
    ]
    assert not any(os.environ.get(k) for k in LOGIN_VARS)  # restored


def test_runs_are_read_from_the_users_experiment(server, mlflow):
    runs = _client(mlflow_token="pat").runs("iris", limit=3)
    assert [r["job_id"] for r in runs] == ["4242"]
    assert ("get_experiment_by_name", "hippocampus/dev/alice/iris") in mlflow
    assert ("search_runs", ["7"], 3) in mlflow
    assert _client(mlflow_token="pat").runs("unknown", limit=3) == []


def test_the_latest_model_version_is_downloaded_with_a_bearer_token(
    server, mlflow, tmp_path
):
    path = _client(mlflow_token="t0k").download_model("iris", dest=tmp_path)
    assert path == tmp_path / "iris-v5"
    uri, login = next((c[1], c[2]) for c in mlflow if c[0] == "download")
    assert uri == "models:/hippocampus.dev.alice.iris/5"
    assert login["MLFLOW_TRACKING_TOKEN"] == "t0k"


def test_listing_models_and_runs_of_later_handles_are_not_available_yet(
    server, mlflow
):
    client = _client(mlflow_token="pat")
    with pytest.raises(Ai4ScienceError, match="Listing models"):
        client.list_models()
    with pytest.raises(Ai4ScienceError, match="submit"):
        client.tracking_run("4242")


def test_the_environment_login_is_restored_after_a_call(server, mlflow, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    _client(mlflow_token="pat").runs("iris", limit=1)
    assert os.environ["MLFLOW_TRACKING_URI"] == "file:./mlruns"
    assert "MLFLOW_TRACKING_TOKEN" not in os.environ
