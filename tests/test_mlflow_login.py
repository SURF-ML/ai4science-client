"""The caller's own MLflow login travels only with tracked apptainer jobs,
from arguments or from MLflow's own environment variables."""

import pytest

import ai4science_client.config  # noqa: F401  (loads .env before the env is patched)
from ai4science_client import Ai4ScienceClient
from tests.test_runtime_jobs import FakeServer

MLFLOW_VARS = (
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
)


@pytest.fixture
def server(monkeypatch):
    return FakeServer(monkeypatch)


@pytest.fixture
def env(monkeypatch):
    for name in MLFLOW_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI4SCIENCE_BASE_URL", "https://srv")
    monkeypatch.setenv("SLURM_USER", "alice")
    monkeypatch.setenv("SLURM_TOKEN", "tok")
    return monkeypatch


def _body(server) -> dict:
    ((_, _, kwargs),) = server.calls
    return kwargs["json"]


def _client(**kwargs) -> Ai4ScienceClient:
    return Ai4ScienceClient(
        base_url="https://srv", user="alice", token="tok", interval=0, **kwargs
    )


def test_a_token_goes_with_a_tracked_v2_job(server):
    _client(mlflow_token="m").submit("print(1)", runtime="apptainer", track=True)
    body = _body(server)
    assert body["mlflow_token"] == "m"
    assert "mlflow_username" not in body


def test_a_username_goes_with_its_token_to_ephemeral_job(server):
    client = _client(mlflow_token="m", mlflow_username="jane.doe")
    client.submit("print(1)", track=True)
    body = _body(server)
    assert server.urls("POST") == ["/ephemeral-job"]
    assert (body["mlflow_token"], body["mlflow_username"]) == ("m", "jane.doe")


@pytest.mark.parametrize(
    "kwargs", [{}, {"runtime": "apptainer"}, {"runtime": "ray", "track": True}]
)
def test_no_login_without_tracking_or_for_ray(server, kwargs):
    _client(mlflow_token="m", mlflow_username="j").submit("print(1)", **kwargs)
    body = _body(server)
    assert "mlflow_token" not in body
    assert "mlflow_username" not in body


def test_a_username_without_a_token_is_dropped(server):
    _client(mlflow_username="j").submit("print(1)", runtime="apptainer", track=True)
    assert "mlflow_username" not in _body(server)


def test_username_and_password_from_the_environment(server, env):
    env.setenv("MLFLOW_TRACKING_USERNAME", "jane.doe")
    env.setenv("MLFLOW_TRACKING_PASSWORD", "pw")
    env.setenv("MLFLOW_TRACKING_TOKEN", "ignored")
    client = Ai4ScienceClient(interval=0)
    assert (client.mlflow_username, client.mlflow_token) == ("jane.doe", "pw")


def test_a_token_from_the_environment(server, env):
    env.setenv("MLFLOW_TRACKING_USERNAME", "jane.doe")  # no password: not basic auth
    env.setenv("MLFLOW_TRACKING_TOKEN", "t")
    client = Ai4ScienceClient(interval=0)
    assert (client.mlflow_username, client.mlflow_token) == (None, "t")


def test_no_mlflow_login_in_the_environment(server, env):
    client = Ai4ScienceClient(interval=0)
    assert (client.mlflow_username, client.mlflow_token) == (None, None)
