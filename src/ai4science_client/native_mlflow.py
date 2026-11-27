"""Your runs and models, read straight from the MLflow server with your own
MLflow login (Ai4ScienceClient's mlflow_token/mlflow_username).

Used when the client has an MLflow login: the deployment's tracking server
then has its own accounts and the ai4science API holds no login there.
The server only says where MLflow is and which name prefixes are yours
(GET /api/v1/mlflow/config). Needs the optional extra:
pip install 'ai4science-client[mlflow]'.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from .exceptions import Ai4ScienceError

_LOGIN_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
)


def _mlflow():
    try:
        import mlflow  # noqa: PLC0415
        import mlflow.artifacts  # noqa: F401, PLC0415
    except ImportError as e:
        raise Ai4ScienceError(
            "Reading runs and models with your MLflow login needs mlflow: "
            "pip install 'ai4science-client[mlflow]'"
        ) from e
    return mlflow


class NativeMlflow:
    """config: the server's GET /api/v1/mlflow/config answer for you."""

    def __init__(self, config: dict[str, str], username: str | None, token: str):
        self.config = config
        self._username = username
        self._token = token

    @contextlib.contextmanager
    def _login(self):
        """mlflow reads its login from the environment (artifact downloads only
        from there): set it for one call, then restore what was there."""
        saved = {name: os.environ.pop(name, None) for name in _LOGIN_VARS}
        os.environ["MLFLOW_TRACKING_URI"] = self.config["tracking_uri"]
        if self._username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._token
        else:
            os.environ["MLFLOW_TRACKING_TOKEN"] = self._token
        try:
            yield _mlflow()
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None)
                if value is not None:
                    os.environ[name] = value

    def _ui_url(self, run) -> str:
        return (
            f"{self.config['ui_base_url']}/#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        )

    def run(self, run_id: str) -> dict[str, Any]:
        """Same shape as GET /api/v1/mlflow/{job_id}/run."""
        with self._login() as mlflow:
            run = mlflow.MlflowClient().get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "tags": dict(run.data.tags),
            "ui_url": self._ui_url(run),
        }

    def runs(self, experiment_name: str, limit: int) -> list[dict[str, Any]]:
        """Same shape as the API's run list: newest first."""
        name = self.config["experiment_prefix"] + experiment_name
        with self._login() as mlflow:
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(name)
            if experiment is None:
                return []
            runs = client.search_runs(
                [experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=limit,
            )
        return [
            {
                "run_id": r.info.run_id,
                "job_id": r.data.tags.get("job_id"),
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": dict(r.data.metrics),
                "ui_url": self._ui_url(r),
            }
            for r in runs
        ]

    def download_model(self, name: str, version: int | str, dest: Path) -> Path:
        """Into dest/<name>-v<n>/, like the API's download."""
        registered = self.config["model_prefix"] + name
        with self._login() as mlflow:
            if str(version) == "latest":
                model = mlflow.MlflowClient().get_registered_model(registered)
                version = max(int(v.version) for v in model.latest_versions)
            target = Path(dest) / f"{name}-v{version}"
            target.mkdir(parents=True, exist_ok=True)
            # The model directory: the path mlflow wrote, inside target.
            return Path(
                mlflow.artifacts.download_artifacts(
                    f"models:/{registered}/{version}", dst_path=str(target)
                )
            )
