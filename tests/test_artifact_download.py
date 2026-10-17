"""Artifacts are downloaded inside the job through the presigned URL the
server returns, with the standard library only: no S3 keys, no boto3.
Bare S3 keys from older servers still use the boto3 path."""

import base64
import http.server
import json
import os
import subprocess
import sys
import threading
import uuid
from unittest.mock import MagicMock

import pytest
import requests

from ai4science_client import Ai4ScienceClient
from ai4science_client import client as client_module
from ai4science_client.script_builder import RESULT_START, build_script


def read_artifact(data_path):
    with open(data_path) as f:
        return [data_path, f.read()]


@pytest.fixture
def artifact_server():
    """Serves one object at /<bucket>/<key>, only with the right signature."""
    key = f"artifacts/alice/{uuid.uuid4().hex}/data.csv"
    seen: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            if self.path != f"/ai4science-artifacts/{key}?X-Amz-Signature=ok":
                self.send_response(403)
                self.end_headers()
                return
            body = b"a,b\n1,2\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/ai4science-artifacts/{key}"
    yield key, base, seen
    server.shutdown()
    local = "/tmp/artifact_" + key.replace("/", "_")
    if os.path.exists(local):
        os.remove(local)


def _run(script: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("S3_")}
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1"
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )


def test_job_downloads_through_the_url_without_any_s3_keys(artifact_server):
    key, base, seen = artifact_server
    url = base + "?X-Amz-Signature=ok"
    proc = _run(build_script(read_artifact, artifacts={"data_path": url}))
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == RESULT_START
    path, content = json.loads(base64.b64decode(lines[1]))
    assert path == "/tmp/artifact_" + key.replace("/", "_")
    assert content == "a,b\n1,2\n"
    assert seen == [f"/ai4science-artifacts/{key}?X-Amz-Signature=ok"]


def test_a_rejected_url_fails_the_job_without_printing_the_url(artifact_server):
    _, base, _ = artifact_server
    url = base + "?X-Amz-Signature=wrong"
    proc = _run(build_script(read_artifact, artifacts={"data_path": url}))
    assert proc.returncode != 0
    assert "403" in proc.stderr
    assert "X-Amz-Signature=wrong" not in proc.stdout + proc.stderr


def test_bare_key_still_uses_boto3():
    script = build_script(
        read_artifact, artifacts={"data_path": "artifacts/u/abc/file.csv"}
    )
    assert "import boto3" in script
    assert '"data_path": "artifacts/u/abc/file.csv"' in script


def _response(body: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 201
    resp.ok = True
    resp.json.return_value = body
    return resp


@pytest.mark.parametrize(
    ("body", "expected", "boto3_added"),
    [
        (
            {"artifact_key": "artifacts/alice/1/d.csv", "download_url": "https://s/x"},
            "https://s/x",
            False,
        ),
        ({"artifact_key": "artifacts/alice/1/d.csv"}, "artifacts/alice/1/d.csv", True),
    ],
)
def test_client_prefers_the_url_and_only_adds_boto3_for_keys(
    monkeypatch, tmp_path, body, expected, boto3_added
):
    local = tmp_path / "d.csv"
    local.write_text("x")
    monkeypatch.setattr(
        client_module.requests, "request", lambda *a, **k: _response(body)
    )
    client = Ai4ScienceClient(base_url="https://srv", user="alice", token="tok")
    assert client._upload_artifacts({"data_path": str(local)}) == {
        "data_path": expected
    }

    submitted = {}

    class Stop(Exception):
        pass

    def fake_submit(script, **kwargs):
        submitted.update(kwargs, script=script)
        raise Stop

    monkeypatch.setattr(client, "submit", fake_submit)
    with pytest.raises(Stop):
        client.run(read_artifact, artifacts={"data_path": str(local)})
    assert ("boto3" in submitted["dependencies"]) is boto3_added
    assert expected in submitted["script"]
