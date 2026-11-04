"""Feature check of ai4science_client against a live deployment (.env).

Runs each feature as a separate check, in parallel where they submit jobs,
and prints a PASS/FAIL table at the end. Jobs are tiny (a few seconds of
Python), so they ask for a small CPU slot (CPU_SMALL, genoa) instead of the
server's default GPU partition, whose queue can take long.

    uv run python client_features.py              # the default set
    uv run python client_features.py v2 failure   # only these
    uv run python client_features.py --list       # what exists
    uv run python client_features.py --server-defaults   # no resources sent

tier sends no resources on purpose: the router picks the cluster/partition.

Checks that are not in the default set (slow or costly): ray, ray_old, byoc.
byoc runs in your own image: AI4S_BYOC_IMAGE (default: the backend's own
image on cr.surf.nl, which is private -- log in once on Snellius first).
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

from ai4science_client import (
    Ai4ScienceAPIError,
    Ai4ScienceClient,
    Ai4ScienceJobFailedError,
    Ai4ScienceMCPClient,
    build_script,
    job,
)
from ai4science_client.schemas import SlurmResourceConfig

CPU_SMALL = SlurmResourceConfig(
    partition="genoa",
    cpus_per_task=2,
    memory_mb=4000,
    time_limit_minutes=10,
    tres_per_node="",
)
EESSI_MODULES = ["foss/2023a", "SciPy-bundle/2023.07-gfbf-2023a"]
BYOC_IMAGE = os.environ.get(
    "AI4S_BYOC_IMAGE", "cr.surf.nl/hpml-ai4science_poc/ai4science_poc:ffd53047"
)
# What the job checks request; None (--server-defaults) = the server's default.
RES: SlurmResourceConfig | None = CPU_SMALL


# --- functions that run on Snellius (self-contained, no closures) ---


def ticks(tag):
    import os
    import socket
    import time

    for i in range(3):
        print(f"{tag}: tick {i}", flush=True)
        time.sleep(15)
    return {"tag": tag, "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID")}


def add(x, y):
    return x + y


def read_artifact(data_path):
    with open(data_path) as f:
        return f.read().strip()


def boom():
    raise RuntimeError("intentional failure from client_features.py")


def cpus_seen():
    import os

    return {
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus": str(len(os.sched_getaffinity(0))),  # CPUs this job may use
        "host": os.uname().nodename,
    }


def python_info():
    import os
    import sys

    return {"python": sys.version.split()[0], "host": os.uname().nodename}


def numpy_sum():
    import numpy as np

    return float(np.ones((100, 100)).sum())


def ray_nodes():
    import ray

    ray.init(address="auto")
    return len(ray.nodes())


# --- checks: each returns a short "what was seen" string or raises ---

c = Ai4ScienceClient()
shared: dict[str, str] = {}  # e.g. the UUID of the v2 job, for later checks


def check_server():
    h = requests.get(f"{c.base_url}/health/readiness", timeout=15)
    paths = requests.get(f"{c.base_url}/openapi.json", timeout=15).json()["paths"]
    v2 = sorted(p for p in paths if p.startswith("/api/v2/jobs"))
    assert h.status_code == 200, f"health {h.status_code}"
    assert len(v2) == 5 and "/logs/{job_id}/stream" in paths, v2
    return f"health 200, {len(v2)} v2 routes, SSE route present"


def check_old():
    result = c.run(ticks, "old", resources=RES)
    assert result["tag"] == "old", result
    return f"old endpoint OK on {result['host']} (Slurm {result['job']})"


def check_v2():
    pieces: list[str] = []
    jobh = c.submit(build_script(ticks, args=("v2",)), runtime="apptainer", resources=RES)
    assert len(jobh.job_id) == 32 and jobh.slurm_job_id, jobh
    shared["uuid"] = jobh.job_id
    r = jobh.wait(stream=True, on_log=pieces.append, timeout=7200)
    assert r.status == "completed" and r.result["tag"] == "v2", r
    text = "".join(pieces)
    assert "v2: tick 2" in text and "[result received:" in text, text[-300:]
    return (
        f"UUID {jobh.job_id[:8]}..., Slurm {jobh.slurm_job_id}, "
        f"log arrived in {len(pieces)} pieces"
    )


def check_reattach():
    uuid = shared.get("uuid")
    assert uuid, "needs the v2 check in the same run"
    again = c.job(uuid)
    data, nxt = c.read_log(uuid, 0)
    tail, nxt2 = c.read_log(uuid, nxt)
    assert again.runtime == "apptainer" and again.slurm_job_id, again
    assert data and nxt == len(data) and tail == b"" and nxt2 == nxt, (len(data), nxt, nxt2)
    rec = requests.get(f"{c.base_url}/api/v2/jobs/{uuid}", headers=c._auth_headers(), timeout=15)
    assert rec.json()["status"] == "COMPLETED", rec.json()
    return f"job({uuid[:8]}...) -> Slurm {again.slurm_job_id}; offset read {nxt} bytes, then 0; record COMPLETED"


def check_decorator():
    @job(
        base_url=c.base_url,
        user=c.user,
        token=c.token,
        runtime="apptainer",
        resources=RES,
    )
    def add_remote(x, y):
        return x + y

    assert add_remote(3, 4) == 7
    return "@job(runtime='apptainer') returned 7"


def check_artifacts():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("hello from an artifact\n")
    try:
        got = c.run(
            read_artifact,
            artifacts={"data_path": f.name},
            runtime="apptainer",
            resources=RES,
        )
    finally:
        os.unlink(f.name)
    assert got == "hello from an artifact", got
    return "uploaded with Slurm headers, read back inside the job"


def check_failure():
    try:
        c.run(boom, runtime="apptainer", resources=RES)
    except Ai4ScienceJobFailedError as e:
        assert e.status == "failed" and "/api/v2/jobs/" in str(e), str(e)
        rec = requests.get(
            f"{c.base_url}/api/v2/jobs/{e.job_id}", headers=c._auth_headers(), timeout=15
        ).json()
        assert rec["status"] == "FAILED", rec
        return f"Ai4ScienceJobFailedError, exit_code={e.exit_code}, record FAILED"
    raise AssertionError("a failing job did not raise")


def check_validation():
    seen = []
    for kwargs in (
        {"runtime": "apptainer", "tier": "auto"},
        {"modules": ["x"]},
        {"container": "base"},
    ):
        try:
            c.submit("print(1)", **kwargs)
        except ValueError:
            seen.append("local")
    try:
        c.submit("print(1)", runtime="apptainer", container="docker://evil/image")
    except Ai4ScienceAPIError as e:
        assert e.status_code == 422, e
        seen.append("422")
    assert seen == ["local", "local", "local", "422"], seen
    return "3 local ValueErrors, server 422 for a user-supplied container"


def check_readauth():
    uuid = shared.get("uuid")
    assert uuid, "needs the v2 check in the same run"
    unknown = c.results("0" * 32)
    wrong = Ai4ScienceClient(base_url=c.base_url, user=c.user, token="not-a-token")
    try:
        wrong.results(uuid)
    except Ai4ScienceAPIError as e:
        assert e.status_code in (401, 403), e
        return f"unknown job -> {unknown.status}; wrong token -> {e.status_code}"
    return f"unknown job -> {unknown.status}; wrong token NOT blocked (read_auth_required off?)"


def check_track():
    jobh = c.submit(
        build_script(add, args=(1, 2)),
        runtime="apptainer",
        resources=RES,
        track=True,
        experiment_name="client-features",
    )
    r = jobh.wait(timeout=7200)
    assert r.status == "completed" and r.result == 3, r
    run = jobh.tracking_run()
    return f"MLflow run {run.get('run_id', '?')[:8]}..., status {run.get('status')}"


def check_resources():
    asked = CPU_SMALL.model_copy(update={"cpus_per_task": 4, "memory_mb": 8000})
    r = c.run(cpus_seen, resources=asked, runtime="apptainer")
    assert r["partition"] == "genoa" and int(r["cpus"]) >= 4, r
    return f"asked genoa/4 CPUs, got {r['partition']}/{r['cpus']} on {r['host']}"


def check_tier():
    r = c.run(add, 2, 3, tier="auto")
    assert r == 5, r
    return "tier='auto' (old endpoint) returned 5"


def check_eessi():
    r = c.run(numpy_sum, runtime="eessi", modules=EESSI_MODULES, resources=RES)
    assert r == 10000.0, r
    return f"EESSI modules {EESSI_MODULES[-1]}: numpy sum {r}"


def check_ray():
    n = c.run(
        ray_nodes,
        runtime="ray",
        resources=SlurmResourceConfig(
            partition="rome", nodes=2, cpus_per_task=8, memory_mb=16000,
            time_limit_minutes=15, tres_per_node="",
        ),
    )
    assert n == 2, n
    return "runtime='ray': 2 nodes"


def check_ray_old():
    n = c.run(
        ray_nodes,
        container="ray",
        resources=SlurmResourceConfig(
            partition="rome", nodes=2, cpus_per_task=8, memory_mb=16000,
            time_limit_minutes=15, tres_per_node="",
        ),
    )
    assert n == 2, n
    return "container='ray' (old endpoint): 2 nodes"


def check_byoc():
    r = c.run(python_info, container=BYOC_IMAGE, resources=RES)
    return f"ran in {BYOC_IMAGE} (Python {r['python']}) on {r['host']}"


def check_mcp():
    async def go():
        async with Ai4ScienceMCPClient(base_url=c.base_url) as mcp:
            tools = await mcp.list_tools()
            community = await mcp.list_community_tools()
            return len(tools), len(community)

    n_tools, n_community = asyncio.run(go())
    assert n_tools > 0, n_tools
    return f"{n_tools} MCP tools, {n_community} community tools"


CHECKS = {
    "server": check_server,
    "validation": check_validation,
    "mcp": check_mcp,
    "old": check_old,
    "v2": check_v2,
    "decorator": check_decorator,
    "artifacts": check_artifacts,
    "failure": check_failure,
    "track": check_track,
    "resources": check_resources,
    "tier": check_tier,
    "eessi": check_eessi,
    "ray": check_ray,
    "ray_old": check_ray_old,
    "byoc": check_byoc,
    # These read what the v2 check produced, so they run after everything else.
    "reattach": check_reattach,
    "readauth": check_readauth,
}
DEFAULT = [n for n in CHECKS if n not in ("ray", "ray_old", "byoc")]
AFTER = ("reattach", "readauth")


def run_check(name):
    start = time.time()
    try:
        return name, "PASS", CHECKS[name](), time.time() - start
    except Exception as e:  # noqa: BLE001 -- report every failure, keep going
        detail = f"{type(e).__name__}: {e}"
        if not isinstance(e, (AssertionError, Ai4ScienceAPIError, Ai4ScienceJobFailedError)):
            detail += "\n" + traceback.format_exc(limit=3)
        return name, "FAIL", detail, time.time() - start


def main(argv):
    global RES
    if "--server-defaults" in argv:
        RES = None
    if "--list" in argv:
        print("\n".join(f"{n}{'' if n in DEFAULT else '  (not in default set)'}" for n in CHECKS))
        return 0
    names = [a for a in argv if not a.startswith("-")] or DEFAULT
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"unknown check(s): {unknown}; see --list")
        return 2
    first = [n for n in names if n not in AFTER]
    later = [n for n in names if n in AFTER]
    where = "server defaults" if RES is None else f"{RES.partition}, {RES.cpus_per_task} CPUs"
    print(f"API {c.base_url} as {c.user}; jobs on: {where}", flush=True)
    print(f"running: {', '.join(names)}\n", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(run_check, first):
            results.append(res)
            print(f"[{res[1]}] {res[0]} ({res[3]:.0f}s)", flush=True)
    for name in later:
        res = run_check(name)
        results.append(res)
        print(f"[{res[1]}] {res[0]} ({res[3]:.0f}s)", flush=True)
    print("\n" + "=" * 72)
    for name, status, detail, secs in results:
        print(f"{status:4}  {name:11} {secs:5.0f}s  {detail}")
    failed = sum(1 for r in results if r[1] == "FAIL")
    print("=" * 72 + f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
