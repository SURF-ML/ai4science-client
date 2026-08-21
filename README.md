# ai4science-client

A thin Python client for running functions on Snellius through the
[Surf-AI4Science](https://github.com/SURF-ML) API. The client uses existing hippocampus infrastructure to submit jobs to slurm.

No SLURM. No object storage. No container config. One API, two calling
styles.

> **Want the full tour?** [`examples/demo_notebook.ipynb`](examples/demo_notebook.ipynb)
> walks through every feature below in one runnable notebook -- dataset
> provisioning, OpenML/HF jobs, artifacts, and automatic tier routing.

> **Full API reference:** the underlying FastAPI service's interactive
> docs are at [ai4science.dev.sdp.surf.nl/docs](https://ai4science.dev.sdp.surf.nl/docs).

## Structure

ai4cience-client/
├── .env
├── .env.example
├── src/
├── tests/
├── examples/
├── pyproject.toml
└── README.md
## Install

Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    'ai4science-client @ git+https://github.com/SURF-ML/ai4science-client.git',
]
```

```bash
uv pip install -e .
```

## Running HPC jobs via ai4science-client

### Example

```python
from ai4science_client import Ai4ScienceClient, job

_BASE_URL = "https://ai4science.dev.sdp.surf.nl"
_USER = ""      # your Snellius username
_TOKEN = ""     # SLURM JWT (scontrol token)

client = Ai4ScienceClient(base_url=_BASE_URL, user=_USER, token=_TOKEN)

def custom_sum(x, y):
    return x + y

print(client.run(custom_sum, 1, 2, stream=True))


@job(base_url=_BASE_URL, user=_USER, token=_TOKEN, stream=True)
def custom_sum_decorated(x, y):
    return x + y

print(custom_sum_decorated(1, 2))
```

`client.run(...)` and `@job(...)` submit the function to Snellius, block
until it finishes, and return the result — same job, two calling
styles. `stream=True` prints live log output while it runs.

### Run it

```bash
uv run python your_script.py
```

### Dependencies

Any non-stdlib packages your function needs are declared per-call via
`dependencies`, installed in an isolated overlay on top of the shared
base image -- nothing to pre-build or configure:

```python
def gpu_benchmark(matrix_size: int = 4096) -> dict:
    import torch  # imported inside the function, not at module level

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    return {"device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None}

result = client.run(gpu_benchmark, 4096, dependencies=["torch"])
```

### Slurm resources

By default your job runs on the server's built-in defaults for its job
type (partition, CPUs, memory, walltime, GPU). To override any subset of
these, pass a `SlurmResourceConfig` -- anything you leave unset falls
back to the server default, nothing is required:

```python
from ai4science_client import Ai4ScienceClient
from ai4science_client.schemas import SlurmResourceConfig

client = Ai4ScienceClient()

resources = SlurmResourceConfig(
    partition="genoa",        # see Snellius partitions: rome/thin, genoa,
                               # fat_rome, fat_genoa, himem_4tb, himem_8tb,
                               # gpu_a100, gpu_h100, gpu_mig, gpu_vis, staging
    cpus_per_task=4,
    memory_mb=8000,
    time_limit_minutes=15,
)

def resource_check() -> dict:
    import os
    return {"partition": os.environ.get("SLURM_JOB_PARTITION")}

result = client.run(resource_check, resources=resources)
```

Works identically with the decorator:

```python
from ai4science_client import job
from ai4science_client.schemas import SlurmResourceConfig

@job(
    base_url="https://ai4science.dev.sdp.surf.nl",
    user="your_snellius_user",
    token=your_slurm_token,
    resources=SlurmResourceConfig(
        partition="gpu_h100",
        cpus_per_task=16,
        memory_mb=96000,
        time_limit_minutes=240,
        tres_per_node="gres:gpu:1",
    ),
)
def train_step(x, y):
    ...

result = train_step(3, 4)
```

`submit()` accepts the same `resources=` argument if you're using the
async/manual submission style below.

### Artifacts

Need your function to work with a local file -- a CSV, an image, a text
file, anything? Pass `artifacts=` mapping a parameter name on your
function to a local path. The file is uploaded automatically before
the job runs; your function just receives the downloaded file's local
path as that parameter -- no special import, no function call, nothing
extra to learn:

```python
from ai4science_client import Ai4ScienceClient

client = Ai4ScienceClient()

def read_csv_artifact(data_path) -> dict:
    with open(data_path) as f:
        lines = f.read().splitlines()
    return {"line_count": len(lines), "first_line": lines[0]}

result = client.run(
    read_csv_artifact,
    artifacts={"data_path": "./my_local_data.csv"},
)
```

Works identically with the decorator:

```python
from ai4science_client import job

@job(
    base_url="https://ai4science.dev.sdp.surf.nl",
    user="your_snellius_user",
    token=your_slurm_token,
    artifacts={"data_path": "./my_local_data.csv"},
)
def read_csv_artifact(data_path) -> dict:
    with open(data_path) as f:
        lines = f.read().splitlines()
    return {"line_count": len(lines), "first_line": lines[0]}

result = read_csv_artifact()
```

Note the artifact dict key (`"data_path"` here) must match the
parameter name exactly -- that's how the downloaded file gets bound to
the right argument. Don't also pass that same name positionally or via
another keyword.

### Streaming

Pass `stream=True` to see log output printed live while the call blocks
(polls `/logs` and prints only what's new each round). Works with both
the client and the decorator:

```python
result = client.run(custom_sum, 3, 4, stream=True, interval=5)
```

```python
@job(base_url=..., user=..., token=..., stream=True, interval=5)
def custom_sum(x, y):
    return x + y
```

### Async

`submit()` returns immediately without blocking. Check in or wait
whenever you like:

```python
from ai4science_client import build_script

script = build_script(custom_sum, args=(3, 4))
job_handle = client.submit(script)   # does not block

job_handle.results()                 # non-blocking status check
result = job_handle.wait()           # block until done, when ready
```

## Examples

- **[`examples/demo_notebook.ipynb`](examples/demo_notebook.ipynb)** --
  an interactive, cell-by-cell tour of the API: dataset provisioning,
  running a job via OpenML, an EESSI module-based job, Hugging Face
  with explicit resources and a local artifact, and automatic tier
  routing across compute clusters. The best starting point if you want
  to see everything in action before writing your own code.

- **`examples/run_bench_hf.py`** -- a GPU benchmark (`torch`, matrix
  multiply timing, real device info), a HuggingFace sentiment-analysis
  pipeline (both declared via `dependencies=[...]` with `stream=True` so
  you can watch the install and run live), a small resource-scoped job
  demonstrating explicit `resources=SlurmResourceConfig(...)`, a
  filesystem test that writes/reads a CSV under `$HOME`, and an
  artifact test that uploads a local CSV via `artifacts=...` and reads
  it back inside the job. Verified against a real H100 node on
  Snellius.

```bash
uv run python examples/run_bench_hf.py
```