# ai4science-client

A thin Python client for running functions on Snellius through the
[Surf-AI4Science](https://github.com/SURF-ML) API. Point it at a running
k8s deployment, and it submits your function, waits for it to
finish on the cluster, and hands you back the return value.

No SLURM. No object storage. No container config. One API, two calling
styles.

## Structure

ai4cience-client/
├── .env
├── .env.example
├── src/
├── tests/
├── examples/
├── pyproject.toml
└── README.md
## Running HPC jobs via ai4science-client

### Install

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

`client.run(...)` and `@job(...)` submit the function to Snellius, block until it finishes, and return the result — same job, two calling styles. `stream=True` prints live log output while it runs.

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

- **`examples/run_bench_hf.py`** -- a GPU benchmark (`torch`, matrix
  multiply timing, real device info) and a HuggingFace sentiment-analysis
  pipeline, both declared via `dependencies=[...]` with `stream=True` so
  you can watch the install and run live. Verified against a real H100
  node on Snellius.

```bash
uv run python examples/run_bench_hf.py
```
