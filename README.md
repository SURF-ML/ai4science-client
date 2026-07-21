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
## Install

```bash
git clone https://github.com/SURF-ML/ai4cience-client.git
cd ai4cience-client
uv sync
```

## Config

Copy the example env file and fill in your real values:

```bash
cp .env.example .env
```

```bash
export AI4SCIENCE_BASE_URL="https://ai4science.dev.sdp.surf.nl"
export SLURM_USER="your_snellius_user"
export SLURM_TOKEN="..."

# optional, default 10 (seconds) -- used by wait()/run() when a call
# doesn't specify its own interval
export AI4SCIENCE_POLL_INTERVAL=10
```

`.env` is gitignored and loaded automatically -- nothing else to configure.

## Usage

### Client

```python
from ai4science_client import Ai4ScienceClient

client = Ai4ScienceClient.from_env()   # reads .env

def custom_sum(x, y):
    return x + y

result = client.run(custom_sum, 3, 4)  # 7
```

Or construct explicitly, without a `.env` file:

```python
client = Ai4ScienceClient()
```

### Decorator

```python
from ai4science_client import job

@job(
    base_url="https://ai4science.dev.sdp.surf.nl",
    user="your_snellius_user",
    token=your_slurm_token,
)
def custom_sum(x, y):
    return x + y

result = custom_sum(3, 4)  # 7
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
