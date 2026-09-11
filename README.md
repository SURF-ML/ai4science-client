python3 << 'PYEOF'
path = "README.md"

with open(path) as f:
    content = f.read()

anchor = '''`submit()` accepts the same `resources=` argument if you're using the
async/manual submission style below.

### Artifacts'''

insertion = '''`submit()` accepts the same `resources=` argument if you're using the
async/manual submission style below.

### Multi-node Ray jobs

Pass `container="ray"` to run your function against a live, multi-node
Ray cluster instead of a single process. The server bootstraps the
cluster (head + workers across the allocated nodes) before your
function runs, so you can call `ray.init(address="auto")` directly --
no need to start Ray yourself:

````python
from ai4science_client import Ai4ScienceClient
from ai4science_client.schemas import SlurmResourceConfig

client = Ai4ScienceClient()

def train_distributed():
    import ray
    ray.init(address="auto")
    return {"nodes": len(ray.nodes()), "resources": ray.cluster_resources()}

result = client.run(
    train_distributed,
    container="ray",
    resources=SlurmResourceConfig(nodes=4, cpus_per_task=16, memory_mb=64000),
    stream=True,
)
````

`resources.nodes` controls cluster size (head + `nodes - 1` workers).
Every other `SlurmResourceConfig` field -- `cpus_per_task`, `memory_mb`,
`tres_per_node`, etc. -- describes what's requested on **each** node,
same fields as a single-node job, just applied per-node instead of once.

Works identically with the decorator:

````python
from ai4science_client import job
from ai4science_client.schemas import SlurmResourceConfig

@job(
    base_url="https://ai4science.dev.sdp.surf.nl",
    user="your_snellius_user",
    token=your_slurm_token,
    container="ray",
    resources=SlurmResourceConfig(nodes=4, cpus_per_task=16),
)
def train_distributed():
    import ray
    ray.init(address="auto")
    return {"nodes": len(ray.nodes())}

result = train_distributed()
````

`dependencies=`, `artifacts=`, and `stream=` all work exactly the same
as single-node jobs -- see the sections above. `tier=`/`cluster=`
(auto-tier-routing) are not supported with `container="ray"` and raise
`ValueError` if combined.

### Artifacts'''

assert anchor in content, "anchor text not found -- README may have changed"
content = content.replace(anchor, insertion, 1)

with open(path, "w") as f:
    f.write(content)

print("README.md updated")
PYEOF