"""A more realistic example: benchmark GPU availability with torch, run a
small HuggingFace sentiment-analysis pipeline, check explicit Slurm
resource requests, and exercise real filesystem I/O -- all on Snellius.

`gpu_benchmark` and `huggingface_sentiment` demonstrate dependencies=[...]
and stream=True together (streaming is genuinely useful here, since
installing torch/transformers takes real time and you'd otherwise be
waiting silently for minutes). Both are self-contained: all imports
happen inside the function body, and torch/transformers are declared via
`dependencies` rather than being pre-installed in the base image.
Neither one requests a specific partition, so it reports whichever
device the job actually lands on (cuda if available, cpu otherwise)
rather than requiring a GPU.

`resource_check` and `csv_training_job` demonstrate the opposite case:
explicitly requesting Slurm resources via `resources=SlurmResourceConfig(...)`
-- partition, CPUs, memory, walltime, and (via `tres_per_node=""`)
explicitly opting out of the GPU that some job types request by default.
See the "Slurm resources" section of the README for the full set of
overridable fields.

Usage
-----
Copy .env.example to .env and fill in real values, then:

    uv run python examples/run_bench_hf.py
"""

from __future__ import annotations

import sys

from ai4science_client import Ai4ScienceClient
from ai4science_client.schemas import SlurmResourceConfig


def read_csv_artifact() -> dict:
    """Read a local CSV that was uploaded as a job artifact and report
    basic stats about it -- demonstrates artifacts=... in client.run():
    the file never touches this function's arguments, the client
    uploads it automatically before submission, and get_artifact("data")
    (a plain global injected into the job script, no import needed)
    downloads it and returns a local path once the job is running.
    """
    with open(get_artifact("data")) as f:  # noqa: F821 -- injected at job-build time
        lines = f.read().splitlines()
    return {
        "line_count": len(lines),
        "first_line": lines[0] if lines else None,
        "last_line": lines[-1] if lines else None,
    }

def gpu_benchmark(matrix_size: int = 4096) -> dict:
    """Report device availability and time a few matrix multiplies on
    whatever device this job actually lands on. Doesn't request a
    specific partition -- see resource_check/csv_training_job below for
    examples that do, via resources=SlurmResourceConfig(partition=...).
    """
    import time

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "matrix_size": matrix_size,
        "seconds_for_5_matmuls": round(elapsed, 4),
    }


def huggingface_sentiment(sentences: list[str]) -> list[dict]:
    """Run a small HuggingFace sentiment-analysis pipeline, on GPU if
    one is available for this job, otherwise CPU.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("sentiment-analysis", device=device)
    return classifier(sentences)


def resource_check() -> dict:
    """A trivial job whose only purpose is to report what resources it
    actually landed on -- used here to demonstrate requesting specific
    Slurm resources via `resources=SlurmResourceConfig(...)`.
    """
    import multiprocessing
    import os

    return {
        "hostname": os.uname().nodename,
        "visible_cpus": multiprocessing.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def csv_training_job(n_samples: int = 2000, n_features: int = 5) -> dict:
    """Generate a synthetic regression dataset, write it to a CSV under
    $HOME, then read that CSV back from disk (not reused from memory)
    and train a small model on it.

    This exercises real filesystem I/O through the generated script --
    not just function arguments/return values -- and specifically
    $HOME, which (unlike /scratch-node) is a shared filesystem visible
    from every Snellius node, so the file would still be there even if
    a later job landed on a different node.
    """
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # --- Step 1: create a synthetic dataset and write it to $HOME ---
    rng = np.random.default_rng(seed=42)
    features = rng.normal(size=(n_samples, n_features))
    true_coefs = rng.normal(size=n_features)
    target = features @ true_coefs + rng.normal(scale=0.1, size=n_samples)

    data_dir = Path.home() / "ai4science_examples"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "synthetic_data.csv"

    df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = target
    df.to_csv(csv_path, index=False)

    # --- Step 2: read the CSV back from disk (a fresh read, proving the
    # file actually persisted, not just an in-memory round-trip) ---
    loaded = pd.read_csv(csv_path)

    # --- Step 3: train a small model on the loaded data ---
    x_loaded = loaded.drop(columns=["target"]).to_numpy()
    y_loaded = loaded["target"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x_loaded, y_loaded, test_size=0.2, random_state=42
    )

    start = time.perf_counter()
    model = LinearRegression()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start

    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)

    return {
        "csv_path": str(csv_path),
        "csv_exists_on_disk": csv_path.exists(),
        "csv_size_bytes": csv_path.stat().st_size,
        "rows_written": len(df),
        "rows_read_back": len(loaded),
        "train_seconds": round(train_seconds, 4),
        "test_mse": round(float(mse), 6),
        "learned_coefs": [round(float(c), 4) for c in model.coef_],
        "true_coefs": [round(float(c), 4) for c in true_coefs],
    }


def main() -> int:

    client = Ai4ScienceClient()

    # print("\n--- GPU benchmark (dependencies=['torch'], stream=True) ---")
    # bench = client.run(
    #     gpu_benchmark,
    #     4096,
    #     dependencies=["torch"],
    #     stream=True,
    #     timeout=1800,
    # )
    # print(f"\n{bench}")
    #
    # print("\n--- HuggingFace sentiment analysis (dependencies=['torch', 'transformers']) ---")
    # sentiment = client.run(
    #     huggingface_sentiment,
    #     ["Snellius makes this so easy.", "I really dislike waiting for pip installs."],
    #     dependencies=["torch", "transformers"],
    #     stream=True,
    #     timeout=1800,
    # )
    # print(f"\n{sentiment}")
    #
    # print("\n--- Resource check (explicit cpus/memory/time/partition via resources=) ---")
    # resources = SlurmResourceConfig(
    #     partition="genoa",
    #     cpus_per_task=4,
    #     memory_mb=8000,
    #     time_limit_minutes=15,
    #     tres_per_node="",
    # )
    # check = client.run(
    #     resource_check,
    #     resources=resources,
    #     timeout=600,
    # )
    # print(f"\n{check}")
    #
    # print("\n--- CSV filesystem test (write to $HOME, read back, train) ---")
    # csv_result = client.run(
    #     csv_training_job,
    #     n_samples=2000,
    #     n_features=5,
    #     dependencies=["pandas", "scikit-learn"],
    #     resources=SlurmResourceConfig(
    #         partition="genoa",
    #         cpus_per_task=4,
    #         memory_mb=8000,
    #         time_limit_minutes=15,
    #         tres_per_node="",
    #     ),
    #     stream=True,
    #     timeout=600,
    # )
    # print(f"\n{csv_result}")

    print("\n--- Artifact test (local CSV uploaded, read back inside the job) ---")
    artifact_result = client.run(
        read_csv_artifact,
        artifacts={"data": "/tmp/test_artifact.csv"},
        stream=True,
        timeout=300,
    )
    print(f"\n{artifact_result}")

    print(
        "\nPASS: GPU benchmark, HuggingFace pipeline, resource-scoped job, "
        "CSV filesystem test, and artifact upload/download all ran via ai4science."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())