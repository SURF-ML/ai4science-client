"""A more realistic example: benchmark GPU availability with torch, and
run a small HuggingFace sentiment-analysis pipeline on Snellius --
demonstrating dependencies=[...] and stream=True together (streaming is
genuinely useful here, since installing torch/transformers takes real
time and you'd otherwise be waiting silently for minutes).

Both functions are self-contained: all imports happen inside the
function body, and torch/transformers are declared via `dependencies`
rather than being pre-installed in the base image.

Neither function assumes a GPU is actually allocated -- the current
/ephemeral-job endpoint doesn't let the client request a specific
partition, so this reports whichever device the job actually lands on
(cuda if available, cpu otherwise) rather than failing if it's CPU-only.

Usage
-----
Copy .env.example to .env and fill in real values, then:

    uv run python examples/run_bench_hf.py
"""

from __future__ import annotations

import sys

from ai4science_client import Ai4ScienceClient


def gpu_benchmark(matrix_size: int = 4096) -> dict:
    """Report device availability and time a few matrix multiplies on
    whatever device this job actually lands on.
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


def main() -> int:

    client = Ai4ScienceClient()

    print("\n--- GPU benchmark (dependencies=['torch'], stream=True) ---")
    bench = client.run(
        gpu_benchmark,
        4096,
        dependencies=["torch"],
        stream=True,
        timeout=1800,
    )
    print(f"\n{bench}")

    print("\n--- HuggingFace sentiment analysis (dependencies=['torch', 'transformers']) ---")
    sentiment = client.run(
        huggingface_sentiment,
        ["Snellius makes this so easy.", "I really dislike waiting for pip installs."],
        dependencies=["torch", "transformers"],
        stream=True,
        timeout=1800,
    )
    print(f"\n{sentiment}")

    print("\nPASS: GPU benchmark and HuggingFace pipeline both ran via ai4science.")
    return 0


if __name__ == "__main__":
    sys.exit(main())