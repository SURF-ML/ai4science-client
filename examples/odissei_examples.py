"""Preprocess on Snellius through Hippocampus: raw CSVs -> Parquet in S3.

One job in the preprocess image (it has boto3, pandas and pyarrow):
  1. download the raw CSVs from s3://odissee/raw/
  2. convert each CSV to Parquet
  3. upload the Parquet files to s3://odissee/processed/

Needs: AI4SCIENCE_BASE_URL, SLURM_USER, SLURM_JWT in the environment, and the
S3 keys in ~/.config/odissee/secrets.env on Snellius.
"""
import os
import sys

from ai4science_client import Ai4ScienceClient, Ai4ScienceJobFailedError
from ai4science_client.schemas import SlurmResourceConfig

PREPROCESS_IMAGE = "cr.surf.nl/mcc-mcc/odissee-poc/preprocess:latest"

DOWNLOAD_RAW = """
import boto3, os
s3 = boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'])
for f in ['benchmark_dataset.csv', 'hlt2_sept_dec_2025.csv']:
    print('Downloading ' + f, flush=True)
    s3.download_file('odissee', 'raw/' + f, f)
"""

CONVERT = """
import pandas as pd
for f in ['benchmark_dataset.csv', 'hlt2_sept_dec_2025.csv']:
    df = pd.read_csv(f)
    df.to_parquet(f.replace('.csv', '.parquet'), index=False)
    print(f'Converted {f}: {len(df)} rows, {len(df.columns)} columns', flush=True)
"""

UPLOAD_PARQUET = """
import boto3, os
s3 = boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'])
for f in ['benchmark_dataset.parquet', 'hlt2_sept_dec_2025.parquet']:
    print('Uploading ' + f, flush=True)
    s3.upload_file(f, 'odissee', 'processed/' + f)
"""


def run_step(commands, workdir, env):
    import os, pwd, subprocess          # imports inside: only this function is shipped

    home = pwd.getpwuid(os.getuid()).pw_dir
    workdir = workdir.replace("{home}", home)   # shared by all calls of one run
    os.makedirs(workdir, exist_ok=True)
    full_env = dict(os.environ, HOME=home, PYTHONUNBUFFERED="1", **env)
    secrets = os.path.join(home, ".config", "odissee", "secrets.env")
    if os.path.exists(secrets):
        for line in open(secrets):
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                full_env[key] = value
    for cmd in commands:
        cmd = [p.replace("{home}", home).replace("{workdir}", workdir) for p in cmd]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=workdir, env=full_env, check=True)  # a failure fails the job


def main(run_name: str) -> None:
    client = Ai4ScienceClient(
        base_url=os.environ["AI4SCIENCE_BASE_URL"],
        user=os.environ["SLURM_USER"],
        token=os.environ["SLURM_JWT"],
    )
    env = {"AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", "https://objectstore.surf.nl")}
    try:
        client.run(
            run_step,
            [
                ["python", "-c", DOWNLOAD_RAW],
                ["python", "-c", CONVERT],
                ["python", "-c", UPLOAD_PARQUET],
            ],
            f"{{home}}/odissee/work/{run_name}",
            env,
            container=PREPROCESS_IMAGE,
            resources=SlurmResourceConfig(
                partition="genoa", cpus_per_task=2, memory_mb=4000,
                time_limit_minutes=10, tres_per_node="",
            ),
            stream=True,
            timeout=6 * 3600,
        )
    except Ai4ScienceJobFailedError as e:
        sys.exit(f"Snellius job failed: {e}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manual")