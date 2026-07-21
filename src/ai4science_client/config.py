"""Config loading from environment variables (and a local .env file, if
one exists -- python-dotenv loads it automatically on import).

Nothing in the rest of this package requires this module.
Ai4ScienceClient can always be constructed directly with explicit
values. This exists purely as a convenience for callers who'd rather
keep base_url/credentials/interval in one place (a .env file) instead
of threading them through every call site.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Ai4ScienceConfig(BaseModel):
    """Config for one ai4science deployment, loaded from the environment.

    Environment variables
    ----------------------
    AI4SCIENCE_BASE_URL      required
    SLURM_USER               required
    SLURM_TOKEN               required
    AI4SCIENCE_POLL_INTERVAL  optional, default 10 (seconds)
    """

    base_url: str
    user: str
    token: str
    interval: int = Field(default=1, ge=1)

    @classmethod
    def from_env(cls) -> "Ai4ScienceConfig":
        missing = [
            name
            for name in ("AI4SCIENCE_BASE_URL", "SLURM_USER", "SLURM_TOKEN")
            if name not in os.environ
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them directly, or in a .env file (see .env.example)."
            )
        return cls(
            base_url=os.environ["AI4SCIENCE_BASE_URL"],
            user=os.environ["SLURM_USER"],
            token=os.environ["SLURM_TOKEN"],
            interval=int(os.environ.get("AI4SCIENCE_POLL_INTERVAL", "10")),
        )


if __name__ == "__main__":
    import unittest.mock as mock

    with mock.patch.dict(
        os.environ,
        {
            "AI4SCIENCE_BASE_URL": "https://example.org",
            "SLURM_USER": "u",
            "SLURM_TOKEN": "t",
        },
        clear=True,
    ):
        cfg = Ai4ScienceConfig.from_env()
        assert cfg.base_url == "https://example.org"
        assert cfg.interval == 1  # default, since AI4SCIENCE_POLL_INTERVAL unset

    with mock.patch.dict(
        os.environ,
        {
            "AI4SCIENCE_BASE_URL": "https://example.org",
            "SLURM_USER": "u",
            "SLURM_TOKEN": "t",
            "AI4SCIENCE_POLL_INTERVAL": "1",
        },
        clear=True,
    ):
        cfg = Ai4ScienceConfig.from_env()
        assert cfg.interval == 1

    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            Ai4ScienceConfig.from_env()
            raise AssertionError("expected ValueError for missing env vars")
        except ValueError as e:
            assert "AI4SCIENCE_BASE_URL" in str(e)

    print("config.py: all sanity checks passed")