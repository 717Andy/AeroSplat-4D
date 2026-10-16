"""Locate repository configs without assuming the process cwd."""

from __future__ import annotations

import os
from pathlib import Path


def find_configs_dir() -> Path:
    """Return the Hydra config directory.

    Search order: ``AEROSPLAT_CONFIGS``, the repo ``configs/`` next to this
    package (editable installs), then ``./configs``.
    """
    env = os.environ.get("AEROSPLAT_CONFIGS")
    if env:
        path = Path(env).expanduser().resolve()
        if not (path / "config.yaml").is_file():
            raise FileNotFoundError(f"AEROSPLAT_CONFIGS={path} has no config.yaml")
        return path

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "configs",
        Path.cwd() / "configs",
    ]
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find configs/config.yaml. Set AEROSPLAT_CONFIGS or run from the repo root."
    )


def find_repo_root() -> Path:
    """Return the directory that contains ``configs/`` and ``aerosplat/``."""
    return find_configs_dir().parent
