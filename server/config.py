"""Workbench settings. Paths are relative to the repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    """Runtime configuration for the API process."""

    repo_root: Path = field(default_factory=repo_root)
    checkpoint: Path | None = None
    model_name: str = "vit_tiny"
    device: str = "auto"
    episode_dirs: list[Path] = field(default_factory=list)
    grid_size: tuple[int, int, int] = (32, 32, 20)
    voxel_size: float = 0.5
    occ_origin: tuple[float, float, float] = (-2.0, -6.0, -1.0)
    host: str = "127.0.0.1"
    port: int = 8000

    def resolve(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.repo_root / candidate).resolve()


def _as_path_list(value: object, root: Path) -> list[Path]:
    if not value:
        return []
    items = value if isinstance(value, list) else str(value).split(os.pathsep)
    paths = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        path = Path(text)
        paths.append(path if path.is_absolute() else (root / path).resolve())
    return paths


def load_settings(path: Path | None = None) -> Settings:
    """YAML file, then environment overrides.

    ``AEROSPLAT_CHECKPOINT``, ``AEROSPLAT_MODEL``, ``AEROSPLAT_DEVICE``, and
    ``AEROSPLAT_EPISODES`` (``os.pathsep`` separated) win over the file.
    """
    root = repo_root()
    file_path = path or root / "server" / "workbench.yaml"
    data: dict = {}
    if file_path.is_file():
        loaded = yaml.safe_load(file_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{file_path} must be a mapping")
        data = loaded
    checkpoint = data.get("checkpoint")
    grid = data.get("grid_size") or [32, 32, 20]
    origin = data.get("occ_origin") or [-2.0, -6.0, -1.0]
    settings = Settings(
        repo_root=root,
        checkpoint=None if checkpoint in (None, "", "null") else Path(str(checkpoint)),
        model_name=str(data.get("model_name", "vit_tiny")),
        device=str(data.get("device", "auto")),
        episode_dirs=_as_path_list(data.get("episode_dirs"), root),
        grid_size=(int(grid[0]), int(grid[1]), int(grid[2])),
        voxel_size=float(data.get("voxel_size", 0.5)),
        occ_origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        host=str(data.get("host", "127.0.0.1")),
        port=int(data.get("port", 8000)),
    )
    if os.environ.get("AEROSPLAT_CHECKPOINT"):
        settings.checkpoint = Path(os.environ["AEROSPLAT_CHECKPOINT"])
    if os.environ.get("AEROSPLAT_MODEL"):
        settings.model_name = os.environ["AEROSPLAT_MODEL"]
    if os.environ.get("AEROSPLAT_DEVICE"):
        settings.device = os.environ["AEROSPLAT_DEVICE"]
    if os.environ.get("AEROSPLAT_EPISODES"):
        settings.episode_dirs = _as_path_list(os.environ["AEROSPLAT_EPISODES"], root)
    if settings.checkpoint is not None and not settings.checkpoint.is_absolute():
        settings.checkpoint = (root / settings.checkpoint).resolve()
    return settings
