"""On-disk frame records before they are synchronized into a sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameRecord:
    """One captured frame. ``R``, ``t`` are world-to-camera at ``timestamp``."""

    frame_index: int
    timestamp: float
    image_path: Path
    depth_path: Path | None
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    depth_kind: str = "planar_z"
    segmentation_path: Path | None = None
    clean_image_path: Path | None = None

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        if self.depth_path is not None:
            self.depth_path = Path(self.depth_path)
        if self.segmentation_path is not None:
            self.segmentation_path = Path(self.segmentation_path)
        if self.clean_image_path is not None:
            self.clean_image_path = Path(self.clean_image_path)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        if self.depth_kind not in {"planar_z", "euclidean"}:
            raise ValueError(f"unknown depth_kind {self.depth_kind!r}")


@dataclass(frozen=True)
class SyncConfig:
    """How per-camera streams are aligned onto the reference clock.

    ``nearest`` keeps each camera's capture pose (geometry matches the image).
    ``interpolate_pose`` still uses the nearest image, but also stores a pose
    interpolated to the query time on ``CameraFrame.R_query``. Capture-time
    ``R`` and ``t`` are left untouched so depth unprojection stays consistent.
    """

    policy: str = "interpolate_pose"
    reference_camera: str = "front"
    max_image_dt: float = 0.05
    require_all_cameras: bool = True
    camera_order: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.policy not in {"nearest", "interpolate_pose"}:
            raise ValueError(
                f"sync policy must be 'nearest' or 'interpolate_pose', got {self.policy!r}"
            )
        if self.max_image_dt < 0:
            raise ValueError("max_image_dt must be >= 0")


def sync_config_from_dict(data: dict) -> SyncConfig:
    order = data.get("camera_order")
    return SyncConfig(
        policy=str(data.get("policy", "interpolate_pose")),
        reference_camera=str(data.get("reference_camera", "front")),
        max_image_dt=float(data.get("max_image_dt", 0.05)),
        require_all_cameras=bool(data.get("require_all_cameras", True)),
        camera_order=tuple(order) if order else None,
    )
