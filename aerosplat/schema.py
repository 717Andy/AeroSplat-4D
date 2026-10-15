"""Common per-timestep sample schema.

One :class:`MultiCameraSample` is one query time on the reference clock. It
holds N camera observations. Cameras may differ in resolution, so images are
not stacked across the rig. Extrinsics are world-to-camera in the flight
frame (ENU, Z up). See ``docs/ARCHITECTURE.md`` for the coordinate contract
and for why capture-time and query-time poses are both stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _as_f32(value: Tensor | object, shape: tuple[int, ...], name: str) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.contiguous()


@dataclass
class CameraFrame:
    """One camera at one capture, expressed in the shared flight frame.

    ``R`` and ``t`` are the world-to-camera pose at ``timestamp`` (when the
    image and depth were captured): ``X_cam = R @ X_world + t``.

    ``R_query`` and ``t_query`` are that camera's pose interpolated onto the
    sample's query time. They match ``R`` and ``t`` when the frame was captured
    at the query time or when the sync policy is ``nearest``. Depth must be
    unprojected with the capture pose, not the query pose.
    """

    name: str
    image: Tensor
    K: Tensor
    R: Tensor
    t: Tensor
    timestamp: float
    depth: Tensor | None = None
    R_query: Tensor | None = None
    t_query: Tensor | None = None
    query_timestamp: float = 0.0
    sync: str = "nearest"
    frame_index: int = 0

    def __post_init__(self) -> None:
        image = torch.as_tensor(self.image, dtype=torch.float32).contiguous()
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{self.name}: image must be [3, H, W], got {tuple(image.shape)}")
        if not torch.isfinite(image).all():
            raise ValueError(f"{self.name}: image contains non-finite values")
        _, height, width = image.shape
        self.image = image
        self.K = _as_f32(self.K, (3, 3), f"{self.name}.K")
        self.R = _as_f32(self.R, (3, 3), f"{self.name}.R")
        self.t = _as_f32(self.t, (3,), f"{self.name}.t")
        self._check_rotation(self.R, f"{self.name}.R")
        if self.R_query is None:
            self.R_query = self.R.clone()
        else:
            self.R_query = _as_f32(self.R_query, (3, 3), f"{self.name}.R_query")
            self._check_rotation(self.R_query, f"{self.name}.R_query")
        if self.t_query is None:
            self.t_query = self.t.clone()
        else:
            self.t_query = _as_f32(self.t_query, (3,), f"{self.name}.t_query")
        if self.depth is not None:
            depth = torch.as_tensor(self.depth, dtype=torch.float32).contiguous()
            if tuple(depth.shape) != (height, width):
                raise ValueError(
                    f"{self.name}: depth shape {tuple(depth.shape)} != image {(height, width)}"
                )
            self.depth = depth
        if self.sync not in {"nearest", "interpolated", "clamped", "exact"}:
            raise ValueError(f"unknown sync state {self.sync!r}")
        if not (self.timestamp == self.timestamp and self.query_timestamp == self.query_timestamp):
            raise ValueError(f"{self.name}: timestamp is NaN")

    @staticmethod
    def _check_rotation(R: Tensor, name: str) -> None:
        eye = torch.eye(3, dtype=R.dtype)
        if not torch.allclose(R.T @ R, eye, atol=1e-3, rtol=1e-3):
            raise ValueError(f"{name} is not orthogonal")
        if torch.det(R) <= 0:
            raise ValueError(f"{name} is not a proper rotation")

    @property
    def height(self) -> int:
        return int(self.image.shape[1])

    @property
    def width(self) -> int:
        return int(self.image.shape[2])

    @property
    def image_dt(self) -> float:
        """Capture time minus query time, in seconds."""
        return float(self.timestamp) - float(self.query_timestamp)


@dataclass
class MultiCameraSample:
    """N synchronized cameras for one query timestamp.

    ``cameras`` keeps rig order. Resolutions may differ across names, so there
    is no ``[N, 3, H, W]`` image tensor on this object.
    """

    scene_id: str
    sample_index: int
    timestamp: float
    cameras: dict[str, CameraFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("MultiCameraSample requires at least one camera")
        if self.timestamp != self.timestamp:
            raise ValueError("sample timestamp is NaN")
        for name, camera in self.cameras.items():
            if camera.name != name:
                raise ValueError(f"camera key {name!r} != frame name {camera.name!r}")

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras.keys())

    def get(self, name: str) -> CameraFrame:
        try:
            return self.cameras[name]
        except KeyError as exc:
            raise KeyError(f"sample has no camera {name!r}; have {self.camera_names}") from exc
