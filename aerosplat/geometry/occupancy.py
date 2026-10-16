"""Voxel occupancy from depth maps and poses.

Depth sees surfaces, not solid interiors. The grid therefore has three states:

* ``unknown`` (0): no ray passed through the voxel and no surface landed in it.
* ``free`` (1): at least one camera ray traversed the voxel before its hit.
* ``occupied`` (2): at least one depth sample landed in the voxel.

Occupied overwrites free. Building interiors stay unknown unless some other
view sees them. A binary occupancy target for the later BCE loss is the
occupied mask; unknown voxels must be masked out, not treated as empty.
Thin wires are occupied only where a depth sample actually hits them, so the
voxel size has to be small enough to catch that sample. A single flight-corridor
grid at centimeter resolution does not fit in memory; ``extent_m`` builds a
local grid around the vehicle instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerosplat.geometry.camera import camera_center

UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


@dataclass
class OccupancyGrid:
    state: np.ndarray
    hit_count: np.ndarray
    origin: np.ndarray
    voxel_size: float
    timestamp: float = 0.0
    scene_id: str = ""

    def __post_init__(self) -> None:
        if self.state.ndim != 3:
            raise ValueError("state must be rank-3")
        if self.hit_count.shape != self.state.shape:
            raise ValueError("hit_count shape must match state")
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)

    @property
    def dims(self) -> tuple[int, int, int]:
        return (
            int(self.state.shape[0]),
            int(self.state.shape[1]),
            int(self.state.shape[2]),
        )

    @property
    def occupied(self) -> np.ndarray:
        return self.state == OCCUPIED

    @property
    def free(self) -> np.ndarray:
        return self.state == FREE

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state=self.state.astype(np.uint8),
            hit_count=self.hit_count.astype(np.uint16),
            origin=self.origin.astype(np.float64),
            voxel_size=np.float64(self.voxel_size),
            timestamp=np.float64(self.timestamp),
            scene_id=np.asarray(self.scene_id),
        )

    @classmethod
    def load(cls, path: Path) -> OccupancyGrid:
        with np.load(path, allow_pickle=False) as data:
            scene = data["scene_id"]
            scene_id = str(scene.item()) if scene.shape == () else str(scene)
            return cls(
                state=data["state"].astype(np.uint8),
                hit_count=data["hit_count"].astype(np.uint16),
                origin=data["origin"].astype(np.float64),
                voxel_size=float(data["voxel_size"]),
                timestamp=float(data["timestamp"]),
                scene_id=scene_id,
            )


def grid_around_point(
    center: np.ndarray,
    *,
    voxel_size: float,
    extent_m: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Axis-aligned grid centered on ``center``. ``extent_m`` is the full side length."""
    if voxel_size <= 0 or extent_m <= 0:
        raise ValueError("voxel_size and extent_m must be positive")
    n = max(1, int(np.ceil(extent_m / voxel_size)))
    origin = np.asarray(center, dtype=np.float64).reshape(3) - 0.5 * n * voxel_size
    return origin, (n, n, n)


def _bin_points(
    points: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    dims: tuple[int, int, int],
) -> np.ndarray:
    """Return flat indices of points that land inside the grid."""
    if points.size == 0:
        return np.zeros((0,), dtype=np.int64)
    ijk = np.floor((points - origin) / voxel_size).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.array(dims, dtype=np.int64)), axis=1)
    ijk = ijk[inside]
    flat = ijk[:, 0] * dims[1] * dims[2] + ijk[:, 1] * dims[2] + ijk[:, 2]
    return flat


def occupancy_from_depths(
    depths: list[np.ndarray],
    Ks: list[np.ndarray],
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    *,
    origin: np.ndarray,
    voxel_size: float,
    dims: tuple[int, int, int],
    max_depth: float = 80.0,
    min_depth: float = 1e-3,
    stride: int = 1,
    depth_kind: str = "planar_z",
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse depth maps into ``(state, hit_count)`` arrays.

    Rays are sampled at half-voxel steps. Samples before the surface vote free.
    The surface sample votes occupied and increments ``hit_count``. Occupied
    wins over free when the two disagree.
    """
    if not (len(depths) == len(Ks) == len(Rs) == len(ts)):
        raise ValueError("depth/pose lists must be the same length")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    state = np.zeros(dims, dtype=np.uint8)
    hit_count = np.zeros(dims, dtype=np.uint16)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    step = 0.5 * float(voxel_size)
    flat_size = int(np.prod(dims))
    chunk = 4096

    for depth, K, R, tvec in zip(depths, Ks, Rs, ts, strict=True):
        z_map = np.asarray(depth, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vs, us = np.mgrid[0 : z_map.shape[0] : stride, 0 : z_map.shape[1] : stride]
        z = z_map[vs, us]
        if depth_kind == "euclidean":
            z = z / np.sqrt(((us - cx) / fx) ** 2 + ((vs - cy) / fy) ** 2 + 1.0)
        elif depth_kind != "planar_z":
            raise ValueError(f"unknown depth_kind {depth_kind!r}")
        valid = np.isfinite(z) & (z > min_depth) & (z <= max_depth)
        if not np.any(valid):
            continue
        u = us[valid].astype(np.float64)
        v = vs[valid].astype(np.float64)
        z = z[valid]
        dirs_c = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(z)], axis=-1)
        dirs_c /= np.linalg.norm(dirs_c, axis=-1, keepdims=True)
        dirs_w = dirs_c @ R
        origin_w = camera_center(R, tvec)
        # Planar z = range * dirs_c[..., 2], so range = z / dirs_c_z.
        range_m = z / np.clip(dirs_c[:, 2], 1e-6, None)
        for start in range(0, range_m.shape[0], chunk):
            sl = slice(start, start + chunk)
            dirs = dirs_w[sl]
            dist = range_m[sl]
            n_steps = int(np.ceil(float(dist.max()) / step)) + 1
            samples = np.arange(1, n_steps, dtype=np.float64) * step
            if samples.size:
                before_hit = samples[None, :] < (dist[:, None] - step)
                if np.any(before_hit):
                    pts = origin_w[None, None, :] + dirs[:, None, :] * samples[None, :, None]
                    free_idx = _bin_points(pts[before_hit], origin, voxel_size, dims)
                    if free_idx.size:
                        free_mask = np.zeros(flat_size, dtype=bool)
                        free_mask[np.unique(free_idx)] = True
                        state[(state == UNKNOWN) & free_mask.reshape(dims)] = FREE
            hit_pts = origin_w[None, :] + dirs * dist[:, None]
            hit_idx = _bin_points(hit_pts, origin, voxel_size, dims)
            if hit_idx.size:
                counts = np.bincount(hit_idx, minlength=flat_size).astype(np.uint32)
                state[counts.reshape(dims) > 0] = OCCUPIED
                summed = hit_count.astype(np.uint32) + counts.reshape(dims)
                hit_count = np.minimum(summed, np.iinfo(np.uint16).max).astype(np.uint16)
    return state, hit_count
