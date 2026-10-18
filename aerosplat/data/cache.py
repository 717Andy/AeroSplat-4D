"""On-disk cache of occupancy grids built from a multi-camera sample."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.occupancy import (
    OccupancyGrid,
    fuse_occupancy,
    grid_around_point,
)
from aerosplat.schema import MultiCameraSample

# A 200³ uint8 volume is 8 MB. Larger than that, the fine wire crop should shrink.
_MAX_FINE_VOXELS = 8_000_000


@dataclass
class CacheConfig:
    root: Path
    voxel_size: float = 0.5
    extent_m: float = 12.0
    max_depth: float = 40.0
    min_depth: float = 0.05
    stride: int = 1
    # Segmentation ids (label maps) or RGB colors (AirSim palettes, 0–255)
    # that mean "this pixel is a cable". Empty means no wire class.
    wire_ids: tuple[int, ...] = ()
    wire_colors: tuple[tuple[int, int, int], ...] = ()
    wire_voxel_size: float = 0.02
    wire_extent_m: float = 4.0

    @classmethod
    def from_hydra(cls, cfg) -> CacheConfig:
        cache = cfg.cache
        ids = _as_tuple(getattr(cache, "wire_ids", ()) or ())
        raw_colors = getattr(cache, "wire_colors", ()) or ()
        colors = tuple(tuple(int(channel) for channel in color) for color in raw_colors)
        return cls(
            root=Path(str(cache.root)),
            voxel_size=float(cache.voxel_size),
            extent_m=float(cache.extent_m),
            max_depth=float(cache.max_depth),
            min_depth=float(cache.min_depth),
            stride=int(cache.stride),
            wire_ids=tuple(int(item) for item in ids),
            wire_colors=colors,  # type: ignore[arg-type]
            wire_voxel_size=float(getattr(cache, "wire_voxel_size", 0.02)),
            wire_extent_m=float(getattr(cache, "wire_extent_m", 4.0)),
        )


class GroundTruthCache:
    """Build or load an occupancy grid for one sample.

    The filename includes the voxel parameters and the query timestamp, so
    changing the grid config does not silently reuse an old file. Regenerating
    the source images at the same timestamps does; delete ``root`` in that case.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config

    def path_for(self, sample: MultiCameraSample) -> Path:
        cfg = self.config
        tag = (
            f"vs{cfg.voxel_size:.4f}_ext{cfg.extent_m:.2f}"
            f"_st{cfg.stride}_md{cfg.max_depth:.1f}_t{sample.timestamp:.6f}"
            f"{_wire_tag(cfg)}"
        )
        safe_tag = tag.replace(".", "p")
        return self.config.root / sample.scene_id / f"{sample.sample_index:06d}_{safe_tag}.npz"

    def get(self, sample: MultiCameraSample, *, center: np.ndarray | None = None) -> OccupancyGrid:
        path = self.path_for(sample)
        if path.is_file():
            grid = OccupancyGrid.load(path)
            if self._matches(grid, sample):
                return grid
        grid = build_occupancy(sample, self.config, center=center)
        grid.save(path)
        return grid

    def _matches(self, grid: OccupancyGrid, sample: MultiCameraSample) -> bool:
        return (
            abs(grid.voxel_size - self.config.voxel_size) < 1e-9
            and abs(grid.timestamp - sample.timestamp) < 1e-6
            and grid.scene_id == sample.scene_id
        )


def build_occupancy(
    sample: MultiCameraSample,
    config: CacheConfig,
    *,
    center: np.ndarray | None = None,
) -> OccupancyGrid:
    """Voxelize every camera that has depth, using capture-time extrinsics."""
    depths: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for frame in sample.cameras.values():
        if frame.depth is None:
            continue
        depths.append(frame.depth.detach().cpu().numpy())
        Ks.append(frame.K.detach().cpu().numpy())
        Rs.append(frame.R.detach().cpu().numpy())
        ts.append(frame.t.detach().cpu().numpy())
    wire_masks: list[np.ndarray | None] = []
    use_wires = bool(config.wire_ids or config.wire_colors)
    for frame in sample.cameras.values():
        if frame.depth is None:
            continue
        if use_wires and frame.segmentation is not None:
            wire_masks.append(
                segmentation_to_wire_mask(frame.segmentation, config.wire_ids, config.wire_colors)
            )
        else:
            wire_masks.append(None)
    if not depths:
        raise ValueError(f"sample {sample.scene_id}#{sample.sample_index} has no depth maps")
    if center is None:
        ref = next(iter(sample.cameras.values()))
        center = camera_center(ref.R.detach().cpu().numpy(), ref.t.detach().cpu().numpy())
    origin, dims = grid_around_point(center, voxel_size=config.voxel_size, extent_m=config.extent_m)
    grid = fuse_occupancy(
        depths,
        Ks,
        Rs,
        ts,
        origin=origin,
        voxel_size=config.voxel_size,
        dims=dims,
        max_depth=config.max_depth,
        min_depth=config.min_depth,
        stride=config.stride,
        depth_kind="planar_z",
        wire_masks=wire_masks if use_wires else None,
    )
    grid.timestamp = sample.timestamp
    grid.scene_id = sample.scene_id
    if grid.wire_points is not None and len(grid.wire_points) > 0 and config.wire_voxel_size > 0:
        fine_origin, fine_dims = grid_around_point(
            center,
            voxel_size=config.wire_voxel_size,
            extent_m=config.wire_extent_m,
        )
        cells = int(np.prod(fine_dims))
        if cells > _MAX_FINE_VOXELS:
            raise ValueError(
                f"wire fine grid {fine_dims} has {cells} cells. "
                "Shrink cache.wire_extent_m or raise cache.wire_voxel_size."
            )
        grid.wire_fine = _rasterize_points(
            grid.wire_points, fine_origin, config.wire_voxel_size, fine_dims
        )
        grid.wire_fine_origin = fine_origin
        grid.wire_fine_voxel_size = float(config.wire_voxel_size)
    return grid


def segmentation_to_wire_mask(segmentation, wire_ids, wire_colors) -> np.ndarray:
    """Boolean HxW mask. Labels match ``wire_ids``. RGB matches ``wire_colors``."""
    if torch.is_tensor(segmentation):
        array = segmentation.detach().cpu().numpy()
    else:
        array = np.asarray(segmentation)
    if array.ndim == 2:
        ids = np.asarray(list(wire_ids), dtype=np.int64)
        if ids.size == 0:
            return np.zeros(array.shape, dtype=bool)
        return np.isin(array.astype(np.int64), ids)
    if array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    elif array.ndim == 3 and array.shape[-1] == 3:
        rgb = array
    else:
        raise ValueError(f"segmentation shape {array.shape} is neither labels nor RGB")
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.round(np.asarray(rgb, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    mask = np.zeros(rgb.shape[:2], dtype=bool)
    for color in wire_colors:
        target = np.asarray(color, dtype=np.uint8).reshape(3)
        mask |= np.all(rgb == target, axis=-1)
    return mask


def _rasterize_points(points, origin, voxel_size, dims) -> np.ndarray:
    grid = np.zeros(dims, dtype=np.uint8)
    if len(points) == 0:
        return grid
    ijk = np.floor((np.asarray(points, dtype=np.float64) - origin) / voxel_size).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(dims)), axis=1)
    ijk = ijk[inside]
    if len(ijk) == 0:
        return grid
    flat = ijk[:, 0] * dims[1] * dims[2] + ijk[:, 1] * dims[2] + ijk[:, 2]
    grid.reshape(-1)[np.unique(flat)] = np.uint8(1)
    return grid


def _wire_tag(config: CacheConfig) -> str:
    if not config.wire_ids and not config.wire_colors:
        return ""
    text = (
        f"{config.wire_ids}|{config.wire_colors}|"
        f"{config.wire_voxel_size:.4f}|{config.wire_extent_m:.2f}"
    )
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"_wire{digest}"


def _as_tuple(value):
    if value is None:
        return ()
    return tuple(value)
