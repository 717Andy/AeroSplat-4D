"""On-disk cache of occupancy grids built from a multi-camera sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.occupancy import OccupancyGrid, grid_around_point, occupancy_from_depths
from aerosplat.schema import MultiCameraSample


@dataclass
class CacheConfig:
    root: Path
    voxel_size: float = 0.5
    extent_m: float = 12.0
    max_depth: float = 40.0
    min_depth: float = 0.05
    stride: int = 1

    @classmethod
    def from_hydra(cls, cfg) -> CacheConfig:
        cache = cfg.cache
        return cls(
            root=Path(str(cache.root)),
            voxel_size=float(cache.voxel_size),
            extent_m=float(cache.extent_m),
            max_depth=float(cache.max_depth),
            min_depth=float(cache.min_depth),
            stride=int(cache.stride),
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
    if not depths:
        raise ValueError(f"sample {sample.scene_id}#{sample.sample_index} has no depth maps")
    if center is None:
        ref = next(iter(sample.cameras.values()))
        center = camera_center(ref.R.detach().cpu().numpy(), ref.t.detach().cpu().numpy())
    origin, dims = grid_around_point(center, voxel_size=config.voxel_size, extent_m=config.extent_m)
    state, hit_count = occupancy_from_depths(
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
    )
    return OccupancyGrid(
        state=state,
        hit_count=hit_count,
        origin=origin,
        voxel_size=config.voxel_size,
        timestamp=sample.timestamp,
        scene_id=sample.scene_id,
    )
