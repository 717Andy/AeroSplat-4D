"""Camera geometry, coordinate frames, and occupancy voxelization."""

from aerosplat.geometry.camera import (
    camera_center,
    interpolate_w2c,
    project_points,
    scale_intrinsics,
    unproject_depth,
)
from aerosplat.geometry.coordinates import (
    airsim_mount_to_w2c,
    tartanair_pose_to_w2c,
)
from aerosplat.geometry.occupancy import OccupancyGrid, occupancy_from_depths

__all__ = [
    "OccupancyGrid",
    "airsim_mount_to_w2c",
    "camera_center",
    "interpolate_w2c",
    "occupancy_from_depths",
    "project_points",
    "scale_intrinsics",
    "tartanair_pose_to_w2c",
    "unproject_depth",
]
