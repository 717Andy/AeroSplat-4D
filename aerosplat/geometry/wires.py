"""Polyline voxelization for cables the depth buffer can miss.

A 1–5 cm wire at flight altitude is often thinner than a pixel, so DepthPlanar
never lands in the cell the cable actually crosses. The synthetic generator
exports the catenary as a 3D polyline and this module stamps that line into
the occupancy cache: the coarse ``wire`` channel, the occupied state (so the
cell is known to ``L_occ``), ``wire_points`` for the point-to-line term, and
the optional 2 cm grid.

The class is the centerline. A cable is not a filled 0.5 m cube; the fine
grid and ``wire_points`` keep the sub-voxel position. Marking the coarse cell
occupied matches what a depth hit would have done.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.geometry.occupancy import OCCUPIED, OccupancyGrid, grid_around_point

WIRE_FORMAT = "aerosplat.wires.v1"
# Same cap as the depth-fused fine grid in ``aerosplat.data.cache``.
_MAX_FINE_VOXELS = 8_000_000


def densify_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """Insert samples so consecutive points are at most ``spacing`` apart."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts.reshape(0, 3)
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if len(pts) == 1:
        return pts.copy()
    out = [pts[0]]
    for start, end in zip(pts[:-1], pts[1:], strict=True):
        length = float(np.linalg.norm(end - start))
        if length < 1e-12:
            continue
        steps = int(np.ceil(length / spacing))
        for index in range(1, steps + 1):
            out.append(start + (end - start) * (index / steps))
    return np.asarray(out, dtype=np.float64)


def rasterize_polyline(
    points: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    dims: tuple[int, int, int],
    *,
    spacing: float | None = None,
) -> np.ndarray:
    """Mark cells the centerline crosses. 1 where the line enters the grid."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    step = float(voxel_size) * 0.5 if spacing is None else float(spacing)
    dense = densify_polyline(points, step)
    grid = np.zeros(dims, dtype=np.uint8)
    if len(dense) == 0:
        return grid
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ijk = np.floor((dense - origin) / float(voxel_size)).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(dims, dtype=np.int64)), axis=1)
    ijk = ijk[inside]
    if len(ijk) == 0:
        return grid
    flat = ijk[:, 0] * dims[1] * dims[2] + ijk[:, 1] * dims[2] + ijk[:, 2]
    grid.reshape(-1)[np.unique(flat)] = np.uint8(1)
    return grid


def polyline_recall(
    points: np.ndarray,
    mask: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> float:
    """Fraction of polyline vertices whose cell is set. Vertices outside are skipped."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return 1.0
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    dims = mask.shape
    ijk = np.floor((pts - origin) / float(voxel_size)).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(dims, dtype=np.int64)), axis=1)
    if not np.any(inside):
        return 1.0
    ijk = ijk[inside]
    values = mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return float(np.mean(values > 0))


def save_wires(path: Path, spans: list[dict]) -> None:
    """Write exact centerlines. ``points_enu`` are flight-frame meters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": WIRE_FORMAT, "frame": "enu_z_up", "spans": spans}
    path.write_text(json.dumps(payload) + "\n")


def load_wires(path: Path) -> list[dict]:
    """Load spans from a wires file. Missing file returns an empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    if payload.get("format") != WIRE_FORMAT:
        raise ValueError(f"{path} format is {payload.get('format')!r}, expected {WIRE_FORMAT}")
    if payload.get("frame") != "enu_z_up":
        raise ValueError(f"{path} is not in the ENU flight frame")
    spans = []
    for span in payload.get("spans", []):
        points = np.asarray(span["points_enu"], dtype=np.float64).reshape(-1, 3)
        spans.append({**span, "points_enu": points})
    return spans


def wires_path_for(root: Path, scene_id: str) -> Path:
    """``root`` is either the wires file or a dataset directory."""
    root = Path(root)
    if root.is_file():
        return root
    return root / scene_id / "wires.json"


def stamp_polylines(
    grid: OccupancyGrid,
    spans: list[dict],
    *,
    center: np.ndarray,
    fine_voxel_size: float = 0.02,
    fine_extent_m: float = 4.0,
) -> OccupancyGrid:
    """Fill the wire class from exact centerlines, including the 2 cm grid.

    Coarse cells the line crosses become occupied and ``wire`` 1. ``wire_points``
    gains the densified samples inside that grid. The fine grid is allocated
    only when it stays within the cache's cell cap.
    """
    if not spans:
        return grid
    assert grid.wire is not None and grid.wire_points is not None
    collected: list[np.ndarray] = []
    for span in spans:
        points = np.asarray(span["points_enu"], dtype=np.float64).reshape(-1, 3)
        mask = rasterize_polyline(points, grid.origin, grid.voxel_size, grid.dims)
        grid.wire = np.maximum(grid.wire, mask)
        grid.state[mask > 0] = OCCUPIED
        dense = densify_polyline(points, grid.voxel_size * 0.5)
        if len(dense):
            ijk = np.floor((dense - grid.origin) / grid.voxel_size).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < np.asarray(grid.dims)), axis=1)
            if np.any(inside):
                collected.append(dense[inside].astype(np.float32))
    if collected:
        extra = np.concatenate(collected, axis=0)
        grid.wire_points = np.concatenate([grid.wire_points, extra], axis=0)
    if fine_voxel_size > 0 and fine_extent_m > 0:
        fine_origin, fine_dims = grid_around_point(
            center, voxel_size=fine_voxel_size, extent_m=fine_extent_m
        )
        cells = int(np.prod(fine_dims))
        if cells <= _MAX_FINE_VOXELS:
            fine = np.zeros(fine_dims, dtype=np.uint8)
            for span in spans:
                points = np.asarray(span["points_enu"], dtype=np.float64).reshape(-1, 3)
                fine = np.maximum(
                    fine,
                    rasterize_polyline(points, fine_origin, fine_voxel_size, fine_dims),
                )
            grid.wire_fine = fine
            grid.wire_fine_origin = fine_origin
            grid.wire_fine_voxel_size = float(fine_voxel_size)
    return grid
