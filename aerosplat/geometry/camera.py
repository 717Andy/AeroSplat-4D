"""Pinhole projection, quaternion pose interpolation, and intrinsics scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method. Returns a unit quaternion with w >= 0."""
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation. ``u`` is clamped to [0, 1] by the caller."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + float(u) * (b - a)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    out = (np.sin((1.0 - u) * theta) * a + np.sin(u * theta) * b) / np.sin(theta)
    return out / np.linalg.norm(out)


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame camera center from world-to-camera extrinsics."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return -R.T @ t


def _lerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    return (1.0 - u) * a + u * b


@dataclass(frozen=True)
class PoseSample:
    timestamp: float
    R: np.ndarray
    t: np.ndarray

    def center(self) -> np.ndarray:
        return camera_center(self.R, self.t)


def interpolate_w2c(
    poses: list[PoseSample],
    time: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Interpolate world-to-camera pose at ``time``.

    Translation is interpolated in the world frame on the camera center, then
    converted back. Interpolating the ``t`` vector of ``X_cam = R X + t``
    directly is not a straight line in the world.

    Returns ``(R, t, status)`` where status is ``exact``, ``interpolated``, or
    ``clamped``. Times outside the pose stream clamp to the nearest end; they
    are not extrapolated.
    """
    if not poses:
        raise ValueError("pose stream is empty")
    ordered = sorted(poses, key=lambda p: p.timestamp)
    times = [p.timestamp for p in ordered]
    if time <= times[0] + 1e-12:
        status = "exact" if abs(time - times[0]) <= 1e-12 else "clamped"
        return ordered[0].R.copy(), ordered[0].t.copy(), status
    if time >= times[-1] - 1e-12:
        status = "exact" if abs(time - times[-1]) <= 1e-12 else "clamped"
        return ordered[-1].R.copy(), ordered[-1].t.copy(), status

    hi = int(np.searchsorted(times, time, side="right"))
    lo = hi - 1
    t0, t1 = times[lo], times[hi]
    if abs(time - t0) <= 1e-12:
        return ordered[lo].R.copy(), ordered[lo].t.copy(), "exact"
    u = (time - t0) / (t1 - t0)
    c0 = ordered[lo].center()
    c1 = ordered[hi].center()
    center = _lerp(c0, c1, u)
    q0 = rotmat_to_quat_wxyz(ordered[lo].R.T)
    q1 = rotmat_to_quat_wxyz(ordered[hi].R.T)
    R_c2w = quat_wxyz_to_rotmat(slerp_wxyz(q0, q1, u))
    R = R_c2w.T
    t = -R @ center
    return R, t, "interpolated"


def scale_intrinsics(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Scale K from ``src_hw`` (H, W) to ``dst_hw``.

    Principal point and focal length scale with the image axes. This matches a
    uniform stretch of the image plane, which is the resize used by the dataset
    transform. It is not a center-crop.
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError("image sizes must be positive")
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    stride: int = 1,
    depth_kind: str = "planar_z",
) -> np.ndarray:
    """Back-project a depth map to world points.

    ``depth_kind="planar_z"`` is the camera-frame Z used by AirSim DepthPlanar,
    TartanAir, and the fixture. ``depth_kind="euclidean"`` is AirSim
    DepthPerspective (range along the ray); it is converted to planar Z before
    unprojection.

    Invalid depths (<= ``min_depth``, above ``max_depth``, or non-finite) are
    dropped. ``stride`` subsamples pixels. Returns ``(N, 3)`` float64, possibly
    empty.
    """
    if depth_kind not in {"planar_z", "euclidean"}:
        raise ValueError(f"unknown depth_kind {depth_kind!r}")
    z_map = np.asarray(depth, dtype=np.float64)
    if z_map.ndim != 2:
        raise ValueError(f"depth must be HxW, got {z_map.shape}")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    tvec = np.asarray(t, dtype=np.float64).reshape(3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vs, us = np.mgrid[0 : z_map.shape[0] : stride, 0 : z_map.shape[1] : stride]
    z = z_map[vs, us]
    if depth_kind == "euclidean":
        ray_scale = np.sqrt(((us - cx) / fx) ** 2 + ((vs - cy) / fy) ** 2 + 1.0)
        z = z / ray_scale
    valid = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        valid &= z <= max_depth
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)
    # X_cam = R X_world + t => X_world = R.T @ (X_cam - t)
    return (pts_cam - tvec) @ R


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points. Returns ``(uv, z_cam)`` with ``uv`` shaped (N, 2)."""
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    tvec = np.asarray(t, dtype=np.float64).reshape(3)
    pts_cam = pts @ R.T + tvec
    z = pts_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-12, np.nan, z)
    u = K[0, 0] * pts_cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], axis=-1), z


def reprojection_error(
    depth: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Per-valid-pixel reprojection error in pixels. Empty if nothing is valid."""
    z_map = np.asarray(depth, dtype=np.float64)
    vs, us = np.mgrid[0 : z_map.shape[0] : stride, 0 : z_map.shape[1] : stride]
    z = z_map[vs, us]
    valid = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        valid &= z <= max_depth
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64)
    points = unproject_depth(
        z_map, K, R, t, min_depth=min_depth, max_depth=max_depth, stride=stride
    )
    uv, _ = project_points(points, K, R, t)
    src = np.stack([us[valid].astype(np.float64), vs[valid].astype(np.float64)], axis=-1)
    return np.linalg.norm(uv - src, axis=-1)
