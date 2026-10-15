"""Software ray/AABB renderer used to build the synthetic fixture.

This is not a stand-in for diff-gaussian-rasterization. It exists so the
checked-in fixture has depth and RGB that are consistent with the stored
intrinsics and extrinsics, and so tests can run without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    bmin: np.ndarray
    bmax: np.ndarray
    albedo: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "bmin", np.asarray(self.bmin, dtype=np.float64).reshape(3))
        object.__setattr__(self, "bmax", np.asarray(self.bmax, dtype=np.float64).reshape(3))
        object.__setattr__(self, "albedo", np.asarray(self.albedo, dtype=np.float64).reshape(3))


def render_boxes(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    boxes: list[Box],
    *,
    max_range: float = 80.0,
    min_range: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize axis-aligned boxes.

    ``R``, ``t`` are world-to-camera. Returns ``rgb`` uint8 ``(H, W, 3)`` and
    planar-Z depth float32 ``(H, W)``. Misses are depth 0 and black.
    """
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    tvec = np.asarray(t, dtype=np.float64).reshape(3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    dirs_c = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    dirs_c /= np.linalg.norm(dirs_c, axis=-1, keepdims=True)
    dirs_w = dirs_c @ R
    origin = -R.T @ tvec

    best_t = np.full((height, width), np.inf, dtype=np.float64)
    best_color = np.zeros((height, width, 3), dtype=np.float64)
    sun = np.array([0.35, 0.25, 0.90], dtype=np.float64)
    sun /= np.linalg.norm(sun)

    for box in boxes:
        t_hit, face = _slab(origin, dirs_w, box.bmin, box.bmax)
        hit = (t_hit > min_range) & (t_hit < max_range) & (t_hit < best_t)
        if not np.any(hit):
            continue
        normal = np.zeros((3,), dtype=np.float64)
        # Shade per hit pixel. Face index 0..5 maps to ±x, ±y, ±z.
        colors = np.zeros((height, width, 3), dtype=np.float64)
        for face_id, axis, sign in (
            (0, 0, 1.0),
            (1, 0, -1.0),
            (2, 1, 1.0),
            (3, 1, -1.0),
            (4, 2, 1.0),
            (5, 2, -1.0),
        ):
            mask = hit & (face == face_id)
            if not np.any(mask):
                continue
            normal[:] = 0.0
            normal[axis] = sign
            shade = 0.30 + 0.70 * max(0.0, float(np.dot(normal, sun)))
            colors[mask] = box.albedo * shade
        best_color[hit] = colors[hit]
        best_t[hit] = t_hit[hit]

    hit_any = np.isfinite(best_t) & (best_t < max_range)
    points = np.zeros((height, width, 3), dtype=np.float64)
    points[hit_any] = origin + dirs_w[hit_any] * best_t[hit_any, None]
    pts_cam = np.zeros((height, width, 3), dtype=np.float64)
    pts_cam[hit_any] = points[hit_any] @ R.T + tvec
    depth = np.zeros((height, width), dtype=np.float32)
    depth[hit_any] = pts_cam[..., 2][hit_any].astype(np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[hit_any] = np.clip(best_color[hit_any] * 255.0, 0, 255).astype(np.uint8)
    return rgb, depth


def _slab(
    origin: np.ndarray,
    dirs: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_hit, face)``. Face is -1 on a miss.

    Face ids: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z, chosen from the entry plane.
    """
    height, width = dirs.shape[:2]
    tnear = np.full((height, width), -np.inf, dtype=np.float64)
    tfar = np.full((height, width), np.inf, dtype=np.float64)
    near_axis = np.zeros((height, width), dtype=np.int64)
    near_sign = np.ones((height, width), dtype=np.float64)
    valid = np.ones((height, width), dtype=bool)
    for axis in range(3):
        d = dirs[..., axis]
        o = float(origin[axis])
        parallel = np.abs(d) < 1e-12
        outside = parallel & ((o < bmin[axis]) | (o > bmax[axis]))
        valid &= ~outside
        inv = np.divide(1.0, d, out=np.ones_like(d), where=~parallel)
        ta = (bmin[axis] - o) * inv
        tb = (bmax[axis] - o) * inv
        t0 = np.minimum(ta, tb)
        t1 = np.maximum(ta, tb)
        # Sign of the entry normal: if we enter through bmin, the outward
        # normal points toward -axis (the ray is moving in +axis).
        enter_min = ta <= tb
        closer = (~parallel) & (t0 > tnear)
        tnear = np.where(parallel, tnear, np.maximum(tnear, t0))
        tfar = np.where(parallel, tfar, np.minimum(tfar, t1))
        near_axis = np.where(closer, axis, near_axis)
        near_sign = np.where(closer, np.where(enter_min, -1.0, 1.0), near_sign)
    hit = valid & (tfar >= tnear) & (tnear > 0.0) & np.isfinite(tnear)
    t_hit = np.where(hit, tnear, np.inf)
    face = np.full((height, width), -1, dtype=np.int64)
    # Map (axis, outward sign) to face id. +axis outward -> even, -axis -> odd.
    face_id = near_axis * 2 + np.where(near_sign > 0.0, 0, 1)
    face[hit] = face_id[hit]
    return t_hit, face
