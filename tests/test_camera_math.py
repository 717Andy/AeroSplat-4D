"""Projection, interpolation, and intrinsics scaling."""

import numpy as np
import pytest

from aerosplat.geometry.camera import (
    PoseSample,
    camera_center,
    interpolate_w2c,
    project_points,
    reprojection_error,
    scale_intrinsics,
    slerp_wxyz,
    unproject_depth,
)
from aerosplat.geometry.coordinates import c2w_to_w2c


def _pose_at(center, yaw_deg=0.0):
    yaw = np.deg2rad(yaw_deg)
    # OpenCV camera looking along world +X when yaw is 0, then yawed about +Z.
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R_c2w = np.stack([right, down, forward], axis=1)
    return c2w_to_w2c(R_c2w, np.asarray(center, dtype=np.float64))


def test_unproject_project_roundtrip():
    K = np.array([[50.0, 0.0, 8.0], [0.0, 40.0, 6.0], [0.0, 0.0, 1.0]])
    R, t = _pose_at([1.0, 2.0, 3.0], yaw_deg=15)
    depth = np.zeros((12, 16), dtype=np.float64)
    depth[2:10, 3:12] = 4.0
    points = unproject_depth(depth, K, R, t)
    uv, z = project_points(points, K, R, t)
    err = reprojection_error(depth, K, R, t)
    assert points.shape[0] == int((depth > 0).sum())
    assert np.allclose(z, 4.0, atol=1e-6)
    assert err.max() < 1e-6
    assert uv.shape == (points.shape[0], 2)


def test_euclidean_depth_on_a_diagonal_ray():
    K = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.zeros(3)
    # Pixel (10, 0) has ray (1, 0, 1) before normalization. Range 2*sqrt(2)
    # corresponds to planar z = 2, because ||dir_unnormalized|| = sqrt(2) and
    # z = range / ||dir||.
    depth = np.zeros((1, 11))
    depth[0, 10] = 2.0 * np.sqrt(2.0)
    pts = unproject_depth(depth, K, R, t, depth_kind="euclidean")
    assert np.allclose(pts, [[2.0, 0.0, 2.0]], atol=1e-6)


def test_slerp_halfway_is_45_degrees():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # +90 deg about Z
    mid = slerp_wxyz(q0, q1, 0.5)
    expected = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
    assert np.allclose(mid, expected, atol=1e-8)


def test_interpolate_center_not_the_w2c_translation():
    R0, t0 = _pose_at([0.0, 0.0, 1.0], yaw_deg=0)
    R1, t1 = _pose_at([2.0, 0.0, 1.0], yaw_deg=90)
    poses = [PoseSample(0.0, R0, t0), PoseSample(1.0, R1, t1)]
    R, t, status = interpolate_w2c(poses, 0.5)
    assert status == "interpolated"
    assert np.allclose(camera_center(R, t), [1.0, 0.0, 1.0], atol=1e-8)
    # 45 deg yaw: forward is (cos45, sin45, 0).
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward[:2], [np.cos(np.pi / 4), np.sin(np.pi / 4)], atol=1e-8)
    R_exact, _, status_exact = interpolate_w2c(poses, 0.0)
    assert status_exact == "exact"
    assert np.allclose(R_exact, R0)
    _, _, status_clamp = interpolate_w2c(poses, -0.2)
    assert status_clamp == "clamped"
    assert np.allclose(camera_center(*interpolate_w2c(poses, -0.2)[:2]), [0.0, 0.0, 1.0])


def test_scale_intrinsics_moves_the_principal_point():
    K = np.array([[100.0, 0.0, 20.0], [0.0, 80.0, 10.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(K, (20, 40), (40, 80))
    assert np.allclose(scaled[0, 0], 200.0)
    assert np.allclose(scaled[1, 1], 160.0)
    assert np.allclose(scaled[0, 2], 40.0)
    assert np.allclose(scaled[1, 2], 20.0)


def test_interpolate_rejects_an_empty_stream():
    with pytest.raises(ValueError):
        interpolate_w2c([], 0.0)
