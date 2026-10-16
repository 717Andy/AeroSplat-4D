"""AirSim / TartanAir conversions into the ENU + OpenCV flight frame."""

import math

import numpy as np

from aerosplat.geometry.camera import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz
from aerosplat.geometry.coordinates import (
    R_ENU_FROM_NED,
    airsim_mount_to_w2c,
    airsim_rpy_matrix,
    enu_to_ned,
    intrinsics_from_hfov,
    ned_to_enu,
    tartanair_pose_to_w2c,
)


def _airsim_reference_quat(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """airsim.utils.to_quaternion, radians in, wxyz out."""
    t0 = math.cos(yaw * 0.5)
    t1 = math.sin(yaw * 0.5)
    t2 = math.cos(roll * 0.5)
    t3 = math.sin(roll * 0.5)
    t4 = math.cos(pitch * 0.5)
    t5 = math.sin(pitch * 0.5)
    return np.array(
        [
            t0 * t2 * t4 + t1 * t3 * t5,
            t0 * t3 * t4 - t1 * t2 * t5,
            t0 * t2 * t5 + t1 * t3 * t4,
            t1 * t2 * t4 - t0 * t3 * t5,
        ]
    )


def test_ned_enu_roundtrip():
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, -4.0]])
    assert np.allclose(enu_to_ned(ned_to_enu(points)), points)
    # NED (north, east, down) -> ENU (east, north, up)
    assert np.allclose(ned_to_enu(np.array([1.0, 2.0, 3.0])), np.array([2.0, 1.0, -3.0]))
    assert np.allclose(R_ENU_FROM_NED @ R_ENU_FROM_NED, np.eye(3))


def test_airsim_rpy_matches_client_and_aviation_signs():
    for pitch, roll, yaw in ((0.0, 0.0, 0.0), (0.4, -0.2, 1.1), (-math.pi / 2, 0.0, 0.3)):
        quat = _airsim_reference_quat(pitch, roll, yaw)
        ours = airsim_rpy_matrix(math.degrees(pitch), math.degrees(roll), math.degrees(yaw))
        assert np.allclose(ours, quat_wxyz_to_rotmat(quat), atol=1e-8)
    # +90 deg pitch points the nose up (NED -Z). +90 deg yaw turns right (NED +Y).
    assert np.allclose(airsim_rpy_matrix(90, 0, 0) @ [1, 0, 0], [0, 0, -1], atol=1e-8)
    assert np.allclose(airsim_rpy_matrix(-90, 0, 0) @ [1, 0, 0], [0, 0, 1], atol=1e-8)
    assert np.allclose(airsim_rpy_matrix(0, 0, 90) @ [1, 0, 0], [0, 1, 0], atol=1e-8)
    assert np.allclose(airsim_rpy_matrix(0, 0, -90) @ [1, 0, 0], [0, -1, 0], atol=1e-8)


def test_identity_vehicle_looks_north_in_enu():
    R, t = airsim_mount_to_w2c(
        np.array([0.0, 0.0, -3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        cam_position_body=np.zeros(3),
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_deg=0.0,
    )
    # Camera center is 3 m above the origin.
    center = -R.T @ t
    assert np.allclose(center, [0.0, 0.0, 3.0], atol=1e-8)
    # OpenCV +Z (forward) is ENU north. A point 1 m ahead projects to (0, 0, 1).
    ahead = center + R.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(ahead, [0.0, 1.0, 3.0], atol=1e-8)
    x_cam = R @ ahead + t
    assert np.allclose(x_cam, [0.0, 0.0, 1.0], atol=1e-8)


def test_downward_mount_looks_down():
    R, t = airsim_mount_to_w2c(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        cam_position_body=np.zeros(3),
        pitch_deg=-90.0,
        roll_deg=0.0,
        yaw_deg=0.0,
    )
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward, [0.0, 0.0, -1.0], atol=1e-8)
    assert abs(np.linalg.det(R) - 1.0) < 1e-8


def test_tartanair_pose_matches_airsim_mount_for_the_same_ned_pose():
    position = np.array([5.0, -2.0, -8.0])
    # 90 deg yaw about NED Z, xyzw in the file, wxyz in the vehicle pose.
    quat_xyzw = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    R_t, t_t = tartanair_pose_to_w2c(position, quat_xyzw)
    R_a, t_a = airsim_mount_to_w2c(
        position,
        quat_wxyz,
        cam_position_body=np.zeros(3),
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_deg=0.0,
    )
    assert np.allclose(R_t, R_a, atol=1e-8)
    assert np.allclose(t_t, t_a, atol=1e-8)


def test_quat_roundtrip_keeps_positive_scalar():
    R = airsim_rpy_matrix(10, -20, 30)
    quat = rotmat_to_quat_wxyz(R)
    assert quat[0] >= 0
    assert np.allclose(quat_wxyz_to_rotmat(quat), R, atol=1e-8)


def test_intrinsics_90_degree_fov():
    K = intrinsics_from_hfov(640, 480, 90)
    assert np.isclose(K[0, 0], 320.0)
    assert np.isclose(K[1, 1], 320.0)
    assert np.isclose(K[0, 2], 320.0)
    assert np.isclose(K[1, 2], 240.0)
