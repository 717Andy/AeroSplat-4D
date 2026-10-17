"""Conversions into the AeroSplat flight frame.

Internal contract
-----------------
* World / flight frame: right-handed ENU, Z up (X east, Y north, Z up).
* Camera frame: OpenCV (X right, Y down, Z forward).
* Extrinsics on every sample are world-to-camera: ``X_cam = R @ X_world + t``.

AirSim, Colosseum, and TartanAir do not use that pair of frames. They use a
NED world (X north, Y east, Z down) and a camera frame with X forward, Y
right, Z down. Loading those datasets without this conversion makes depth
unprojection disagree with the rasterizer, which expects OpenCV cameras.

Quaternion layout in this module is wxyz. TartanAir files store xyzw; the
adapter swaps before calling into here. AirSim recordings store wxyz.
"""

from __future__ import annotations

import numpy as np

# p_enu = R_ENU_FROM_NED @ p_ned
R_ENU_FROM_NED = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# AirSim camera (x forward, y right, z down) -> OpenCV (x right, y down, z forward).
R_CV_FROM_AIRSIM_CAM = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
R_AIRSIM_CAM_FROM_CV = R_CV_FROM_AIRSIM_CAM.T.copy()


def ned_to_enu(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ R_ENU_FROM_NED.T


def enu_to_ned(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return pts @ R_ENU_FROM_NED


def airsim_rpy_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation that takes AirSim-camera vectors into the vehicle body frame.

    Matches ``airsim.utils.to_quaternion(pitch, roll, yaw)``: yaw about Z,
    pitch about Y, roll about X, composed as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    Positive pitch points the camera up. Positive yaw points it to the vehicle's
    right. Angles are degrees.
    """
    pitch, roll, yaw = np.deg2rad([pitch_deg, roll_deg, yaw_deg])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def c2w_to_w2c(R_c2w: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_c2w = np.asarray(R_c2w, dtype=np.float64)
    t_c2w = np.asarray(t_c2w, dtype=np.float64).reshape(3)
    R = R_c2w.T
    t = -R @ t_c2w
    return R, t


def w2c_to_c2w(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R_c2w = R.T
    t_c2w = -R_c2w @ t
    return R_c2w, t_c2w


def _ned_airsim_cam_pose_to_w2c(
    position_ned: np.ndarray,
    R_ned_from_airsim_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-NED pose of an AirSim-frame camera -> ENU/OpenCV world-to-camera."""
    R_c2w_ned = np.asarray(R_ned_from_airsim_cam, dtype=np.float64) @ R_AIRSIM_CAM_FROM_CV
    t_c2w_ned = np.asarray(position_ned, dtype=np.float64).reshape(3)
    R_c2w = R_ENU_FROM_NED @ R_c2w_ned
    t_c2w = R_ENU_FROM_NED @ t_c2w_ned
    return c2w_to_w2c(R_c2w, t_c2w)


def airsim_mount_to_w2c(
    vehicle_position_ned: np.ndarray,
    vehicle_quat_wxyz: np.ndarray,
    *,
    cam_position_body: np.ndarray,
    pitch_deg: float,
    roll_deg: float,
    yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle pose plus a rigid camera mount -> world-to-camera ENU/OpenCV.

    ``vehicle_quat_wxyz`` rotates body vectors into the NED world. The mount
    position is in the body frame (meters). Mount angles follow AirSim settings.
    """
    from aerosplat.geometry.camera import quat_wxyz_to_rotmat

    R_ned_from_body = quat_wxyz_to_rotmat(vehicle_quat_wxyz)
    t_body = np.asarray(cam_position_body, dtype=np.float64).reshape(3)
    R_body_from_cam = airsim_rpy_matrix(pitch_deg, roll_deg, yaw_deg)
    R_ned_from_cam = R_ned_from_body @ R_body_from_cam
    t_ned = R_ned_from_body @ t_body + np.asarray(vehicle_position_ned, dtype=np.float64).reshape(3)
    return _ned_airsim_cam_pose_to_w2c(t_ned, R_ned_from_cam)


def tartanair_pose_to_w2c(
    position_ned: np.ndarray,
    quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TartanAir ``tx ty tz qx qy qz qw`` camera-to-NED pose -> ENU/OpenCV w2c.

    The file quaternion is xyzw and rotates the AirSim camera frame into NED.
    """
    from aerosplat.geometry.camera import quat_wxyz_to_rotmat

    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    R_ned_from_cam = quat_wxyz_to_rotmat(quat_wxyz)
    return _ned_airsim_cam_pose_to_w2c(position_ned, R_ned_from_cam)


def opencv_cam2world_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OpenCV cam2world 4x4. The translation column is the camera center, meters.

    ``R``, ``t`` are world-to-camera, ``X_cam = R @ X_world + t``.
    """
    R_c2w, center = w2c_to_c2w(R, t)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R_c2w
    matrix[:3, 3] = center
    return matrix


def nerf_c2w_to_opencv_w2c(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NeRF / OpenGL cam2world (4x4 or 3x4) to OpenCV world-to-camera.

    NeRF cameras point with x right, y up, z backward. OpenCV is x right, y
    down, z forward. The camera center is unchanged. MatrixCity per-block files
    need the rotation scaled by 100 before this call; that fix lives in the
    MatrixCity adapter because it is a file bug, not a camera convention.
    """
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.shape == (3, 4):
        full = np.eye(4, dtype=np.float64)
        full[:3] = raw
        raw = full
    if raw.shape != (4, 4):
        raise ValueError(f"cam2world must be 3x4 or 4x4, got {raw.shape}")
    flip = np.diag([1.0, -1.0, -1.0])
    R_cv = raw[:3, :3] @ flip
    center = raw[:3, 3]
    return c2w_to_w2c(R_cv, center)


def intrinsics_from_hfov(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Pinhole K from a horizontal FOV, square pixels, principal point at (W/2, H/2).

    This is the AirSim convention. Pixel coordinates used with this K are column
    and row indices (origin at the top-left pixel's sample location).
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if not 0.0 < fov_degrees < 180.0:
        raise ValueError(f"horizontal FOV must be in (0, 180), got {fov_degrees}")
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_degrees) / 2.0)
    K = np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K
