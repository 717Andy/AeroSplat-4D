"""Synthetic four-camera rig used by tests and the sanity script.

The rig is defined directly in the ENU/OpenCV flight frame. It is not an
AirSim recording. AirSim and TartanAir conversions are tested separately.

Frame rates are intentionally unequal: front and starboard at 10 Hz, port at
10 Hz with a 20 ms offset, down at 5 Hz. Resolutions differ per camera so the
loader cannot cheat by stacking a single image tensor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aerosplat.data.adapters.canonical import write_canonical
from aerosplat.geometry.camera import scale_intrinsics
from aerosplat.geometry.coordinates import c2w_to_w2c, intrinsics_from_hfov
from aerosplat.geometry.raycast import Box, render_boxes
from aerosplat.paths import find_repo_root

FOV_DEGREES = 130.0
SCENE_ID = "flight0"

# Columns are the camera axes in the world: right, down, forward.
_R_C2W = {
    "front": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64),
    "down": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    "port": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64),
    "starboard": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64),
}
_OFFSETS = {
    "front": np.array([0.20, 0.00, 0.00], dtype=np.float64),
    "down": np.array([0.00, 0.00, -0.05], dtype=np.float64),
    "port": np.array([0.00, 0.15, 0.00], dtype=np.float64),
    "starboard": np.array([0.00, -0.15, 0.00], dtype=np.float64),
}
RESOLUTIONS = {
    "front": (80, 60),
    "down": (48, 48),
    "port": (64, 48),
    "starboard": (40, 56),
}
STREAM_TIMES = {
    "front": (0.0, 0.1, 0.2, 0.3, 0.4),
    "down": (0.0, 0.2, 0.4),
    "port": (0.02, 0.12, 0.22, 0.32, 0.42),
    "starboard": (0.0, 0.1, 0.2, 0.3, 0.4),
}

BOXES = [
    Box(np.array([-1.0, -5.0, -0.20]), np.array([8.0, 5.0, 0.0]), np.array([0.35, 0.55, 0.30])),
    Box(np.array([2.2, -1.2, 0.0]), np.array([4.2, 1.2, 3.0]), np.array([0.75, 0.25, 0.20])),
    Box(np.array([3.40, 1.55, 0.0]), np.array([3.55, 1.70, 4.5]), np.array([0.15, 0.25, 0.85])),
]


def vehicle_center(time: float) -> np.ndarray:
    """Eastbound flight at 4 m/s, 3 m altitude. ``time`` is seconds."""
    return np.array([4.0 * float(time), 0.0, 3.0], dtype=np.float64)


def camera_pose(name: str, time: float) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera extrinsics of one fixture camera at ``time``."""
    R_c2w = _R_C2W[name]
    center = vehicle_center(time) + _OFFSETS[name]
    return c2w_to_w2c(R_c2w, center)


def intrinsics(name: str) -> np.ndarray:
    width, height = RESOLUTIONS[name]
    return intrinsics_from_hfov(width, height, FOV_DEGREES)


def scaled_intrinsics(name: str, width: int, height: int) -> np.ndarray:
    src_w, src_h = RESOLUTIONS[name]
    return scale_intrinsics(intrinsics(name), (src_h, src_w), (height, width))


def generate_fixture(root: Path) -> Path:
    """Render the tiny rig and write a canonical recording. Returns ``root``."""
    root = Path(root)
    cameras: dict[str, list[dict]] = {}
    for name, times in STREAM_TIMES.items():
        width, height = RESOLUTIONS[name]
        K = intrinsics(name)
        frames = []
        for time in times:
            R, t = camera_pose(name, time)
            rgb, depth = render_boxes(K, R, t, width, height, BOXES)
            frames.append(
                {
                    "timestamp": float(time),
                    "image": rgb,
                    "depth": depth,
                    "K": K,
                    "R": R,
                    "t": t,
                }
            )
        cameras[name] = frames
    write_canonical(root, {SCENE_ID: cameras})
    return root


def default_fixture_root() -> Path:
    return find_repo_root() / "fixtures" / "tiny_rig"


def _assert_rotations() -> None:
    for name, R in _R_C2W.items():
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-8):
            raise RuntimeError(f"fixture rotation {name} is not orthogonal")
        if abs(np.linalg.det(R) - 1.0) > 1e-8:
            raise RuntimeError(f"fixture rotation {name} is not a proper rotation")


_assert_rotations()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic four-camera fixture.")
    parser.add_argument("--root", type=Path, default=None, help="Output directory.")
    args = parser.parse_args(argv)
    root = args.root or default_fixture_root()
    generate_fixture(root)
    print(f"wrote {root}")


if __name__ == "__main__":
    main()
