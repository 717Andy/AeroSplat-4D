"""TartanAir adapter.

TartanAir (https://theairlab.org/tartanair-dataset/) was rendered in AirSim, so
its pose and depth conventions match the AirSim family (Colosseum and
Cosys-AirSim): NED world, camera X forward / Y right / Z down, planar depth in
meters, and a ``tx ty tz qx qy qz qw`` line per frame. That shared convention
is why this adapter exists instead of a Mid-Air one. See ``docs/DATASETS.md``.

The public stereo pair is a forward left/right rig, not a four-camera UAM rig.
Camera names in the sample are ``left`` and ``right``. Trajectories have no
capture clock; frame ``i`` is assigned ``timestamp = i / fps``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.records import FrameRecord
from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.coordinates import tartanair_pose_to_w2c

_FRAME_RE = re.compile(r"(\d+)")


def read_tartanair(root: Path, cfg) -> dict[str, dict[str, list[FrameRecord]]]:
    """Load every trajectory under ``root`` (or ``root`` itself if it is one)."""
    root = Path(root)
    trajectories = discover_trajectories(root)
    if not trajectories:
        raise DataError(
            f"{root} is not a TartanAir trajectory and does not contain one. "
            "Expected pose_left.txt, image_left/, and depth_left/."
        )
    intrinsics = cfg.intrinsics
    base_K = np.array(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
            [0.0, float(intrinsics.fy), float(intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    base_size = (int(intrinsics.width), int(intrinsics.height))
    fps = float(cfg.fps)
    baseline = float(cfg.stereo_baseline_m)
    depth_kind = str(cfg.depth_kind)
    if depth_kind not in {"planar_z", "euclidean"}:
        raise DataError(f"tartanair depth_kind must be planar_z or euclidean, got {depth_kind!r}")
    scenes: dict[str, dict[str, list[FrameRecord]]] = {}
    for traj in trajectories:
        scenes[_scene_id(root, traj)] = _read_trajectory(
            traj,
            base_K=base_K,
            base_size=base_size,
            fps=fps,
            baseline=baseline,
            depth_kind=depth_kind,
        )
    return scenes


def discover_trajectories(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "pose_left.txt").is_file():
        return [root]
    found = sorted({path.parent for path in root.glob("**/pose_left.txt")})
    return found


def _scene_id(root: Path, traj: Path) -> str:
    try:
        rel = traj.resolve().relative_to(root.resolve())
    except ValueError:
        rel = traj.name
    text = str(rel).replace("\\", "/")
    return text if text != "." else traj.name


def _read_trajectory(
    traj: Path,
    *,
    base_K: np.ndarray,
    base_size: tuple[int, int],
    fps: float,
    baseline: float,
    depth_kind: str,
) -> dict[str, list[FrameRecord]]:
    left_poses = _read_pose_file(traj / "pose_left.txt")
    right_path = traj / "pose_right.txt"
    right_poses = _read_pose_file(right_path) if right_path.is_file() else None
    left_images = _indexed_files(traj / "image_left", (".png", ".jpg", ".jpeg"))
    right_images = _indexed_files(traj / "image_right", (".png", ".jpg", ".jpeg"))
    left_depth = _indexed_files(traj / "depth_left", (".npy", ".png", ".pfm"))
    right_depth = _indexed_files(traj / "depth_right", (".npy", ".png", ".pfm"))
    if not left_images:
        raise DataError(f"{traj} has no files in image_left/")
    keys = sorted(left_images)
    if len(keys) != len(left_poses):
        raise DataError(f"{traj}: {len(left_images)} left images but {len(left_poses)} pose lines")
    if right_poses is not None and len(right_poses) != len(left_poses):
        raise DataError(f"{traj}: pose_left and pose_right have different lengths")
    streams: dict[str, list[FrameRecord]] = {"left": [], "right": []}
    for index, key in enumerate(keys):
        timestamp = index / fps
        left_R, left_t = tartanair_pose_to_w2c(*left_poses[index])
        if right_poses is not None:
            if key not in right_images:
                raise DataError(f"{traj}: missing right image for frame id {key}")
            right_R, right_t = tartanair_pose_to_w2c(*right_poses[index])
        else:
            right_R, right_t = _right_from_baseline(left_R, left_t, baseline)
            if key not in right_images:
                raise DataError(f"{traj}: missing right image for frame id {key}")
        streams["left"].append(
            _frame(
                index,
                timestamp,
                left_images[key],
                left_depth.get(key),
                left_R,
                left_t,
                base_K,
                base_size,
                depth_kind,
            )
        )
        streams["right"].append(
            _frame(
                index,
                timestamp,
                right_images[key],
                right_depth.get(key),
                right_R,
                right_t,
                base_K,
                base_size,
                depth_kind,
            )
        )
    return streams


def _frame(
    index: int,
    timestamp: float,
    image_path: Path,
    depth_path: Path | None,
    R: np.ndarray,
    t: np.ndarray,
    base_K: np.ndarray,
    base_size: tuple[int, int],
    depth_kind: str,
) -> FrameRecord:
    from PIL import Image

    from aerosplat.geometry.camera import scale_intrinsics

    with Image.open(image_path) as img:
        width, height = img.size
    if (width, height) != base_size:
        K = scale_intrinsics(base_K, (base_size[1], base_size[0]), (height, width))
    else:
        K = base_K.copy()
    return FrameRecord(
        frame_index=index,
        timestamp=timestamp,
        image_path=image_path,
        depth_path=depth_path,
        K=K,
        R=R,
        t=t,
        depth_kind=depth_kind,
    )


def _right_from_baseline(
    R_left: np.ndarray, t_left: np.ndarray, baseline: float
) -> tuple[np.ndarray, np.ndarray]:
    """Place the right camera ``baseline`` meters along AirSim camera +Y (right).

    Both cameras share orientation. The offset is expressed in the OpenCV
    camera frame (X right), which is where a +Y AirSim offset lands after the
    frame conversion: AirSim +Y maps to OpenCV +X.
    """
    center = camera_center(R_left, t_left)
    # OpenCV +X is camera-right. AirSim's +Y (right) lands on that axis after
    # the frame conversion, so a +0.25 m AirSim baseline is +X in OpenCV.
    right_axis = R_left.T @ np.array([1.0, 0.0, 0.0])
    right_center = center + float(baseline) * right_axis
    t_right = -R_left @ right_center
    return R_left.copy(), t_right


def _read_pose_file(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    if not path.is_file():
        raise DataError(f"missing TartanAir pose file {path}")
    poses = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            raise DataError(f"{path}:{line_no} expected 7 values, got {len(parts)}")
        try:
            vals = [float(part) for part in parts]
        except ValueError as exc:
            raise DataError(f"{path}:{line_no} is not numeric") from exc
        position = np.array(vals[0:3], dtype=np.float64)
        quat_xyzw = np.array(vals[3:7], dtype=np.float64)
        poses.append((position, quat_xyzw))
    if not poses:
        raise DataError(f"{path} is empty")
    return poses


def _indexed_files(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        match = _FRAME_RE.search(path.stem)
        if match is None:
            raise DataError(f"cannot find a frame index in {path.name}")
        key = match.group(1)
        if key in found:
            raise DataError(f"duplicate frame id {key} in {directory}")
        found[key] = path
    return found
