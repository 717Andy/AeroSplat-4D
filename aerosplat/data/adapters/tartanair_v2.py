"""TartanAir V2 loader.

V2 is a 12-camera AirSim rig: left and right centers 0.25 m apart, each with
front/back/left/right/top/bottom. Images are 640x640, 90° FOV, ``fx = fy = 320``,
``cx = cy = 320``, synchronized at 10 Hz. Poses are ``tx ty tz qx qy qz qw`` in
NED, the same line as V1. Depth is float32 meters packed into an RGBA PNG.

The deep-dive flags the body-to-camera axis convention as unconfirmed and says
it follows V1. This loader uses :func:`tartanair_pose_to_w2c`, which is the V1
AirSim-camera conversion into ENU / OpenCV. A reprojection check against a
real V2 trajectory is still owed; see the open questions in
``docs/ARCHITECTURE.md``.

Camera names in the sample are the file stems (``lcam_front``, ``rcam_left``,
...). Every camera at one frame index shares a timestamp, so this is a real
synchronized rig, unlike AerialMegaDepth.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.specs import SpecDataset, ViewGroup, ViewSpec
from aerosplat.geometry.camera import scale_intrinsics
from aerosplat.geometry.coordinates import tartanair_pose_to_w2c

_POSE_RE = re.compile(r"pose_(.+)\.txt$")
_FRAME_RE = re.compile(r"(\d+)")

# Official cube camera. Other resolutions scale this K.
_BASE_SIZE = (640, 640)
_BASE_K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 320.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_tartanair_v2(root: Path, cfg) -> SpecDataset:
    root = Path(root)
    trajectories = discover_trajectories(root)
    if not trajectories:
        raise DataError(
            f"{root} is not a TartanAir V2 trajectory. Expected pose_<camera>.txt, "
            "image_<camera>/, and depth_<camera>/."
        )
    requested = getattr(cfg, "cameras", None)
    camera_filter = tuple(requested) if requested else None
    max_frames = getattr(cfg, "max_frames", None)
    max_frames = None if max_frames in (None, "null") else int(max_frames)
    fps = float(getattr(cfg, "fps", 10.0))
    groups: list[ViewGroup] = []
    for traj in trajectories:
        groups.extend(
            _read_trajectory(
                traj,
                root=root,
                camera_filter=camera_filter,
                max_frames=max_frames,
                fps=fps,
            )
        )
    return SpecDataset(groups)


def discover_trajectories(root: Path) -> list[Path]:
    root = Path(root)
    if list(root.glob("pose_*.txt")):
        return [root]
    return sorted({path.parent for path in root.glob("**/pose_*.txt")})


def _read_trajectory(
    traj: Path,
    *,
    root: Path,
    camera_filter: tuple[str, ...] | None,
    max_frames: int | None,
    fps: float,
) -> list[ViewGroup]:
    cameras = []
    for pose_path in sorted(traj.glob("pose_*.txt")):
        match = _POSE_RE.search(pose_path.name)
        if match is None:
            continue
        name = match.group(1)
        if camera_filter is not None and name not in camera_filter:
            continue
        cameras.append(name)
    if not cameras:
        raise DataError(f"{traj} has no pose_<camera>.txt files")
    streams: dict[str, list[ViewSpec]] = {}
    counts = []
    for name in cameras:
        views = _read_camera(traj, name, fps=fps, max_frames=max_frames)
        streams[name] = views
        counts.append(len(views))
    if len(set(counts)) != 1:
        raise DataError(
            f"{traj} cameras do not have the same number of frames: {dict(zip(cameras, counts))}"
        )
    n_frames = counts[0]
    try:
        scene_id = str(traj.resolve().relative_to(root.resolve()))
    except ValueError:
        scene_id = traj.name
    if scene_id == ".":
        scene_id = traj.name
    groups = []
    for index in range(n_frames):
        groups.append(
            ViewGroup(
                scene_id=scene_id,
                views=[streams[name][index] for name in cameras],
                timestamp=index / fps,
            )
        )
    return groups


def _read_camera(traj: Path, name: str, *, fps: float, max_frames: int | None) -> list[ViewSpec]:
    poses = _read_poses(traj / f"pose_{name}.txt")
    images = _indexed(traj / f"image_{name}", (".png", ".jpg", ".jpeg"))
    depths = _indexed(traj / f"depth_{name}", (".png", ".npy"))
    if not images:
        raise DataError(f"{traj} has no images for camera {name}")
    keys = sorted(images, key=lambda item: int(item))
    if max_frames is not None:
        keys = keys[:max_frames]
        poses = poses[:max_frames]
    if len(keys) != len(poses):
        raise DataError(f"{traj} camera {name}: {len(keys)} images but {len(poses)} pose lines")
    views = []
    for index, key in enumerate(keys):
        R, t = tartanair_pose_to_w2c(*poses[index])
        image_path = images[key]
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        if (width, height) != _BASE_SIZE:
            K = scale_intrinsics(_BASE_K, (_BASE_SIZE[1], _BASE_SIZE[0]), (height, width))
        else:
            K = _BASE_K.copy()
        depth_path = depths.get(key)
        views.append(
            ViewSpec(
                name=name,
                image_path=image_path,
                depth_path=depth_path,
                K=K,
                R=R,
                t=t,
                depth_scale=1.0,
                depth_format="tartan_v2" if depth_path is not None else "auto",
            )
        )
    return views


def _read_poses(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    if not path.is_file():
        raise DataError(f"missing TartanAir V2 pose file {path}")
    poses = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            raise DataError(f"{path}:{line_no} expected tx ty tz qx qy qz qw")
        vals = [float(part) for part in parts]
        poses.append((np.array(vals[0:3], dtype=np.float64), np.array(vals[3:7], dtype=np.float64)))
    if not poses:
        raise DataError(f"{path} is empty")
    return poses


def _indexed(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        match = _FRAME_RE.search(path.stem)
        if match is None:
            raise DataError(f"cannot find a frame index in {path.name}")
        found[match.group(1)] = path
    return found
