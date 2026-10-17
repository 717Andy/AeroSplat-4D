"""AerialMegaDepth loader.

The release is a COLMAP reconstruction in the OpenCV camera frame, metric
scale, plus an official DUSt3R preprocess that stores ``intrinsics`` and
``cam2world``. This adapter reads either:

* ``views.json`` listing images, depths, ``K``, and OpenCV ``cam2world`` (or ``w2c``)
* COLMAP ``cameras.txt`` + ``images.txt`` next to ``images/`` and ``depths/``
* a directory of ``.npz`` files with ``intrinsics`` and ``cam2world``

Depth ``.h5`` (key ``depth``) needs h5py. ``.npy`` is read directly. Values are
treated as meters, which matches the metric reconstruction; the deep-dive notes
that the unit is inferred from the "metric" label and the sample ranges.

Views are not a synchronized rig. They are chunked into virtual N-view samples
in filename order. The Google Earth frames and the real MegaDepth photos in one
scene share a coordinate frame, which is what the chunk uses. The dataset
license is non-commercial. This module does not download it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.specs import SpecDataset, ViewGroup, ViewSpec, chunk_views
from aerosplat.geometry.camera import quat_wxyz_to_rotmat
from aerosplat.geometry.coordinates import c2w_to_w2c, opencv_cam2world_matrix


def load_aerial_megadepth(root: Path, cfg) -> SpecDataset:
    root = Path(root)
    views_per_sample = int(getattr(cfg, "views_per_sample", 4))
    groups: list[ViewGroup] = []
    scenes = discover_scenes(root)
    if not scenes:
        raise DataError(
            f"{root} has no AerialMegaDepth scene. Expected views.json, COLMAP "
            "cameras.txt, or per-image npz files with intrinsics and cam2world."
        )
    for scene in scenes:
        views = _read_scene(scene)
        scene_id = scene.name if scene != root else root.name
        groups.extend(chunk_views(views, views_per_sample, scene_id))
    return SpecDataset(groups)


def discover_scenes(root: Path) -> list[Path]:
    root = Path(root)
    if _is_scene(root):
        return [root]
    found: list[Path] = []
    seen: set[Path] = set()
    for manifest in sorted(root.glob("**/views.json")):
        parent = manifest.parent
        if parent not in seen:
            found.append(parent)
            seen.add(parent)
    for cameras in sorted(root.glob("**/cameras.txt")):
        parent = cameras.parent
        # sparse/ and sparse-txt/ hold the COLMAP files; the scene is the parent
        # that also has images/, which is how the localized_dense_metric tree is laid out.
        scene = parent if (parent / "images").is_dir() else parent.parent
        if scene not in seen and _is_scene(scene):
            found.append(scene)
            seen.add(scene)
    return found


def _is_scene(path: Path) -> bool:
    if (path / "views.json").is_file():
        return True
    if list(path.glob("*.npz")):
        return True
    for cameras in (
        path / "cameras.txt",
        path / "sparse" / "cameras.txt",
        path / "sparse-txt" / "cameras.txt",
    ):
        if cameras.is_file() and (path / "images").is_dir():
            return True
    return False


def _read_scene(scene: Path) -> list[ViewSpec]:
    manifest = scene / "views.json"
    if manifest.is_file():
        return _read_manifest(scene, manifest)
    npz_files = sorted(scene.glob("*.npz"))
    if npz_files and not _colmap_cameras(scene):
        return [_read_npz(path) for path in npz_files]
    return _read_colmap(scene)


def _read_manifest(scene: Path, manifest: Path) -> list[ViewSpec]:
    payload = json.loads(manifest.read_text())
    views = []
    for index, entry in enumerate(payload["views"]):
        image_path = scene / entry["image"]
        depth_path = scene / entry["depth"] if entry.get("depth") else None
        K = np.asarray(entry["K"], dtype=np.float64).reshape(3, 3)
        if "cam2world" in entry:
            matrix = np.asarray(entry["cam2world"], dtype=np.float64).reshape(4, 4)
            R, t = c2w_to_w2c(matrix[:3, :3], matrix[:3, 3])
        elif "w2c" in entry:
            matrix = np.asarray(entry["w2c"], dtype=np.float64).reshape(4, 4)
            R, t = matrix[:3, :3], matrix[:3, 3]
        else:
            raise DataError(f"{manifest} view {index} needs cam2world or w2c")
        views.append(
            ViewSpec(
                name=str(entry.get("name", f"view{index}")),
                image_path=image_path,
                depth_path=depth_path,
                K=K,
                R=R,
                t=t,
                depth_scale=float(entry.get("depth_scale", 1.0)),
                depth_format=str(entry.get("depth_format", "auto")),
            )
        )
    if not views:
        raise DataError(f"{manifest} lists no views")
    return views


def _read_npz(path: Path) -> ViewSpec:
    with np.load(path) as data:
        if "intrinsics" not in data or "cam2world" not in data:
            raise DataError(f"{path} needs 'intrinsics' and 'cam2world' arrays")
        K = np.asarray(data["intrinsics"], dtype=np.float64).reshape(3, 3)
        matrix = np.asarray(data["cam2world"], dtype=np.float64)
        image_key = "image" if "image" in data.files else None
        depth_key = "depth" if "depth" in data.files else None
    if matrix.shape == (3, 4):
        full = np.eye(4)
        full[:3] = matrix
        matrix = full
    R, t = c2w_to_w2c(matrix[:3, :3], matrix[:3, 3])
    image_path = path.with_suffix(".png")
    if not image_path.is_file():
        image_path = path.with_suffix(".jpg")
    depth_path = path.with_name(path.stem + "_depth.npy")
    if not depth_path.is_file():
        depth_path = None
    if image_key is not None or not image_path.is_file():
        raise DataError(
            f"{path} must sit beside {path.stem}.png (or .jpg). Embedded image arrays "
            "are not written out by this loader."
        )
    del image_key, depth_key
    return ViewSpec(name=path.stem, image_path=image_path, depth_path=depth_path, K=K, R=R, t=t)


def _colmap_cameras(scene: Path) -> Path | None:
    for candidate in (
        scene / "cameras.txt",
        scene / "sparse" / "cameras.txt",
        scene / "sparse-txt" / "cameras.txt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _read_colmap(scene: Path) -> list[ViewSpec]:
    cameras_path = _colmap_cameras(scene)
    if cameras_path is None:
        raise DataError(f"{scene} has no COLMAP cameras.txt")
    images_path = cameras_path.with_name("images.txt")
    if not images_path.is_file():
        raise DataError(f"{scene} has cameras.txt but no images.txt beside it")
    cameras = parse_colmap_cameras(cameras_path.read_text())
    images = parse_colmap_images(images_path.read_text())
    views = []
    for image in images:
        cam = cameras.get(image["camera_id"])
        if cam is None:
            raise DataError(f"{images_path} refers to unknown camera {image['camera_id']}")
        image_path = _find_image(scene, image["name"])
        depth_path = _find_depth(scene, image_path)
        R = quat_wxyz_to_rotmat(image["quat_wxyz"])
        views.append(
            ViewSpec(
                name=Path(image["name"]).stem,
                image_path=image_path,
                depth_path=depth_path,
                K=cam,
                R=R,
                t=image["t"],
            )
        )
    if not views:
        raise DataError(f"{images_path} has no images")
    return views


def parse_colmap_cameras(text: str) -> dict[int, np.ndarray]:
    """SIMPLE_PINHOLE and PINHOLE. Radial distortion is ignored (the release is pinhole)."""
    cameras: dict[int, np.ndarray] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            cam_id = int(parts[0])
            model = parts[1]
            params = [float(value) for value in parts[4:]]
        except (IndexError, ValueError) as exc:
            raise DataError(f"cameras.txt:{line_no} is not a COLMAP camera line") from exc
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        else:
            raise DataError(f"cameras.txt:{line_no} model {model} is not supported")
        cameras[cam_id] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return cameras


def parse_colmap_images(text: str) -> list[dict]:
    """World-to-camera quaternion (wxyz) and translation, OpenCV convention.

    COLMAP writes two lines per image. The second line is the 2D points and can
    be long, so it is skipped when it does not look like another image header
    (a header ends in a non-numeric file name).
    """
    lines = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    images = []
    index = 0
    while index < len(lines):
        parts = lines[index].split()
        if not _is_colmap_image_header(parts):
            raise DataError(f"images.txt line {lines[index][:48]!r} is not a COLMAP image pose")
        quat = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        translation = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
        images.append(
            {
                "quat_wxyz": quat,
                "t": translation,
                "camera_id": int(parts[8]),
                "name": parts[9],
            }
        )
        index += 1
        if index < len(lines) and not _is_colmap_image_header(lines[index].split()):
            index += 1
    return images


def _is_colmap_image_header(parts: list[str]) -> bool:
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        [float(part) for part in parts[1:8]]
        int(parts[8])
        float(parts[9])
        return False
    except ValueError:
        return len(parts) >= 10 and not _is_float(parts[9])


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _find_image(scene: Path, name: str) -> Path:
    candidates = [scene / "images" / name, scene / name, scene / "images" / Path(name).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DataError(f"COLMAP image {name} was not found under {scene}")


def _find_depth(scene: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    for folder in (scene / "depths", scene / "depth", image_path.parent):
        for suffix in (".npy", ".h5", ".hdf5"):
            candidate = folder / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def cam2world_of(spec: ViewSpec) -> np.ndarray:
    """OpenCV cam2world, meters. Exposed for tests and for callers who want the 4x4."""
    return opencv_cam2world_matrix(spec.R, spec.t)
