"""Capture a four-camera rig from an AirSim-family simulator.

``simulator`` selects the profile in :mod:`aerosplat.airsim_family`:

* ``colosseum`` — CodexLabsLLC/Colosseum, now archived. SettingsVersion 1.2.
  Scene RGB is stored in the order ``simGetImages`` returns.
* ``cosys`` — Cosys-Lab/Cosys-AirSim, the maintained option. SettingsVersion 2.0.
  Uncompressed Scene RGB is vertically flipped to undo the documented
  ``simGetImages`` orientation. Depth and segmentation are not flipped.

Both write the canonical recording read by ``data.source=canonical``, plus the
``settings.json`` the simulator must load. Vehicle pose is NED; the mount is
the body frame (X forward, Y right, Z down). ``External: true`` cameras are
rejected because their pose is Unreal world (or local NED when
``ExternalLocal`` is set), which is a different frame from the body mount.

``--dry-run`` only writes ``settings.json``. The Python client (``airsim`` /
``colosseum``, or ``cosysairsim``) is not a dependency of this package and is
not installed in CI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from aerosplat.airsim_family import (
    DEPTH_PLANAR,
    SCENE,
    SEGMENTATION,
    SIMULATORS,
    SimulatorProfile,
    get_profile,
)
from aerosplat.config import load_config
from aerosplat.data.adapters.canonical import write_canonical
from aerosplat.data.errors import DataError
from aerosplat.data.io import save_rgb
from aerosplat.geometry.coordinates import airsim_mount_to_w2c, intrinsics_from_hfov


class RigClient(Protocol):
    def vehicle_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """NED position (meters) and body-to-NED quaternion wxyz."""

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 HWC and planar-Z depth float32 HW, in meters.

        The image is already in display order. The Cosys Scene-RGB flip is
        applied inside the RPC client, before this method returns.
        """

    def timestamp(self) -> float:
        """Seconds on a monotonic clock for this capture session."""


def build_settings(
    rig: dict,
    simulator: str = "colosseum",
    *,
    record_segmentation: bool = False,
) -> dict:
    """``settings.json`` for ``simulator``.

    ImageType 0 is Scene (RGB). ImageType 1 is DepthPlanar (camera-frame Z in
    meters). DepthPerspective (type 2) is Euclidean range and is left out so
    the rest of the stack sees planar Z. DepthVis and DisparityNormalized are
    not metric depth and are never requested.

    ``record_segmentation`` adds ImageType 5. On Colosseum that image is a
    mesh-object color palette. On Cosys-AirSim it is an instance color index.
    The canonical sample does not carry this image; it is a sidecar.
    """
    profile = get_profile(simulator)
    fov = float(rig["fov_degrees"])
    cameras = {}
    for name, cam in rig["cameras"].items():
        width, height = int(cam["width"]), int(cam["height"])
        captures = [
            _capture_setting(SCENE, width, height, fov),
            _capture_setting(DEPTH_PLANAR, width, height, fov),
        ]
        if record_segmentation:
            captures.append(_capture_setting(SEGMENTATION, width, height, fov))
        camera = {
            "CaptureSettings": captures,
            "X": float(cam["x"]),
            "Y": float(cam["y"]),
            "Z": float(cam["z"]),
            "Pitch": float(cam["pitch"]),
            "Roll": float(cam["roll"]),
            "Yaw": float(cam["yaw"]),
        }
        if profile.name == "cosys":
            # Vehicle mount. External true would pose the camera in Unreal world.
            camera["External"] = False
            camera["ExternalLocal"] = False
        cameras[name] = camera
    settings = {
        "SettingsVersion": profile.settings_version,
        "SimMode": "Multirotor",
        "ViewMode": "FlyWithMe",
    }
    if profile.name == "cosys":
        # Leave the Unreal origin where the map put it. MoveWorldOrigin true
        # relocates that origin to the player start and changes External coordinates.
        settings["MoveWorldOrigin"] = False
        # Random per-mesh color index at startup. This is instance segmentation,
        # which is a different id space from Colosseum's mesh-name object IDs.
        settings["InitialInstanceSegmentation"] = True
    settings["Vehicles"] = {
        str(rig.get("vehicle_name", "Drone1")): {
            "VehicleType": "SimpleFlight",
            "AutoCreate": True,
            "Cameras": cameras,
        }
    }
    return settings


def capture_episode(
    client: RigClient,
    rig: dict,
    output_dir: Path,
    *,
    scene_id: str = "flight",
    num_frames: int = 1,
    simulator: str = "colosseum",
    record_segmentation: bool = False,
) -> Path:
    """Grab ``num_frames`` from every camera and write a canonical dataset."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    profile = get_profile(simulator)
    output_dir = Path(output_dir)
    settings = build_settings(rig, profile.name, record_segmentation=record_segmentation)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    streams: dict[str, list[dict]] = {name: [] for name in rig["cameras"]}
    segmentation: dict[str, list[np.ndarray]] = {name: [] for name in rig["cameras"]}
    for _ in range(num_frames):
        position, quat = client.vehicle_pose()
        timestamp = float(client.timestamp())
        for name, cam in rig["cameras"].items():
            rgb, depth = client.grab(name)
            rgb = np.asarray(rgb)
            depth = np.asarray(depth)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise DataError(f"{name}: grab() RGB must be HxWx3, got {rgb.shape}")
            if depth.shape != rgb.shape[:2]:
                raise DataError(f"{name}: depth shape {depth.shape} != image {rgb.shape[:2]}")
            expected = (int(cam["height"]), int(cam["width"]))
            if rgb.shape[0] != expected[0] or rgb.shape[1] != expected[1]:
                raise DataError(
                    f"{name}: simulator returned {rgb.shape[1]}x{rgb.shape[0]}, "
                    f"settings requested {expected[1]}x{expected[0]}"
                )
            R, t = airsim_mount_to_w2c(
                np.asarray(position, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
                cam_position_body=np.array([cam["x"], cam["y"], cam["z"]], dtype=np.float64),
                pitch_deg=float(cam["pitch"]),
                roll_deg=float(cam["roll"]),
                yaw_deg=float(cam["yaw"]),
            )
            K = intrinsics_from_hfov(
                int(cam["width"]), int(cam["height"]), float(rig["fov_degrees"])
            )
            streams[name].append(
                {
                    "timestamp": timestamp,
                    "image": rgb,
                    "depth": depth.astype(np.float32),
                    "K": K,
                    "R": R,
                    "t": t,
                }
            )
            if record_segmentation:
                segmentation[name].append(_read_segmentation(client, name, expected, profile))
    write_canonical(output_dir, {scene_id: streams})
    if record_segmentation:
        _write_segmentation(output_dir, scene_id, segmentation, profile)
    return output_dir


def decode_scene_rgb(buffer, height: int, width: int, simulator: str) -> np.ndarray:
    """Decode uncompressed Scene RGB from ``simGetImages``.

    Cosys-AirSim's image API documents ``np.flipud`` on this buffer. Colosseum
    captures in this repo are stored without that flip, and existing recordings
    stay that way.
    """
    profile = get_profile(simulator)
    rgb = _as_uint8(buffer, height * width * 3).reshape(height, width, 3)
    if profile.flip_scene_rgb:
        rgb = np.ascontiguousarray(np.flipud(rgb))
    return rgb


def decode_planar_depth(values, height: int, width: int, simulator: str) -> np.ndarray:
    """Decode DepthPlanar float pixels as camera-plane Z in meters.

    The buffer is row-major. Neither profile flips it: Cosys's documented
    ``flipud`` is on the uncompressed RGB example, and the float depth path
    writes the array in capture order.
    """
    profile = get_profile(simulator)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size != height * width:
        raise DataError(
            f"planar depth has {flat.size} values, expected {height}x{width} = {height * width}"
        )
    depth = flat.reshape(height, width)
    if profile.flip_depth:
        depth = np.ascontiguousarray(np.flipud(depth))
    return depth


def decode_segmentation(buffer, height: int, width: int, simulator: str) -> np.ndarray:
    """Decode ImageType 5. The Cosys segmentation example is not flipped."""
    profile = get_profile(simulator)
    image = _as_uint8(buffer, height * width * 3).reshape(height, width, 3)
    if profile.flip_segmentation:
        image = np.ascontiguousarray(np.flipud(image))
    return image


def images_from_responses(
    rgb_response, depth_response, simulator: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the simulator's image conventions to one Scene + DepthPlanar pair."""
    if rgb_response.height == 0 or rgb_response.width == 0:
        raise DataError("camera returned an empty image. Is that name in settings.json?")
    rgb = decode_scene_rgb(
        rgb_response.image_data_uint8,
        int(rgb_response.height),
        int(rgb_response.width),
        simulator,
    )
    depth = decode_planar_depth(
        depth_response.image_data_float,
        int(depth_response.height),
        int(depth_response.width),
        simulator,
    )
    if depth.shape != rgb.shape[:2]:
        raise DataError(f"depth shape {depth.shape} != image {rgb.shape[:2]}")
    return rgb, depth


class _RpcClient:
    """Live AirSim-family client. Constructed only from the CLI."""

    def __init__(self, ip: str, simulator: str) -> None:
        self._profile = get_profile(simulator)
        self._api, self._mod = _connect(ip, self._profile)
        self._t0 = time.perf_counter()

    def vehicle_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pose = self._api.simGetVehiclePose()
        position = np.array(
            [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            dtype=np.float64,
        )
        quat = pose.orientation
        # The Python Quaternionr stores xyzw fields. The recording file stores wxyz.
        quat_wxyz = np.array([quat.w_val, quat.x_val, quat.y_val, quat.z_val], dtype=np.float64)
        return position, quat_wxyz

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        responses = self._api.simGetImages(
            [
                self._mod.ImageRequest(camera, self._mod.ImageType.Scene, False, False),
                self._mod.ImageRequest(camera, self._mod.ImageType.DepthPlanar, True, False),
            ]
        )
        rgb, depth = images_from_responses(responses[0], responses[1], self._profile.name)
        if rgb.size == 0:
            raise DataError(
                f"camera {camera!r} returned an empty image. Is that name in settings.json?"
            )
        return rgb, depth

    def grab_segmentation(self, camera: str) -> np.ndarray:
        responses = self._api.simGetImages(
            [self._mod.ImageRequest(camera, self._mod.ImageType.Segmentation, False, False)]
        )
        response = responses[0]
        if response.height == 0 or response.width == 0:
            raise DataError(f"camera {camera!r} returned an empty segmentation image")
        return decode_segmentation(
            response.image_data_uint8,
            int(response.height),
            int(response.width),
            self._profile.name,
        )

    def timestamp(self) -> float:
        return time.perf_counter() - self._t0


def _connect(ip: str, profile: SimulatorProfile):
    module = None
    errors = []
    for name in profile.python_modules:
        try:
            module = __import__(name)
            break
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    if module is None:
        status = "archived, read-only" if profile.archived else "maintained"
        raise DataError(
            f"Could not import the {profile.name} client "
            f"(tried {', '.join(profile.python_modules)}). "
            f"The {profile.name} repo is {status}: {profile.install_url} . "
            "Re-run with --dry-run to write settings.json without a simulator. "
            + " | ".join(errors)
        )
    client = module.MultirotorClient(ip=ip) if ip else module.MultirotorClient()
    client.confirmConnection()
    return client, module


def _capture_setting(image_type: int, width: int, height: int, fov: float) -> dict:
    return {
        "ImageType": image_type,
        "Width": width,
        "Height": height,
        "FOV_Degrees": fov,
    }


def _as_uint8(buffer, expected: int) -> np.ndarray:
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        values = np.frombuffer(buffer, dtype=np.uint8)
    else:
        values = np.asarray(buffer, dtype=np.uint8).reshape(-1)
    if values.size != expected:
        raise DataError(f"image buffer has {values.size} bytes, expected {expected}")
    return values.copy()


def _read_segmentation(
    client: RigClient,
    camera: str,
    expected_hw: tuple[int, int],
    profile: SimulatorProfile,
) -> np.ndarray:
    method = getattr(client, "grab_segmentation", None)
    if method is None:
        raise DataError(
            f"record_segmentation is set but {type(client).__name__} has no "
            f"grab_segmentation(). Camera {camera!r}, simulator {profile.name}."
        )
    image = np.asarray(method(camera))
    if image.ndim != 3 or image.shape[2] != 3:
        raise DataError(f"{camera}: segmentation must be HxWx3, got {image.shape}")
    if image.shape[0] != expected_hw[0] or image.shape[1] != expected_hw[1]:
        raise DataError(
            f"{camera}: segmentation is {image.shape[1]}x{image.shape[0]}, "
            f"settings requested {expected_hw[1]}x{expected_hw[0]}"
        )
    return image


def _write_segmentation(
    output_dir: Path,
    scene_id: str,
    streams: dict[str, list[np.ndarray]],
    profile: SimulatorProfile,
) -> None:
    """Sidecar only. Segmentation is not a depth channel and not part of [B, N, 62]."""
    manifest = {
        "simulator": profile.name,
        "image_type": SEGMENTATION,
        "id_space": profile.segmentation_id_space,
        "encoding": "rgb_color_index",
        "vertical_flip_applied": profile.flip_segmentation,
        "note": profile.segmentation_note,
        "cameras": {},
    }
    for name, frames in streams.items():
        rels = []
        for index, image in enumerate(frames):
            rel = f"{scene_id}/{name}/segmentation/{index:06d}.png"
            save_rgb(output_dir / rel, image)
            rels.append(rel)
        manifest["cameras"][name] = rels
    (output_dir / "segmentation.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture a 4-camera AirSim-family rig.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--ip", default="", help="RPC host. Empty uses the client default.")
    parser.add_argument("--scene-id", default="flight")
    parser.add_argument(
        "--simulator",
        choices=list(SIMULATORS),
        default="colosseum",
        help=(
            "colosseum is archived (SettingsVersion 1.2). "
            "cosys is Cosys-AirSim, the maintained option (SettingsVersion 2.0)."
        ),
    )
    parser.add_argument(
        "--segmentation",
        action="store_true",
        help=(
            "Also record ImageType 5 as a sidecar. Colosseum colors are mesh-object "
            "ids. Cosys-AirSim colors are instance ids."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write settings.json and exit without connecting to a simulator.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example rig=uam_quad.")
    args = parser.parse_args(argv)
    cfg = load_config(args.overrides)
    from omegaconf import OmegaConf

    rig = OmegaConf.to_container(cfg.rig, resolve=True)
    settings = build_settings(rig, args.simulator, record_segmentation=args.segmentation)
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"wrote {settings_path} ({args.simulator})")
    if args.dry_run or args.frames <= 0:
        if args.simulator == "cosys":
            where = "the Cosys-AirSim settings path (Documents/AirSim/settings.json)"
        else:
            where = "the Colosseum settings path"
        print(
            f"Dry run: copy settings.json into {where}, start the simulator, "
            "then re-run with --frames N."
        )
        return
    client = _RpcClient(args.ip, args.simulator)
    capture_episode(
        client,
        rig,
        args.output,
        scene_id=args.scene_id,
        num_frames=args.frames,
        simulator=args.simulator,
        record_segmentation=args.segmentation,
    )
    print(f"wrote canonical recording under {args.output}")


if __name__ == "__main__":
    main()
