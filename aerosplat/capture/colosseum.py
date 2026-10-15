"""Capture a four-camera rig from Colosseum (the maintained AirSim fork).

The script writes the canonical recording read by ``data.source=canonical``.
It also writes ``settings.json``, which is what the simulator must load so
the mounts match the extrinsics stored next to each image.

``--dry-run`` only writes that settings file. Connecting requires the
``airsim`` or ``colosseum`` Python client, which is not a dependency of this
package and is not installed in CI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from aerosplat.config import load_config
from aerosplat.data.adapters.canonical import write_canonical
from aerosplat.data.errors import DataError
from aerosplat.geometry.coordinates import airsim_mount_to_w2c, intrinsics_from_hfov

SCENE = 0
DEPTH_PLANAR = 1


class RigClient(Protocol):
    def vehicle_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """NED position (meters) and body-to-NED quaternion wxyz."""

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 HWC and planar-Z depth float32 HW, in meters."""

    def timestamp(self) -> float:
        """Seconds on a monotonic clock for this capture session."""


def build_settings(rig: dict) -> dict:
    """AirSim/Colosseum ``settings.json`` for the configured mounts.

    ImageType 0 is Scene (RGB). ImageType 1 is DepthPlanar, which is camera-frame
    Z in meters. DepthPerspective (type 2) is intentionally not requested:
    the rest of the stack treats depth as planar Z unless a raw log says otherwise.
    """
    fov = float(rig["fov_degrees"])
    cameras = {}
    for name, cam in rig["cameras"].items():
        width, height = int(cam["width"]), int(cam["height"])
        cameras[name] = {
            "CaptureSettings": [
                {
                    "ImageType": SCENE,
                    "Width": width,
                    "Height": height,
                    "FOV_Degrees": fov,
                },
                {
                    "ImageType": DEPTH_PLANAR,
                    "Width": width,
                    "Height": height,
                    "FOV_Degrees": fov,
                },
            ],
            "X": float(cam["x"]),
            "Y": float(cam["y"]),
            "Z": float(cam["z"]),
            "Pitch": float(cam["pitch"]),
            "Roll": float(cam["roll"]),
            "Yaw": float(cam["yaw"]),
        }
    return {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ViewMode": "FlyWithMe",
        "Vehicles": {
            str(rig.get("vehicle_name", "Drone1")): {
                "VehicleType": "SimpleFlight",
                "AutoCreate": True,
                "Cameras": cameras,
            }
        },
    }


def capture_episode(
    client: RigClient,
    rig: dict,
    output_dir: Path,
    *,
    scene_id: str = "flight",
    num_frames: int = 1,
) -> Path:
    """Grab ``num_frames`` from every camera and write a canonical dataset."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    output_dir = Path(output_dir)
    settings = build_settings(rig)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    streams: dict[str, list[dict]] = {name: [] for name in rig["cameras"]}
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
    write_canonical(output_dir, {scene_id: streams})
    return output_dir


class _RpcClient:
    """Live Colosseum/AirSim client. Constructed only from the CLI."""

    def __init__(self, ip: str) -> None:
        self._api, self._mod = _connect(ip)
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
        rgb_resp, depth_resp = responses
        if rgb_resp.height == 0 or rgb_resp.width == 0:
            raise DataError(
                f"camera {camera!r} returned an empty image. Is that name in settings.json?"
            )
        rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8).copy()
        rgb = rgb.reshape(rgb_resp.height, rgb_resp.width, 3)
        depth = np.asarray(depth_resp.image_data_float, dtype=np.float32).reshape(
            depth_resp.height, depth_resp.width
        )
        return rgb, depth

    def timestamp(self) -> float:
        return time.perf_counter() - self._t0


def _connect(ip: str):
    module = None
    errors = []
    for name in ("airsim", "colosseum"):
        try:
            module = __import__(name)
            break
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    if module is None:
        raise DataError(
            "Could not import the Colosseum/AirSim client (tried airsim, colosseum). "
            "Install the client from https://github.com/CodexLabsLLC/Colosseum "
            "or re-run with --dry-run. " + " | ".join(errors)
        )
    client = module.MultirotorClient(ip=ip) if ip else module.MultirotorClient()
    client.confirmConnection()
    return client, module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture a 4-camera Colosseum/AirSim rig.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--ip", default="", help="RPC host. Empty uses the client default.")
    parser.add_argument("--scene-id", default="flight")
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
    settings = build_settings(rig)
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"wrote {settings_path}")
    if args.dry_run or args.frames <= 0:
        print(
            "Dry run: copy settings.json into the Colosseum/AirSim settings path, "
            "start the simulator, then re-run with --frames N."
        )
        return
    client = _RpcClient(args.ip)
    capture_episode(client, rig, args.output, scene_id=args.scene_id, num_frames=args.frames)
    print(f"wrote canonical recording under {args.output}")


if __name__ == "__main__":
    main()
