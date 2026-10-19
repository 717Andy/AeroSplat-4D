"""Simulator session for one synthetic episode.

The live client speaks the AirSim-family RPC already used by
``aerosplat.capture.airsim`` (Colosseum by default, Cosys-AirSim when
selected). The mock client renders disks and hard-rasterized cables so CI
can run the same episode loop without Unreal.

Lens flare is not a first-class RPC on either simulator. ``set_lens_effects``
tries a post-process method, then a console command, and reports
``unavailable`` when neither exists. The episode loop then composites a
sun-aligned blob, which is the documented fallback.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from aerosplat.capture.airsim import (
    _connect,
    decode_planar_depth,
    decode_scene_rgb,
    decode_segmentation,
)
from aerosplat.data.errors import DataError
from aerosplat.geometry.coordinates import airsim_mount_to_w2c, intrinsics_from_hfov, ned_to_enu
from aerosplat.synth.palette import segmentation_color
from aerosplat.synth.render import render_hard_frame

# AirSim WeatherParameter order.
_WEATHER = {
    "rain": 0,
    "road_wetness": 1,
    "snow": 2,
    "road_snow": 3,
    "maple_leaf": 4,
    "road_leaf": 5,
    "dust": 6,
    "fog": 7,
}


class ScenarioClient(Protocol):
    def pause(self, paused: bool) -> None: ...

    def step(self, dt: float) -> None: ...

    def set_vehicle_pose_ned(
        self, position_ned: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> None: ...

    def spawn_actor(
        self,
        name: str,
        asset: str,
        position_ned: np.ndarray,
        quaternion_wxyz: np.ndarray,
        scale_m: np.ndarray,
    ) -> None: ...

    def set_actor_pose_ned(
        self, name: str, position_ned: np.ndarray, quaternion_wxyz: np.ndarray
    ) -> None: ...

    def spawn_wire_segment(
        self,
        name: str,
        asset: str,
        position_ned: np.ndarray,
        quaternion_wxyz: np.ndarray,
        scale_m: np.ndarray,
    ) -> None: ...

    def set_segmentation_id(self, name: str, object_id: int) -> None: ...

    def set_time_of_day(self, datetime_text: str) -> None: ...

    def set_weather(self, weather: dict[str, float]) -> None: ...

    def load_level(self, name: str) -> bool: ...

    def reset_spawned(self) -> None: ...

    def set_lens_effects(self, enabled: bool) -> str: ...

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class MockScenarioClient:
    """In-process rig. Poses are stored. Images are the hard rasterizer."""

    def __init__(self, rig: dict, *, supersample: int = 1) -> None:
        self.rig = rig
        self.supersample = int(supersample)
        self.paused = False
        self.steps = 0
        self.grabs = 0
        self.vehicle_ned = np.zeros(3, dtype=np.float64)
        self.vehicle_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.actors: dict[str, dict] = {}
        self.segments: dict[str, dict] = {}
        self.weather: dict[str, float] = {}
        self.datetime_text = ""
        self.level = ""
        self.lens_enabled = False
        self.lens_method = "unavailable"
        self.spawn_failures: list[str] = []
        self.world_actors: list[dict] = []
        self.world_wires: list[dict] = []
        self.sun_world = np.array([0.0, 0.2, 0.98], dtype=np.float64)

    def set_world(self, actors: list[dict], wires: list[dict], sun_world: np.ndarray) -> None:
        """Analytic scene the rasterizer draws. The episode loop updates this per frame."""
        self.world_actors = actors
        self.world_wires = wires
        self.sun_world = np.asarray(sun_world, dtype=np.float64).reshape(3)

    def pause(self, paused: bool) -> None:
        self.paused = bool(paused)

    def step(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be >= 0")
        self.steps += 1

    def set_vehicle_pose_ned(self, position_ned: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self.vehicle_ned = np.asarray(position_ned, dtype=np.float64).reshape(3).copy()
        self.vehicle_quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4).copy()

    def spawn_actor(self, name, asset, position_ned, quaternion_wxyz, scale_m) -> None:
        self.actors[name] = {
            "asset": asset,
            "position_ned": np.asarray(position_ned, dtype=np.float64).reshape(3),
            "quaternion_wxyz": np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4),
            "scale_m": np.asarray(scale_m, dtype=np.float64).reshape(3),
        }

    def set_actor_pose_ned(self, name, position_ned, quaternion_wxyz) -> None:
        slot = self.actors.get(name)
        if slot is None:
            raise KeyError(name)
        slot["position_ned"] = np.asarray(position_ned, dtype=np.float64).reshape(3)
        slot["quaternion_wxyz"] = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)

    def spawn_wire_segment(self, name, asset, position_ned, quaternion_wxyz, scale_m) -> None:
        self.segments[name] = {"asset": asset, "scale_m": np.asarray(scale_m, dtype=np.float64)}

    def set_segmentation_id(self, name: str, object_id: int) -> None:
        if name in self.actors:
            self.actors[name]["segmentation_id"] = int(object_id)
        elif name in self.segments:
            self.segments[name]["segmentation_id"] = int(object_id)

    def set_time_of_day(self, datetime_text: str) -> None:
        self.datetime_text = str(datetime_text)

    def set_weather(self, weather: dict[str, float]) -> None:
        self.weather = {key: float(value) for key, value in weather.items()}

    def load_level(self, name: str) -> bool:
        self.level = str(name)
        return True

    def reset_spawned(self) -> None:
        self.actors.clear()
        self.segments.clear()

    def set_lens_effects(self, enabled: bool) -> str:
        self.lens_enabled = bool(enabled)
        return self.lens_method

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam = self.rig["cameras"][camera]
        height, width = int(cam["height"]), int(cam["width"])
        rotation, translation = airsim_mount_to_w2c(
            self.vehicle_ned,
            self.vehicle_quat,
            cam_position_body=np.array([cam["x"], cam["y"], cam["z"]], dtype=np.float64),
            pitch_deg=float(cam["pitch"]),
            roll_deg=float(cam["roll"]),
            yaw_deg=float(cam["yaw"]),
        )
        intrinsics = intrinsics_from_hfov(width, height, float(self.rig["fov_degrees"]))
        wire_color = segmentation_color(self._wire_id())
        rgb, depth, segmentation = render_hard_frame(
            self.world_actors,
            self.world_wires,
            intrinsics=intrinsics,
            rotation_w2c=rotation,
            translation_w2c=translation,
            height=height,
            width=width,
            sun_world=self.sun_world,
            wire_color=wire_color,
        )
        self.grabs += 1
        return rgb, depth, segmentation

    def _wire_id(self) -> int:
        if not self.world_wires:
            return 250
        return int(self.world_wires[0]["segmentation_id"])


class AirSimScenarioClient:
    """Live Colosseum or Cosys-AirSim session. Constructed only from the CLI."""

    def __init__(self, ip: str, simulator: str) -> None:
        from aerosplat.airsim_family import get_profile

        self.simulator = simulator
        self._profile = get_profile(simulator)
        self._api, self._mod = _connect(ip, self._profile)
        self._pose = self._mod.Pose
        self._vector = self._mod.Vector3r
        self._spawned: list[str] = []

    def pause(self, paused: bool) -> None:
        self._api.simPause(bool(paused))

    def step(self, dt: float) -> None:
        # Continues a paused sim for ``dt`` seconds and leaves it paused.
        self._api.simContinueForTime(float(dt))

    def set_vehicle_pose_ned(self, position_ned: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        self._api.simSetVehiclePose(self._airsim_pose(position_ned, quaternion_wxyz), True)

    def spawn_actor(self, name, asset, position_ned, quaternion_wxyz, scale_m) -> None:
        self._spawn(name, asset, position_ned, quaternion_wxyz, scale_m)

    def set_actor_pose_ned(self, name, position_ned, quaternion_wxyz) -> None:
        pose = self._airsim_pose(position_ned, quaternion_wxyz)
        ok = self._api.simSetObjectPose(name, pose, True)
        if ok is False:
            raise DataError(f"simSetObjectPose failed for {name!r}")

    def spawn_wire_segment(self, name, asset, position_ned, quaternion_wxyz, scale_m) -> None:
        self._spawn(name, asset, position_ned, quaternion_wxyz, scale_m)

    def set_segmentation_id(self, name: str, object_id: int) -> None:
        self._api.simSetSegmentationObjectID(str(name), int(object_id), False)

    def set_time_of_day(self, datetime_text: str) -> None:
        # update_interval_secs is small so a single step applies the new sun
        # while the sim is frame-locked. The default of 60 s would skip it.
        self._api.simSetTimeOfDay(True, datetime_text, False, 1.0, 0.001, True)

    def set_weather(self, weather: dict[str, float]) -> None:
        enable = getattr(self._api, "simEnableWeather", None)
        if enable is not None:
            enable(True)
        enum = getattr(self._mod, "WeatherParameter", None)
        for name, value in weather.items():
            if name not in _WEATHER:
                continue
            parameter = _WEATHER[name]
            if enum is not None:
                try:
                    parameter = getattr(enum, _ENUM_NAMES[name])
                except AttributeError:
                    parameter = _WEATHER[name]
            self._api.simSetWeatherParameter(parameter, float(value))

    def load_level(self, name: str) -> bool:
        method = getattr(self._api, "simLoadLevel", None)
        if method is None or not name or name == "current":
            return False
        method(name)
        return True

    def reset_spawned(self) -> None:
        destroy = getattr(self._api, "simDestroyObject", None)
        if destroy is None:
            self._spawned.clear()
            return
        for name in self._spawned:
            try:
                destroy(name)
            except Exception:
                continue
        self._spawned.clear()

    def set_lens_effects(self, enabled: bool) -> str:
        flag = bool(enabled)
        for method_name in ("simSetLensFlare", "simSetBloom", "simSetPostProcessSettings"):
            method = getattr(self._api, method_name, None)
            if method is None:
                continue
            method(flag)
            return "rpc"
        console = getattr(self._api, "simRunConsoleCommand", None)
        if console is not None:
            quality = "1" if flag else "0"
            console(f"r.DefaultFeature.LensFlare {quality}")
            console(f"r.BloomQuality {quality}")
            return "console"
        return "unavailable"

    def grab(self, camera: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        requests = [
            self._mod.ImageRequest(camera, self._mod.ImageType.Scene, False, False),
            self._mod.ImageRequest(camera, self._mod.ImageType.DepthPlanar, True, False),
            self._mod.ImageRequest(camera, self._mod.ImageType.Segmentation, False, False),
        ]
        responses = self._api.simGetImages(requests)
        if len(responses) != 3:
            raise DataError(f"{camera}: expected 3 images, got {len(responses)}")
        rgb_response, depth_response, seg_response = responses
        if rgb_response.height == 0 or rgb_response.width == 0:
            raise DataError(f"camera {camera!r} returned an empty image. Is settings.json loaded?")
        rgb = decode_scene_rgb(
            rgb_response.image_data_uint8,
            int(rgb_response.height),
            int(rgb_response.width),
            self.simulator,
        )
        depth = decode_planar_depth(
            depth_response.image_data_float,
            int(depth_response.height),
            int(depth_response.width),
            self.simulator,
        )
        segmentation = decode_segmentation(
            seg_response.image_data_uint8,
            int(seg_response.height),
            int(seg_response.width),
            self.simulator,
        )
        return rgb, depth, segmentation

    def _spawn(self, name, asset, position_ned, quaternion_wxyz, scale_m) -> None:
        scale = np.asarray(scale_m, dtype=np.float64).reshape(3)
        try:
            self._api.simSpawnObject(
                str(name),
                str(asset),
                self._airsim_pose(position_ned, quaternion_wxyz),
                self._vector(float(scale[0]), float(scale[1]), float(scale[2])),
                False,
                False,
            )
            self._spawned.append(str(name))
        except Exception as exc:
            raise DataError(
                f"Could not spawn {name!r} from asset {asset!r}. "
                "The open Unreal level has to contain that mesh (Blocks usually has Cube). "
                f"The simulator said: {exc}"
            ) from exc

    def _airsim_pose(self, position_ned, quaternion_wxyz):
        position = np.asarray(position_ned, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        # Quaternionr stores xyzw. Recordings and this package store wxyz.
        return self._pose(
            self._vector(float(position[0]), float(position[1]), float(position[2])),
            self._mod.Quaternionr(float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])),
        )


_ENUM_NAMES = {
    "rain": "Rain",
    "road_wetness": "Roadwetness",
    "snow": "Snow",
    "road_snow": "RoadSnow",
    "maple_leaf": "MapleLeaf",
    "road_leaf": "RoadLeaf",
    "dust": "Dust",
    "fog": "Fog",
}


def vehicle_pose_to_enu(position_ned: np.ndarray) -> np.ndarray:
    return ned_to_enu(np.asarray(position_ned, dtype=np.float64).reshape(3))
