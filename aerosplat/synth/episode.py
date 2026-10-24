"""Capture one planned episode and write a canonical shard.

The loop is frame-locked: pause, teleport the rig and every actor, step the
simulator clock, teleport again so a physics step cannot drift the script,
then grab every camera before the clock moves. Depth and segmentation come
from the flare-off grab. The flare-on frame is a second grab when the
simulator can toggle lens effects, and a sun-aligned composite otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.capture.airsim import build_settings
from aerosplat.geometry.coordinates import airsim_mount_to_w2c, enu_to_ned, intrinsics_from_hfov
from aerosplat.geometry.sun import sun_in_camera
from aerosplat.synth.client import ScenarioClient
from aerosplat.synth.motion import quat_align_body_x
from aerosplat.synth.render import (
    downsample_depth,
    downsample_rgb,
    downsample_segmentation,
    paint_sun_flare,
    project_dynamics,
)
from aerosplat.synth.scenario import EpisodePlan, PlannedActor, broadcast_actor_velocity
from aerosplat.synth.writer import (
    dynamics_document,
    segmentation_document,
    sun_document,
    write_shard,
)

_KIND_RADIUS = {"drone": 0.5, "bird": 0.5, "vehicle": 0.5}


def scale_rig(rig: dict, factor: int) -> dict:
    """Multiply every camera's width and height. Mounts and FOV stay put."""
    if factor < 1:
        raise ValueError("supersample factor must be >= 1")
    if factor == 1:
        return rig
    cameras = {}
    for name, cam in rig["cameras"].items():
        cameras[name] = {
            **cam,
            "width": int(cam["width"]) * factor,
            "height": int(cam["height"]) * factor,
        }
    return {**rig, "cameras": cameras}


def run_episode(
    client: ScenarioClient,
    plan: EpisodePlan,
    rig: dict,
    output_dir: Path,
    *,
    scene_id: str,
    frame_slice: slice | None = None,
) -> dict:
    """Write one shard covering ``frame_slice`` of ``plan`` (default: all frames)."""
    frames = plan.frames[frame_slice] if frame_slice is not None else plan.frames
    if not frames:
        raise ValueError("episode slice has no frames")
    factor = int(plan.supersample)
    capture_rig = scale_rig(rig, factor)
    settings = build_settings(
        capture_rig,
        plan.simulator,
        record_segmentation=True,
        origin_geopoint={
            "latitude_deg": plan.location.latitude_deg,
            "longitude_deg": plan.location.longitude_deg,
            "altitude_m": plan.location.altitude_m,
        },
    )
    client.pause(True)
    client.set_weather(plan.weather.as_dict())
    if plan.autoload_map and plan.map_name and plan.map_name != "current":
        client.load_level(plan.map_name)
    client.reset_spawned()
    _spawn_world(client, plan)
    lens_method = client.set_lens_effects(False)
    glare_source = _glare_source(plan.glare_mode, lens_method)
    streams: dict[str, list[dict]] = {name: [] for name in rig["cameras"]}
    dynamic_frames = []
    sun_frames = []
    for frame in frames:
        _pose_frame(client, frame, plan.actors)
        client.set_time_of_day(frame.datetime_utc.strftime("%Y-%m-%d %H:%M:%S"))
        client.step(plan.dt)
        _pose_frame(client, frame, plan.actors)
        _push_render_state(client, frame, plan)
        actor_records = _actor_records(frame, plan.actors)
        velocities = [
            np.asarray(record["velocity_enu"], dtype=np.float64) for record in actor_records
        ]
        broadcast = broadcast_actor_velocity(velocities)
        cameras_sun = {}
        for name, cam in rig["cameras"].items():
            rotation, translation = airsim_mount_to_w2c(
                frame.ego_position_ned,
                frame.ego_quaternion_wxyz,
                cam_position_body=np.array([cam["x"], cam["y"], cam["z"]], dtype=np.float64),
                pitch_deg=float(cam["pitch"]),
                roll_deg=float(cam["roll"]),
                yaw_deg=float(cam["yaw"]),
            )
            intrinsics = intrinsics_from_hfov(
                int(cam["width"]), int(cam["height"]), float(rig["fov_degrees"])
            )
            sun_camera = sun_in_camera(frame.sun_world_enu, rotation)
            clean, depth, segmentation = _grab_downsampled(
                client, capture_rig, name, factor, wire_color_of(plan)
            )
            if clean.shape[0] != int(cam["height"]) or clean.shape[1] != int(cam["width"]):
                raise ValueError(
                    f"{name}: image is {clean.shape[1]}x{clean.shape[0]}, "
                    f"rig is {cam['width']}x{cam['height']}"
                )
            observed, used = _flare_on(
                client,
                capture_rig,
                name,
                factor,
                plan,
                clean,
                sun_camera,
                intrinsics,
                glare_source,
            )
            actors, wires = _projection_lists(frame, plan)
            instance, velocity, coverage = project_dynamics(
                actors,
                wires,
                intrinsics=intrinsics,
                rotation_w2c=rotation,
                translation_w2c=translation,
                height=int(cam["height"]),
                width=int(cam["width"]),
            )
            streams[name].append(
                {
                    "timestamp": frame.timestamp,
                    "image": observed,
                    "depth": depth,
                    "segmentation": segmentation,
                    "clean_image": clean,
                    "K": intrinsics,
                    "R": rotation,
                    "t": translation,
                    "velocity": velocity,
                    "instance": instance,
                    "wire_coverage": coverage if plan.record_wire_visibility else None,
                }
            )
            cameras_sun[name] = {
                "sun_camera": sun_camera.tolist(),
                "in_front": bool(sun_camera[2] > 0.0),
            }
        dynamic_frames.append(
            {
                "index": frame.index,
                "timestamp": frame.timestamp,
                "ego_position_enu": frame.ego_position_enu,
                "ego_velocity_enu": frame.ego_velocity_enu,
                "ego_quaternion_wxyz": frame.ego_quaternion_wxyz,
                "gt_velocity_enu": None if broadcast is None else broadcast,
                "actors": actor_records,
            }
        )
        sun_frames.append(
            {
                "index": frame.index,
                "timestamp": frame.timestamp,
                "datetime_utc": frame.datetime_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sun_world_enu": frame.sun_world_enu,
                "glare_source": used,
                "cameras": cameras_sun,
            }
        )
    ids = {actor.name: actor.segmentation_id for actor in plan.actors}
    if plan.wires:
        ids["wire"] = plan.wires[0].segmentation_id
    write_shard(
        output_dir,
        scene_id,
        streams,
        dynamics=dynamics_document(
            latitude_deg=plan.location.latitude_deg,
            longitude_deg=plan.location.longitude_deg,
            scene_id=scene_id,
            frames=dynamic_frames,
        ),
        sun=sun_document(
            latitude_deg=plan.location.latitude_deg,
            longitude_deg=plan.location.longitude_deg,
            frames=sun_frames,
        ),
        wires=[
            {
                "name": wire.name,
                "kind": wire.kind,
                "segmentation_id": wire.segmentation_id,
                "diameter_m": wire.diameter_m,
                "sag_m": wire.sag_m,
                "points_enu": wire.points_enu,
            }
            for wire in plan.wires
        ],
        segmentation_ids=segmentation_document(ids),
        settings=settings,
        episode_meta={
            "scenario": plan.scenario_name,
            "simulator": plan.simulator,
            "map": plan.map_name,
            "seed": plan.seed,
            "scene_id": scene_id,
            "frames": [frame.index for frame in frames],
            "supersample": factor,
            "glare_source": glare_source,
            "aliasing": (
                "Cables of 1-5 cm are often under half a pixel at 30-300 m. "
                "wire_coverage stores the analytic fraction. supersample renders "
                "the hard segmentation at a finer grid and keeps a wire-colored "
                "subpixel when the block is reduced. Depth downsample keeps the "
                "closest positive return."
            ),
        },
    )
    return {
        "scene_id": scene_id,
        "frames": len(frames),
        "cameras": len(rig["cameras"]),
        "glare_source": glare_source,
        "lens_method": lens_method,
    }


def wire_color_of(plan: EpisodePlan) -> tuple[int, int, int]:
    from aerosplat.synth.palette import segmentation_color

    if not plan.wires:
        return segmentation_color(250)
    return segmentation_color(plan.wires[0].segmentation_id)


def _flare_on(
    client,
    capture_rig,
    name,
    factor,
    plan,
    clean,
    sun_camera,
    intrinsics,
    glare_source,
):
    if glare_source == "simulator":
        client.set_lens_effects(True)
        observed, _, _ = _grab_downsampled(client, capture_rig, name, factor, wire_color_of(plan))
        client.set_lens_effects(False)
        return observed, "simulator"
    if glare_source == "synthetic":
        return paint_sun_flare(clean, sun_camera, intrinsics), "synthetic"
    return clean.copy(), "identical"


def _grab_downsampled(client, capture_rig, name, factor, wire_color):
    rgb, depth, segmentation = client.grab(name)
    expected = capture_rig["cameras"][name]
    if rgb.shape[0] != int(expected["height"]) or rgb.shape[1] != int(expected["width"]):
        raise ValueError(
            f"{name}: simulator returned {rgb.shape[1]}x{rgb.shape[0]}, "
            f"settings asked for {expected['width']}x{expected['height']}. "
            "Copy this shard's settings.json into the simulator and restart the level."
        )
    if factor == 1:
        return rgb, depth, segmentation
    return (
        downsample_rgb(rgb, factor),
        downsample_depth(depth, factor),
        downsample_segmentation(segmentation, factor, wire_color=wire_color),
    )


def _glare_source(mode: str, lens_method: str) -> str:
    if mode == "synthetic":
        return "synthetic"
    if mode == "simulator":
        return "simulator" if lens_method != "unavailable" else "identical"
    if lens_method == "unavailable":
        return "synthetic"
    return "simulator"


def _spawn_world(client: ScenarioClient, plan: EpisodePlan) -> None:
    for actor in plan.actors:
        client.spawn_actor(
            actor.name,
            actor.asset,
            actor.position_ned[0],
            actor.quaternion_wxyz[0],
            actor.scale_m,
        )
        client.set_segmentation_id(actor.name, actor.segmentation_id)
    for wire in plan.wires:
        points = wire.points_enu
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue
            name = f"{wire.name}_{index:03d}"
            midpoint = 0.5 * (start + end)
            direction_ned = enu_to_ned(delta)
            client.spawn_wire_segment(
                name,
                wire.asset,
                enu_to_ned(midpoint),
                quat_align_body_x(direction_ned),
                np.array([length, wire.diameter_m, wire.diameter_m], dtype=np.float64),
            )
            client.set_segmentation_id(name, wire.segmentation_id)


def _pose_frame(client: ScenarioClient, frame, actors: list[PlannedActor]) -> None:
    client.set_vehicle_pose_ned(frame.ego_position_ned, frame.ego_quaternion_wxyz)
    for actor in actors:
        client.set_actor_pose_ned(
            actor.name,
            actor.position_ned[frame.index],
            actor.quaternion_wxyz[frame.index],
        )


def _push_render_state(client, frame, plan: EpisodePlan) -> None:
    setter = getattr(client, "set_world", None)
    if setter is None:
        return
    actors, wires = _projection_lists(frame, plan)
    setter(actors, wires, frame.sun_world_enu)


def _projection_lists(frame, plan: EpisodePlan):
    actors = []
    for actor in plan.actors:
        scale = actor.scale_m
        actors.append(
            {
                "position_enu": actor.position_enu[frame.index],
                "velocity_enu": actor.velocity_enu[frame.index],
                "segmentation_id": actor.segmentation_id,
                "radius_m": _KIND_RADIUS.get(actor.kind, 0.5) * float(np.max(scale)),
            }
        )
    wires = [
        {
            "points_enu": wire.points_enu,
            "diameter_m": wire.diameter_m,
            "segmentation_id": wire.segmentation_id,
        }
        for wire in plan.wires
    ]
    return actors, wires


def _actor_records(frame, actors: list[PlannedActor]) -> list[dict]:
    records = []
    for actor in actors:
        records.append(
            {
                "name": actor.name,
                "kind": actor.kind,
                "segmentation_id": actor.segmentation_id,
                "position_enu": actor.position_enu[frame.index],
                "quaternion_wxyz": actor.quaternion_wxyz[frame.index],
                "velocity_enu": actor.velocity_enu[frame.index],
            }
        )
    return records


def read_sun(path: Path) -> dict:
    return json.loads(Path(path).read_text())
