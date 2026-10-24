"""CPU tests for the synthetic generator. No Unreal client is imported."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from aerosplat.config import load_config
from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.dataset import build_dataset
from aerosplat.geometry.camera import quat_wxyz_to_rotmat
from aerosplat.geometry.coordinates import enu_to_ned, intrinsics_from_hfov
from aerosplat.geometry.occupancy import OCCUPIED, OccupancyGrid
from aerosplat.geometry.wires import polyline_recall, rasterize_polyline, stamp_polylines
from aerosplat.models.gaussians import GaussianCloud, GaussianLayout
from aerosplat.models.temporal import velocity_from_positions as torch_velocity
from aerosplat.synth.batch import generate_batch
from aerosplat.synth.catenary import catenary_parameter, catenary_polyline
from aerosplat.synth.motion import velocities_from_positions
from aerosplat.synth.palette import segmentation_color
from aerosplat.synth.render import (
    HARD_WIRE_PIXELS,
    downsample_segmentation,
    project_dynamics,
    render_hard_frame,
    wire_pixel_thickness,
)
from aerosplat.synth.scenario import (
    ALTITUDE_MAX_M,
    ALTITUDE_MIN_M,
    _fit_altitude,
    broadcast_actor_velocity,
    load_scenario,
    materialize,
    plan_episode,
)
from aerosplat.synth.velocity import slot_velocity_from_maps
from aerosplat.synth.writer import estimate_shard_bytes

ROOT = Path(__file__).parents[1]
SCENARIO = ROOT / "configs" / "scenario" / "urban_corridor.yaml"


def _tiny_rig() -> dict:
    camera = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "yaw": 0.0,
        "width": 32,
        "height": 24,
    }
    down = {**camera, "pitch": -90.0, "width": 16, "height": 16}
    return {
        "name": "tiny",
        "vehicle_name": "Drone1",
        "fov_degrees": 90.0,
        "cameras": {"front": camera, "down": down},
    }


def _one_actor_yaml(path: Path) -> None:
    path.write_text(
        """
schema: aerosplat.scenario.v1
name: one_actor
simulator: colosseum
map: Blocks
autoload_map: false
seed: 7
rig: uam_quad
location:
  name: chicago
  latitude_deg: 41.8781
  longitude_deg: -87.6298
  altitude_m: 180
time:
  start: "2026-06-21T14:00:00Z"
  end: "2026-06-21T14:00:04Z"
weather:
  fog: 0.0
flight:
  frames: 4
  frame_rate_hz: 10
  speed_mps: 8
  yaw: toward_sun
  waypoints_enu:
    - [0.0, 0.0, 80.0]
    - [0.0, 40.0, 80.0]
actors:
  - name: escort
    kind: drone
    asset: Cube
    segmentation_id: 10
    speed_mps: 4
    scale_m: [1.0, 1.0, 1.0]
    waypoints_enu:
      - [0.0, 12.0, 80.0]
      - [0.0, 40.0, 80.0]
wires:
  segmentation_id: 250
  asset: Cube
  spans:
    - name: feeder
      kind: power_line
      start_enu: [-2.0, 8.0, 78.0]
      end_enu: [2.0, 8.0, 78.0]
      sag_m: 0.2
      diameter_m: 0.02
      samples: 8
glare:
  mode: auto
capture:
  supersample: 1
  pause_step: true
  record_wire_visibility: true
randomization:
  enabled: false
output:
  shard_frames: 100
"""
    )


def test_shipped_scenario_parses_and_plans():
    scenario = load_scenario(SCENARIO)
    assert scenario.simulator == "colosseum"
    assert scenario.map_name == "Blocks"
    assert scenario.flight.yaw == "toward_sun"
    assert {actor.kind for actor in scenario.actors} == {"drone", "bird", "vehicle"}
    plan = plan_episode(scenario)
    altitude = plan.frames[0].ego_position_enu[2]
    assert ALTITUDE_MIN_M <= altitude <= ALTITUDE_MAX_M
    for wire in plan.wires:
        assert 0.01 <= wire.diameter_m <= 0.05
        assert wire.points_enu.shape[1] == 3
    # Three moving actors cannot share one broadcast vector.
    velocities = [actor.velocity_enu[0] for actor in plan.actors]
    assert broadcast_actor_velocity(velocities) is None
    assert plan.frames[0].timestamp > 1e9


def test_parser_rejects_altitude_and_diameter(tmp_path: Path):
    scenario = yaml.safe_load(SCENARIO.read_text())
    low = tmp_path / "low.yaml"
    scenario["flight"]["waypoints_enu"] = [[0.0, 0.0, 20.0], [10.0, 0.0, 40.0]]
    low.write_text(yaml.safe_dump(scenario))
    with pytest.raises(ValueError, match="altitudes"):
        load_scenario(low)

    thick = tmp_path / "thick.yaml"
    scenario = yaml.safe_load(SCENARIO.read_text())
    scenario["wires"]["spans"][0]["diameter_m"] = 0.08
    thick.write_text(yaml.safe_dump(scenario))
    with pytest.raises(ValueError, match="diameter"):
        load_scenario(thick)

    thin = tmp_path / "thin.yaml"
    scenario["wires"]["spans"][0]["diameter_m"] = 0.005
    thin.write_text(yaml.safe_dump(scenario))
    with pytest.raises(ValueError, match="diameter"):
        load_scenario(thin)


def test_catenary_sag_and_endpoints():
    start = np.array([0.0, 0.0, 10.0])
    end = np.array([20.0, 0.0, 10.0])
    points = catenary_polyline(start, end, sag_m=2.0, samples=21)
    assert np.allclose(points[0], start)
    assert np.allclose(points[-1], end)
    assert points[10, 2] == pytest.approx(8.0, abs=1e-6)
    assert np.allclose(points[:, 2], points[::-1, 2])
    assert np.all(points[:, 2] <= 10.0 + 1e-8)
    span = catenary_parameter(20.0, 2.0)
    assert span * (math.cosh(10.0 / span) - 1.0) == pytest.approx(2.0, rel=1e-8)
    straight = catenary_polyline(start, end, sag_m=0.0, samples=5)
    assert np.allclose(straight[:, 2], 10.0)

    high = np.array([0.0, 0.0, 20.0])
    low = np.array([10.0, 0.0, 8.0])
    unequal = catenary_polyline(high, low, sag_m=1.5, samples=11)
    chord_mid = 0.5 * (high + low)
    assert np.allclose(unequal[0], high)
    assert np.allclose(unequal[-1], low)
    assert unequal[5, 2] == pytest.approx(chord_mid[2] - 1.5, abs=1e-6)


def test_velocity_matches_the_training_difference():
    dt = 0.1
    positions = np.stack([np.array([index * 4.0 * dt, 0.0, 80.0]) for index in range(5)])
    got = velocities_from_positions(positions, dt)
    assert np.allclose(got, np.array([4.0, 0.0, 0.0]))
    for index in range(len(positions) - 1):
        expected = torch_velocity(
            torch.tensor(positions[index]),
            torch.tensor(positions[index + 1]),
            dt,
        )
        assert np.allclose(got[index], expected.numpy())


def test_same_seed_is_identical_and_stays_in_the_altitude_band():
    scenario = load_scenario(SCENARIO)
    scenario.randomization.enabled = True
    first = materialize(scenario, 11, randomize=True)
    again = materialize(scenario, 11, randomize=True)
    other = materialize(scenario, 12, randomize=True)
    assert np.allclose(first.flight.waypoints_enu, again.flight.waypoints_enu)
    assert first.flight.speed_mps == again.flight.speed_mps
    changed = (
        not np.allclose(first.flight.waypoints_enu, other.flight.waypoints_enu)
        or first.flight.speed_mps != other.flight.speed_mps
        or first.weather.fog != other.weather.fog
    )
    assert changed
    for concrete in (first, other):
        altitude = concrete.flight.waypoints_enu[:, 2]
        assert altitude.min() >= ALTITUDE_MIN_M
        assert altitude.max() <= ALTITUDE_MAX_M
        plan_episode(concrete)


def test_altitude_profile_slides_into_the_band():
    tall = np.array([[0.0, 0.0, 20.0], [1.0, 0.0, 40.0]])
    fitted = _fit_altitude(tall)
    assert fitted[:, 2].min() >= ALTITUDE_MIN_M
    assert np.allclose(fitted[:, 2], [30.0, 50.0])
    too_tall = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 280.0]])
    with pytest.raises(ValueError, match="profile"):
        _fit_altitude(too_tall)


def test_toward_sun_points_the_nose_at_the_horizontal_sun():
    scenario = load_scenario(SCENARIO)
    plan = plan_episode(materialize(scenario, 0))
    sun = plan.frames[0].sun_world_enu
    horizontal = np.array([sun[0], sun[1], 0.0])
    horizontal = horizontal / np.linalg.norm(horizontal)
    rotation = quat_wxyz_to_rotmat(plan.frames[0].ego_quaternion_wxyz)
    body_x_ned = rotation[:, 0]
    assert np.allclose(body_x_ned, enu_to_ned(horizontal), atol=1e-5)


def test_wire_voxel_recall_on_the_two_centimeter_grid():
    samples = np.linspace(0.0, 1.6, 40)
    points = np.stack([samples, np.zeros_like(samples), np.full_like(samples, 0.4)], axis=1)
    origin = np.array([-0.05, -0.05, -0.05])
    voxel = 0.02
    dims = (90, 20, 40)
    mask = rasterize_polyline(points, origin, voxel, dims)
    assert polyline_recall(points, mask, origin, voxel) >= 0.99

    center = points.mean(axis=0)
    grid = OccupancyGrid(
        state=np.zeros((16, 16, 16), dtype=np.uint8),
        hit_count=np.zeros((16, 16, 16), dtype=np.uint8),
        origin=center - 4.0,
        voxel_size=0.5,
    )
    stamp_polylines(
        grid,
        [{"points_enu": points}],
        center=center,
        fine_voxel_size=0.02,
        fine_extent_m=4.0,
    )
    assert grid.wire is not None and int(grid.wire.sum()) > 0
    assert np.all(grid.state[grid.wire > 0] == OCCUPIED)
    assert grid.wire_fine is not None
    assert (
        polyline_recall(points, grid.wire_fine, grid.wire_fine_origin, grid.wire_fine_voxel_size)
        >= 0.99
    )


def test_subpixel_wire_is_recorded_and_supersampling_keeps_it():
    intrinsics = intrinsics_from_hfov(32, 32, 90.0)
    assert float(intrinsics[0, 0]) == pytest.approx(16.0)
    thickness = wire_pixel_thickness(0.05, 10.0, float(intrinsics[0, 0]))
    assert thickness == pytest.approx(0.08)
    assert thickness < HARD_WIRE_PIXELS
    wire = {
        "points_enu": np.array([[-1.0, 0.0, 10.0], [1.0, 0.0, 10.0]]),
        "diameter_m": 0.05,
        "segmentation_id": 250,
    }
    color = segmentation_color(250)
    eye = np.eye(3)
    zero = np.zeros(3)
    _, _, hard = render_hard_frame(
        [],
        [wire],
        intrinsics=intrinsics,
        rotation_w2c=eye,
        translation_w2c=zero,
        height=32,
        width=32,
        sun_world=np.array([0.0, 0.0, 1.0]),
        wire_color=color,
    )
    assert not np.any(np.all(hard == np.asarray(color), axis=-1))
    _, _, coverage = project_dynamics(
        [],
        [wire],
        intrinsics=intrinsics,
        rotation_w2c=eye,
        translation_w2c=zero,
        height=32,
        width=32,
    )
    assert float(coverage.max()) > 0.0
    assert float(coverage.max()) < 1.0

    fine_k = intrinsics_from_hfov(256, 256, 90.0)
    assert wire_pixel_thickness(0.05, 10.0, float(fine_k[0, 0])) == pytest.approx(0.64)
    _, _, fine = render_hard_frame(
        [],
        [wire],
        intrinsics=fine_k,
        rotation_w2c=eye,
        translation_w2c=zero,
        height=256,
        width=256,
        sun_world=np.array([0.0, 0.0, 1.0]),
        wire_color=color,
    )
    assert np.any(np.all(fine == np.asarray(color), axis=-1))
    reduced = downsample_segmentation(fine, 8, wire_color=color)
    assert np.any(np.all(reduced == np.asarray(color), axis=-1))


def test_mock_shard_round_trips_through_the_dataset_loader(tmp_path: Path):
    scenario_path = tmp_path / "one_actor.yaml"
    _one_actor_yaml(scenario_path)
    output = tmp_path / "out"
    manifest = generate_batch(
        scenario_path,
        output,
        episodes=1,
        frames=4,
        mock=True,
        rig=_tiny_rig(),
    )
    assert manifest["format"] == "aerosplat.synthetic.v1"
    shard = output / "shard_00000_00"
    assert (shard / "DONE").is_file()
    assert (shard / "dataset.json").is_file()
    estimate = json.loads((output / "disk_estimate.json").read_text())
    assert estimate["bytes_total_upper"] > 0
    throughput = (output / "throughput.jsonl").read_text().strip().splitlines()
    assert len(throughput) == 1
    assert json.loads(throughput[0])["camera_frames_per_sec"] > 0

    scenes = read_canonical(shard)
    front = scenes["ep000000"]["front"][0]
    assert front.timestamp > 1e9
    assert front.depth_path is not None and front.depth_path.is_file()
    assert front.segmentation_path is not None and front.segmentation_path.is_file()
    assert front.clean_image_path is not None and front.clean_image_path.is_file()
    assert front.K.shape == (3, 3)
    assert front.R.shape == (3, 3)
    assert abs(np.linalg.det(front.R) - 1.0) < 1e-5

    cfg = load_config(
        [
            "data=synthetic",
            f"data.root={shard}",
            "data.load_segmentation=true",
        ]
    )
    dataset = build_dataset(cfg)
    sample = dataset[0]
    assert sample.latitude_deg == pytest.approx(41.8781)
    assert sample.longitude_deg == pytest.approx(-87.6298)
    assert sample.gt_velocity is not None
    assert torch.allclose(sample.gt_velocity, torch.tensor([0.0, 4.0, 0.0]), atol=1e-4)
    frame = sample.cameras["front"]
    assert frame.velocity is not None and tuple(frame.velocity.shape) == (3, 24, 32)
    assert frame.instance is not None and tuple(frame.instance.shape) == (24, 32)
    assert frame.clean_image is not None
    assert frame.segmentation is not None
    assert frame.depth is not None
    sun = json.loads((shard / "sun.json").read_text())
    sun_camera = np.asarray(sun["frames"][0]["cameras"]["front"]["sun_camera"])
    assert sun_camera.shape == (3,)
    assert sun_camera[2] > 0.05
    assert not torch.equal(frame.image, frame.clean_image)
    wires = json.loads((shard / "ep000000" / "wires.json").read_text())
    assert wires["format"] == "aerosplat.wires.v1"
    assert len(wires["spans"][0]["points_enu"]) >= 2


def test_resume_does_not_rewrite_a_finished_shard(tmp_path: Path):
    scenario_path = tmp_path / "one_actor.yaml"
    _one_actor_yaml(scenario_path)
    output = tmp_path / "out"
    rig = _tiny_rig()
    generate_batch(scenario_path, output, episodes=1, frames=4, mock=True, rig=rig)
    image = output / "shard_00000_00" / "ep000000" / "front" / "images" / "000000.png"
    original = image.read_bytes()
    image.write_bytes(original + b"\x00")
    generate_batch(scenario_path, output, episodes=1, frames=4, mock=True, rig=rig)
    assert image.read_bytes() == original + b"\x00"
    assert len((output / "throughput.jsonl").read_text().strip().splitlines()) == 1


def test_slot_velocity_samples_the_map_at_the_gaussian_pixel():
    batch, count = 1, 1
    cloud = GaussianCloud.from_activated(
        xyz=torch.zeros(batch, count, 3),
        scale=torch.ones(batch, count, 3) * 0.1,
        quat=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        opacity=torch.ones(batch, count, 1) * 0.5,
        sh=torch.zeros(batch, count, GaussianLayout.SH_COEFFS, 3),
        velocity=torch.zeros(batch, count, 3),
        valid=torch.ones(batch, count, dtype=torch.bool),
        wire_prob=torch.zeros(batch, count),
        slot_id=torch.zeros(count, dtype=torch.long),
        camera_index=torch.zeros(count, dtype=torch.long),
        row=torch.zeros(count, dtype=torch.long),
        col=torch.zeros(count, dtype=torch.long),
        subslot=torch.zeros(count, dtype=torch.long),
        uv=torch.tensor([[3.2, 4.6]]),
        camera_order=("front",),
        anchor_valid=torch.ones(batch, count, dtype=torch.bool),
    )
    maps = {"front": torch.zeros(1, 3, 8, 8)}
    maps["front"][0, :, 5, 3] = torch.tensor([1.0, 2.0, 3.0])
    sampled = slot_velocity_from_maps(cloud, maps)
    assert sampled is not None
    assert torch.allclose(sampled[0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_disk_estimate_is_positive():
    size = estimate_shard_bytes(_tiny_rig(), frames=4, supersample=2)
    assert size > 0
