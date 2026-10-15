"""Colosseum raw logs, the capture writer, and TartanAir."""

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from aerosplat.capture.colosseum import build_settings, capture_episode
from aerosplat.data.adapters.airsim import read_airsim_raw
from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.adapters.tartanair import read_tartanair
from aerosplat.data.dataset import MultiCameraDataset
from aerosplat.data.records import SyncConfig
from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.coordinates import airsim_mount_to_w2c


def _tiny_rig():
    return {
        "vehicle_name": "Drone1",
        "fov_degrees": 90.0,
        "cameras": {
            "front": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "width": 10,
                "height": 8,
            },
            "down": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "pitch": -90.0,
                "roll": 0.0,
                "yaw": 0.0,
                "width": 10,
                "height": 8,
            },
        },
    }


def test_settings_request_scene_and_depth_planar():
    settings = build_settings(_tiny_rig())
    front = settings["Vehicles"]["Drone1"]["Cameras"]["front"]
    types = [entry["ImageType"] for entry in front["CaptureSettings"]]
    assert types == [0, 1]
    assert front["Yaw"] == 0.0
    down = settings["Vehicles"]["Drone1"]["Cameras"]["down"]
    assert down["Pitch"] == -90.0


def test_capture_episode_roundtrips_through_the_canonical_loader(tmp_path: Path):
    rig = _tiny_rig()

    class Scripted:
        """Pose is sampled once per frame; the index advances after the last grab."""

        def __init__(self):
            self.frame = 0
            self.grabs = 0

        def vehicle_pose(self):
            ned = np.array([0.0, float(self.frame), -3.0])
            return ned, np.array([1.0, 0.0, 0.0, 0.0])

        def timestamp(self) -> float:
            return 0.1 * self.frame

        def grab(self, camera: str):
            del camera
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            rgb[..., 1] = 40 + self.frame
            depth = np.full((8, 10), 6.0, dtype=np.float32)
            self.grabs += 1
            if self.grabs == len(rig["cameras"]):
                self.grabs = 0
                self.frame += 1
            return rgb, depth

    capture_episode(Scripted(), rig, tmp_path / "rec", scene_id="flight", num_frames=2)
    scenes = read_canonical(tmp_path / "rec")
    dataset = MultiCameraDataset(
        scenes,
        SyncConfig(
            policy="nearest",
            reference_camera="front",
            max_image_dt=0.0,
            camera_order=("front", "down"),
        ),
    )
    assert len(dataset) == 2
    first = dataset[0].cameras["front"]
    expected_R, expected_t = airsim_mount_to_w2c(
        np.array([0.0, 0.0, -3.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        cam_position_body=np.zeros(3),
        pitch_deg=0.0,
        roll_deg=0.0,
        yaw_deg=0.0,
    )
    assert np.allclose(first.R.numpy(), expected_R, atol=1e-5)
    assert np.allclose(first.t.numpy(), expected_t, atol=1e-5)
    assert np.allclose(camera_center(first.R.numpy(), first.t.numpy()), [0.0, 0.0, 3.0], atol=1e-5)
    second = dataset[1].cameras["front"]
    assert second.timestamp == pytest.approx(0.1)
    # Second pose is 1 m east in NED (+Y) -> ENU +X.
    assert np.allclose(
        camera_center(second.R.numpy(), second.t.numpy()), [1.0, 0.0, 3.0], atol=1e-5
    )
    down = dataset[0].cameras["down"]
    forward = down.R.numpy().T @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward, [0.0, 0.0, -1.0], atol=1e-5)


def test_airsim_rec_parser(tmp_path: Path):
    rig = _tiny_rig()
    settings = build_settings(rig)
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    # Two cameras, semicolon-separated, timestamps in seconds via the unit flag.
    lines = [
        "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile",
        "Drone1\t0\t0\t0\t-3\t1\t0\t0\t0\timg/front_0.png;img/down_0.png",
        "Drone1\t0.2\t0\t1\t-3\t1\t0\t0\t0\timg/front_1.png;img/down_1.png",
    ]
    (tmp_path / "airsim_rec.txt").write_text("\n".join(lines) + "\n")
    for name in ("front_0", "front_1", "down_0", "down_1"):
        path = tmp_path / "img" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8), mode="RGB").save(path)
        np.save(tmp_path / "img" / f"{name}_depth.npy", np.ones((8, 10), dtype=np.float32))
    streams = read_airsim_raw(
        tmp_path / "airsim_rec.txt",
        tmp_path / "settings.json",
        timestamp_unit="seconds",
    )
    assert list(streams) == ["front", "down"]
    assert streams["front"][1].timestamp == pytest.approx(0.2)
    assert streams["front"][1].depth_path is not None
    center = camera_center(streams["front"][1].R, streams["front"][1].t)
    assert np.allclose(center, [1.0, 0.0, 3.0], atol=1e-5)


def test_tartanair_stereo_and_baseline_fallback(tmp_path: Path):
    traj = tmp_path / "env" / "Easy" / "P000"
    _write_traj(traj, with_right_pose=True)
    cfg = _tartan_cfg()
    scenes = read_tartanair(tmp_path, cfg)
    assert list(scenes) == ["env/Easy/P000"]
    left = scenes["env/Easy/P000"]["left"]
    right = scenes["env/Easy/P000"]["right"]
    assert len(left) == 2
    assert left[1].timestamp == pytest.approx(0.1)
    assert np.allclose(camera_center(left[0].R, left[0].t), [0.0, 0.0, 3.0], atol=1e-5)
    assert np.allclose(camera_center(right[0].R, right[0].t), [0.25, 0.0, 3.0], atol=1e-5)
    # Image is 16x8 rather than 640x480, so K scales.
    assert left[0].K[0, 0] == pytest.approx(320.0 * (16 / 640))

    bare = tmp_path / "bare"
    _write_traj(bare, with_right_pose=False)
    derived = read_tartanair(bare, cfg)
    derived_right = derived["bare"]["right"][0]
    assert np.allclose(camera_center(derived_right.R, derived_right.t), [0.25, 0.0, 3.0], atol=1e-5)

    dataset = MultiCameraDataset(
        scenes,
        SyncConfig(policy="nearest", reference_camera="left", max_image_dt=1e-6),
    )
    sample = dataset[0]
    assert sample.camera_names == ["left", "right"]
    assert sample.cameras["left"].image.shape[0] == 3


def _tartan_cfg():
    return OmegaConf.create(
        {
            "intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 240.0,
            },
            "fps": 10.0,
            "stereo_baseline_m": 0.25,
            "depth_kind": "planar_z",
        }
    )


def _write_traj(traj: Path, *, with_right_pose: bool) -> None:
    (traj / "image_left").mkdir(parents=True)
    (traj / "image_right").mkdir()
    (traj / "depth_left").mkdir()
    (traj / "depth_right").mkdir()
    for index in (0, 1):
        rgb = np.zeros((8, 16, 3), dtype=np.uint8)
        rgb[..., 0] = 20 * (index + 1)
        Image.fromarray(rgb, mode="RGB").save(traj / "image_left" / f"{index:06d}_left.png")
        Image.fromarray(rgb, mode="RGB").save(traj / "image_right" / f"{index:06d}_right.png")
        np.save(
            traj / "depth_left" / f"{index:06d}_left_depth.npy", np.full((8, 16), 5.0, np.float32)
        )
        np.save(
            traj / "depth_right" / f"{index:06d}_right_depth.npy", np.full((8, 16), 5.0, np.float32)
        )
    # Identity quaternion xyzw, NED position (0, 0, -3) is 3 m up.
    left_lines = ["0 0 -3 0 0 0 1", "1 0 -3 0 0 0 1"]
    (traj / "pose_left.txt").write_text("\n".join(left_lines) + "\n")
    if with_right_pose:
        right_lines = ["0 0.25 -3 0 0 0 1", "1 0.25 -3 0 0 0 1"]
        (traj / "pose_right.txt").write_text("\n".join(right_lines) + "\n")
