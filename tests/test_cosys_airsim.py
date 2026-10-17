"""Fixture tests for the shared Colosseum and Cosys-AirSim backend.

No simulator and no CUDA. Clients are in-process fakes. Raw logs are tiny files.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aerosplat.airsim_family import ANNOTATION, LIGHTING, get_profile
from aerosplat.capture import airsim as family
from aerosplat.capture import colosseum
from aerosplat.capture.airsim import (
    build_settings,
    capture_episode,
    decode_planar_depth,
    decode_scene_rgb,
    decode_segmentation,
    images_from_responses,
    main,
)
from aerosplat.data.adapters.airsim import parse_settings, read_airsim_raw
from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.errors import DataError
from aerosplat.geometry.camera import camera_center


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


class _Scripted:
    def __init__(self, rig):
        self._n = len(rig["cameras"])
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
        rgb[0, 0] = (255, 0, 0)
        rgb[..., 1] = 40 + self.frame
        depth = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
        self.grabs += 1
        if self.grabs == self._n:
            self.grabs = 0
            self.frame += 1
        return rgb, depth

    def grab_segmentation(self, camera: str):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[0, 1] = (0, 255, 0) if camera == "front" else (0, 0, 255)
        return image


def test_colosseum_entry_is_the_shared_backend():
    assert colosseum.build_settings is family.build_settings
    assert colosseum.capture_episode is family.capture_episode
    assert colosseum.main is family.main
    assert get_profile("colosseum").archived is True
    assert get_profile("cosys").archived is False
    assert get_profile("cosys").python_modules == ("cosysairsim",)
    assert get_profile("colosseum").python_modules == ("airsim", "colosseum")
    with pytest.raises(DataError, match="unknown AirSim-family"):
        get_profile("blocks")


def test_both_simulators_roundtrip_the_same_body_mount(tmp_path: Path):
    rig = _tiny_rig()
    centers = {}
    for simulator in ("colosseum", "cosys"):
        root = tmp_path / simulator
        capture_episode(
            _Scripted(rig),
            rig,
            root,
            scene_id="flight",
            num_frames=2,
            simulator=simulator,
        )
        scenes = read_canonical(root)
        front = scenes["flight"]["front"][1]
        centers[simulator] = camera_center(front.R, front.t)
        image = np.asarray(Image.open(root / "flight" / "front" / "images" / "000000.png"))
        # The RPC flip is not applied twice. grab() already returns display order.
        assert image[0, 0, 0] == 255
        settings = json.loads((root / "settings.json").read_text())
        types = [
            entry["ImageType"]
            for entry in settings["Vehicles"]["Drone1"]["Cameras"]["front"]["CaptureSettings"]
        ]
        assert LIGHTING not in types
        assert ANNOTATION not in types
        assert types == [0, 1]
    assert np.allclose(centers["colosseum"], centers["cosys"], atol=1e-5)
    assert np.allclose(centers["cosys"], [1.0, 0.0, 3.0], atol=1e-5)


def test_scene_rgb_flip_is_cosys_only_and_depth_is_not():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    raw = rgb.tobytes()
    colosseum_rgb = decode_scene_rgb(raw, 4, 6, "colosseum")
    cosys_rgb = decode_scene_rgb(raw, 4, 6, "cosys")
    assert colosseum_rgb[0, 0, 0] == 255
    assert cosys_rgb[-1, 0, 0] == 255
    assert cosys_rgb[0, 0, 0] == 0

    depth = np.arange(24, dtype=np.float32)
    for simulator in ("colosseum", "cosys"):
        decoded = decode_planar_depth(depth, 4, 6, simulator)
        assert decoded[0, 0] == 0
        assert decoded[-1, -1] == 23

    seg = np.zeros((4, 6, 3), dtype=np.uint8)
    seg[0, 2] = (7, 8, 9)
    for simulator in ("colosseum", "cosys"):
        decoded_seg = decode_segmentation(seg.tobytes(), 4, 6, simulator)
        assert tuple(decoded_seg[0, 2]) == (7, 8, 9)


def test_rpc_response_helper_applies_the_cosys_rgb_flip():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[0, 1] = (3, 0, 0)
    depth = np.full((4, 6), 2.5, dtype=np.float32)

    class Response:
        def __init__(self, height, width, rgb_bytes=None, depth_values=None):
            self.height = height
            self.width = width
            self.image_data_uint8 = rgb_bytes if rgb_bytes is not None else b""
            self.image_data_float = [] if depth_values is None else depth_values

    rgb_response = Response(4, 6, rgb.tobytes())
    depth_response = Response(4, 6, depth_values=depth.reshape(-1))
    upright, planar = images_from_responses(rgb_response, depth_response, "cosys")
    assert upright[-1, 1, 0] == 3
    assert planar[0, 0] == pytest.approx(2.5)
    stored, _ = images_from_responses(rgb_response, depth_response, "colosseum")
    assert stored[0, 1, 0] == 3


def test_segmentation_sidecar_records_the_id_space(tmp_path: Path):
    rig = _tiny_rig()
    expected = {"colosseum": "mesh_object", "cosys": "instance"}
    for simulator, id_space in expected.items():
        root = tmp_path / simulator
        capture_episode(
            _Scripted(rig),
            rig,
            root,
            num_frames=1,
            simulator=simulator,
            record_segmentation=True,
        )
        manifest = json.loads((root / "segmentation.json").read_text())
        assert manifest["simulator"] == simulator
        assert manifest["image_type"] == 5
        assert manifest["id_space"] == id_space
        assert manifest["vertical_flip_applied"] is False
        image = np.asarray(Image.open(root / manifest["cameras"]["front"][0]))
        assert tuple(image[0, 1]) == (0, 255, 0)
        settings = json.loads((root / "settings.json").read_text())
        types = [
            entry["ImageType"]
            for entry in settings["Vehicles"]["Drone1"]["Cameras"]["down"]["CaptureSettings"]
        ]
        assert types == [0, 1, 5]
        record_path = root / "flight" / "front" / "frames.jsonl"
        record = json.loads(record_path.read_text().splitlines()[0])
        assert "segmentation" not in record


def test_cosys_log_keeps_pose_when_extra_columns_are_present(tmp_path: Path):
    rig = _tiny_rig()
    settings = build_settings(rig, "cosys")
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    header = (
        "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\t"
        "ImageFile\tLatitude\tLongitude\tAltitude\tPressure\tAccX\tAccY\tAccZ"
    )
    row = (
        "Drone1\t0\t0\t1\t-3\t1\t0\t0\t0\timg/front.png;img/down.png\t1.2\t3.4\t5\t1013\t0\t0\t9.8"
    )
    (tmp_path / "airsim_rec.txt").write_text(header + "\n" + row + "\n")
    for name in ("front", "down"):
        path = tmp_path / "img" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8), mode="RGB").save(path)
    streams = read_airsim_raw(
        tmp_path / "airsim_rec.txt",
        tmp_path / "settings.json",
        timestamp_unit="seconds",
        simulator="cosys",
    )
    assert streams["front"][0].depth_kind == "planar_z"
    assert streams["front"][0].timestamp == pytest.approx(0.0)
    center = camera_center(streams["front"][0].R, streams["front"][0].t)
    assert np.allclose(center, [1.0, 0.0, 3.0], atol=1e-5)
    mounts = parse_settings(tmp_path / "settings.json")
    assert mounts["front"]["simulator"] == "cosys"
    assert mounts["front"]["segmentation_id_space"] == "instance"


def test_external_cameras_and_the_wrong_settings_version_are_rejected(tmp_path: Path):
    rig = _tiny_rig()
    settings = build_settings(rig, "cosys")
    settings["Vehicles"]["Drone1"]["Cameras"]["front"]["External"] = True
    settings["Vehicles"]["Drone1"]["Cameras"]["front"]["ExternalLocal"] = False
    settings["MoveWorldOrigin"] = True
    path = tmp_path / "external.json"
    path.write_text(json.dumps(settings))
    with pytest.raises(DataError, match="Unreal NED"):
        parse_settings(path, simulator="cosys")

    settings["Vehicles"]["Drone1"]["Cameras"]["front"]["ExternalLocal"] = True
    path.write_text(json.dumps(settings))
    with pytest.raises(DataError, match="local NED"):
        parse_settings(path, simulator="cosys")

    colosseum_settings = build_settings(rig, "colosseum")
    colosseum_settings["Vehicles"]["Drone1"]["Cameras"]["down"]["External"] = True
    colosseum_path = tmp_path / "colosseum_external.json"
    colosseum_path.write_text(json.dumps(colosseum_settings))
    with pytest.raises(DataError, match="External true"):
        parse_settings(colosseum_path, simulator="colosseum")

    with pytest.raises(DataError, match="SettingsVersion"):
        parse_settings(path, simulator="colosseum")
    plain = tmp_path / "colosseum.json"
    plain.write_text(json.dumps(build_settings(rig, "colosseum")))
    with pytest.raises(DataError, match="SettingsVersion"):
        parse_settings(plain, simulator="cosys")


def test_depth_kind_follows_the_captured_image_type(tmp_path: Path):
    rig = _tiny_rig()
    settings = build_settings(rig, "cosys")
    _blank_recording(tmp_path)

    def write(mutate) -> Path:
        data = json.loads(json.dumps(settings))
        mutate(data)
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(data))
        return path

    def only(image_type: int):
        def mutate(data):
            for camera in data["Vehicles"]["Drone1"]["Cameras"].values():
                camera["CaptureSettings"] = [
                    entry for entry in camera["CaptureSettings"] if entry["ImageType"] == 0
                ]
                camera["CaptureSettings"].append(
                    {"ImageType": image_type, "Width": 10, "Height": 8, "FOV_Degrees": 90}
                )

        return mutate

    perspective = write(only(2))
    streams = read_airsim_raw(
        tmp_path / "airsim_rec.txt", perspective, timestamp_unit="seconds", simulator="cosys"
    )
    assert streams["front"][0].depth_kind == "euclidean"

    both = write(
        lambda data: [
            camera["CaptureSettings"].append(
                {"ImageType": 2, "Width": 10, "Height": 8, "FOV_Degrees": 90}
            )
            for camera in data["Vehicles"]["Drone1"]["Cameras"].values()
        ]
    )
    streams = read_airsim_raw(
        tmp_path / "airsim_rec.txt", both, timestamp_unit="seconds", simulator="cosys"
    )
    assert streams["front"][0].depth_kind == "planar_z"

    vis = write(only(3))
    with pytest.raises(DataError, match="DepthVis"):
        read_airsim_raw(
            tmp_path / "airsim_rec.txt", vis, timestamp_unit="seconds", simulator="cosys"
        )

    disparity = write(only(4))
    with pytest.raises(DataError, match="DisparityNormalized"):
        read_airsim_raw(
            tmp_path / "airsim_rec.txt", disparity, timestamp_unit="seconds", simulator="cosys"
        )

    planar = write(only(1))
    with pytest.raises(DataError, match="euclidean"):
        read_airsim_raw(
            tmp_path / "airsim_rec.txt",
            planar,
            timestamp_unit="seconds",
            simulator="cosys",
            depth_kind="euclidean",
        )


def test_dry_run_selects_the_simulator_settings(tmp_path: Path):
    main(["--dry-run", "--simulator", "cosys", "--output", str(tmp_path / "cosys")])
    cosys = json.loads((tmp_path / "cosys" / "settings.json").read_text())
    assert cosys["SettingsVersion"] == 2.0
    assert cosys["InitialInstanceSegmentation"] is True
    main(
        [
            "--dry-run",
            "--simulator",
            "colosseum",
            "--segmentation",
            "--output",
            str(tmp_path / "colosseum"),
        ]
    )
    colosseum_settings = json.loads((tmp_path / "colosseum" / "settings.json").read_text())
    assert colosseum_settings["SettingsVersion"] == 1.2
    front = colosseum_settings["Vehicles"]["Drone1"]["Cameras"]["front"]
    assert [entry["ImageType"] for entry in front["CaptureSettings"]] == [0, 1, 5]
    assert "External" not in front


def _blank_recording(root: Path) -> None:
    lines = [
        "VehicleName\tTimeStamp\tPOS_X\tPOS_Y\tPOS_Z\tQ_W\tQ_X\tQ_Y\tQ_Z\tImageFile",
        "Drone1\t0\t0\t0\t-3\t1\t0\t0\t0\timg/front.png;img/down.png",
    ]
    (root / "airsim_rec.txt").write_text("\n".join(lines) + "\n")
    for name in ("front", "down"):
        path = root / "img" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8), mode="RGB").save(path)
