"""API tests. Inference uses the random-init tiny model; no checkpoint and no GPU."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from aerosplat.models.gaussians import GaussianLayout
from aerosplat.schema import CameraFrame, MultiCameraSample
from server.app import create_app
from server.bake import bake_sample
from server.codec import pack_gaussians, split_infer_blob, unpack_gaussians, unpack_occupancy
from server.config import Settings
from server.demo import build_demo


def _png(width: int = 8, height: int = 8, color: tuple[int, int, int] = (40, 80, 120)) -> str:
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _camera(name: str, translation: list[float]) -> dict:
    return {
        "name": name,
        "timestamp": 0.0,
        "K": [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "t": translation,
        "image_png_base64": _png(),
    }


def _client(tmp_path, episodes: list | None = None) -> TestClient:
    settings = Settings(
        episode_dirs=episodes or [],
        model_name="vit_tiny",
        device="cpu",
        checkpoint=None,
        grid_size=(8, 8, 8),
        voxel_size=0.5,
        occ_origin=(-2.0, -2.0, -1.0),
    )
    return TestClient(create_app(settings))


def test_gaussian_stream_roundtrip():
    packed = np.zeros((1, 2, 62), dtype=np.float32)
    packed[0, 0, 0:3] = (1.5, -2.0, 0.25)
    packed[0, 0, 3:6] = 0.1
    packed[0, 0, 6] = 1.0
    packed[0, 0, 10] = 0.8
    packed[0, 1, 59:62] = (0.0, 1.0, 0.0)
    valid = np.array([[True, False]])
    wire = np.array([[0.2, 0.9]], dtype=np.float32)
    sun = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    blob = pack_gaussians(packed, valid, wire, timestamp=1.25, dt=0.1, source_sun=sun)
    decoded = unpack_gaussians(blob)
    assert decoded["batch"] == 1 and decoded["count"] == 2
    assert decoded["timestamp"] == np.float32(1.25)
    assert np.allclose(decoded["source_sun"], sun)
    assert np.allclose(decoded["packed"][0, 0, 0:3], packed[0, 0, 0:3], atol=1e-3)
    assert not bool(decoded["valid"][0, 1])
    assert abs(float(decoded["wire_prob"][0, 1]) - 0.9) < 1e-3


def test_schema_matches_the_core_layout(tmp_path):
    client = _client(tmp_path)
    body = client.get("/api/v1/schema").json()
    assert body["channels"] == list(GaussianLayout.channel_names())
    assert len(body["channels"]) == 62
    assert body["slices"]["sh"] == [11, 59]
    assert body["slices"]["velocity"] == [59, 62]


def test_health_reports_cpu_before_any_inference(tmp_path):
    client = _client(tmp_path)
    body = client.get("/api/v1/health").json()
    assert body["gpu"]["available"] is False
    assert body["gpu"]["name"] == "cpu"
    assert body["gpu"]["vram_total_bytes"] is None
    assert body["model"]["loaded"] is False
    assert body["throughput_fps"] is None
    assert body["inferences"] == 0


def test_baked_episode_frame_cloud_and_collision(tmp_path):
    root = build_demo(tmp_path / "demo")
    client = _client(tmp_path, episodes=[root])
    listed = client.get("/api/v1/episodes").json()
    assert listed[0]["id"] == "demo"
    assert listed[0]["samples"] >= 2
    frame = client.get("/api/v1/episodes/demo/frames/0").json()
    assert frame["dt"] > 0
    assert {camera["name"] for camera in frame["cameras"]} == {"front", "down", "port", "starboard"}
    image = client.get(frame["cameras"][0]["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    glare = client.get("/api/v1/episodes/demo/frames/0/cameras/front/glare")
    assert glare.status_code == 200
    cloud = client.get("/api/v1/episodes/demo/clouds/0")
    decoded = unpack_gaussians(cloud.content)
    assert decoded["count"] > 100
    assert int(decoded["valid"].sum()) == decoded["count"]
    velocity = decoded["packed"][0, :, 59:62]
    assert float(np.linalg.norm(velocity, axis=1).max()) > 1.0
    occupancy = client.get(
        "/api/v1/occupancy",
        params={"episode": "demo", "index": 0, "radius": 0.6},
    ).json()
    assert occupancy["collision"]["wire_risk"] > 0.5
    assert occupancy["collision"]["risk"] > 0.5
    assert occupancy["collision"]["nearest_distance_m"] < 0.6
    binary = client.get(
        "/api/v1/occupancy",
        params={"episode": "demo", "index": 0, "format": "binary"},
    )
    parsed = unpack_occupancy(binary.content)
    assert parsed["collision"]["wire_risk"] > 0.5
    assert len(parsed["cells"]) > 0


def test_empty_depth_still_bakes_the_cable():
    image = torch.zeros(3, 8, 8)
    depth = torch.zeros(8, 8)
    rotation = torch.eye(3)
    frame = CameraFrame(
        name="front",
        image=image,
        K=[[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
        R=rotation,
        t=[0.0, 0.0, 0.0],
        timestamp=0.0,
        depth=depth,
    )
    baked = bake_sample(
        MultiCameraSample(scene_id="s", sample_index=0, timestamp=0.0, cameras={"front": frame})
    )
    assert float(baked.wire_prob.max()) == 1.0
    assert baked.glare["front"].shape[-1] == 3


def test_infer_json_binary_and_multipart(tmp_path):
    client = _client(tmp_path)
    payload = {
        "format": "json",
        "sun_world": [0.2, 0.1, 0.97],
        "cameras": [
            _camera("front", [0.0, 0.0, 0.0]),
            _camera("starboard", [0.0, -0.2, 0.0]),
        ],
    }
    response = client.post("/api/v1/infer", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["channels"] == 62
    assert body["count"] > 0
    assert body["model"]["init"] == "random"
    assert body["sun_source"] == "request"
    assert set(body["glare"]) == {"front", "starboard"}
    assert body["latency_ms"]["lifter_ms"] >= 0
    assert body["latency_ms"]["total_ms"] > 0
    decoded = unpack_gaussians(base64.b64decode(body["gaussian_b64"]))
    assert decoded["count"] == body["count"]
    assert decoded["packed"].shape[-1] == 62
    health = client.get("/api/v1/health").json()
    assert health["model"]["loaded"] is True
    assert health["inferences"] == 1
    assert health["throughput_fps"] > 0
    assert health["latency_ms"]["gaussians_ms"] >= 0

    binary = client.post("/api/v1/infer?format=binary", json={**payload, "format": "binary"})
    assert binary.status_code == 200
    meta_raw, stream = split_infer_blob(binary.content)
    meta = json.loads(meta_raw)
    assert "gaussian_b64" not in meta
    assert unpack_gaussians(stream)["count"] == meta["count"]

    manifest = {
        "cameras": [
            {
                key: value
                for key, value in payload["cameras"][0].items()
                if key != "image_png_base64"
            },
            {
                key: value
                for key, value in payload["cameras"][1].items()
                if key != "image_png_base64"
            },
        ],
        "sun_world": [0.0, 0.0, 1.0],
    }
    files = {
        "manifest": (None, json.dumps(manifest), "application/json"),
        "front": (
            "front.png",
            base64.b64decode(payload["cameras"][0]["image_png_base64"]),
            "image/png",
        ),
        "starboard": (
            "starboard.png",
            base64.b64decode(payload["cameras"][1]["image_png_base64"]),
            "image/png",
        ),
    }
    multipart = client.post("/api/v1/infer", files=files)
    assert multipart.status_code == 200, multipart.text
    assert multipart.json()["count"] == body["count"]

    latest = client.get("/api/v1/occupancy").json()
    assert latest["dims"] == [8, 8, 8]
    missing = client.get("/api/v1/episodes/nope/frames/0")
    assert missing.status_code == 404
