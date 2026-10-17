"""Render-benchmark report, without a simulator."""

import json
from pathlib import Path

from aerosplat.capture.benchmark import (
    RESOLUTIONS,
    build_benchmark_settings,
    measure_resolution,
    query_nvidia_smi,
    render_markdown,
    write_report,
)
from aerosplat.config import load_config


class _Response:
    def __init__(self, height, width, image_type, time_stamp):
        self.height = height
        self.width = width
        self.image_type = image_type
        self.time_stamp = time_stamp


class FakeSim:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.calls: list[tuple] = []
        self.paused = False

    def pause(self, paused: bool) -> None:
        self.paused = paused
        self.calls.append(("pause", paused))

    def step(self, dt: float) -> None:
        assert self.paused
        self.calls.append(("step", dt))

    def grab(self, requests):
        assert self.paused
        self.calls.append(("grab", tuple(requests)))
        return [
            _Response(self.height, self.width, image_type, time_stamp=7)
            for _camera, image_type in requests
        ]


def test_measure_is_frame_locked_and_writes_a_report(tmp_path: Path):
    height, width = RESOLUTIONS["720p"]
    client = FakeSim(height, width)
    run = measure_resolution(
        client,
        ["front", "down", "port", "starboard"],
        height=height,
        width=width,
        frames=3,
        warmup=1,
        dt=0.05,
    )
    assert run["matched"]
    assert set(run["modalities"]) == {"rgb", "depth_planar", "segmentation", "all"}
    all_stats = run["modalities"]["all"]
    # 4 cameras x 3 image types.
    assert all_stats["responses_per_call"] == 12
    assert all_stats["fps"] > 0
    assert all_stats["latency_ms"]["p95"] >= all_stats["latency_ms"]["p50"]
    # pause, then (step, grab) per modality per frame, then unpause.
    assert client.calls[0] == ("pause", True)
    assert client.calls[-1] == ("pause", False)
    grabs = [call for call in client.calls if call[0] == "grab"]
    steps = [call for call in client.calls if call[0] == "step"]
    assert len(grabs) == len(steps) == 4 * (3 + 1)
    # Every grab is a single paused tick: the previous call is the step.
    for index, call in enumerate(client.calls):
        if call[0] == "grab":
            assert client.calls[index - 1][0] == "step"

    mismatched = FakeSim(height // 2, width // 2)
    bad = measure_resolution(
        mismatched,
        ["front"],
        height=height,
        width=width,
        frames=1,
        warmup=0,
        dt=0.01,
    )
    assert bad["matched"] is False
    report = {
        "client": "fake",
        "gpu": query_nvidia_smi(),
        "cameras": ["front", "down", "port", "starboard"],
        "runs": [run, bad],
    }
    json_path, md_path = write_report(tmp_path, report)
    payload = json.loads(json_path.read_text())
    assert payload["runs"][0]["modalities"]["rgb"]["fps"] > 0
    text = md_path.read_text()
    assert "720p" in text
    assert "segmentation" in text
    assert "DID NOT MATCH" in text


def test_benchmark_settings_add_segmentation_at_both_resolutions():
    cfg = load_config()
    from omegaconf import OmegaConf

    rig = OmegaConf.to_container(cfg.rig, resolve=True)
    for label, (height, width) in RESOLUTIONS.items():
        settings = build_benchmark_settings(rig, height, width)
        cameras = settings["Vehicles"]["Drone1"]["Cameras"]
        assert list(cameras) == ["front", "down", "port", "starboard"]
        types = [entry["ImageType"] for entry in cameras["front"]["CaptureSettings"]]
        assert types == [0, 1, 5]
        assert cameras["port"]["CaptureSettings"][0]["Width"] == width
        assert cameras["port"]["CaptureSettings"][0]["Height"] == height
    text = render_markdown({"gpu": None, "client": "airsim", "cameras": ["front"], "runs": []})
    assert "GPU:" in text
