"""Config composition, the checked-in rig file, and the CPU CI policy."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from aerosplat.capture.colosseum import build_settings
from aerosplat.config import load_config
from aerosplat.data.fixture import camera_pose, default_fixture_root
from aerosplat.data.records import SyncConfig, sync_config_from_dict


def test_hydra_groups_switch_dataset():
    fixture = load_config()
    assert fixture.data.source == "canonical"
    assert fixture.data.sync.reference_camera == "front"
    assert fixture.rig.cameras.front.yaw == 0
    assert fixture.rig.cameras.down.pitch == -90
    tartan = load_config(["data=tartanair", "data.root=/tmp/tartan"])
    assert tartan.data.source == "tartanair"
    assert tartan.data.sync.reference_camera == "left"
    assert tartan.data.intrinsics.fx == 320
    assert tartan.rig.name == "uam_quad"


def test_checked_in_settings_match_the_rig_yaml():
    cfg = load_config()
    rig = OmegaConf.to_container(cfg.rig, resolve=True)
    generated = build_settings(rig)
    committed = json.loads(
        (Path(__file__).parents[1] / "configs" / "rig" / "colosseum_settings.json").read_text()
    )
    assert generated == committed
    names = list(generated["Vehicles"]["Drone1"]["Cameras"])
    assert names == ["front", "down", "port", "starboard"]


def test_ci_does_not_install_a_cuda_rasterizer():
    text = (Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml").read_text()
    install_lines = [
        line for line in text.splitlines() if "pip install" in line or "pip3 install" in line
    ]
    blob = "\n".join(install_lines)
    assert "diff-gaussian" not in blob
    assert "gsplat" not in blob
    assert "nvcc" not in text
    assert "whl/cpu" in text


def test_committed_fixture_matches_the_generator():
    root = default_fixture_root()
    manifest = json.loads((root / "dataset.json").read_text())
    assert manifest["format"] == "aerosplat.multicam.v1"
    assert manifest["world_frame"] == "enu_z_up"
    front = (root / "flight0" / "front" / "frames.jsonl").read_text().splitlines()[0]
    record = json.loads(front)
    R, t = camera_pose("front", 0.0)
    assert record["timestamp"] == 0.0
    assert np_allclose(record["R"], R)
    assert np_allclose(record["t"], t)


def test_sync_config_rejects_unknown_policy():
    import pytest

    with pytest.raises(ValueError):
        SyncConfig(policy="bilinear")
    parsed = sync_config_from_dict(
        {"policy": "nearest", "reference_camera": "left", "max_image_dt": 0.01}
    )
    assert parsed.reference_camera == "left"


def np_allclose(a, b):
    import numpy as np

    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=1e-5)
