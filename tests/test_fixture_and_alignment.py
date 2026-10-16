"""Synthetic rig: variable resolution, unequal rates, geometric alignment."""

from pathlib import Path

import numpy as np

from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.collate import collate_samples
from aerosplat.data.dataset import MultiCameraDataset
from aerosplat.data.fixture import RESOLUTIONS, camera_pose, generate_fixture, vehicle_center
from aerosplat.data.records import SyncConfig
from aerosplat.data.transforms import resize_camera_frame
from aerosplat.geometry.camera import camera_center
from aerosplat.vis.alignment import check_alignment, save_report


def _dataset(root: Path, **sync_kwargs) -> MultiCameraDataset:
    defaults = dict(
        policy="interpolate_pose",
        reference_camera="front",
        max_image_dt=0.15,
        require_all_cameras=True,
        camera_order=("front", "down", "port", "starboard"),
    )
    defaults.update(sync_kwargs)
    return MultiCameraDataset(read_canonical(root), SyncConfig(**defaults))


def test_fixture_rates_resolutions_and_interpolated_port_pose(tmp_path: Path):
    root = generate_fixture(tmp_path / "rig")
    dataset = _dataset(root)
    assert len(dataset) == 5
    sample = dataset[2]
    assert sample.timestamp == 0.2
    assert sample.camera_names == ["front", "down", "port", "starboard"]
    shapes = {name: (frame.height, frame.width) for name, frame in sample.cameras.items()}
    assert shapes == {name: (h, w) for name, (w, h) in RESOLUTIONS.items()}
    assert len({shapes[name] for name in shapes}) >= 3

    port = sample.cameras["port"]
    assert port.timestamp == 0.22
    assert port.sync == "interpolated"
    R_cap, t_cap = camera_pose("port", 0.22)
    R_query, t_query = camera_pose("port", 0.2)
    assert np.allclose(port.R.numpy(), R_cap, atol=1e-5)
    assert np.allclose(port.t.numpy(), t_cap, atol=1e-5)
    assert np.allclose(port.R_query.numpy(), R_query, atol=1e-5)
    assert np.allclose(port.t_query.numpy(), t_query, atol=1e-5)
    expected_center = vehicle_center(0.2) + np.array([0.0, 0.15, 0.0])
    got_center = camera_center(port.R_query.numpy(), port.t_query.numpy())
    assert np.allclose(got_center, expected_center)

    front = sample.cameras["front"]
    assert front.sync == "exact"
    assert front.depth is not None
    assert front.depth.shape == (front.height, front.width)


def test_tight_sync_drops_down_camera_frames(tmp_path: Path):
    root = generate_fixture(tmp_path / "rig")
    dataset = _dataset(root, max_image_dt=0.05)
    # Down is 5 Hz. Queries at 0.1 and 0.3 have no downward frame within 50 ms.
    assert [dataset[i].timestamp for i in range(len(dataset))] == [0.0, 0.2, 0.4]


def test_alignment_and_resize(tmp_path: Path):
    root = generate_fixture(tmp_path / "rig")
    sample = _dataset(root)[2]
    report = check_alignment(sample, stride=1, agreement_m=0.35)
    for name, err in report.reprojection_max_px.items():
        assert err < 1e-2, name
    overlapping = [
        pair for pair in report.pairs if pair.overlap >= 20 and pair.inlier_fraction >= 0.2
    ]
    assert overlapping, [(p.source, p.target, p.overlap, p.inlier_fraction) for p in report.pairs]
    front_down = [
        pair
        for pair in report.pairs
        if {pair.source, pair.target} == {"front", "down"} and pair.overlap > 0
    ]
    assert front_down

    resized = resize_camera_frame(sample.cameras["front"], 30, 40)
    assert resized.image.shape == (3, 30, 40)
    assert resized.depth.shape == (30, 40)
    # 80x60 -> 40x30 halves both axes. fx and cx scale by 0.5.
    assert np.isclose(resized.K[0, 0].item(), sample.cameras["front"].K[0, 0].item() * 0.5)
    assert np.isclose(resized.K[0, 2].item(), sample.cameras["front"].K[0, 2].item() * 0.5)

    batch = collate_samples([_dataset(root)[0], _dataset(root)[2]])
    assert batch["cameras"]["front"]["image"].shape[0] == 2
    assert (
        batch["cameras"]["down"]["image"].shape[-2:]
        != batch["cameras"]["front"]["image"].shape[-2:]
    )
    assert "depth" in batch["cameras"]["port"]

    save_report(report, tmp_path / "viz", sample)
    assert (tmp_path / "viz" / "alignment.json").is_file()
    assert (tmp_path / "viz" / "alignment.png").is_file()
    assert (tmp_path / "viz" / "points.npz").is_file()
