"""Depth decoding and the free / occupied / unknown voxel cache."""

from pathlib import Path

import numpy as np
import pytest

from aerosplat.data.cache import CacheConfig, GroundTruthCache
from aerosplat.data.io import load_depth, read_pfm, write_pfm
from aerosplat.geometry.occupancy import FREE, OCCUPIED, UNKNOWN, occupancy_from_depths


def test_pfm_roundtrip(tmp_path: Path):
    depth = np.array([[0.0, 1.5], [2.0, 4.25]], dtype=np.float32)
    path = tmp_path / "depth.pfm"
    write_pfm(path, depth)
    loaded = read_pfm(path)
    assert loaded.shape == depth.shape
    assert np.allclose(loaded, depth)
    assert np.allclose(load_depth(path), depth)


def test_npy_depth_roundtrip(tmp_path: Path):
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "depth.npy"
    np.save(path, depth)
    assert np.allclose(load_depth(path), depth)


def test_surface_is_occupied_and_the_far_side_stays_unknown():
    # Identity camera at the origin looking +Z. A  patch at planar z = 4.
    height = width = 32
    K = np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((height, width), dtype=np.float64)
    depth[14:19, 14:19] = 4.0
    origin = np.array([-2.0, -2.0, 0.0])
    dims = (8, 8, 20)
    state, hits = occupancy_from_depths(
        [depth],
        [K],
        [np.eye(3)],
        [np.zeros(3)],
        origin=origin,
        voxel_size=0.5,
        dims=dims,
        max_depth=20.0,
        min_depth=1e-3,
        stride=1,
    )
    # World (0, 0, z) -> voxel index (4, 4, z / 0.5).
    assert state[4, 4, 4] == FREE  # z = 2, in front of the surface
    assert state[4, 4, 8] == OCCUPIED  # z = 4, the hit
    assert state[4, 4, 12] == UNKNOWN  # z = 6, behind the surface
    assert hits[4, 4, 8] > 0
    assert int(state[state == OCCUPIED].size) >= 1


def test_cache_roundtrip_uses_capture_pose(tmp_path: Path):
    from aerosplat.data.adapters.canonical import read_canonical
    from aerosplat.data.dataset import MultiCameraDataset
    from aerosplat.data.fixture import generate_fixture
    from aerosplat.data.records import SyncConfig

    root = generate_fixture(tmp_path / "rig")
    dataset = MultiCameraDataset(
        read_canonical(root),
        SyncConfig(
            policy="nearest",
            reference_camera="front",
            max_image_dt=0.05,
            camera_order=("front", "starboard"),
        ),
        load_depth=True,
    )
    sample = dataset[0]
    config = CacheConfig(root=tmp_path / "cache", voxel_size=0.5, extent_m=8.0, stride=2)
    cache = GroundTruthCache(config)
    first = cache.get(sample)
    second = cache.get(sample)
    assert cache.path_for(sample).is_file()
    assert first.scene_id == sample.scene_id
    assert int(first.occupied.sum()) > 0
    assert int(first.free.sum()) > 0
    assert np.array_equal(first.state, second.state)
    assert first.dims[0] == pytest.approx(int(np.ceil(8.0 / 0.5)))
