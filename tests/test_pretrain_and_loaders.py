"""Pretraining loaders and the CPU depth loop on the tiny fixture."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from aerosplat.config import load_config
from aerosplat.data.adapters.aerial_megadepth import load_aerial_megadepth, parse_colmap_images
from aerosplat.data.adapters.matrixcity import fix_matrixcity_rotation, load_matrixcity
from aerosplat.data.adapters.tartanair_v2 import load_tartanair_v2
from aerosplat.data.dataset import build_dataset
from aerosplat.data.exr import read_exr, write_exr
from aerosplat.data.fixture import default_fixture_root
from aerosplat.data.specs import write_tartanair_v2_depth
from aerosplat.geometry.camera import quat_wxyz_to_rotmat
from aerosplat.geometry.coordinates import (
    intrinsics_from_hfov,
    nerf_c2w_to_opencv_w2c,
    opencv_cam2world_matrix,
    tartanair_pose_to_w2c,
)
from aerosplat.models.factory import build_lifter
from aerosplat.models.pretrain import prepare_rig, run_depth_pretrain


def _png(path: Path, color: tuple[int, int, int], size=(8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:] = color
    Image.fromarray(arr, mode="RGB").save(path)


def test_exr_roundtrip_none_and_zip(tmp_path: Path):
    depth = np.linspace(100.0, 400.0, 12, dtype=np.float32).reshape(3, 4)
    for compression in ("none", "zip", "zips"):
        path = tmp_path / f"depth_{compression}.exr"
        write_exr(path, {"R": depth, "G": depth / 2}, compression=compression)
        planes = read_exr(path)
        assert planes["R"].shape == (3, 4)
        assert np.allclose(planes["R"], depth, atol=1e-4)
        assert np.allclose(planes["G"], depth / 2, atol=1e-4)


def test_reader_matches_openexr_when_installed(tmp_path: Path):
    pytest.importorskip("OpenEXR")
    import Imath
    import OpenEXR

    depth = np.arange(20, dtype=np.float32).reshape(4, 5) * 10 + 250
    path = tmp_path / "from_openexr.exr"
    header = OpenEXR.Header(5, 4)
    header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels({"R": depth.tobytes(), "G": depth.tobytes(), "B": depth.tobytes()})
    out.close()
    planes = read_exr(path)
    assert np.allclose(planes["R"], depth, atol=1e-5)


def test_aerial_megadepth_manifest_is_opencv_cam2world_meters(tmp_path: Path):
    K = intrinsics_from_hfov(8, 6, 60.0)
    center = np.array([10.0, -2.0, 80.0])
    cam2world = np.eye(4)
    cam2world[:3, 3] = center
    # A 90° yaw so the matrix is not the identity rotation.
    cam2world[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    scene = tmp_path / "0001"
    _png(scene / "images" / "a.png", (20, 40, 60))
    _png(scene / "images" / "b.png", (60, 40, 20))
    (scene / "depths").mkdir(parents=True)
    np.save(scene / "depths" / "a.npy", np.full((6, 8), 120.0, np.float32))
    np.save(scene / "depths" / "b.npy", np.full((6, 8), 140.0, np.float32))
    manifest = {
        "views": [
            {
                "image": "images/a.png",
                "depth": "depths/a.npy",
                "K": K.tolist(),
                "cam2world": cam2world.tolist(),
            },
            {
                "image": "images/b.png",
                "depth": "depths/b.npy",
                "K": K.tolist(),
                "cam2world": cam2world.tolist(),
            },
        ]
    }
    (scene / "views.json").write_text(json.dumps(manifest))
    dataset = load_aerial_megadepth(scene, SimpleNamespace(views_per_sample=2))
    sample = dataset[0]
    frame = sample.cameras["view0"]
    matrix = opencv_cam2world_matrix(frame.R.numpy(), frame.t.numpy())
    assert np.allclose(matrix[:3, 3], center, atol=1e-5)
    assert np.allclose(matrix[:3, :3], cam2world[:3, :3], atol=1e-5)
    assert float(frame.depth[0, 0]) == pytest.approx(120.0)


def test_colmap_pose_roundtrip(tmp_path: Path):
    # Identity OpenCV camera sitting at (0, 0, 5): w2c translation is (0, 0, -5).
    cameras = "1 SIMPLE_PINHOLE 8 6 40 4 3\n"
    images = "1 1 0 0 0 0 0 -5 1 frame.png\n\n"
    scene = tmp_path / "colmap"
    (scene / "sparse").mkdir(parents=True)
    (scene / "sparse" / "cameras.txt").write_text(cameras)
    (scene / "sparse" / "images.txt").write_text(images)
    _png(scene / "images" / "frame.png", (1, 2, 3))
    (scene / "depths").mkdir(parents=True)
    np.save(scene / "depths" / "frame.npy", np.full((6, 8), 30.0, np.float32))
    # A second view so the chunk has two cameras.
    images2 = images + "2 1 0 0 0 0 0 -6 1 other.png\n\n"
    (scene / "sparse" / "images.txt").write_text(images2)
    _png(scene / "images" / "other.png", (3, 2, 1))
    np.save(scene / "depths" / "other.npy", np.full((6, 8), 31.0, np.float32))
    parsed = parse_colmap_images(images2)
    assert np.allclose(quat_wxyz_to_rotmat(parsed[0]["quat_wxyz"]), np.eye(3))
    dataset = load_aerial_megadepth(scene, SimpleNamespace(views_per_sample=2))
    frame = dataset[0].cameras["view0"]
    matrix = opencv_cam2world_matrix(frame.R.numpy(), frame.t.numpy())
    assert np.allclose(matrix[:3, 3], [0, 0, 5], atol=1e-5)
    assert float(frame.depth.mean()) == pytest.approx(30.0)


def test_tartanair_v2_depth_and_pose(tmp_path: Path):
    traj = tmp_path / "ArchViz" / "Data_easy" / "P000"
    depth = np.linspace(0.5, 4.0, 16, dtype=np.float32).reshape(4, 4)
    for name, color in (("lcam_front", (10, 20, 30)), ("lcam_right", (30, 20, 10))):
        _png(
            traj / f"image_{name}" / "000000_lcam_front.png".replace("lcam_front", name),
            color,
            (4, 4),
        )
        # The filename stem only needs to contain the index.
        image = traj / f"image_{name}" / f"000000_{name}.png"
        if not image.is_file():
            _png(image, color, (4, 4))
        write_tartanair_v2_depth(traj / f"depth_{name}" / f"000000_{name}_depth.png", depth)
        # NED position and an identity quaternion, xyzw.
        (traj / f"pose_{name}.txt").write_text("1.0 2.0 3.0 0 0 0 1\n")
    dataset = load_tartanair_v2(traj, SimpleNamespace(cameras=None, max_frames=None, fps=10.0))
    sample = dataset[0]
    assert set(sample.cameras) == {"lcam_front", "lcam_right"}
    frame = sample.cameras["lcam_front"]
    assert frame.depth.shape == (4, 4)
    assert np.allclose(frame.depth.numpy(), depth, atol=1e-5)
    expected_R, expected_t = tartanair_pose_to_w2c(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0])
    )
    assert np.allclose(frame.R.numpy(), expected_R, atol=1e-5)
    assert np.allclose(frame.t.numpy(), expected_t, atol=1e-5)
    matrix = opencv_cam2world_matrix(frame.R.numpy(), frame.t.numpy())
    # Camera +Z in the OpenCV cam2world matrix is the third column of R_c2w.
    forward = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(forward, matrix[:3, 2], atol=1e-6)
    assert abs(np.linalg.det(matrix[:3, :3]) - 1.0) < 1e-5


def test_matrixcity_rotation_scale_and_centimeter_depth(tmp_path: Path):
    # Build an OpenCV cam2world, convert to NeRF, then divide the rotation by 100.
    center = np.array([12.0, -4.0, 350.0])
    R_cv = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    R_nerf = R_cv @ np.diag([1.0, -1.0, -1.0])
    broken = np.eye(4)
    broken[:3, :3] = R_nerf / 100.0
    broken[:3, 3] = center
    already = np.eye(4)
    already[:3, :3] = R_nerf
    already[:3, 3] = center
    fixed = fix_matrixcity_rotation(broken[:3, :3], "auto")
    assert np.allclose(fixed, R_nerf, atol=1e-6)
    untouched = fix_matrixcity_rotation(already[:3, :3], "auto")
    assert np.allclose(untouched, R_nerf, atol=1e-6)

    folder = tmp_path / "block_1"
    folder.mkdir()
    _png(folder / "0000.png", (8, 8, 8), (6, 4))
    _png(folder / "0001.png", (9, 9, 9), (6, 4))
    depth_cm = np.full((4, 6), 25400.0, np.float32)  # 254 m
    from aerosplat.data.exr import write_exr

    write_exr(folder / "0000.exr", {"R": depth_cm})
    write_exr(folder / "0001.exr", {"R": depth_cm + 100.0})
    transforms = {
        "camera_angle_x": 0.7853981633974483,
        "frames": [
            {"file_path": "0000.png", "transform_matrix": broken.tolist()},
            {"file_path": "0001.png", "transform_matrix": broken.tolist()},
        ],
    }
    (folder / "transforms.json").write_text(json.dumps(transforms))
    dataset = load_matrixcity(
        folder,
        SimpleNamespace(
            views_per_sample=2,
            rotation_fix="auto",
            translation_scale=1.0,
            depth_scale=0.01,
        ),
    )
    sample = dataset[0]
    frame = sample.cameras["view0"]
    assert float(frame.depth[0, 0]) == pytest.approx(254.0, abs=1e-2)
    got = nerf_c2w_to_opencv_w2c(
        np.block([[R_nerf, center.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])
    )
    assert np.allclose(frame.R.numpy(), got[0], atol=1e-4)
    assert np.allclose(frame.t.numpy(), got[1], atol=1e-4)
    matrix = opencv_cam2world_matrix(frame.R.numpy(), frame.t.numpy())
    assert np.allclose(matrix[:3, 3], center, atol=1e-3)
    # OpenCV forward is +Z, which for this rotation is world +X.
    assert np.allclose(matrix[:3, 2], [1.0, 0.0, 0.0], atol=1e-4)


def test_depth_pretrain_runs_on_the_fixture_cpu():
    cfg = load_config(["model=vit_tiny", "data=fixture", f"data.root={default_fixture_root()}"])
    dataset = build_dataset(cfg)
    model = build_lifter(cfg)
    sample = prepare_rig(dataset[0], int(cfg.model.img_size), int(cfg.model.patch_size))
    losses = run_depth_pretrain(model, [sample], steps=2, lr=1e-3, device="cpu")
    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.abs().sum())
    assert grad_norm > 0


def test_model_configs_name_the_laptop_vram_budget():
    from omegaconf import OmegaConf

    root = Path(__file__).parents[1] / "configs" / "model"
    small = OmegaConf.load(root / "vit_small.yaml")
    base = OmegaConf.load(root / "vit_base.yaml")
    assert small.memory.min_vram_gb == 8
    assert small.memory.max_vram_gb == 16
    assert base.memory.min_vram_gb == 8
    assert base.memory.max_vram_gb == 16
    assert small.memory.recommended_resolution == 224
    assert "8" in small.memory.note and "16" in small.memory.note
    assert small.embed_dim == 384 and base.embed_dim == 768
    large = OmegaConf.load(root / "vit_large.yaml")
    assert large.embed_dim == 1024 and large.depth == 24 and large.patch_size == 16
