"""Rasterizer principal point, ViT gradients, configs, and a CPU overfit."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from aerosplat.config import load_config
from aerosplat.data.dataset import build_dataset
from aerosplat.models.factory import ModelConfig, build_lifter
from aerosplat.models.gaussians import pack_gaussians, sh_dc_from_rgb
from aerosplat.models.rasterizer import (
    GaussianRasterizer,
    diff_gaussian_camera_tensors,
    kernel_pixels,
    load_diff_gaussian_rasterization,
)
from aerosplat.models.splats import FeedforwardSplatter, SplatConfig, SplatPredictor
from aerosplat.train import load_checkpoint, run_training, save_checkpoint, set_seed


def test_kernel_pixels_follow_opencv_when_the_principal_point_is_off_center():
    intrinsics = torch.tensor([[20.0, 0.0, 10.0], [0.0, 18.0, 4.0], [0.0, 0.0, 1.0]])
    angle = 0.4
    rotation = torch.tensor(
        [
            [torch.cos(torch.tensor(angle)), 0.0, torch.sin(torch.tensor(angle))],
            [0.0, 1.0, 0.0],
            [-torch.sin(torch.tensor(angle)), 0.0, torch.cos(torch.tensor(angle))],
        ]
    )
    translation = torch.tensor([0.2, -0.3, 0.15])
    xyz = torch.tensor([[0.0, 0.0, 3.0], [0.5, -0.25, 2.2], [-0.4, 0.3, 4.5]])
    height, width = 12, 16
    pixels = kernel_pixels(xyz, intrinsics, rotation, translation, height, width)
    camera = xyz @ rotation.T + translation
    expected_u = intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2]
    expected_v = intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2]
    assert torch.allclose(pixels[:, 0], expected_u, atol=1e-4)
    assert torch.allclose(pixels[:, 1], expected_v, atol=1e-4)
    # cx = 10 on a width-16 image is not (W-1)/2 = 7.5, so the projection
    # shift is nonzero. tanfov stays (W/2)/fx so the kernel focal equals fx.
    packed = diff_gaussian_camera_tensors(intrinsics, rotation, translation, height, width)
    assert abs(packed["tanfovx"] - (8.0 / 20.0)) < 1e-6
    assert abs(float(packed["projection"][0, 2]) - ((2 * 10.0 - 16 + 1) / 16)) < 1e-5


def test_reference_render_ignores_velocity_and_reaches_the_vit():
    packed = _one_gaussian((0.0, 0.0, 2.0), (0.9, 0.1, 0.1), 0.12, 0.8)
    other = packed.clone()
    other[..., 59:] = 4.0
    cameras = {"front": _pose(1, 17, 17, 18.0)}
    renderer = GaussianRasterizer(backend="reference")
    first = renderer(packed, cameras)["front"]
    second = renderer(other, cameras)["front"]
    assert torch.allclose(first, second, atol=1e-6)

    model = _tiny_splatter()
    images, poses = _one_view(16)
    model.train()
    cloud = model(images, poses)
    rendered = renderer(cloud.packed, _with_size(poses, 16), valid=cloud.valid)["front"]
    rendered.sum().backward()
    weight = model.lifter.backbone.encoder.patch_embed.proj.weight
    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or load_diff_gaussian_rasterization() is None,
    reason="CUDA diff-gaussian-rasterization is not installed; CPU CI skips this",
)
def test_cuda_raster_matches_the_reference_on_a_tiny_scene():
    torch.manual_seed(0)
    packed = _one_gaussian((0.04, -0.03, 2.0), (0.8, 0.2, 0.1), 0.08, 0.85).cuda()
    packed.requires_grad_(True)
    cameras = {
        name: {
            key: value.cuda() if torch.is_tensor(value) else value for key, value in camera.items()
        }
        for name, camera in {"front": _pose(1, 21, 21, 30.0)}.items()
    }
    reference = GaussianRasterizer(backend="reference").cuda()
    cuda = GaussianRasterizer(backend="diff_gaussian").cuda()
    image_ref = reference(packed, cameras)["front"]
    image_cuda = cuda(packed, cameras)["front"]
    assert image_ref.shape == image_cuda.shape
    assert torch.allclose(image_ref, image_cuda, atol=5e-2, rtol=0.1)
    loss_ref = (image_ref - 0.3).pow(2).mean()
    loss_cuda = (image_cuda - 0.3).pow(2).mean()
    grad_ref = torch.autograd.grad(loss_ref, packed, retain_graph=True)[0]
    grad_cuda = torch.autograd.grad(loss_cuda, packed)[0]
    # Velocity is not in the kernel. Both gradients on that slice are zero.
    assert torch.allclose(grad_ref[..., 59:], torch.zeros_like(grad_ref[..., 59:]), atol=1e-6)
    assert torch.allclose(grad_cuda[..., 59:], torch.zeros_like(grad_cuda[..., 59:]), atol=1e-6)
    ref_vec = grad_ref[..., :59].reshape(-1)
    cuda_vec = grad_cuda[..., :59].reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(ref_vec, cuda_vec, dim=0)
    assert float(cosine) > 0.9


def test_experiment_configs_select_the_laptop_and_ddp_recipes():
    default = load_config()
    assert default.train.lambda_photo == 0.8
    assert default.train.lambda_ssim == 0.2
    assert default.train.lambda_occ == 0.0
    assert default.train.lambda_velocity == 0.0
    assert default.train.tau_max == 0.008
    small = load_config(config_name="experiment/laptop_8gb")
    assert small.model.name == "vit_small"
    assert small.model.embed_dim == 384
    assert int(small.model.img_size) == 224
    assert bool(small.model.gradient_checkpointing) is True
    assert bool(small.train.gradient_checkpointing) is True
    assert bool(small.train.amp) is True
    assert int(small.train.batch_size) == 1
    assert small.train.backend == "auto"
    large = load_config(config_name="experiment/laptop_16gb")
    assert large.model.name == "vit_base"
    assert large.model.embed_dim == 768
    assert int(large.model.memory.min_vram_gb) == 8
    assert bool(large.train.amp) is True
    ddp = load_config(config_name="experiment/ddp")
    assert bool(ddp.train.distributed) is True
    assert ddp.train.backend == "diff_gaussian"
    assert ddp.train.device == "cuda"
    assert ddp.model.name == "vit_small"


def test_overfit_on_the_fixture_reduces_photo_with_the_cpu_renderer(tmp_path):
    cfg = load_config(
        [
            "model=vit_tiny",
            "model.img_size=16",
            "model.lift_stride=8",
            "model.splats.slots_per_cell=1",
            "model.gradient_checkpointing=true",
            "train.steps=4",
            "train.device=cpu",
            "train.backend=reference",
            "train.amp=false",
            "train.lambda_photo=1",
            "train.lambda_ssim=0",
            "train.lambda_temp=0",
            "train.lambda_sparse=0",
            "train.lambda_occ=0",
            "train.lambda_velocity=0",
            "train.lr=0.05",
            "train.weight_decay=0",
            "train.log_every=1",
            "train.ckpt_every=0",
            "train.save_last=true",
            "train.gradient_checkpointing=true",
            "train.camera_names=[front]",
            "train.wandb=false",
            "train.batch_size=1",
            "train.grad_accum_steps=1",
            "train.num_workers=0",
            f"train.log_dir={tmp_path}",
            "train.seed=0",
        ]
    )
    base = build_dataset(load_config())
    history = run_training(cfg, dataset=_Only(base[0]))
    assert len(history) == 4
    assert history[-1]["photo"] < history[0]["photo"]
    metrics = (Path(tmp_path) / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 4
    assert (Path(tmp_path) / "render_000001.png").is_file()
    assert (Path(tmp_path) / "last.pt").is_file()
    # Same seed, same first photo, on a fresh process-local run of one step.
    set_seed(0)
    again = load_config(
        [
            "model=vit_tiny",
            "model.img_size=16",
            "model.lift_stride=8",
            "model.splats.slots_per_cell=1",
            "train.steps=1",
            "train.device=cpu",
            "train.backend=reference",
            "train.amp=false",
            "train.lambda_photo=1",
            "train.lambda_ssim=0",
            "train.lambda_temp=0",
            "train.lambda_sparse=0",
            "train.lambda_occ=0",
            "train.lambda_velocity=0",
            "train.lr=0.05",
            "train.log_every=0",
            "train.ckpt_every=0",
            "train.save_last=false",
            "train.gradient_checkpointing=false",
            "train.camera_names=[front]",
            "train.wandb=false",
            f"train.log_dir={tmp_path}/seed",
            "seed=0",
        ]
    )
    # The first run already consumed the global RNG. Re-seed inside run_training
    # via cfg.seed, which this config sets, and compare two fresh calls.
    first = run_training(again, dataset=_Only(base[0]))
    second = run_training(again, dataset=_Only(base[0]))
    assert abs(first[0]["photo"] - second[0]["photo"]) < 1e-6


def test_checkpoint_roundtrip_restores_the_step(tmp_path):
    model = _tiny_splatter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "step.pt"
    save_checkpoint(path, model, optimizer, scaler, step=7, seed=3)
    other = _tiny_splatter()
    other_opt = torch.optim.AdamW(other.parameters(), lr=1e-3)
    step = load_checkpoint(path, other, other_opt, scaler)
    assert step == 7
    left = model.state_dict()
    right = other.state_dict()
    for key, value in left.items():
        assert torch.equal(value, right[key])


class _Only(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def _tiny_splatter():
    model_cfg = ModelConfig(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_indices=(1,),
        cross_view_layers=1,
        pose_encoding="plucker",
        pose_zero_init=False,
        normalize="none",
        gradient_checkpointing=False,
        depth_planes=4,
        depth_near=0.5,
        depth_far=8.0,
        depth_spacing="linear",
        lift_stride=8,
    )
    return FeedforwardSplatter(
        build_lifter(model_cfg),
        SplatPredictor(16, SplatConfig(slots_per_cell=1, max_extra_gaussians=0)),
    )


def _one_view(size):
    image = torch.rand(1, 3, size, size)
    intrinsics = torch.eye(3).view(1, 3, 3)
    intrinsics[0, 0, 0] = 12.0
    intrinsics[0, 1, 1] = 12.0
    intrinsics[0, 0, 2] = size / 2
    intrinsics[0, 1, 2] = size / 2
    cameras = {
        "front": {
            "K": intrinsics,
            "R": torch.eye(3).view(1, 3, 3),
            "t": torch.zeros(1, 3),
        }
    }
    return {"front": image}, cameras


def _with_size(cameras, size):
    out = {}
    for name, camera in cameras.items():
        out[name] = {**camera, "height": size, "width": size}
    return out


def _pose(batch, height, width, focal):
    intrinsics = torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    return {
        "K": intrinsics,
        "R": torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1),
        "t": torch.zeros(batch, 3),
        "height": height,
        "width": width,
    }


def _one_gaussian(xyz, rgb, scale, opacity):
    sh = torch.zeros(1, 1, 16, 3)
    sh[0, 0, 0] = sh_dc_from_rgb(torch.tensor(rgb))
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    return pack_gaussians(
        torch.tensor(xyz, dtype=torch.float32).view(1, 1, 3),
        torch.full((1, 1, 3), scale),
        quat,
        torch.tensor([[[opacity]]]),
        sh,
        torch.zeros(1, 1, 3),
    )
