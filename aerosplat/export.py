"""ONNX export of the network, and a TensorRT build for Jetson Orin or RTX.

The exported graph is the encoder through the Gaussian cloud, the glare
layers, and the occupancy volume. The rasterizer is not in the graph: it is a
CUDA extension with its own camera matrices, and ONNX would not run it on the
Jetson. Render ``clean`` with the rasterizer, then add the exported glare.

Shapes are static. The trace uses the batch, camera count, resolution, slot
count, and grid size of the example, and ``dynamic_axes`` is left unset.

Sub-millisecond is a realistic budget for :func:`aerosplat.models.collision.query_collision`
on the occupancy volume. It is not a realistic budget for the ViT. See
``docs/PHASE5.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from aerosplat.models.gaussians import GaussianLayout
from aerosplat.models.occupancy_head import OccupancyHead
from aerosplat.models.relight import RelightHead
from aerosplat.train import SplatTrainer


class InferenceModule(nn.Module):
    """Stacked-tensor wrapper around :class:`SplatTrainer`.

    Inputs are dense tensors so ONNX can see them. Camera ``i`` is entry ``i``
    of ``camera_names``. The rasterizer is not called.
    """

    def __init__(
        self,
        trainer: SplatTrainer,
        camera_names: tuple[str, ...],
        grid_size: tuple[int, int, int],
        voxel_size: float,
        origin: Tensor,
    ) -> None:
        super().__init__()
        if not camera_names:
            raise ValueError("camera_names must be non-empty")
        self.trainer = trainer
        self.camera_names = tuple(camera_names)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        self.voxel_size = float(voxel_size)
        origin = torch.as_tensor(origin, dtype=torch.float32).reshape(1, 3)
        self.register_buffer("origin", origin)

    def forward(
        self,
        images: Tensor,
        intrinsics: Tensor,
        rotation: Tensor,
        translation: Tensor,
        sun_world: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if images.ndim != 5:
            raise ValueError(f"images must be [B, V, 3, H, W], got {tuple(images.shape)}")
        if images.shape[1] != len(self.camera_names):
            raise ValueError(
                f"images have {images.shape[1]} cameras, module has {len(self.camera_names)}"
            )
        views: dict[str, Tensor] = {}
        cameras: dict[str, dict[str, Tensor]] = {}
        for index, name in enumerate(self.camera_names):
            views[name] = images[:, index]
            cameras[name] = {
                "K": intrinsics[:, index],
                "R": rotation[:, index],
                "t": translation[:, index],
            }
        origin = self.origin.to(dtype=images.dtype).expand(images.shape[0], 3)
        cloud, glare, volume = self.trainer(
            views,
            cameras,
            sun_world,
            origin,
            self.voxel_size,
            self.grid_size,
        )
        assert volume is not None
        glare_stack = torch.stack([glare[name] for name in self.camera_names], dim=1)
        return (
            cloud.packed,
            cloud.valid.to(dtype=cloud.packed.dtype),
            cloud.wire_prob,
            volume.occupancy_logits,
            volume.wire_logits,
            volume.wire_direction,
            volume.wire_offset,
            glare_stack,
        )


class RelightOnnx(nn.Module):
    """Static-shape glare head. Output height and width are baked in."""

    def __init__(self, feature_dim: int, height: int, width: int, hidden: int = 32) -> None:
        super().__init__()
        self.head = RelightHead(feature_dim, hidden=hidden)
        self.height = int(height)
        self.width = int(width)

    def forward(self, features: Tensor, sh: Tensor, sun_cam: Tensor) -> Tensor:
        return self.head(features, sh, sun_cam, (self.height, self.width))


class OccupancyOnnx(nn.Module):
    """Static-shape occupancy head. The grid size is baked in.

    Returns tensors, not the dataclass, so ONNX can name the outputs. ``valid``
    is 0/1 float: the training path passes a bool mask and the head casts it.
    """

    def __init__(self, grid_size: tuple[int, int, int], hidden: int = 16) -> None:
        super().__init__()
        self.head = OccupancyHead(hidden=hidden)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))

    def forward(
        self,
        xyz: Tensor,
        opacity: Tensor,
        wire_prob: Tensor,
        valid: Tensor,
        origin: Tensor,
        voxel_size: Tensor,
        scale: Tensor,
        sh_dc: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        volume = self.head(
            xyz,
            opacity,
            wire_prob,
            valid,
            origin,
            voxel_size,
            self.grid_size,
            scale,
            sh_dc,
        )
        return (
            volume.occupancy_logits,
            volume.wire_logits,
            volume.wire_direction,
            volume.wire_offset,
        )


def export_onnx(
    module: nn.Module,
    args: tuple,
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Write a static-shape ONNX file. Rendering is not part of ``module``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": 17,
        "dynamo": False,
        "dynamic_axes": None,
    }
    try:
        torch.onnx.export(module, args, str(path), **kwargs)
    except TypeError:
        kwargs.pop("dynamo", None)
        kwargs.pop("dynamic_axes", None)
        torch.onnx.export(module, args, str(path), **kwargs)


def relight_example(batch: int, channels: int, height: int, width: int) -> tuple:
    features = torch.zeros(batch, channels, height, width)
    sh = torch.zeros(batch, GaussianLayout.SH_COEFFS, 3)
    sun = torch.zeros(batch, 3)
    sun[:, 2] = 1.0
    return (features, sh, sun, (height, width))


def occupancy_example(batch: int, count: int, grid: tuple[int, int, int]) -> tuple:
    xyz = torch.zeros(batch, count, 3)
    xyz[..., 0] = 0.1
    opacity = torch.full((batch, count), 0.5)
    wire = torch.zeros(batch, count)
    valid = torch.ones(batch, count, dtype=torch.bool)
    origin = torch.zeros(batch, 3)
    voxel = torch.tensor(0.5)
    scale = torch.full((batch, count, 3), 0.1)
    sh_dc = torch.zeros(batch, count, 3)
    return (xyz, opacity, wire, valid, origin, voxel, grid, scale, sh_dc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export AeroSplat heads or the full network to ONNX."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--what",
        choices=("relight", "occupancy", "network"),
        default="network",
        help="ViT through Gaussians, glare, and occupancy. The rasterizer is omitted.",
    )
    parser.add_argument("--config-name", default="config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    if args.what == "relight":
        module = RelightOnnx(feature_dim=8, height=16, width=16, hidden=8)
        features, sh, sun, _hw = relight_example(1, 8, 16, 16)
        export_onnx(
            module,
            (features, sh, sun),
            args.output,
            ["features", "sh", "sun_cam"],
            ["glare"],
        )
        return
    if args.what == "occupancy":
        grid = (8, 8, 8)
        module = OccupancyOnnx(grid, hidden=8)
        example = occupancy_example(1, 4, grid)
        xyz, opacity, wire, valid, origin, voxel, _grid, scale, sh_dc = example
        export_onnx(
            module,
            (
                xyz,
                opacity,
                wire,
                valid.to(dtype=torch.float32),
                origin,
                voxel.reshape(()),
                scale,
                sh_dc,
            ),
            args.output,
            ["xyz", "opacity", "wire_prob", "valid", "origin", "voxel_size", "scale", "sh_dc"],
            ["occupancy_logits", "wire_logits", "wire_direction", "wire_offset"],
        )
        return
    from aerosplat.config import load_config
    from aerosplat.models.factory import build_model

    cfg = load_config(args.overrides, config_name=args.config_name)
    splatter = build_model(cfg)
    splatter.lifter.backbone.encoder.gradient_checkpointing = False
    trainer = SplatTrainer(
        splatter,
        RelightHead(feature_dim=int(cfg.model.embed_dim)),
        OccupancyHead(),
    )
    trainer.eval()
    batch = 1
    cameras = 1
    size = int(cfg.model.img_size)
    grid_n = 8
    module = InferenceModule(
        trainer,
        ("front",),
        (grid_n, grid_n, grid_n),
        0.5,
        torch.zeros(3),
    )
    images = torch.zeros(batch, cameras, 3, size, size)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    intrinsics[..., 0, 0] = size
    intrinsics[..., 1, 1] = size
    intrinsics[..., 0, 2] = size / 2
    intrinsics[..., 1, 2] = size / 2
    rotation = torch.eye(3).view(1, 1, 3, 3).repeat(batch, cameras, 1, 1)
    translation = torch.zeros(batch, cameras, 3)
    sun = torch.tensor([[0.0, 0.0, 1.0]])
    export_onnx(
        module,
        (images, intrinsics, rotation, translation, sun),
        args.output,
        ["images", "intrinsics", "rotation", "translation", "sun_world"],
        [
            "packed",
            "valid",
            "wire_prob",
            "occupancy_logits",
            "wire_logits",
            "wire_direction",
            "wire_offset",
            "glare",
        ],
    )


if __name__ == "__main__":
    main()
