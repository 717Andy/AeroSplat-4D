"""Shared multi-camera ViT with cross-view attention.

Each camera runs through the same :class:`~aerosplat.models.encoder.DinoEncoder`.
Pose tokens are added before the transformer blocks. A short stack of
cross-view blocks then lets every patch attend to every other camera. There is
no learned camera-index embedding: which way a camera looks is the pose, so
permuting the rig permutes the outputs.

The returned tokens are dense and multi-scale. Scale 1 is the patch grid after
cross-view attention. Scales 2 and 4 are average-pooled when the grid divides.
Phase 3 reads these maps; this module does not predict Gaussians.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aerosplat.models.encoder import Attention, DinoEncoder, Mlp
from aerosplat.models.pose import PoseInjection

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class BackboneTokens:
    """Per-camera dense tokens.

    ``features`` is the cross-view output at the patch grid, ``[B, C, Gh, Gw]``.
    ``multiscale`` maps a stride (in patches) to a pooled map. ``layers`` holds
    the shared encoder's intermediate patch tokens, before cross-view mixing,
    keyed by block index.
    """

    features: dict[str, Tensor]
    multiscale: dict[str, dict[int, Tensor]]
    layers: dict[str, dict[int, Tensor]]
    names: list[str]
    patch_size: int


class CrossViewBlock(nn.Module):
    """Self-attention over the concatenated tokens of every camera."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        lengths = [token.shape[1] for token in tokens]
        packed = torch.cat(tokens, dim=1)
        hidden = packed + self.attn(self.norm1(packed))
        hidden = hidden + self.mlp(self.norm2(hidden))
        return list(hidden.split(lengths, dim=1))


class ViTBackbone(nn.Module):
    """Shared encoder over a dict of camera images.

    ``images`` maps a camera name to ``[B, 3, H, W]``. Resolutions may differ.
    ``cameras`` maps the same names to ``K`` ``[B, 3, 3]``, ``R`` ``[B, 3, 3]``,
    and ``t`` ``[B, 3]`` (world-to-camera). Heights and widths must be multiples
    of the patch size; pad or resize before calling. The output order follows
    the ``images`` dict.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        out_indices: tuple[int, ...] = (11,),
        cross_view_layers: int = 2,
        pose_encoding: str = "plucker",
        pose_zero_init: bool = False,
        num_frequencies: int = 6,
        normalize: str = "none",
        gradient_checkpointing: bool = False,
        init_values: float = 1.0,
    ) -> None:
        super().__init__()
        if normalize not in {"none", "imagenet"}:
            raise ValueError(f"normalize must be none or imagenet, got {normalize!r}")
        if cross_view_layers < 1:
            raise ValueError("cross_view_layers must be >= 1")
        self.patch_size = patch_size
        self.normalize = normalize
        self.encoder = DinoEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_indices=out_indices,
            init_values=init_values,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pose = PoseInjection(
            embed_dim,
            mode=pose_encoding,
            num_frequencies=num_frequencies,
            zero_init=pose_zero_init,
        )
        self.cross_view = nn.ModuleList(
            CrossViewBlock(embed_dim, num_heads, mlp_ratio) for _ in range(cross_view_layers)
        )

    def forward(
        self, images: dict[str, Tensor], cameras: dict[str, dict[str, Tensor]]
    ) -> BackboneTokens:
        if not images:
            raise ValueError("ViTBackbone expected at least one camera image")
        names = list(images)
        for name in names:
            if name not in cameras:
                raise KeyError(f"cameras is missing {name!r}")
        per_camera: list[Tensor] = []
        layers: dict[str, dict[int, Tensor]] = {}
        grids: dict[str, tuple[int, int]] = {}
        for name in names:
            image = images[name]
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{name}: image must be [B, 3, H, W], got {tuple(image.shape)}")
            height, width = image.shape[-2:]
            if height % self.patch_size or width % self.patch_size:
                raise ValueError(
                    f"{name}: image {height}x{width} is not a multiple of patch size "
                    f"{self.patch_size}. Resize or pad before the backbone."
                )
            view = _normalize(image, self.normalize)
            pose = cameras[name]
            bias = self.pose(pose["K"], pose["R"], pose["t"], height, width, self.patch_size)
            tokens, layer_tokens, grid_h, grid_w = self.encoder.forward_tokens(view, bias)
            per_camera.append(tokens)
            layers[name] = {
                index: _to_map(token, grid_h, grid_w) for index, token in layer_tokens.items()
            }
            grids[name] = (grid_h, grid_w)
        mixed = per_camera
        for block in self.cross_view:
            mixed = block(mixed)
        features: dict[str, Tensor] = {}
        multiscale: dict[str, dict[int, Tensor]] = {}
        for name, tokens, (grid_h, grid_w) in zip(names, mixed, grids.values(), strict=True):
            feature_map = _to_map(tokens, grid_h, grid_w)
            features[name] = feature_map
            multiscale[name] = _pyramid(feature_map)
        return BackboneTokens(
            features=features,
            multiscale=multiscale,
            layers=layers,
            names=names,
            patch_size=self.patch_size,
        )


def _normalize(image: Tensor, mode: str) -> Tensor:
    if mode == "none":
        return image
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def _to_map(tokens: Tensor, grid_h: int, grid_w: int) -> Tensor:
    batch, length, channels = tokens.shape
    if length != grid_h * grid_w:
        raise RuntimeError(f"cannot reshape {length} tokens into {grid_h}x{grid_w}")
    return tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)


def _pyramid(feature_map: Tensor) -> dict[int, Tensor]:
    scales = {1: feature_map}
    height, width = feature_map.shape[-2:]
    if height % 2 == 0 and width % 2 == 0 and height >= 2:
        scales[2] = F.avg_pool2d(feature_map, kernel_size=2)
    if height % 4 == 0 and width % 4 == 0 and height >= 4:
        scales[4] = F.avg_pool2d(feature_map, kernel_size=4)
    return scales


# Re-export the block type so a checkpoint loader can see it beside the encoder.
__all__ = ["BackboneTokens", "CrossViewBlock", "ViTBackbone"]
