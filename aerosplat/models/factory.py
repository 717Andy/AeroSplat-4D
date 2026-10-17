"""Build the Phase 2 lifter from a Hydra model config."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from aerosplat.models.backbone import ViTBackbone
from aerosplat.models.lift import EpipolarLift
from aerosplat.models.pretrained import load_pretrained_weights
from aerosplat.models.splats import FeedforwardSplatter, SplatConfig, SplatPredictor


@dataclass
class ModelConfig:
    name: str = "vit_small"
    img_size: int = 224
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    out_indices: tuple[int, ...] = (3, 7, 11)
    cross_view_layers: int = 2
    pose_encoding: str = "plucker+sinusoidal"
    pose_zero_init: bool = True
    num_frequencies: int = 6
    normalize: str = "imagenet"
    gradient_checkpointing: bool = True
    init_values: float = 1.0
    depth_planes: int = 32
    depth_near: float = 0.5
    depth_far: float = 80.0
    depth_spacing: str = "linear"
    lift_stride: int = 14
    temperature: float = 0.02
    pretrained: str | None = None
    pretrained_allow_download: bool = False


class MultiViewLifter(nn.Module):
    """Backbone plus plane-sweep lift. Phase 3 calls this and reads anchors."""

    def __init__(self, backbone: ViTBackbone, lift: EpipolarLift) -> None:
        super().__init__()
        self.backbone = backbone
        self.lift = lift

    def forward(self, images, cameras, match: str = "hybrid"):
        tokens = self.backbone(images, cameras)
        return self.lift(tokens, cameras, images=images, match=match)


def model_config_from(cfg) -> ModelConfig:
    node = cfg.model if hasattr(cfg, "model") else cfg
    indices = _get(node, "out_indices", (11,))
    pretrained = _get(node, "pretrained", None)
    if pretrained in {"", "null", "none", "None"}:
        pretrained = None
    return ModelConfig(
        name=str(_get(node, "name", "vit_small")),
        img_size=int(_get(node, "img_size", 224)),
        patch_size=int(_get(node, "patch_size", 14)),
        embed_dim=int(_get(node, "embed_dim", 384)),
        depth=int(_get(node, "depth", 12)),
        num_heads=int(_get(node, "num_heads", 6)),
        mlp_ratio=float(_get(node, "mlp_ratio", 4.0)),
        out_indices=tuple(int(i) for i in indices),
        cross_view_layers=int(_get(node, "cross_view_layers", 2)),
        pose_encoding=str(_get(node, "pose_encoding", "plucker+sinusoidal")),
        pose_zero_init=bool(_get(node, "pose_zero_init", True)),
        num_frequencies=int(_get(node, "num_frequencies", 6)),
        normalize=str(_get(node, "normalize", "imagenet")),
        gradient_checkpointing=bool(_get(node, "gradient_checkpointing", True)),
        init_values=float(_get(node, "init_values", 1.0)),
        depth_planes=int(_get(node, "depth_planes", 32)),
        depth_near=float(_get(node, "depth_near", 0.5)),
        depth_far=float(_get(node, "depth_far", 80.0)),
        depth_spacing=str(_get(node, "depth_spacing", "linear")),
        lift_stride=int(_get(node, "lift_stride", 14)),
        temperature=float(_get(node, "temperature", 0.02)),
        pretrained=None if pretrained is None else str(pretrained),
        pretrained_allow_download=bool(_get(node, "pretrained_allow_download", False)),
    )


def build_lifter(cfg) -> MultiViewLifter:
    """Compose a lifter and, when ``pretrained`` is set, load encoder weights."""
    model_cfg = cfg if isinstance(cfg, ModelConfig) else model_config_from(cfg)
    backbone = ViTBackbone(
        img_size=model_cfg.img_size,
        patch_size=model_cfg.patch_size,
        embed_dim=model_cfg.embed_dim,
        depth=model_cfg.depth,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        out_indices=model_cfg.out_indices,
        cross_view_layers=model_cfg.cross_view_layers,
        pose_encoding=model_cfg.pose_encoding,
        pose_zero_init=model_cfg.pose_zero_init,
        num_frequencies=model_cfg.num_frequencies,
        normalize=model_cfg.normalize,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        init_values=model_cfg.init_values,
    )
    lift = EpipolarLift(
        embed_dim=model_cfg.embed_dim,
        num_planes=model_cfg.depth_planes,
        depth_near=model_cfg.depth_near,
        depth_far=model_cfg.depth_far,
        depth_spacing=model_cfg.depth_spacing,
        stride=model_cfg.lift_stride,
        temperature=model_cfg.temperature,
    )
    lifter = MultiViewLifter(backbone, lift)
    if model_cfg.pretrained:
        load_pretrained_weights(
            lifter.backbone,
            model_cfg.pretrained,
            allow_download=model_cfg.pretrained_allow_download,
        )
    return lifter


def splat_config_from(cfg) -> SplatConfig:
    """Read ``model.splats`` from a Hydra config. Missing block -> defaults."""
    if isinstance(cfg, SplatConfig):
        return cfg
    node = cfg.model if hasattr(cfg, "model") else cfg
    splats = _get(node, "splats", None)
    if splats is None:
        return SplatConfig()
    max_extra = _get(splats, "max_extra_gaussians", None)
    if isinstance(max_extra, str) and max_extra.lower() in {"", "null", "none"}:
        max_extra = None
    return SplatConfig(
        slots_per_cell=int(_get(splats, "slots_per_cell", 4)),
        max_extra_gaussians=None if max_extra is None else int(max_extra),
        edge_threshold=float(_get(splats, "edge_threshold", 0.4)),
        offset_radius_m=float(_get(splats, "offset_radius_m", 0.5)),
        along_radius_m=float(_get(splats, "along_radius_m", 0.4)),
        min_scale_m=float(_get(splats, "min_scale_m", 1e-4)),
        max_scale_m=float(_get(splats, "max_scale_m", 2.0)),
        wire_transverse_max_m=float(_get(splats, "wire_transverse_max_m", 0.03)),
        wire_long_max_m=float(_get(splats, "wire_long_max_m", 1.0)),
        base_scale_m=float(_get(splats, "base_scale_m", 0.2)),
        extra_thin_scale_m=float(_get(splats, "extra_thin_scale_m", 0.01)),
        extra_long_scale_m=float(_get(splats, "extra_long_scale_m", 0.25)),
        opacity_init=float(_get(splats, "opacity_init", 0.1)),
    )


def build_splat_predictor(cfg) -> SplatPredictor:
    """Phase 3 heads sized to the ViT width in ``cfg``."""
    model_cfg = cfg if isinstance(cfg, ModelConfig) else model_config_from(cfg)
    return SplatPredictor(model_cfg.embed_dim, splat_config_from(cfg))


def build_model(cfg) -> FeedforwardSplatter:
    """Lifter plus splat predictor. The forward returns a :class:`GaussianCloud`."""
    return FeedforwardSplatter(build_lifter(cfg), build_splat_predictor(cfg))


def _get(node, key: str, default):
    if node is None:
        return default
    try:
        if key not in node:
            return default
        value = node[key]
    except TypeError:
        value = getattr(node, key, default)
    if value is None:
        return default
    return value
