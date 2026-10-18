"""Model interfaces. Phase 2 is the lifter. Phase 3 packs the cloud. Phase 4 renders it."""

from aerosplat.models.backbone import ViTBackbone
from aerosplat.models.factory import MultiViewLifter, build_lifter, build_model
from aerosplat.models.gaussians import (
    GAUSSIAN_DIM,
    GaussianCloud,
    GaussianLayout,
    pack_gaussians,
    unpack_gaussians,
)
from aerosplat.models.lift import EpipolarLift, LiftedGeometry
from aerosplat.models.rasterizer import GaussianRasterizer, ReferenceSplatRenderer
from aerosplat.models.splats import FeedforwardSplatter, SplatPredictor

__all__ = [
    "GAUSSIAN_DIM",
    "EpipolarLift",
    "FeedforwardSplatter",
    "GaussianCloud",
    "GaussianLayout",
    "GaussianRasterizer",
    "LiftedGeometry",
    "MultiViewLifter",
    "ReferenceSplatRenderer",
    "SplatPredictor",
    "ViTBackbone",
    "build_lifter",
    "build_model",
    "pack_gaussians",
    "unpack_gaussians",
]
