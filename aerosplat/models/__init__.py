"""Model interfaces. Phase 2 is the ViT and the plane-sweep lift."""

from aerosplat.models.backbone import ViTBackbone
from aerosplat.models.factory import MultiViewLifter, build_lifter
from aerosplat.models.gaussians import (
    GAUSSIAN_DIM,
    GaussianLayout,
    pack_gaussians,
    unpack_gaussians,
)
from aerosplat.models.lift import EpipolarLift, LiftedGeometry

__all__ = [
    "GAUSSIAN_DIM",
    "EpipolarLift",
    "GaussianLayout",
    "LiftedGeometry",
    "MultiViewLifter",
    "ViTBackbone",
    "build_lifter",
    "pack_gaussians",
    "unpack_gaussians",
]
