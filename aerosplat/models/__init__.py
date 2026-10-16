"""Model interfaces. Phase 1 ships the tensor contract and stubs only."""

from aerosplat.models.gaussians import (
    GAUSSIAN_DIM,
    GaussianLayout,
    pack_gaussians,
    unpack_gaussians,
)

__all__ = [
    "GAUSSIAN_DIM",
    "GaussianLayout",
    "pack_gaussians",
    "unpack_gaussians",
]
