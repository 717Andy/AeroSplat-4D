"""AeroSplat-4D: feedforward 4D Gaussian splatting for urban air mobility."""

from aerosplat.models.gaussians import GAUSSIAN_DIM, GaussianLayout
from aerosplat.schema import CameraFrame, MultiCameraSample

__version__ = "0.3.0"

__all__ = [
    "GAUSSIAN_DIM",
    "CameraFrame",
    "GaussianLayout",
    "MultiCameraSample",
    "__version__",
]
