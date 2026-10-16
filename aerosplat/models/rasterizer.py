"""Rasterizer boundary. The CUDA extension is deliberately not imported.

Phase 4 should bind either ``diff_gaussian_rasterization`` (the library named
in the spec) or a maintained replacement such as ``gsplat``. Both are compiled
CUDA extensions. Importing either here would make CPU tests and CI depend on
a GPU toolchain, so this module only checks the packed-tensor contract and
refuses to run.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from aerosplat.models.gaussians import GaussianLayout, rasterizer_view


class GaussianRasterizer(nn.Module):
    """Project packed Gaussians into each camera.

    Planned call::

        images = rasterizer(packed, cameras)

    ``packed`` is ``[B, N, 62]``. The wrapper strips velocity before the CUDA
    call. ``cameras`` is the collated multi-camera batch (per-name K, R, t,
    and image size). One forward renders every camera in the rig so the
    photometric loss can compare against each view.
    """

    def forward(self, packed: Tensor, cameras) -> Tensor:
        del cameras
        rasterizer_view(packed)
        raise NotImplementedError(
            "Phase 4: bind diff-gaussian-rasterization (or gsplat) here. "
            f"Rasterizer inputs are packed[..., :{GaussianLayout.RASTER_DIM}]; "
            "velocity stays in the packed tensor for the temporal loss only. "
            "This module does not import the CUDA extension."
        )
