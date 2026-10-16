"""Phase 3 and Phase 5 heads.

The rasterizer tensor is the concatenation of geometry, appearance, and
dynamics. Relighting and occupancy are side outputs: they are not extra
channels in ``[B, N, 62]``. Relight consumes the degree-3 SH coefficients.
Occupancy consumes the lifted feature volume and is supervised by the cached
voxel grid, not by a per-Gaussian channel.
"""

from __future__ import annotations

import torch.nn as nn

# Name -> output fields. Geometry/appearance/dynamics pack into GaussianLayout.
HEAD_OUTPUTS: dict[str, tuple[str, ...]] = {
    "geometry": ("xyz", "scale", "quat"),
    "appearance": ("opacity", "sh"),
    "dynamics": ("velocity",),
    "relight": ("glare_rgb",),
    "occupancy": ("voxel_logits",),
}


class _UnimplementedHead(nn.Module):
    phase = "later"
    head_name = "head"

    def forward(self, features):
        del features
        raise NotImplementedError(
            f"{self.phase}: {self.head_name} head is not implemented. "
            f"Planned outputs: {', '.join(HEAD_OUTPUTS[self.head_name])}."
        )


class GeometryHead(_UnimplementedHead):
    """XYZ, positive scales, and a wxyz quaternion per Gaussian."""

    phase = "Phase 3"
    head_name = "geometry"


class AppearanceHead(_UnimplementedHead):
    """Opacity and degree-3 RGB spherical harmonics (16 x 3)."""

    phase = "Phase 3"
    head_name = "appearance"


class DynamicsHead(_UnimplementedHead):
    """World-frame velocity in meters per second."""

    phase = "Phase 3"
    head_name = "dynamics"


class RelightHead(_UnimplementedHead):
    """Screen-space glare residual predicted from SH and a sun direction.

    Degree-3 SH can represent low-frequency view-dependent color. It does not
    by itself isolate a sharp lens flare. This head is the place that residual
    has to live; packing more SH bands into the 62 would change the rasterizer
    contract and still would not be a flare model.
    """

    phase = "Phase 5"
    head_name = "relight"


class OccupancyHead(_UnimplementedHead):
    """Local voxel logits. Supervise with the cached grid, masking unknown cells."""

    phase = "Phase 5"
    head_name = "occupancy"
