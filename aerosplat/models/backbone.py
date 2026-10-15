"""Phase 2 interfaces. No weights and no forward implementation in Phase 1."""

from __future__ import annotations

import torch.nn as nn


class ViTBackbone(nn.Module):
    """Shared (or cross-attention) Vision Transformer over the N camera images.

    Planned call::

        tokens = backbone(images, cameras)

    ``images`` is a dict of ``[B, 3, H, W]`` tensors, one per camera, resolutions
    allowed to differ. ``cameras`` carries K, R, t so a later pose-conditioning
    layer can add sinusoidal or learned embeddings of the extrinsics. The
    backbone returns multi-scale tokens; it does not predict Gaussians.
    """

    def forward(self, images, cameras):
        del images, cameras
        raise NotImplementedError("Phase 2: ViT backbone is not implemented")


class EpipolarLift(nn.Module):
    """Cost volume / frustum lift across overlapping fields of view.

    Planned call::

        volume = lift(tokens, cameras)

    Overlap between a forward camera and a side camera is what resolves depth.
    A rig whose cameras do not overlap cannot feed this module useful matches;
    that is a property of the mount, not of the loss.
    """

    def forward(self, tokens, cameras):
        del tokens, cameras
        raise NotImplementedError("Phase 2: epipolar lift is not implemented")
