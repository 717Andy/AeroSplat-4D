"""Resolution changes that keep intrinsics consistent with the image."""

from __future__ import annotations

from dataclasses import replace

import torch.nn.functional as F

from aerosplat.geometry.camera import scale_intrinsics
from aerosplat.schema import CameraFrame, MultiCameraSample


def resize_camera_frame(frame: CameraFrame, height: int, width: int) -> CameraFrame:
    """Resize RGB bilinearly and depth with nearest neighbors, and scale K.

    Nearest depth avoids blending a hit with a miss into a flying pixel. The
    scaled K matches a uniform stretch of the image plane (``align_corners``
    is left at PyTorch's default, which is the same stretch).
    """
    if height == frame.height and width == frame.width:
        return frame
    image = F.interpolate(
        frame.image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    depth = None
    if frame.depth is not None:
        depth = (
            F.interpolate(
                frame.depth.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
        )
    K = scale_intrinsics(
        frame.K.detach().cpu().numpy(),
        (frame.height, frame.width),
        (height, width),
    )
    segmentation = _resize_segmentation(frame.segmentation, height, width)
    clean = None
    if frame.clean_image is not None:
        clean = F.interpolate(
            frame.clean_image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    velocity = _resize_map(frame.velocity, height, width)
    instance = _resize_instance(frame.instance, height, width)
    return replace(
        frame,
        image=image,
        K=K,
        depth=depth,
        segmentation=segmentation,
        clean_image=clean,
        velocity=velocity,
        instance=instance,
    )


def resize_sample(
    sample: MultiCameraSample, sizes: dict[str, tuple[int, int]]
) -> MultiCameraSample:
    """Resize named cameras. ``sizes`` maps camera name -> ``(height, width)``."""
    cameras = {}
    for name, frame in sample.cameras.items():
        if name in sizes:
            height, width = sizes[name]
            cameras[name] = resize_camera_frame(frame, height, width)
        else:
            cameras[name] = frame
    return replace(sample, cameras=cameras)


def _resize_map(velocity, height: int, width: int):
    """Nearest, so a moving actor is not blended with a static neighbor."""
    if velocity is None:
        return None
    return F.interpolate(velocity.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)


def _resize_instance(instance, height: int, width: int):
    if instance is None:
        return None
    resized = F.interpolate(
        instance.float().unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).to(dtype=instance.dtype)


def _resize_segmentation(segmentation, height: int, width: int):
    if segmentation is None:
        return None
    if segmentation.ndim == 2:
        resized = F.interpolate(
            segmentation.float().unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )
        return resized.squeeze(0).squeeze(0).to(dtype=segmentation.dtype)
    resized = F.interpolate(
        segmentation.float().unsqueeze(0),
        size=(height, width),
        mode="nearest",
    )
    return resized.squeeze(0).to(dtype=segmentation.dtype)
