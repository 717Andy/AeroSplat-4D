"""Resolution changes that keep intrinsics consistent with the image."""

from __future__ import annotations

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
    return CameraFrame(
        name=frame.name,
        image=image,
        K=K,
        R=frame.R,
        t=frame.t,
        timestamp=frame.timestamp,
        depth=depth,
        R_query=frame.R_query,
        t_query=frame.t_query,
        query_timestamp=frame.query_timestamp,
        sync=frame.sync,
        frame_index=frame.frame_index,
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
    return MultiCameraSample(
        scene_id=sample.scene_id,
        sample_index=sample.sample_index,
        timestamp=sample.timestamp,
        cameras=cameras,
        gt_velocity=sample.gt_velocity,
    )
