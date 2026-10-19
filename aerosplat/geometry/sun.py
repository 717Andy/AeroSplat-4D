"""Sun direction in the flight frame.

Glare is a camera artifact, so the relight head needs the sun in the camera
that took the picture. Two sources are supported:

* Ephemeris. A UTC timestamp plus latitude and longitude, turned into a unit
  vector in ENU (X east, Y north, Z up) by a compact solar-position formula.
  That vector is accurate to a fraction of a degree, which is enough to place
  a glare lobe. It is not a spacecraft ephemeris.
* An image estimate, used when the time or the location is missing. The
  brightest pixel is unprojected with ``K`` and rotated into the flight frame
  by ``R^T``. A white roof or a cloud deck can win that argmax. Callers should
  prefer the ephemeris when they have it.

``R`` is world-to-camera, so a direction transforms as ``d_cam = R @ d_world``.
OpenCV camera axes: X right, Y down, Z forward. A sun in front of the lens has
positive camera Z.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

# Timestamps below this are treated as clip-relative seconds, not Unix time.
_UNIX_FLOOR = 1_000_000_000.0


def sun_direction_enu(
    unix_seconds: float,
    latitude_deg: float,
    longitude_deg: float,
) -> np.ndarray:
    """Unit vector toward the sun in ENU. ``unix_seconds`` is UTC.

    The formula is the usual low-precision solar position: mean longitude,
    equation of center, obliquity, hour angle, then altitude and azimuth.
    Azimuth is measured from north toward east, which is Y toward X in ENU.
    """
    if not math.isfinite(unix_seconds):
        raise ValueError("unix_seconds must be finite")
    if not -90.0 <= float(latitude_deg) <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {latitude_deg}")
    if not -180.0 <= float(longitude_deg) <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude_deg}")
    julian = float(unix_seconds) / 86400.0 + 2440587.5
    day = julian - 2451545.0
    mean_longitude = math.radians((280.460 + 0.9856474 * day) % 360.0)
    anomaly = math.radians((357.528 + 0.9856003 * day) % 360.0)
    ecliptic = mean_longitude + math.radians(
        1.915 * math.sin(anomaly) + 0.020 * math.sin(2.0 * anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * day)
    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic),
        math.cos(ecliptic),
    )
    declination = math.asin(math.sin(obliquity) * math.sin(ecliptic))
    gmst = math.radians((280.46061837 + 360.98564736629 * day) % 360.0)
    hour_angle = gmst + math.radians(float(longitude_deg)) - right_ascension
    latitude = math.radians(float(latitude_deg))
    sin_alt = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(
        declination
    ) * math.cos(hour_angle)
    altitude = math.asin(max(-1.0, min(1.0, sin_alt)))
    cos_alt = math.cos(altitude)
    denom = cos_alt * math.cos(latitude)
    if abs(denom) < 1e-8:
        azimuth = 0.0
    else:
        cos_az = (math.sin(declination) - math.sin(altitude) * math.sin(latitude)) / denom
        azimuth = math.acos(max(-1.0, min(1.0, cos_az)))
        if math.sin(hour_angle) > 0.0:
            azimuth = 2.0 * math.pi - azimuth
    vector = np.array(
        [
            cos_alt * math.sin(azimuth),
            cos_alt * math.cos(azimuth),
            math.sin(altitude),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise RuntimeError("sun vector vanished; check the timestamp and the location")
    return vector / norm


def sun_in_camera(sun_world: np.ndarray, rotation_w2c: np.ndarray) -> np.ndarray:
    """Rotate an ENU sun vector into an OpenCV camera. ``rotation_w2c`` is ``R``."""
    sun = np.asarray(sun_world, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_w2c, dtype=np.float64).reshape(3, 3)
    camera = rotation @ sun
    norm = float(np.linalg.norm(camera))
    if norm < 1e-8:
        raise ValueError("camera sun vector vanished")
    return camera / norm


def looks_like_unix(timestamp: float) -> bool:
    """True when ``timestamp`` is large enough to be a UTC Unix second."""
    return math.isfinite(timestamp) and float(timestamp) > _UNIX_FLOOR


def estimate_sun_from_image(image: Tensor, intrinsics: Tensor, rotation_w2c: Tensor) -> Tensor:
    """World-frame sun guessed from the brightest pixel.

    ``image`` is ``[B, 3, H, W]`` or ``[3, H, W]``. ``intrinsics`` and
    ``rotation_w2c`` are ``[B, 3, 3]`` or ``[3, 3]``. The return is ``[B, 3]``
    in the flight frame. The argmax is detached: this is an input, not a
    prediction.
    """
    single = image.ndim == 3
    if single:
        image = image.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
        rotation_w2c = rotation_w2c.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
    # A single camera matrix applies to every image in the batch.
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(image.shape[0], -1, -1)
    if rotation_w2c.ndim == 2:
        rotation_w2c = rotation_w2c.unsqueeze(0).expand(image.shape[0], -1, -1)
    luminance = 0.2126 * image[:, 0] + 0.7152 * image[:, 1] + 0.0722 * image[:, 2]
    batch, _, width = luminance.shape[0], luminance.shape[1], luminance.shape[2]
    index = luminance.reshape(batch, -1).argmax(dim=1)
    row = torch.div(index, width, rounding_mode="floor").to(dtype=image.dtype)
    col = (index - row.long() * width).to(dtype=image.dtype)
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    ones = torch.ones_like(fx)
    direction = torch.stack(((col - cx) / fx, (row - cy) / fy, ones), dim=-1)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    world = torch.matmul(rotation_w2c.transpose(1, 2), direction.unsqueeze(-1)).squeeze(-1)
    world = world / world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return world.squeeze(0) if single else world


def resolve_sun_world(
    timestamp: float,
    latitude_deg: float | None,
    longitude_deg: float | None,
    image: Tensor,
    intrinsics: Tensor,
    rotation_w2c: Tensor,
) -> tuple[Tensor, str]:
    """Ephemeris when time and location are present, otherwise the image estimate.

    Returns ``(sun_world, source)`` with ``source`` in ``{"ephemeris", "estimated"}``.
    ``sun_world`` has the same batch shape as ``image``.
    """
    if latitude_deg is not None and longitude_deg is not None and looks_like_unix(timestamp):
        vector = sun_direction_enu(timestamp, latitude_deg, longitude_deg)
        sun = torch.tensor(vector, dtype=image.dtype, device=image.device)
        if image.ndim == 4:
            sun = sun.view(1, 3).expand(image.shape[0], 3).contiguous()
        return sun, "ephemeris"
    return estimate_sun_from_image(image, intrinsics, rotation_w2c), "estimated"
