"""Analytic projection for ground truth the rasterizer can alias away.

Actor disks and cable centerlines are projected with the same OpenCV
``K`` and world-to-camera ``R|t`` the dataset stores. A cable thinner than a
pixel still gets a coverage fraction. The hard segmentation paint uses a
0.5 pixel threshold, which is the aliasing a real renderer hits; supersampling
renders that hard paint at a finer grid and keeps a wire-colored subpixel
when the block is reduced.
"""

from __future__ import annotations

import numpy as np

from aerosplat.synth.palette import segmentation_color

# A hard rasterizer drops fragments thinner than half a pixel.
HARD_WIRE_PIXELS = 0.5


def paint_sun_flare(
    rgb: np.ndarray,
    sun_camera: np.ndarray,
    intrinsics: np.ndarray,
    *,
    gain: float = 0.9,
) -> np.ndarray:
    """Add a sun-aligned blob. The sun behind the camera leaves the image unchanged.

    This is the documented fallback when the simulator cannot toggle lens
    flare. The blob is not a measured lens PSF. It is a positive additive
    layer so ``paired_glare_target`` has a mask on frames that look toward
    the sun.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    sun = np.asarray(sun_camera, dtype=np.float64).reshape(3)
    if float(sun[2]) <= 0.05:
        return image.copy()
    height, width = image.shape[:2]
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    column = K[0, 0] * sun[0] / sun[2] + K[0, 2]
    row = K[1, 1] * sun[1] / sun[2] + K[1, 2]
    yy, xx = np.mgrid[0:height, 0:width]
    radius = max(width, height) * 0.08
    falloff = np.exp(-((xx - column) ** 2 + (yy - row) ** 2) / (2.0 * radius * radius))
    # A second, wider veil so the layer is not a single saturated pixel.
    veil = 0.25 * np.exp(-((xx - column) ** 2 + (yy - row) ** 2) / (2.0 * (radius * 3.0) ** 2))
    layer = (falloff + veil) * float(gain)
    added = image.astype(np.float32) / 255.0
    added += layer[..., None] * np.array([1.0, 0.95, 0.8], dtype=np.float32)
    return np.clip(added * 255.0, 0, 255).astype(np.uint8)


def project_dynamics(
    actors: list[dict],
    wires: list[dict],
    *,
    intrinsics: np.ndarray,
    rotation_w2c: np.ndarray,
    translation_w2c: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(instance HxW int32, velocity 3xHxW float32, wire_coverage HxW float32)``.

    The closest surface wins. Background instance is 0 and velocity is 0.
    Wire pixels keep the cable id and a zero velocity. Coverage is the
    analytic pixel fraction of the cylinder, including sub-pixel cables.
    """
    instance = np.zeros((height, width), dtype=np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    velocity = np.zeros((3, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(rotation_w2c, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_w2c, dtype=np.float64).reshape(3)
    for actor in actors:
        _paint_sphere(
            instance,
            depth,
            velocity,
            center_enu=np.asarray(actor["position_enu"], dtype=np.float64),
            radius_m=float(actor["radius_m"]),
            object_id=int(actor["segmentation_id"]),
            velocity_enu=np.asarray(actor["velocity_enu"], dtype=np.float64),
            intrinsics=K,
            rotation=rotation,
            translation=translation,
        )
    for wire in wires:
        _paint_wire_analytic(
            instance,
            depth,
            coverage,
            points_enu=np.asarray(wire["points_enu"], dtype=np.float64),
            diameter_m=float(wire["diameter_m"]),
            object_id=int(wire["segmentation_id"]),
            intrinsics=K,
            rotation=rotation,
            translation=translation,
        )
    return instance, velocity, coverage


def render_hard_frame(
    actors: list[dict],
    wires: list[dict],
    *,
    intrinsics: np.ndarray,
    rotation_w2c: np.ndarray,
    translation_w2c: np.ndarray,
    height: int,
    width: int,
    sun_world: np.ndarray,
    wire_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Software stand-in for one paused sim grab.

    RGB is a flat sky. Depth is planar Z. Segmentation is painted only where
    a disk or a cable at least half a pixel wide wins the z-buffer. Thin
    cables are omitted here on purpose.
    """
    sky = _sky_color(sun_world)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = sky
    depth = np.zeros((height, width), dtype=np.float32)
    segmentation = np.zeros((height, width, 3), dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(rotation_w2c, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_w2c, dtype=np.float64).reshape(3)
    for actor in actors:
        _splat_disk(
            rgb,
            depth,
            segmentation,
            zbuffer,
            center_enu=np.asarray(actor["position_enu"], dtype=np.float64),
            radius_m=float(actor["radius_m"]),
            color=segmentation_color(int(actor["segmentation_id"])),
            intrinsics=K,
            rotation=rotation,
            translation=translation,
        )
    for wire in wires:
        _splat_wire_hard(
            rgb,
            depth,
            segmentation,
            zbuffer,
            points_enu=np.asarray(wire["points_enu"], dtype=np.float64),
            diameter_m=float(wire["diameter_m"]),
            color=wire_color,
            intrinsics=K,
            rotation=rotation,
            translation=translation,
        )
    return rgb, depth, segmentation


def downsample_rgb(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    view = _blocks(image, factor)
    return np.clip(view.mean(axis=(1, 3)), 0, 255).astype(np.uint8)


def downsample_depth(depth: np.ndarray, factor: int) -> np.ndarray:
    """Keep the closest positive return so a wire subpixel is not averaged away."""
    if factor == 1:
        return depth
    view = _blocks(depth.astype(np.float32), factor)
    positive = np.where(view > 0, view, np.inf)
    closest = positive.min(axis=(1, 3))
    closest[~np.isfinite(closest)] = 0.0
    return closest.astype(np.float32)


def downsample_segmentation(
    image: np.ndarray,
    factor: int,
    *,
    wire_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """A wire-colored subpixel wins the block. Otherwise the majority non-zero color."""
    if factor == 1:
        return image
    height, width = image.shape[:2]
    out_h, out_w = height // factor, width // factor
    view = image.reshape(out_h, factor, out_w, factor, 3)
    flat = np.moveaxis(view, (1, 3), (2, 3)).reshape(out_h, out_w, factor * factor, 3)
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    wire = None if wire_color is None else np.asarray(wire_color, dtype=np.uint8)
    for row in range(out_h):
        for col in range(out_w):
            pixels = flat[row, col]
            if wire is not None and np.any(np.all(pixels == wire, axis=1)):
                out[row, col] = wire
                continue
            nonzero = pixels[np.any(pixels != 0, axis=1)]
            if len(nonzero) == 0:
                continue
            colors, counts = np.unique(nonzero, axis=0, return_counts=True)
            out[row, col] = colors[int(np.argmax(counts))]
    return out


def wire_pixel_thickness(
    diameter_m: float,
    depth_m: float,
    focal_px: float,
) -> float:
    """Cylinder width in pixels. Below 1 the cable is sub-pixel."""
    if depth_m <= 1e-6:
        return 0.0
    return float(diameter_m) / float(depth_m) * float(focal_px)


def _paint_sphere(
    instance,
    depth,
    velocity,
    *,
    center_enu,
    radius_m,
    object_id,
    velocity_enu,
    intrinsics,
    rotation,
    translation,
) -> None:
    camera = rotation @ center_enu + translation
    if camera[2] <= radius_m:
        return
    column, row, radius_px = _project_radius(camera, radius_m, intrinsics)
    height, width = instance.shape
    c0 = max(int(np.floor(column - radius_px)), 0)
    c1 = min(int(np.ceil(column + radius_px)) + 1, width)
    r0 = max(int(np.floor(row - radius_px)), 0)
    r1 = min(int(np.ceil(row + radius_px)) + 1, height)
    if c0 >= c1 or r0 >= r1:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = (xx - column) ** 2 + (yy - row) ** 2 <= radius_px**2
    closer = camera[2] < depth[r0:r1, c0:c1]
    hit = inside & closer
    if not np.any(hit):
        return
    depth_block = depth[r0:r1, c0:c1]
    depth_block[hit] = camera[2]
    instance_block = instance[r0:r1, c0:c1]
    instance_block[hit] = object_id
    for channel, component in enumerate(velocity_enu):
        velocity_block = velocity[channel, r0:r1, c0:c1]
        velocity_block[hit] = np.float32(component)


def _paint_wire_analytic(
    instance,
    depth,
    coverage,
    *,
    points_enu,
    diameter_m,
    object_id,
    intrinsics,
    rotation,
    translation,
) -> None:
    if len(points_enu) < 2:
        return
    height, width = instance.shape
    focal = float(intrinsics[0, 0])
    for start, end in zip(points_enu[:-1], points_enu[1:], strict=True):
        start_c = rotation @ start + translation
        end_c = rotation @ end + translation
        if start_c[2] <= 1e-3 and end_c[2] <= 1e-3:
            continue
        samples = max(2, int(np.ceil(np.linalg.norm(end - start) * 4.0)))
        for weight in np.linspace(0.0, 1.0, samples):
            camera = (1.0 - weight) * start_c + weight * end_c
            if camera[2] <= 1e-3:
                continue
            column = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
            row = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
            thickness = wire_pixel_thickness(diameter_m, camera[2], focal)
            half = 0.5 * thickness + 0.5
            c0 = max(int(np.floor(column - half)), 0)
            c1 = min(int(np.ceil(column + half)) + 1, width)
            r0 = max(int(np.floor(row - half)), 0)
            r1 = min(int(np.ceil(row + half)) + 1, height)
            if c0 >= c1 or r0 >= r1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            dist = np.sqrt((xx - column) ** 2 + (yy - row) ** 2)
            falloff = np.clip(1.0 - dist / max(half, 1e-6), 0.0, 1.0)
            fraction = np.clip(thickness, 0.0, 1.0) * falloff
            block = coverage[r0:r1, c0:c1]
            update = fraction > block
            block[update] = fraction[update].astype(np.float32)
            closer = (fraction > 0) & (camera[2] < depth[r0:r1, c0:c1])
            if np.any(closer):
                depth[r0:r1, c0:c1][closer] = camera[2]
                instance[r0:r1, c0:c1][closer] = object_id


def _splat_disk(
    rgb,
    depth,
    segmentation,
    zbuffer,
    *,
    center_enu,
    radius_m,
    color,
    intrinsics,
    rotation,
    translation,
):
    camera = rotation @ center_enu + translation
    if camera[2] <= max(radius_m, 1e-3):
        return
    column, row, radius_px = _project_radius(camera, radius_m, intrinsics)
    height, width = depth.shape
    c0 = max(int(np.floor(column - radius_px)), 0)
    c1 = min(int(np.ceil(column + radius_px)) + 1, width)
    r0 = max(int(np.floor(row - radius_px)), 0)
    r1 = min(int(np.ceil(row + radius_px)) + 1, height)
    if c0 >= c1 or r0 >= r1:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1]
    inside = (xx - column) ** 2 + (yy - row) ** 2 <= radius_px**2
    hit = inside & (camera[2] < zbuffer[r0:r1, c0:c1])
    if not np.any(hit):
        return
    zbuffer[r0:r1, c0:c1][hit] = camera[2]
    depth[r0:r1, c0:c1][hit] = np.float32(camera[2])
    segmentation[r0:r1, c0:c1][hit] = np.asarray(color, dtype=np.uint8)
    shade = rgb[r0:r1, c0:c1].astype(np.int16)
    shade[hit] = np.clip(shade[hit] + np.asarray(color, dtype=np.int16) // 3, 0, 255)
    rgb[r0:r1, c0:c1] = shade.astype(np.uint8)


def _splat_wire_hard(
    rgb,
    depth,
    segmentation,
    zbuffer,
    *,
    points_enu,
    diameter_m,
    color,
    intrinsics,
    rotation,
    translation,
):
    del rgb
    if len(points_enu) < 2:
        return
    height, width = depth.shape
    focal = float(intrinsics[0, 0])
    color_u8 = np.asarray(color, dtype=np.uint8)
    for start, end in zip(points_enu[:-1], points_enu[1:], strict=True):
        start_c = rotation @ start + translation
        end_c = rotation @ end + translation
        samples = max(2, int(np.ceil(np.linalg.norm(end - start) * 4.0)))
        for weight in np.linspace(0.0, 1.0, samples):
            camera = (1.0 - weight) * start_c + weight * end_c
            if camera[2] <= 1e-3:
                continue
            thickness = wire_pixel_thickness(diameter_m, camera[2], focal)
            if thickness < HARD_WIRE_PIXELS:
                continue
            column = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
            row = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
            half = 0.5 * thickness
            c0 = max(int(np.floor(column - half)), 0)
            c1 = min(int(np.ceil(column + half)) + 1, width)
            r0 = max(int(np.floor(row - half)), 0)
            r1 = min(int(np.ceil(row + half)) + 1, height)
            if c0 >= c1 or r0 >= r1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            dist = np.sqrt((xx - column) ** 2 + (yy - row) ** 2)
            hit = (dist <= half) & (camera[2] < zbuffer[r0:r1, c0:c1])
            if not np.any(hit):
                continue
            zbuffer[r0:r1, c0:c1][hit] = camera[2]
            depth[r0:r1, c0:c1][hit] = np.float32(camera[2])
            segmentation[r0:r1, c0:c1][hit] = color_u8


def _project_radius(camera: np.ndarray, radius_m: float, intrinsics: np.ndarray):
    column = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
    row = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
    radius_px = max(float(radius_m) / float(camera[2]) * float(intrinsics[0, 0]), 0.5)
    return float(column), float(row), radius_px


def _sky_color(sun_world: np.ndarray) -> tuple[int, int, int]:
    elevation = float(np.clip(np.asarray(sun_world, dtype=np.float64).reshape(3)[2], -1.0, 1.0))
    daylight = np.clip(elevation, 0.0, 1.0)
    red = int(40 + 80 * daylight)
    green = int(60 + 90 * daylight)
    blue = int(90 + 110 * daylight)
    return red, green, blue


def _blocks(image: np.ndarray, factor: int) -> np.ndarray:
    if image.ndim == 2:
        height, width = image.shape
        if height % factor or width % factor:
            raise ValueError(f"cannot downsample {width}x{height} by {factor}")
        return image.reshape(height // factor, factor, width // factor, factor)
    height, width, channels = image.shape
    if height % factor or width % factor:
        raise ValueError(f"cannot downsample {width}x{height} by {factor}")
    return image.reshape(height // factor, factor, width // factor, factor, channels)
