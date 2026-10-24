"""Compact binary layouts for the workbench.

Gaussian stream (``AGS1``), little-endian::

    offset  size  field
    0       4     magic ``AGS1``
    4       2     version uint16 = 1
    6       2     flags uint16 (bit 0 = float16 body)
    8       4     batch uint32
    12      4     count uint32  (N)
    16      2     channels uint16 = 62
    18      2     sh_coeffs uint16 = 16
    20      4     timestamp float32 (seconds)
    24      4     dt float32 (seconds toward the next frame; 0 if unknown)
    28      4     header_bytes uint32 = 64
    32      12    source sun xyz float32, flight frame (the SH lobe's sun)
    44      20    reserved zeros

Body, tightly packed, C order:

    packed     float16[B, N, 62]
    valid      uint8[B, N]
    wire_prob  float16[B, N]

Channel order is :meth:`aerosplat.models.gaussians.GaussianLayout.channel_names`.
SH inside the 62 is coefficient-major then RGB, ``packed[..., 11:59]`` as
``[16, 3]``.

Occupancy stream (``AOC1``), little-endian, sparse cells only::

    offset  size  field
    0       4     magic ``AOC1``
    4       2     version uint16 = 1
    6       2     flags uint16 (bit 0 = collision fields are valid)
    8       12    origin xyz float32 (flight frame, voxel (0,0,0) corner)
    20      4     voxel_size float32 (meters)
    24      12    dims uint32[3] (X, Y, Z)
    36      4     cell_count uint32
    40      4     risk float32
    44      4     nearest_distance_m float32 (IEEE inf when nothing is occupied)
    48      4     wire_risk float32
    52      4     solid_risk float32
    56      4     min_prob float32 (cells at or below this were dropped)
    60      4     reserved

Each cell is 24 bytes:

    i, j, k        uint16
    occupancy      float16 probability
    wire           float16 probability
    direction xyz  float16 unit tangent
    offset xyz     float16, voxel units in [-0.5, 0.5]
    pad            uint16 = 0

``i`` is the X index. A wire segment for drawing is the cell center plus
``offset * voxel_size``, extended by ``0.5 * voxel_size * sqrt(3)`` along
``direction``. That is the segment :func:`aerosplat.models.collision.query_collision`
measures.
"""

from __future__ import annotations

import struct

import numpy as np

from aerosplat.models.gaussians import GaussianLayout

GAUSSIAN_MAGIC = b"AGS1"
OCCUPANCY_MAGIC = b"AOC1"
GAUSSIAN_HEADER_BYTES = 64
OCCUPANCY_HEADER_BYTES = 64
GAUSSIAN_VERSION = 1
OCCUPANCY_VERSION = 1
CELL_BYTES = 24

_CELL_DTYPE = np.dtype(
    [
        ("i", "<u2"),
        ("j", "<u2"),
        ("k", "<u2"),
        ("occupancy", "<f2"),
        ("wire", "<f2"),
        ("dx", "<f2"),
        ("dy", "<f2"),
        ("dz", "<f2"),
        ("ox", "<f2"),
        ("oy", "<f2"),
        ("oz", "<f2"),
        ("pad", "<u2"),
    ]
)


def gaussian_layout_names() -> list[str]:
    return list(GaussianLayout.channel_names())


def pack_gaussians(
    packed: np.ndarray,
    valid: np.ndarray,
    wire_prob: np.ndarray,
    *,
    timestamp: float = 0.0,
    dt: float = 0.0,
    source_sun: np.ndarray | None = None,
) -> bytes:
    """Pack ``[B, N, 62]`` plus the side channels into an ``AGS1`` blob."""
    array = np.asarray(packed, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != GaussianLayout.DIM:
        raise ValueError(f"packed must be [B, N, {GaussianLayout.DIM}], got {array.shape}")
    batch, count, _ = array.shape
    mask = np.asarray(valid)
    if mask.shape != (batch, count):
        raise ValueError(f"valid must be {(batch, count)}, got {mask.shape}")
    wire = np.asarray(wire_prob, dtype=np.float32)
    if wire.shape != (batch, count):
        raise ValueError(f"wire_prob must be {(batch, count)}, got {wire.shape}")
    if not np.isfinite(array).all() or not np.isfinite(wire).all():
        raise ValueError("gaussian stream contains non-finite values")
    sun = (
        np.zeros(3, dtype=np.float32)
        if source_sun is None
        else np.asarray(source_sun, dtype=np.float32)
    )
    if sun.shape != (3,):
        raise ValueError(f"source_sun must have shape (3,), got {sun.shape}")
    header = bytearray(GAUSSIAN_HEADER_BYTES)
    header[0:4] = GAUSSIAN_MAGIC
    struct.pack_into(
        "<HHIIHH",
        header,
        4,
        GAUSSIAN_VERSION,
        1,
        int(batch),
        int(count),
        GaussianLayout.DIM,
        GaussianLayout.SH_COEFFS,
    )
    struct.pack_into("<ffI", header, 20, float(timestamp), float(dt), GAUSSIAN_HEADER_BYTES)
    struct.pack_into("<fff", header, 32, float(sun[0]), float(sun[1]), float(sun[2]))
    body = b"".join(
        (
            np.ascontiguousarray(array.astype("<f2")).tobytes(),
            np.ascontiguousarray(mask.astype(np.uint8)).tobytes(),
            np.ascontiguousarray(wire.astype("<f2")).tobytes(),
        )
    )
    return bytes(header) + body


def unpack_gaussians(blob: bytes) -> dict[str, np.ndarray | float | int]:
    """Inverse of :func:`pack_gaussians`. Arrays are float32 / bool."""
    if len(blob) < GAUSSIAN_HEADER_BYTES:
        raise ValueError("gaussian stream is shorter than the header")
    if blob[0:4] != GAUSSIAN_MAGIC:
        raise ValueError(f"bad gaussian magic {blob[0:4]!r}")
    version, flags, batch, count, channels, sh_coeffs = struct.unpack_from("<HHIIHH", blob, 4)
    timestamp, dt, header_bytes = struct.unpack_from("<ffI", blob, 20)
    sun = struct.unpack_from("<fff", blob, 32)
    if version != GAUSSIAN_VERSION:
        raise ValueError(f"unsupported gaussian version {version}")
    if channels != GaussianLayout.DIM or sh_coeffs != GaussianLayout.SH_COEFFS:
        raise ValueError(f"unexpected layout channels={channels} sh={sh_coeffs}")
    if header_bytes != GAUSSIAN_HEADER_BYTES:
        raise ValueError(f"unexpected header length {header_bytes}")
    need = batch * count * channels * 2 + batch * count + batch * count * 2
    body = blob[header_bytes:]
    if len(body) != need:
        raise ValueError(f"gaussian body is {len(body)} bytes, expected {need}")
    offset = 0
    packed_n = batch * count * channels
    packed = np.frombuffer(body, dtype="<f2", count=packed_n, offset=offset).astype(np.float32)
    packed = packed.reshape(batch, count, channels)
    offset += packed_n * 2
    valid = np.frombuffer(body, dtype=np.uint8, count=batch * count, offset=offset).reshape(
        batch, count
    )
    offset += batch * count
    wire = np.frombuffer(body, dtype="<f2", count=batch * count, offset=offset).astype(np.float32)
    wire = wire.reshape(batch, count)
    return {
        "packed": packed.copy(),
        "valid": valid.astype(bool).copy(),
        "wire_prob": wire.copy(),
        "timestamp": float(timestamp),
        "dt": float(dt),
        "source_sun": np.asarray(sun, dtype=np.float32),
        "flags": int(flags),
        "batch": int(batch),
        "count": int(count),
    }


def pack_occupancy(
    origin: np.ndarray,
    voxel_size: float,
    occupancy: np.ndarray,
    wire: np.ndarray,
    direction: np.ndarray,
    offset: np.ndarray,
    *,
    min_prob: float = 0.5,
    collision: dict[str, float] | None = None,
) -> bytes:
    """Sparse ``AOC1`` stream.

    ``occupancy`` and ``wire`` are probabilities ``[X, Y, Z]``. ``direction``
    and ``offset`` are ``[X, Y, Z, 3]``. Cells with both probabilities at or
    below ``min_prob`` are omitted.
    """
    occ = np.asarray(occupancy, dtype=np.float32)
    cab = np.asarray(wire, dtype=np.float32)
    direc = np.asarray(direction, dtype=np.float32)
    off = np.asarray(offset, dtype=np.float32)
    if occ.ndim != 3 or cab.shape != occ.shape:
        raise ValueError("occupancy and wire must be [X, Y, Z] and match")
    if direc.shape != occ.shape + (3,) or off.shape != occ.shape + (3,):
        raise ValueError("direction and offset must be [X, Y, Z, 3]")
    origin_v = np.asarray(origin, dtype=np.float32).reshape(3)
    keep = (occ > min_prob) | (cab > min_prob)
    coords = np.argwhere(keep)
    cells = np.zeros(coords.shape[0], dtype=_CELL_DTYPE)
    for index, (ix, iy, iz) in enumerate(coords):
        cells[index]["i"] = int(ix)
        cells[index]["j"] = int(iy)
        cells[index]["k"] = int(iz)
        cells[index]["occupancy"] = np.float16(occ[ix, iy, iz])
        cells[index]["wire"] = np.float16(cab[ix, iy, iz])
        cells[index]["dx"] = np.float16(direc[ix, iy, iz, 0])
        cells[index]["dy"] = np.float16(direc[ix, iy, iz, 1])
        cells[index]["dz"] = np.float16(direc[ix, iy, iz, 2])
        cells[index]["ox"] = np.float16(off[ix, iy, iz, 0])
        cells[index]["oy"] = np.float16(off[ix, iy, iz, 1])
        cells[index]["oz"] = np.float16(off[ix, iy, iz, 2])
    header = bytearray(OCCUPANCY_HEADER_BYTES)
    header[0:4] = OCCUPANCY_MAGIC
    struct.pack_into("<HH", header, 4, OCCUPANCY_VERSION, 1 if collision else 0)
    struct.pack_into(
        "<fff",
        header,
        8,
        float(origin_v[0]),
        float(origin_v[1]),
        float(origin_v[2]),
    )
    dims = occ.shape
    struct.pack_into(
        "<fIII",
        header,
        20,
        float(voxel_size),
        int(dims[0]),
        int(dims[1]),
        int(dims[2]),
    )
    risk = nearest = wire_risk = solid_risk = 0.0
    if collision:
        risk = float(collision.get("risk", 0.0))
        nearest = float(collision.get("nearest_distance_m", float("inf")))
        wire_risk = float(collision.get("wire_risk", 0.0))
        solid_risk = float(collision.get("solid_risk", 0.0))
    struct.pack_into(
        "<Ifffff",
        header,
        36,
        int(cells.shape[0]),
        risk,
        nearest,
        wire_risk,
        solid_risk,
        float(min_prob),
    )
    return bytes(header) + cells.tobytes()


def unpack_occupancy(blob: bytes) -> dict[str, object]:
    """Inverse of :func:`pack_occupancy`. Cells are a structured numpy array."""
    if len(blob) < OCCUPANCY_HEADER_BYTES:
        raise ValueError("occupancy stream is shorter than the header")
    if blob[0:4] != OCCUPANCY_MAGIC:
        raise ValueError(f"bad occupancy magic {blob[0:4]!r}")
    version, flags = struct.unpack_from("<HH", blob, 4)
    if version != OCCUPANCY_VERSION:
        raise ValueError(f"unsupported occupancy version {version}")
    origin = struct.unpack_from("<fff", blob, 8)
    voxel, dx, dy, dz = struct.unpack_from("<fIII", blob, 20)
    count, risk, nearest, wire_risk, solid_risk, min_prob = struct.unpack_from("<Ifffff", blob, 36)
    body = blob[OCCUPANCY_HEADER_BYTES:]
    if len(body) != count * CELL_BYTES:
        raise ValueError(f"occupancy body is {len(body)} bytes, expected {count * CELL_BYTES}")
    cells = np.frombuffer(body, dtype=_CELL_DTYPE, count=count).copy()
    collision = None
    if flags & 1:
        collision = {
            "risk": float(risk),
            "nearest_distance_m": float(nearest),
            "wire_risk": float(wire_risk),
            "solid_risk": float(solid_risk),
        }
    return {
        "origin": np.asarray(origin, dtype=np.float32),
        "voxel_size": float(voxel),
        "dims": (int(dx), int(dy), int(dz)),
        "cells": cells,
        "min_prob": float(min_prob),
        "collision": collision,
        "flags": int(flags),
    }


def frame_infer_blob(meta_json: bytes, gaussians: bytes) -> bytes:
    """``uint32 meta_len`` + UTF-8 JSON + ``AGS1`` bytes."""
    if len(meta_json) > 2**32 - 1:
        raise ValueError("infer metadata is too large")
    return struct.pack("<I", len(meta_json)) + meta_json + gaussians


def split_infer_blob(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 4:
        raise ValueError("infer blob is empty")
    (length,) = struct.unpack_from("<I", blob, 0)
    end = 4 + length
    if end > len(blob):
        raise ValueError("infer metadata length overruns the blob")
    return blob[4:end], blob[end:]
