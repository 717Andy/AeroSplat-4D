"""Minimal OpenEXR scanline reader and writer.

MatrixCity aerial depth is a float32 EXR (ZIP or uncompressed). CI does not
install the OpenEXR Python package, so this module speaks the scanline subset
those files use: one part, not tiled, compression NONE, ZIPS, or ZIP, channels
stored as float32 or float16.

Scanlines are planar: within each line the channels follow the header order,
and each channel stores the whole row. The ZIP shuffle matches OpenEXR's
``ImfZip``: even bytes are packed in front of the odd bytes, then a +128
delta predictor, then zlib.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_MAGIC = b"\x76\x2f\x31\x01"
_COMPRESSION_NONE = 0
_COMPRESSION_ZIPS = 2
_COMPRESSION_ZIP = 3
_FLOAT = 2
_HALF = 1


class ExrError(ValueError):
    """The file is not a scanline EXR this reader can decode."""


def read_exr(path: Path) -> dict[str, np.ndarray]:
    """Read float channels. Each array is float32 ``(H, W)``."""
    data = Path(path).read_bytes()
    if data[:4] != _MAGIC:
        raise ExrError(f"{path} is not an OpenEXR file")
    version = struct.unpack_from("<I", data, 4)[0]
    if version & 0x200:
        raise ExrError(f"{path} is tiled; re-save it as a scanline EXR")
    if version & 0x1000:
        raise ExrError(f"{path} is multipart; this reader expects one part")
    attrs, offset = _read_header(data, 8)
    channels = _parse_chlist(attrs.get("channels"))
    if not channels:
        raise ExrError(f"{path} has no channels")
    x_min, y_min, x_max, y_max = _parse_box2i(attrs.get("dataWindow"))
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        raise ExrError(f"{path} has an empty data window")
    compression = _parse_u8(attrs.get("compression"), default=0)
    if compression not in {_COMPRESSION_NONE, _COMPRESSION_ZIPS, _COMPRESSION_ZIP}:
        raise ExrError(
            f"{path} uses EXR compression code {compression}. "
            "Re-save as ZIP, ZIPS, or uncompressed. MatrixCity float32 depth is ZIP."
        )
    for channel in channels:
        if channel["x_sampling"] != 1 or channel["y_sampling"] != 1:
            raise ExrError(f"{path} channel {channel['name']} is subsampled")
        if channel["pixel_type"] not in {_HALF, _FLOAT}:
            raise ExrError(f"{path} channel {channel['name']} is not float or half")
    lines_per_block = 16 if compression == _COMPRESSION_ZIP else 1
    n_blocks = (height + lines_per_block - 1) // lines_per_block
    offsets = [struct.unpack_from("<Q", data, offset + 8 * i)[0] for i in range(n_blocks)]
    bytes_per_pixel = sum(2 if c["pixel_type"] == _HALF else 4 for c in channels)
    raw_rows = [b""] * height
    for block, chunk_at in enumerate(offsets):
        y, packed_size = struct.unpack_from("<ii", data, chunk_at)
        packed = data[chunk_at + 8 : chunk_at + 8 + packed_size]
        row0 = y - y_min
        n_lines = min(lines_per_block, height - row0)
        raw_size = n_lines * width * bytes_per_pixel
        if compression == _COMPRESSION_NONE:
            raw = packed
        else:
            raw = _zip_decompress(packed, raw_size)
        if len(raw) != raw_size:
            raise ExrError(f"{path} block {block} decoded to {len(raw)} bytes, expected {raw_size}")
        row_bytes = width * bytes_per_pixel
        for line in range(n_lines):
            raw_rows[row0 + line] = raw[line * row_bytes : (line + 1) * row_bytes]
    planes = {channel["name"]: np.empty((height, width), dtype=np.float32) for channel in channels}
    for y, row in enumerate(raw_rows):
        if len(row) != width * bytes_per_pixel:
            raise ExrError(f"{path} is missing scanline {y}")
        # OpenEXR stores a scanline channel-planar, in header order: every
        # sample of channel 0, then every sample of channel 1, and so on.
        cursor = 0
        for channel in channels:
            sample_bytes = 2 if channel["pixel_type"] == _HALF else 4
            count = width * sample_bytes
            chunk = bytes(row[cursor : cursor + count])
            cursor += count
            if channel["pixel_type"] == _HALF:
                values = np.frombuffer(chunk, dtype="<f2").astype(np.float32)
            else:
                values = np.frombuffer(chunk, dtype="<f4").copy()
            planes[channel["name"]][y] = values
    return planes


def read_exr_depth(path: Path, channel: str = "R") -> np.ndarray:
    """Return one channel as float32 ``(H, W)``. Falls back to the first channel."""
    planes = read_exr(path)
    if channel in planes:
        return planes[channel]
    first = next(iter(planes))
    return planes[first]


def write_exr(path: Path, planes: dict[str, np.ndarray], *, compression: str = "none") -> None:
    """Write float32 scanlines. ``compression`` is ``none``, ``zips``, or ``zip``."""
    if not planes:
        raise ExrError("no channels to write")
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in planes.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ExrError("EXR channels must share one HxW shape")
    height, width = next(iter(shapes))
    code = {"none": _COMPRESSION_NONE, "zips": _COMPRESSION_ZIPS, "zip": _COMPRESSION_ZIP}[
        compression
    ]
    names = list(arrays)
    header = _build_header(names, width, height, code)
    lines_per_block = 16 if code == _COMPRESSION_ZIP else 1
    n_blocks = (height + lines_per_block - 1) // lines_per_block
    chunks: list[bytes] = []
    for block in range(n_blocks):
        row0 = block * lines_per_block
        n_lines = min(lines_per_block, height - row0)
        raw = bytearray()
        for y in range(row0, row0 + n_lines):
            for name in names:
                raw += np.ascontiguousarray(arrays[name][y]).astype("<f4", copy=False).tobytes()
        packed = bytes(raw) if code == _COMPRESSION_NONE else _zip_compress(bytes(raw))
        chunks.append(struct.pack("<ii", row0, len(packed)) + packed)
    offset_table_at = len(header)
    offsets: list[int] = []
    cursor = offset_table_at + 8 * n_blocks
    body = bytearray()
    for chunk in chunks:
        offsets.append(cursor)
        body += chunk
        cursor += len(chunk)
    table = b"".join(struct.pack("<Q", value) for value in offsets)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + table + body)


def _build_header(names: list[str], width: int, height: int, compression: int) -> bytes:
    channels = bytearray()
    for name in names:
        channels += name.encode("ascii") + b"\x00"
        channels += struct.pack("<iBxxxii", _FLOAT, 0, 1, 1)
    channels += b"\x00"
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    parts = [
        _attr("channels", "chlist", bytes(channels)),
        _attr("compression", "compression", struct.pack("<B", compression)),
        _attr("dataWindow", "box2i", box),
        _attr("displayWindow", "box2i", box),
        _attr("lineOrder", "lineOrder", struct.pack("<B", 0)),
        _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    ]
    return _MAGIC + struct.pack("<I", 2) + b"".join(parts) + b"\x00"


def _attr(name: str, type_name: str, value: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + type_name.encode("ascii")
        + b"\x00"
        + struct.pack("<i", len(value))
        + value
    )


def _read_header(data: bytes, offset: int) -> tuple[dict[str, tuple[str, bytes]], int]:
    attrs: dict[str, tuple[str, bytes]] = {}
    while True:
        name, offset = _cstring(data, offset)
        if name == "":
            return attrs, offset
        type_name, offset = _cstring(data, offset)
        size = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        attrs[name] = (type_name, data[offset : offset + size])
        offset += size


def _cstring(data: bytes, offset: int) -> tuple[str, int]:
    end = data.index(b"\x00", offset)
    return data[offset:end].decode("ascii"), end + 1


def _parse_chlist(attr: tuple[str, bytes] | None) -> list[dict]:
    if attr is None:
        return []
    raw = attr[1]
    channels = []
    cursor = 0
    while cursor < len(raw):
        end = raw.index(b"\x00", cursor)
        name = raw[cursor:end].decode("ascii")
        cursor = end + 1
        if name == "":
            break
        pixel_type, _linear, x_sampling, y_sampling = struct.unpack_from("<iBxxxii", raw, cursor)
        cursor += 16
        channels.append(
            {
                "name": name,
                "pixel_type": pixel_type,
                "x_sampling": x_sampling,
                "y_sampling": y_sampling,
            }
        )
    return channels


def _parse_box2i(attr: tuple[str, bytes] | None) -> tuple[int, int, int, int]:
    if attr is None:
        raise ExrError("EXR header has no dataWindow")
    return struct.unpack("<iiii", attr[1])


def _parse_u8(attr: tuple[str, bytes] | None, default: int) -> int:
    if attr is None:
        return default
    return attr[1][0]


def _zip_compress(raw: bytes) -> bytes:
    """OpenEXR ``ImfZip``: split even/odd bytes, then a +128 delta, then zlib."""
    return zlib.compress(_apply_predictor(_split_bytes(raw)))


def _zip_decompress(data: bytes, raw_size: int) -> bytes:
    inflated = zlib.decompress(data)
    if len(inflated) < raw_size:
        raise ExrError(f"ZIP block inflated to {len(inflated)} bytes, expected {raw_size}")
    return _interleave_bytes(_undo_predictor(inflated[:raw_size]))


def _split_bytes(raw: bytes) -> bytes:
    half = (len(raw) + 1) // 2
    out = bytearray(len(raw))
    out[:half] = raw[0::2]
    out[half:] = raw[1::2]
    return bytes(out)


def _interleave_bytes(shuffled: bytes) -> bytes:
    half = (len(shuffled) + 1) // 2
    out = bytearray(len(shuffled))
    out[0::2] = shuffled[:half]
    out[1::2] = shuffled[half:]
    return bytes(out)


def _apply_predictor(data: bytes) -> bytes:
    if not data:
        return data
    out = bytearray(data)
    previous = out[0]
    for index in range(1, len(out)):
        original = out[index]
        out[index] = (original - previous + 128) & 0xFF
        previous = original
    return bytes(out)


def _undo_predictor(data: bytes) -> bytes:
    if not data:
        return data
    out = bytearray(data)
    previous = out[0]
    for index in range(1, len(out)):
        encoded = out[index]
        previous = (previous + encoded - 128) & 0xFF
        out[index] = previous
    return bytes(out)
