"""AirSim segmentation palette.

Colosseum paints mesh ids with the PASCAL VOC bit-interleaved colormap: class
1 is ``(128, 0, 0)``. Cosys-AirSim can return a different map from
``simGetSegmentationColorMap``. The generator records the colors it actually
used in ``segmentation_ids.json`` so ``cache.wire_colors`` does not have to
guess.
"""

from __future__ import annotations


def segmentation_color(object_id: int) -> tuple[int, int, int]:
    """RGB in 0–255 for one segmentation id. Ids wrap at 256, matching AirSim."""
    index = int(object_id) % 256
    red = green = blue = 0
    for bit in range(8):
        red |= ((index >> (3 * bit)) & 1) << (7 - bit)
        green |= ((index >> (3 * bit + 1)) & 1) << (7 - bit)
        blue |= ((index >> (3 * bit + 2)) & 1) << (7 - bit)
    return int(red), int(green), int(blue)


def color_table(ids: list[int]) -> dict[int, tuple[int, int, int]]:
    return {int(object_id): segmentation_color(object_id) for object_id in ids}
