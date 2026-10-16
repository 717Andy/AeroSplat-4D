"""Dataset adapters. Each one emits camera streams of :class:`FrameRecord`."""

from aerosplat.data.adapters.airsim import read_airsim_raw
from aerosplat.data.adapters.canonical import read_canonical, write_canonical
from aerosplat.data.adapters.tartanair import read_tartanair

__all__ = [
    "read_airsim_raw",
    "read_canonical",
    "read_tartanair",
    "write_canonical",
]
