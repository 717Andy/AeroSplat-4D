"""Dataset adapters. Each one emits camera streams or pretraining view groups."""

from aerosplat.data.adapters.aerial_megadepth import load_aerial_megadepth
from aerosplat.data.adapters.airsim import read_airsim_raw
from aerosplat.data.adapters.canonical import read_canonical, write_canonical
from aerosplat.data.adapters.matrixcity import load_matrixcity
from aerosplat.data.adapters.tartanair import read_tartanair
from aerosplat.data.adapters.tartanair_v2 import load_tartanair_v2

__all__ = [
    "load_aerial_megadepth",
    "load_matrixcity",
    "load_tartanair_v2",
    "read_airsim_raw",
    "read_canonical",
    "read_tartanair",
    "write_canonical",
]
