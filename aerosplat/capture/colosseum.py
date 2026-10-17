"""Colosseum capture entry point.

Colosseum (https://github.com/CodexLabsLLC/Colosseum) is archived and read-only.
This module stays so existing recordings and ``scripts/capture_colosseum_rig.py``
keep working. New captures should use Cosys-AirSim (``--simulator cosys`` or
``scripts/capture_cosys_rig.py``). Both simulators share
:mod:`aerosplat.capture.airsim`.
"""

from aerosplat.capture.airsim import build_settings, capture_episode, main

__all__ = ["build_settings", "capture_episode", "main"]
