"""Synthetic episode generator for Colosseum and Cosys-AirSim.

The simulator client is imported by the scripts that talk to Unreal. Tests
and the dataset loader do not import it.
"""

from aerosplat.synth.scenario import load_scenario, materialize

__all__ = ["load_scenario", "materialize"]
