"""Time a running Colosseum or Cosys-AirSim 4-camera rig.

See docs/BENCHMARK.md for the Windows / Unreal setup. This file is the entry
point; the measurement lives in aerosplat.capture.benchmark so tests can drive
a fake client without a simulator.
"""

from aerosplat.capture.benchmark import main

if __name__ == "__main__":
    main()
