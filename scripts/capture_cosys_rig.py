"""Write a Cosys-AirSim settings.json and, if that sim is running, record a 4-camera rig.

Cosys-AirSim (https://github.com/Cosys-Lab/Cosys-AirSim) is the maintained
AirSim-family simulator. The Python client is ``cosysairsim`` and is not
installed in CI. ``--dry-run`` only writes settings.json.
"""

import sys

from aerosplat.capture.airsim import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--simulator" not in argv:
        argv = ["--simulator", "cosys", *argv]
    main(argv)
