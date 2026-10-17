"""Write a Colosseum settings.json and, if that sim is running, record a 4-camera rig.

Colosseum is archived. This script remains for existing setups and defaults to
``--simulator colosseum``. Cosys-AirSim, the maintained option, is
``scripts/capture_cosys_rig.py``.
"""

from aerosplat.capture.colosseum import main

if __name__ == "__main__":
    main()
