"""Generate a synthetic multi-camera dataset from a scenario YAML file.

Colosseum is the default simulator. Cosys-AirSim is ``--simulator cosys``.
``--mock`` writes the same on-disk format without Unreal, which is the first
check on a new machine. The Windows steps are in ``docs/SYNTHETIC.md``.
"""

from aerosplat.synth.batch import main

if __name__ == "__main__":
    main()
