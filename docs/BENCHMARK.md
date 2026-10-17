# Render benchmark on a Windows laptop

This measures how fast Unreal can produce the 4-camera rig: RGB, planar depth,
and segmentation, together and one at a time, at 1080p and at 720p. The number
decides how large a synthetic set you can record. It is not the training speed
of the network. Training memory targets are in `configs/model/` (ViT-S at 224px
for an 8 GB GPU, ViT-B at 224px for 16 GB).

The machine this was written for is an MSI Vector 17 HX laptop: Windows, an
NVIDIA RTX GPU, Unreal Engine installed. The script also runs against any
machine that can launch the sim. It does not start Unreal for you.

You need two programs talking to each other:

1. The simulator (Unreal, with the AirSim plugin), already running with the
   drone in the level.
2. `scripts/benchmark_sim_render.py`, which pauses that sim, steps one tick,
   and times the image request.

## Which sim to install

Two simulators are supported. They share the pause / step / `simGetImages`
calls, and they do not share a Python package or a settings file. Colosseum is
the script default. Cosys-AirSim is the maintained one. Pick one section below
and stay in it: a Colosseum `settings.json` (version 1.2) is refused by
Cosys-AirSim, and `import airsim` does not load the Cosys client.

Do not use the MatrixCity plugin for this benchmark. That plugin is pinned to
Unreal 5.0.3 and renders through Movie Render Queue, which is an offline path.
The benchmark is the live AirSim capture path.

Start with the **Blocks** environment that ships with the sim. It is a few
cubes in an empty world, which is enough to see whether the rig and the script
work. A city map (City Sample, a marketplace city) is heavier: more RAM, lower
frame rate. Benchmark Blocks first, then the map you actually want to record.

## Install Unreal and the plugin

Epic's docs change, so treat this as the order of operations rather than a
pixel-perfect click path.

1. Create an Epic Games account and install the **Epic Games Launcher**.
2. In the launcher, install **Unreal Engine 5.8** if you are on Cosys-AirSim,
   or **5.6** if you are on Colosseum. Use the version the sim's current README
   names if it has moved on. You do not need a source build of the engine for
   the prebuilt Blocks environment; you do need it if you are compiling the
   plugin yourself. The sim's Windows instructions say which one.
3. Install a current **NVIDIA Game Ready or Studio driver**. Open a terminal
   (`PowerShell` or `Command Prompt`) and run `nvidia-smi`. You should see the
   GPU name and a VRAM number (8 GB, 12 GB, 16 GB, …). If the command is not
   found, the driver is not on the PATH yet; install it and open a new terminal.
4. Clone the sim repository from the section you picked and follow its
   **Windows** build or install page. Cosys-AirSim documents a `build.cmd` (or
   the equivalent in that README) that registers the plugin and produces the
   Blocks project. The first build is long. Leave the laptop plugged in.
5. Install that sim's Python client, from the section below, into the
   environment you use for this project. This repo does not pip-depend on
   either client, because CI has no simulator.

### Cosys-AirSim

Maintained. Use this when you are installing a sim from scratch.

* Repository: https://github.com/Cosys-Lab/Cosys-AirSim
* Unreal Engine **5.8** as of October 2026. A plugin built for 5.8 will not
  open in 5.4 or 5.6. If the sim README has moved on, use the version it names.
* License: MIT. The city or map you load has its own license.
* Settings file: **SettingsVersion 2.0**. Each camera is a vehicle mount
  (`External` false, `ExternalLocal` false) and `MoveWorldOrigin` is false.
  Segmentation (image type 5) is an instance color index, not a Colosseum
  mesh-object id.
* Python package: `cosysairsim`. Installing the Cosys client does **not**
  make `import airsim` work, and the benchmark will not try `airsim` in this
  mode. A leftover Colosseum install must not answer a Cosys session.

```powershell
cd Cosys-AirSim\PythonClient
pip install -e .
python -c "import cosysairsim; print('cosysairsim', cosysairsim.__file__)"
```

```powershell
python scripts\benchmark_sim_render.py --simulator cosys --write-settings --output outputs\benchmark
```

That writes `settings_cosys_1080p.json` and `settings_cosys_720p.json`.

### Colosseum

Archived (read-only). This is the script default, for a machine that already
has a Colosseum project. Its GitHub repo will not get fixes. The main branch's
own README says it supports **Unreal Engine 5.6** only.

* Repository: https://github.com/CodexLabsLLC/Colosseum
* Settings file: **SettingsVersion 1.2**. No Cosys `External` or
  `MoveWorldOrigin` flags. Segmentation (image type 5) is an RGB palette of
  mesh-name object ids.
* Python package: the Colosseum `PythonClient` installs as `airsim`. Some
  builds also expose `colosseum`. The benchmark tries `import airsim`, then
  `import colosseum`. It does not import `cosysairsim` in this mode.

```powershell
cd Colosseum\PythonClient
pip install -e .
python -c "import airsim; print('airsim', airsim.__file__)"
```

```powershell
python scripts\benchmark_sim_render.py --simulator colosseum --write-settings --output outputs\benchmark
```

`--simulator colosseum` is the default, so omitting the flag writes the same
files: `settings_colosseum_1080p.json` and `settings_colosseum_720p.json`.

## Where settings.json goes

AirSim, Colosseum, and Cosys-AirSim all read one file at startup:

```
C:\Users\<your name>\Documents\AirSim\settings.json
```

That is the Documents folder of the Windows user, not the Unreal project
folder. If `Documents\AirSim` does not exist, create it.

The file has to be in place **before** you press Play. Changing it while the
game is running does nothing until you stop and press Play again. 1080p and
720p are two separate launches for that reason. The RPC cannot resize a camera
that was created from the old file.

Generate both resolutions from the repo (this does not need the sim running).
Use the flag that matches the sim you installed. See the two sections above
for the exact command. You get a pair named for that simulator:

* `settings_<simulator>_1080p.json` — 1920×1080
* `settings_<simulator>_720p.json` — 1280×720

`<simulator>` is `cosys` or `colosseum`. Copy the file from the sim you are
actually launching. A 1.2 file will not start a Cosys session.

Each one mounts the four cameras from `configs/rig/uam_quad.yaml`:

| Camera | Which way it looks |
| --- | --- |
| front | along the nose |
| down | straight down |
| port | to the vehicle's left |
| starboard | to the vehicle's right |

Horizontal field of view is 90°. Each camera requests three image types: Scene
(RGB), DepthPlanar (distance straight out of the camera, in meters), and
Segmentation (an object id per pixel).

Copy **one** of those files to `Documents\AirSim\settings.json`. To switch
resolution, copy the other file over it and restart Play.

## Launch the simulator

1. Open the Blocks project (the `.uproject` the sim's build step produced) in
   the Unreal Editor. If Windows asks which engine version, pick the one you
   installed above.
2. Wait until the editor finishes loading shaders. The first time on a laptop
   can take several minutes. The GPU fan spinning is normal.
3. You should see the Blocks map in the viewport: a floor and some cubes.
4. Press the green **Play** button (Alt+P). The view switches to the game.
   The drone is spawned because `settings.json` says `AutoCreate`.
5. Leave Play running. Do not close the editor. The Python API listens on
   `127.0.0.1` port **41451**. If a firewall prompt appears, allow it for
   private networks.

If Play starts and Python cannot connect, the usual causes are: Play is not
actually running (you are still in the editor viewport), `settings.json` is in
the wrong folder, or another program already grabbed port 41451.

## Run the benchmark

From the repo, with Play still running:

```powershell
python scripts\benchmark_sim_render.py --simulator cosys --resolution 1080p --output outputs\benchmark
```

Use `--simulator colosseum` instead when Play was started from the Colosseum
settings file. The flag has to match the sim that is running. Then stop Play,
copy `settings_<simulator>_720p.json` onto `Documents\AirSim\settings.json`,
press Play again, and run the same command with `--resolution 720p` and a
separate output folder:

```powershell
python scripts\benchmark_sim_render.py --simulator cosys --resolution 720p --output outputs\benchmark_720p
```

`--resolution both` tries both sizes in one process. The second size will be
marked as a mismatch unless you restarted Play in between, because the sim
cannot change resolution live. A mismatch still writes the report, and the
script exits with status 1 so you notice.

What the script does, for RGB, for planar depth, for segmentation, and for all
three in one request:

1. Pause the simulation.
2. Advance it by one fixed step (`--dt`, default 0.05 s) with
   `simContinueForTime`, which returns to the pause. Every camera sees that
   same tick. That is the frame lock.
3. Time `simGetImages` for the four cameras.
4. Unpause when the run finishes.

Useful flags:

| flag | meaning |
| --- | --- |
| `--frames 20` | timed captures per modality (default 10) |
| `--warmup 2` | captures thrown away before the timing |
| `--ip 127.0.0.1` | the machine running Unreal. Use the laptop's address only if Python is on another computer |
| `--dt 0.05` | simulated seconds per step |
| `--simulator colosseum` | default. Imports `airsim` then `colosseum`, SettingsVersion 1.2 |
| `--simulator cosys` | Imports `cosysairsim` only, SettingsVersion 2.0 and the Cosys camera flags |

## Read the report

The script writes `benchmark.json` and `benchmark.md` in the output folder.

* **fps** is the end-to-end rate: one scripted step plus the image grab. That
  is how fast you can record a frame-locked dataset.
* **grab mean / p50 / p95** are only the `simGetImages` call, in milliseconds.
  That is the render and the copy back to Python.
* The GPU block is `nvidia-smi`: name, total VRAM, VRAM in use, driver. If it
  says nvidia-smi returned nothing, the driver is not visible to the terminal
  you launched Python from.
* `matched: false` means the images came back at a different size than you
  asked for. The timing is still real, but it is not 1080p. Copy the right
  settings file and restart Play.

A Blocks map on an RTX laptop is usually much faster than a city. Record both
if you can. The city number is the one that sizes the 4D dataset.

## If something fails

* `Could not import the cosys client` — `cosysairsim` is not installed in this
  environment. Installing Colosseum's `airsim` package does not fix it. See
  the Cosys-AirSim section above.
* `Could not import the colosseum client` — neither `airsim` nor `colosseum`
  is installed. See the Colosseum section above. Do not point this flag at a
  Cosys install.
* `empty image` or a camera name error — `settings.json` was not the file the
  sim loaded. Check `Documents\AirSim\settings.json` and restart Play.
* The editor crashes on Play — the map is too heavy for the laptop's RAM, or
  the engine version does not match the plugin. Go back to Blocks.
* Segmentation comes back black — image type 5 is still a valid timing
  measurement. Object ids are assigned by the sim; an empty level can look
  dark. The benchmark does not check the colors, only the size and the clock.

## What this does not measure

Training the ViT. An 8 GB GPU should use `configs/model/vit_small.yaml` at
224px, batch size 1, four cameras, gradient checkpointing. A 16 GB GPU can use
`configs/model/vit_base.yaml` at the same 224px. ViT-L (the aerial DUSt3R /
MASt3R width) does not fit that budget unfrozen. None of those configs train
at 1080p.
