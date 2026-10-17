"""Frame-locked render benchmark for a Colosseum or Cosys-AirSim rig.

The script is meant for the developer's Windows laptop (an MSI Vector 17 HX
with an NVIDIA RTX GPU and Unreal installed). It does not start the simulator.
It connects to one that is already running, pauses it, steps the clock, and
times ``simGetImages`` for the four-camera rig.

Resolution is whatever ``settings.json`` baked in when the session started.
AirSim does not resize cameras over RPC, so 1080p and 720p are two launches.
``--write-settings`` writes both files. A run whose returned images do not
match the requested size is recorded as a mismatch instead of being labeled
with the wrong resolution.

Image types, classic AirSim numbering:

* 0 Scene (RGB)
* 1 DepthPlanar (camera-frame Z, meters)
* 5 Segmentation

``--simulator`` selects the shared AirSim-family profile. ``colosseum`` (the
default) imports ``airsim`` then ``colosseum`` and writes SettingsVersion 1.2.
``cosys`` imports ``cosysairsim`` only and writes SettingsVersion 2.0 with the
Cosys camera flags. The two clients are not interchangeable: a leftover
Colosseum install must not answer a Cosys session.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from aerosplat.airsim_family import SIMULATORS, get_profile
from aerosplat.capture.airsim import _connect, build_settings
from aerosplat.data.errors import DataError

RGB = 0
DEPTH_PLANAR = 1
SEGMENTATION = 5

MODALITIES: dict[str, tuple[int, ...]] = {
    "rgb": (RGB,),
    "depth_planar": (DEPTH_PLANAR,),
    "segmentation": (SEGMENTATION,),
    "all": (RGB, DEPTH_PLANAR, SEGMENTATION),
}

RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1080p": (1080, 1920),
    "720p": (720, 1280),
}


class Grab(Protocol):
    height: int
    width: int
    image_type: int
    time_stamp: int


class BenchClient(Protocol):
    def pause(self, paused: bool) -> None:
        """Pause or unpause the simulator."""

    def step(self, dt: float) -> None:
        """Advance the paused sim by ``dt`` seconds and return to the pause."""

    def grab(self, requests: list[tuple[str, int]]) -> list:
        """Return one response per ``(camera_name, image_type)`` request."""


@dataclass
class GrabResult:
    """What the timing loop needs from a response. Tests build these directly."""

    height: int
    width: int
    image_type: int
    time_stamp: int = 0


def build_benchmark_settings(
    rig: dict,
    height: int,
    width: int,
    simulator: str = "colosseum",
) -> dict:
    """Rig settings at one resolution, with RGB, planar depth, and segmentation.

    The mount poses stay the ones in ``rig``. Only the capture size changes.
    ``simulator`` is ``colosseum`` (default, SettingsVersion 1.2) or ``cosys``
    (SettingsVersion 2.0, ``External`` false, ``MoveWorldOrigin`` false).
    Segmentation is ImageType 5 on both; the id space differs and is recorded
    by the shared settings builder.
    """
    if height < 1 or width < 1:
        raise ValueError("resolution must be positive")
    adjusted = {
        "vehicle_name": rig.get("vehicle_name", "Drone1"),
        "fov_degrees": rig["fov_degrees"],
        "cameras": {},
    }
    for name, cam in rig["cameras"].items():
        adjusted["cameras"][name] = {**cam, "width": width, "height": height}
    return build_settings(adjusted, simulator, record_segmentation=True)


def query_nvidia_smi() -> dict | None:
    """GPU name and VRAM from ``nvidia-smi``. ``None`` when the tool is absent."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        completed = subprocess.run(
            [
                binary,
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 4:
        return None
    # GPU names do not contain commas. The last three fields are numeric.
    name = ",".join(parts[:-3]).strip()
    try:
        total = float(parts[-3])
        used = float(parts[-2])
    except ValueError:
        return None
    return {
        "name": name,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "driver": parts[-1],
    }


def measure_resolution(
    client: BenchClient,
    cameras: list[str],
    *,
    height: int,
    width: int,
    frames: int = 10,
    warmup: int = 2,
    dt: float = 0.05,
) -> dict:
    """Pause, step, and time each modality. The sim is unpaused afterwards."""
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    runs: dict[str, dict] = {}
    client.pause(True)
    try:
        for name, types in MODALITIES.items():
            requests = [(camera, image_type) for camera in cameras for image_type in types]
            samples = []
            matched = True
            actual = None
            timestamps_aligned = True
            for index in range(warmup + frames):
                client.step(dt)
                started = time.perf_counter()
                responses = client.grab(requests)
                elapsed = time.perf_counter() - started
                if len(responses) != len(requests):
                    raise DataError(
                        f"{name}: sim returned {len(responses)} images, expected {len(requests)}"
                    )
                shapes = [(int(response.height), int(response.width)) for response in responses]
                if any(shape != (height, width) for shape in shapes):
                    matched = False
                actual = shapes[0]
                stamps = [int(getattr(response, "time_stamp", 0)) for response in responses]
                if len(set(stamps)) > 1:
                    timestamps_aligned = False
                if index >= warmup:
                    samples.append(elapsed)
            runs[name] = _summarize(samples, dt=dt, response_count=len(requests))
            runs[name]["matched_resolution"] = matched
            runs[name]["actual_hw"] = list(actual) if actual is not None else None
            runs[name]["timestamps_aligned"] = timestamps_aligned
    finally:
        client.pause(False)
    return {
        "requested_hw": [height, width],
        "label": _label(height, width),
        "cameras": list(cameras),
        "frames": frames,
        "warmup": warmup,
        "step_dt_s": dt,
        "modalities": runs,
        "matched": all(item["matched_resolution"] for item in runs.values()),
    }


def render_markdown(report: dict) -> str:
    """A short human-readable report next to the JSON."""
    lines = ["# Sim render benchmark", ""]
    gpu = report.get("gpu")
    if gpu:
        lines.append(
            f"GPU: {gpu['name']} ({gpu['memory_total_mib']:.0f} MiB total, "
            f"{gpu['memory_used_mib']:.0f} MiB in use, driver {gpu['driver']})"
        )
    else:
        lines.append("GPU: nvidia-smi did not return a device.")
    lines.append("")
    lines.append(f"Simulator: {report.get('simulator', 'unspecified')}")
    lines.append(f"Sim client: {report.get('client', 'unspecified')}")
    lines.append(f"Cameras: {', '.join(report.get('cameras', []))}")
    lines.append("")
    for run in report.get("runs", []):
        label = run.get("label", "resolution")
        requested = run.get("requested_hw")
        status = (
            "matched the running sim" if run.get("matched") else "DID NOT MATCH the running sim"
        )
        lines.append(f"## {label} {requested} ({status})")
        lines.append("")
        lines.append("| modality | fps (step+grab) | grab mean ms | p50 ms | p95 ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, stats in run.get("modalities", {}).items():
            latency = stats["latency_ms"]
            lines.append(
                f"| {name} | {stats['fps']:.2f} | {latency['mean']:.2f} | "
                f"{latency['p50']:.2f} | {latency['p95']:.2f} |"
            )
        lines.append("")
        if not run.get("matched"):
            lines.append(
                "The images came back at a different size. Copy the matching "
                "`settings_*.json` into the AirSim settings folder and restart Play, "
                "then run this resolution again."
            )
            lines.append("")
    lines.append(
        "Training-time VRAM targets for this laptop are in `configs/model/`, not in "
        "this render report. ViT-S at 224 px is the 8 GB setting; "
        "ViT-B at 224 px is the 16 GB setting."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(directory: Path, report: dict) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "benchmark.json"
    md_path = directory / "benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


class AirSimBenchClient:
    """Pause / step / grab against the client ``connect`` imported."""

    def __init__(self, api, module) -> None:
        self.api = api
        self.module = module

    def pause(self, paused: bool) -> None:
        self.api.simPause(bool(paused))

    def step(self, dt: float) -> None:
        # simContinueForTime advances a paused sim by dt and leaves it paused.
        # That is the frame lock: every camera in the following grab sees one tick.
        if hasattr(self.api, "simContinueForTime"):
            self.api.simContinueForTime(float(dt))
            return
        self.api.simPause(False)
        time.sleep(dt)
        self.api.simPause(True)

    def grab(self, requests: list[tuple[str, int]]) -> list:
        image_requests = []
        for camera, image_type in requests:
            pixels_as_float = image_type == DEPTH_PLANAR
            image_requests.append(
                self.module.ImageRequest(camera, image_type, pixels_as_float, False)
            )
        return list(self.api.simGetImages(image_requests))


def connect(ip: str, simulator: str = "colosseum"):
    """Import the profile's client and confirm the RPC connection.

    ``colosseum`` tries ``airsim`` then ``colosseum``. ``cosys`` tries
    ``cosysairsim`` only. Both go through the shared AirSim-family connector.
    """
    profile = get_profile(simulator)
    api, module = _connect(ip, profile)
    return api, module, module.__name__


def _summarize(samples: list[float], *, dt: float, response_count: int) -> dict:
    grab = np.asarray(samples, dtype=np.float64)
    # End-to-end frame time is the scripted step plus the grab. The step is the
    # sim's own dt; the grab is what nvidia-smi's GPU is spending on the render.
    frame = grab + dt
    return {
        "responses_per_call": response_count,
        "fps": float(1.0 / np.mean(frame)),
        "grab_fps": float(1.0 / np.mean(grab)) if np.mean(grab) > 0 else None,
        "latency_ms": {
            "mean": float(np.mean(grab) * 1e3),
            "p50": _percentile(grab, 50) * 1e3,
            "p95": _percentile(grab, 95) * 1e3,
            "min": float(np.min(grab) * 1e3),
            "max": float(np.max(grab) * 1e3),
        },
    }


def _percentile(values: np.ndarray, percent: float) -> float:
    if values.size == 0:
        return math.nan
    ordered = np.sort(values)
    index = (len(ordered) - 1) * (percent / 100.0)
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return float(ordered[low])
    weight = index - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _label(height: int, width: int) -> str:
    for name, (h, w) in RESOLUTIONS.items():
        if (h, w) == (height, width):
            return name
    return f"{width}x{height}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a running Colosseum or Cosys-AirSim 4-camera rig."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument(
        "--simulator",
        choices=list(SIMULATORS),
        default="colosseum",
        help=(
            "colosseum (default) imports airsim or colosseum and writes "
            "SettingsVersion 1.2. cosys imports cosysairsim only and writes "
            "SettingsVersion 2.0 with the Cosys camera flags."
        ),
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.05, help="Sim step in seconds while paused.")
    parser.add_argument(
        "--resolution",
        default="both",
        help="1080p, 720p, or both. The sim must already be running at that size.",
    )
    parser.add_argument(
        "--write-settings",
        action="store_true",
        help="Write settings_1080p.json and settings_720p.json, then exit.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example rig=uam_quad.")
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf

    from aerosplat.config import load_config

    cfg = load_config(args.overrides)
    rig = OmegaConf.to_container(cfg.rig, resolve=True)
    cameras = list(rig["cameras"])
    args.output.mkdir(parents=True, exist_ok=True)
    for label, (height, width) in RESOLUTIONS.items():
        settings = build_benchmark_settings(rig, height, width, args.simulator)
        path = args.output / f"settings_{args.simulator}_{label}.json"
        path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"wrote {path}")
    if args.write_settings:
        print(
            f"Copy the {args.simulator} file you want to "
            "Documents\\AirSim\\settings.json and restart Play."
        )
        return

    selected = _selected_resolutions(args.resolution)
    api, module, client_name = connect(args.ip, args.simulator)
    client = AirSimBenchClient(api, module)
    runs = []
    for height, width in selected:
        print(f"measuring {width}x{height} ...")
        runs.append(
            measure_resolution(
                client,
                cameras,
                height=height,
                width=width,
                frames=args.frames,
                warmup=args.warmup,
                dt=args.dt,
            )
        )
    report = {
        "simulator": args.simulator,
        "client": client_name,
        "ip": args.ip,
        "gpu": query_nvidia_smi(),
        "cameras": cameras,
        "runs": runs,
    }
    json_path, md_path = write_report(args.output, report)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not all(run["matched"] for run in runs):
        raise SystemExit(1)


def _selected_resolutions(name: str) -> list[tuple[int, int]]:
    if name == "both":
        return [RESOLUTIONS["1080p"], RESOLUTIONS["720p"]]
    if name not in RESOLUTIONS:
        raise SystemExit(f"unknown resolution {name!r}. Use 1080p, 720p, or both.")
    return [RESOLUTIONS[name]]


if __name__ == "__main__":
    main()
