"""Generate N episodes, resumable and deterministic in the seed.

Each episode is ``base_seed + episode_index``. A finished shard has a
``DONE`` file and is skipped on the next run. Throughput is camera-frames
per second, appended to ``throughput.jsonl``. The disk estimate is an
uncompressed upper bound printed before the first grab.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aerosplat.config import load_config
from aerosplat.synth.client import AirSimScenarioClient, MockScenarioClient
from aerosplat.synth.episode import run_episode, scale_rig
from aerosplat.synth.scenario import load_scenario, materialize, plan_episode
from aerosplat.synth.writer import estimate_shard_bytes, format_bytes

MANIFEST_FORMAT = "aerosplat.synthetic.v1"


def generate_batch(
    scenario_path: Path,
    output: Path,
    *,
    episodes: int,
    episode_start: int = 0,
    simulator: str | None = None,
    randomize: bool | None = None,
    frames: int | None = None,
    seed: int | None = None,
    resume: bool = True,
    mock: bool = False,
    ip: str = "",
    rig: dict | None = None,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if episode_start < 0:
        raise ValueError("episode_start must be >= 0")
    scenario = load_scenario(scenario_path)
    if simulator:
        scenario.simulator = simulator
    if frames is not None:
        if frames < 1:
            raise ValueError("frames must be >= 1")
        scenario.flight.frames = int(frames)
    base_seed = scenario.seed if seed is None else int(seed)
    if rig is None:
        rig = _load_rig(scenario.rig)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    per_episode = estimate_shard_bytes(
        rig, scenario.flight.frames, supersample=scenario.capture.supersample
    )
    estimate = {
        "episodes": int(episodes),
        "frames_per_episode": int(scenario.flight.frames),
        "cameras": len(rig["cameras"]),
        "bytes_per_episode_upper": per_episode,
        "bytes_total_upper": per_episode * int(episodes),
        "bytes_total_human": format_bytes(per_episode * int(episodes)),
        "note": (
            "Uncompressed upper bound. PNG compression makes RGB smaller. "
            "Supersampled grabs are not stored."
        ),
    }
    (output / "disk_estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
    print(
        f"disk estimate {estimate['bytes_total_human']} upper bound for "
        f"{episodes} episodes x {scenario.flight.frames} frames x {len(rig['cameras'])} cameras"
    )
    manifest = _load_manifest(output) if resume else _empty_manifest(scenario, base_seed)
    manifest["simulator"] = scenario.simulator
    manifest["seed"] = base_seed
    client = None
    if not mock:
        client = AirSimScenarioClient(ip, scenario.simulator)
    for offset in range(episodes):
        episode_index = episode_start + offset
        episode_seed = base_seed + episode_index
        concrete = materialize(scenario, episode_seed, randomize=randomize)
        plan = plan_episode(concrete)
        step = max(int(concrete.output.shard_frames), 1)
        shard_count = int(np.ceil(len(plan.frames) / step))
        for part, start in enumerate(range(0, len(plan.frames), step)):
            if shard_count == 1:
                scene_id = f"ep{episode_index:06d}"
            else:
                scene_id = f"ep{episode_index:06d}_s{part:02d}"
            shard_dir = output / f"shard_{episode_index:05d}_{part:02d}"
            if resume and (shard_dir / "DONE").is_file():
                print(f"skip {shard_dir.name} (already done)")
                _record(manifest, episode_index, episode_seed, scene_id, shard_dir.name, "complete")
                continue
            capture_rig = scale_rig(rig, concrete.capture.supersample)
            if mock:
                client = MockScenarioClient(capture_rig)
            assert client is not None
            started = time.perf_counter()
            result = run_episode(
                client,
                plan,
                rig,
                shard_dir,
                scene_id=scene_id,
                frame_slice=slice(start, start + step),
            )
            elapsed = max(time.perf_counter() - started, 1e-6)
            camera_frames = result["frames"] * result["cameras"]
            record = {
                "episode": episode_index,
                "shard": shard_dir.name,
                "scene_id": scene_id,
                "frames": result["frames"],
                "cameras": result["cameras"],
                "seconds": elapsed,
                "camera_frames_per_sec": camera_frames / elapsed,
                "rig_frames_per_sec": result["frames"] / elapsed,
                "glare_source": result["glare_source"],
                "mock": mock,
            }
            with (output / "throughput.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{shard_dir.name}: {record['camera_frames_per_sec']:.2f} camera-frames/s "
                f"({record['rig_frames_per_sec']:.2f} rig-frames/s), glare={result['glare_source']}"
            )
            _record(manifest, episode_index, episode_seed, scene_id, shard_dir.name, "complete")
            _write_manifest(output, manifest)
    _write_manifest(output, manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic AeroSplat episodes from Colosseum (default) or Cosys-AirSim."
        )
    )
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="Override frames per episode.")
    parser.add_argument("--simulator", choices=["colosseum", "cosys"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ip", default="")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Software renderer. No Unreal. Writes a real canonical dataset for a pipeline check.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example rig=uam_quad.")
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    rig = _load_rig(scenario.rig, args.overrides)
    generate_batch(
        args.scenario,
        args.output,
        episodes=args.episodes,
        episode_start=args.episode_start,
        simulator=args.simulator,
        randomize=True if args.randomize else None,
        frames=args.frames,
        seed=args.seed,
        resume=not args.no_resume,
        mock=args.mock,
        ip=args.ip,
        rig=rig,
    )


def _load_rig(name: str, overrides: list[str] | None = None) -> dict:
    from omegaconf import OmegaConf

    cfg = load_config([f"rig={name}", *(overrides or [])])
    rig = OmegaConf.to_container(cfg.rig, resolve=True)
    if not isinstance(rig, dict):
        raise ValueError("rig config did not resolve to a mapping")
    return rig


def _empty_manifest(scenario, seed: int) -> dict:
    return {
        "format": MANIFEST_FORMAT,
        "scenario": scenario.name,
        "simulator": scenario.simulator,
        "map": scenario.map_name,
        "seed": int(seed),
        "episodes": [],
    }


def _load_manifest(output: Path) -> dict:
    path = output / "manifest.json"
    if not path.is_file():
        return {
            "format": MANIFEST_FORMAT,
            "scenario": "",
            "simulator": "",
            "map": "",
            "seed": 0,
            "episodes": [],
        }
    return json.loads(path.read_text())


def _write_manifest(output: Path, manifest: dict) -> None:
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _record(manifest, episode_index, seed, scene_id, shard, status) -> None:
    episodes = manifest.setdefault("episodes", [])
    for item in episodes:
        if item["index"] == episode_index and shard in item["shards"]:
            item["status"] = status
            return
    found = None
    for item in episodes:
        if item["index"] == episode_index:
            found = item
            break
    if found is None:
        episodes.append(
            {
                "index": episode_index,
                "seed": seed,
                "scene_id": scene_id,
                "shards": [shard],
                "status": status,
            }
        )
        return
    if shard not in found["shards"]:
        found["shards"].append(shard)
    found["status"] = status


if __name__ == "__main__":
    main()
