"""Scenario YAML: a concrete episode after the seed is applied.

The file describes a map, a rig trajectory, the sun, the weather, dynamic
actors, and catenary spans. ``materialize`` applies domain randomization with
``numpy`` Generator(seed) so episode ``k`` is ``base_seed + k`` and a resumed
shard repeats the same poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

from aerosplat.airsim_family import SIMULATORS
from aerosplat.geometry.sun import sun_direction_enu
from aerosplat.synth.catenary import catenary_polyline
from aerosplat.synth.motion import (
    enu_position_to_ned,
    path_tangent,
    quat_from_enu_direction,
    sample_polyline,
    velocities_from_positions,
    yaw_rad_from_enu_direction,
)

ALTITUDE_MIN_M = 30.0
ALTITUDE_MAX_M = 300.0
DIAMETER_MIN_M = 0.01
DIAMETER_MAX_M = 0.05
SCHEMA = "aerosplat.scenario.v1"


@dataclass
class Location:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    name: str = ""


@dataclass
class TimeWindow:
    start: datetime
    end: datetime


@dataclass
class Weather:
    rain: float = 0.0
    road_wetness: float = 0.0
    snow: float = 0.0
    road_snow: float = 0.0
    maple_leaf: float = 0.0
    road_leaf: float = 0.0
    dust: float = 0.0
    fog: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "rain": self.rain,
            "road_wetness": self.road_wetness,
            "snow": self.snow,
            "road_snow": self.road_snow,
            "maple_leaf": self.maple_leaf,
            "road_leaf": self.road_leaf,
            "dust": self.dust,
            "fog": self.fog,
        }


@dataclass
class Flight:
    frames: int
    frame_rate_hz: float
    speed_mps: float
    yaw: str
    waypoints_enu: np.ndarray


@dataclass
class ActorSpec:
    name: str
    kind: str
    asset: str
    segmentation_id: int
    speed_mps: float
    scale_m: np.ndarray
    waypoints_enu: np.ndarray


@dataclass
class WireSpan:
    name: str
    kind: str
    start_enu: np.ndarray
    end_enu: np.ndarray
    sag_m: float
    diameter_m: float
    samples: int


@dataclass
class WireSpec:
    segmentation_id: int
    asset: str
    spans: list[WireSpan]


@dataclass
class GlareSpec:
    mode: str = "auto"


@dataclass
class CaptureSpec:
    supersample: int = 1
    pause_step: bool = True
    record_wire_visibility: bool = True


@dataclass
class Randomization:
    enabled: bool = False
    altitude_m: tuple[float, float] = (ALTITUDE_MIN_M, ALTITUDE_MAX_M)
    speed_mps: tuple[float, float] = (4.0, 18.0)
    fog: tuple[float, float] = (0.0, 0.35)
    hour_utc: tuple[float, float] = (7.0, 18.0)
    wire_diameter_m: tuple[float, float] = (DIAMETER_MIN_M, DIAMETER_MAX_M)
    wire_sag_m: tuple[float, float] = (0.4, 3.0)
    lateral_offset_m: tuple[float, float] = (-15.0, 15.0)


@dataclass
class OutputSpec:
    shard_frames: int = 100


@dataclass
class Scenario:
    name: str
    simulator: str
    map_name: str
    autoload_map: bool
    seed: int
    rig: str
    location: Location
    time: TimeWindow
    weather: Weather
    flight: Flight
    actors: list[ActorSpec]
    wires: WireSpec
    glare: GlareSpec
    capture: CaptureSpec
    randomization: Randomization
    output: OutputSpec


@dataclass
class PlannedActor:
    name: str
    kind: str
    asset: str
    segmentation_id: int
    scale_m: np.ndarray
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    quaternion_wxyz: np.ndarray
    position_ned: np.ndarray


@dataclass
class PlannedWire:
    name: str
    kind: str
    segmentation_id: int
    asset: str
    diameter_m: float
    sag_m: float
    points_enu: np.ndarray


@dataclass
class FramePlan:
    index: int
    timestamp: float
    datetime_utc: datetime
    ego_position_enu: np.ndarray
    ego_velocity_enu: np.ndarray
    ego_quaternion_wxyz: np.ndarray
    ego_position_ned: np.ndarray
    sun_world_enu: np.ndarray
    actors: list[tuple[PlannedActor, int]] = field(default_factory=list)


@dataclass
class EpisodePlan:
    scenario_name: str
    simulator: str
    map_name: str
    seed: int
    location: Location
    weather: Weather
    glare_mode: str
    supersample: int
    record_wire_visibility: bool
    autoload_map: bool
    dt: float
    wires: list[PlannedWire]
    actors: list[PlannedActor]
    frames: list[FramePlan]


def load_scenario(path: Path) -> Scenario:
    """Parse and validate a scenario file. Randomization is not applied yet."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    schema = str(raw.get("schema", ""))
    if schema != SCHEMA:
        raise ValueError(f"{path} schema is {schema!r}, expected {SCHEMA}")
    simulator = str(raw.get("simulator", "colosseum")).strip().lower()
    if simulator not in SIMULATORS:
        raise ValueError(f"simulator must be one of {SIMULATORS}, got {simulator!r}")
    flight = _flight(raw.get("flight") or {})
    actors = [_actor(item, index) for index, item in enumerate(raw.get("actors") or [])]
    wires = _wires(raw.get("wires") or {})
    time_window = _time_window(raw.get("time") or {})
    location = _location(raw.get("location") or {})
    _validate_flight_altitude(flight.waypoints_enu)
    _validate_ids(actors, wires)
    return Scenario(
        name=str(raw.get("name", path.stem)),
        simulator=simulator,
        map_name=str(raw.get("map", "Blocks")),
        autoload_map=bool(raw.get("autoload_map", False)),
        seed=int(raw.get("seed", 0)),
        rig=str(raw.get("rig", "uam_quad")),
        location=location,
        time=time_window,
        weather=_weather(raw.get("weather") or {}),
        flight=flight,
        actors=actors,
        wires=wires,
        glare=GlareSpec(mode=_glare_mode(raw.get("glare") or {})),
        capture=_capture(raw.get("capture") or {}),
        randomization=_randomization(raw.get("randomization") or {}),
        output=OutputSpec(shard_frames=int((raw.get("output") or {}).get("shard_frames", 100))),
    )


def materialize(scenario: Scenario, seed: int, *, randomize: bool | None = None) -> Scenario:
    """Copy ``scenario`` with domain randomization from ``seed``.

    ``randomize`` overrides the file. The same seed returns the same episode.
    """
    enabled = scenario.randomization.enabled if randomize is None else bool(randomize)
    if not enabled:
        return replace(scenario, seed=int(seed))
    rng = np.random.default_rng(int(seed))
    knobs = scenario.randomization
    altitude = float(rng.uniform(*knobs.altitude_m))
    shifted = _shift_altitude(scenario.flight.waypoints_enu, altitude)
    offset = float(rng.uniform(*knobs.lateral_offset_m))
    shifted = shifted.copy()
    shifted[:, 1] += offset
    shifted = _fit_altitude(shifted)
    speed = float(rng.uniform(*knobs.speed_mps))
    flight = replace(scenario.flight, speed_mps=speed, waypoints_enu=shifted)
    fog = float(rng.uniform(*knobs.fog))
    weather = replace(scenario.weather, fog=fog)
    hour = float(rng.uniform(*knobs.hour_utc))
    start = _replace_hour(scenario.time.start, hour)
    duration = scenario.time.end - scenario.time.start
    time_window = TimeWindow(start=start, end=start + duration)
    spans = []
    for span in scenario.wires.spans:
        diameter = float(
            np.clip(rng.uniform(*knobs.wire_diameter_m), DIAMETER_MIN_M, DIAMETER_MAX_M)
        )
        sag = float(max(0.0, rng.uniform(*knobs.wire_sag_m)))
        spans.append(replace(span, diameter_m=diameter, sag_m=sag))
    wires = replace(scenario.wires, spans=spans)
    _validate_diameters(wires)
    return replace(
        scenario,
        seed=int(seed),
        flight=flight,
        weather=weather,
        time=time_window,
        wires=wires,
    )


def plan_episode(scenario: Scenario) -> EpisodePlan:
    """Expand waypoints into per-frame poses, velocities, sun vectors, and cables."""
    _validate_flight_altitude(scenario.flight.waypoints_enu)
    _validate_diameters(scenario.wires)
    frames = int(scenario.flight.frames)
    if frames < 1:
        raise ValueError("flight.frames must be >= 1")
    rate = float(scenario.flight.frame_rate_hz)
    if rate <= 0:
        raise ValueError("frame_rate_hz must be positive")
    dt = 1.0 / rate
    times = np.arange(frames, dtype=np.float64) * dt
    ego = sample_polyline(scenario.flight.waypoints_enu, scenario.flight.speed_mps, times)
    ego_velocity = velocities_from_positions(ego, dt)
    tangents = path_tangent(scenario.flight.waypoints_enu, scenario.flight.speed_mps, times)
    unix = _unix_times(scenario.time, frames)
    suns = np.stack(
        [
            sun_direction_enu(
                float(stamp),
                scenario.location.latitude_deg,
                scenario.location.longitude_deg,
            )
            for stamp in unix
        ],
        axis=0,
    )
    ego_quat = np.stack(
        [
            _ego_quaternion(scenario.flight.yaw, tangents[index], suns[index])
            for index in range(frames)
        ],
        axis=0,
    )
    actors = [_plan_actor(actor, times, dt) for actor in scenario.actors]
    wires = [_plan_wire(span, scenario.wires) for span in scenario.wires.spans]
    planned_frames = []
    for index in range(frames):
        planned_frames.append(
            FramePlan(
                index=index,
                timestamp=float(unix[index]),
                datetime_utc=_from_unix(float(unix[index])),
                ego_position_enu=ego[index],
                ego_velocity_enu=ego_velocity[index],
                ego_quaternion_wxyz=ego_quat[index],
                ego_position_ned=enu_position_to_ned(ego[index]),
                sun_world_enu=suns[index],
                actors=[(actor, index) for actor in actors],
            )
        )
    return EpisodePlan(
        scenario_name=scenario.name,
        simulator=scenario.simulator,
        map_name=scenario.map_name,
        seed=int(scenario.seed),
        location=scenario.location,
        weather=scenario.weather,
        glare_mode=scenario.glare.mode,
        supersample=int(scenario.capture.supersample),
        record_wire_visibility=bool(scenario.capture.record_wire_visibility),
        autoload_map=bool(scenario.autoload_map),
        dt=dt,
        wires=wires,
        actors=actors,
        frames=planned_frames,
    )


def broadcast_actor_velocity(velocities: list[np.ndarray]) -> np.ndarray | None:
    """The one vector ``gt_velocity`` can hold, or None when actors disagree.

    The optional loss broadcasts a single flight-frame vector onto every
    Gaussian. That is the right target only when one actor is moving. Several
    moving actors leave it unset; the per-pixel maps are the supervision.
    """
    moving = [np.asarray(velocity, dtype=np.float64).reshape(3) for velocity in velocities]
    moving = [velocity for velocity in moving if float(np.linalg.norm(velocity)) > 1e-3]
    if len(moving) != 1:
        return None
    return moving[0]


def _plan_actor(actor: ActorSpec, times: np.ndarray, dt: float) -> PlannedActor:
    position = sample_polyline(actor.waypoints_enu, actor.speed_mps, times)
    velocity = velocities_from_positions(position, dt)
    tangent = path_tangent(actor.waypoints_enu, actor.speed_mps, times)
    quaternion = np.stack([quat_from_enu_direction(direction) for direction in tangent], axis=0)
    return PlannedActor(
        name=actor.name,
        kind=actor.kind,
        asset=actor.asset,
        segmentation_id=int(actor.segmentation_id),
        scale_m=np.asarray(actor.scale_m, dtype=np.float64).reshape(3),
        position_enu=position,
        velocity_enu=velocity,
        quaternion_wxyz=quaternion,
        position_ned=np.stack([enu_position_to_ned(point) for point in position], axis=0),
    )


def _plan_wire(span: WireSpan, wires: WireSpec) -> PlannedWire:
    points = catenary_polyline(span.start_enu, span.end_enu, span.sag_m, span.samples)
    return PlannedWire(
        name=span.name,
        kind=span.kind,
        segmentation_id=int(wires.segmentation_id),
        asset=wires.asset,
        diameter_m=float(span.diameter_m),
        sag_m=float(span.sag_m),
        points_enu=points,
    )


def _ego_quaternion(mode: str, tangent: np.ndarray, sun_world: np.ndarray) -> np.ndarray:
    if mode == "toward_sun":
        horizontal = np.array([sun_world[0], sun_world[1], 0.0], dtype=np.float64)
        if float(np.linalg.norm(horizontal)) < 1e-8:
            horizontal = tangent
        return quat_from_enu_direction(horizontal)
    if mode == "fixed":
        return quat_from_enu_direction(np.array([0.0, 1.0, 0.0]))
    if mode == "tangent":
        return quat_from_enu_direction(tangent)
    raise ValueError(f"yaw must be tangent, toward_sun, or fixed, got {mode!r}")


def _unix_times(window: TimeWindow, frames: int) -> np.ndarray:
    start = window.start.timestamp()
    end = window.end.timestamp()
    if frames == 1:
        return np.array([start], dtype=np.float64)
    weights = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    return start + weights * (end - start)


def _from_unix(stamp: float) -> datetime:
    return datetime.fromtimestamp(stamp, tz=timezone.utc)


def _flight(raw: dict) -> Flight:
    waypoints = _points(raw.get("waypoints_enu"), "flight.waypoints_enu")
    yaw = str(raw.get("yaw", "tangent"))
    if yaw not in {"tangent", "toward_sun", "fixed"}:
        raise ValueError(f"flight.yaw must be tangent, toward_sun, or fixed, got {yaw!r}")
    frames = int(raw.get("frames", 1))
    rate = float(raw.get("frame_rate_hz", 10.0))
    speed = float(raw.get("speed_mps", 8.0))
    if speed < 0:
        raise ValueError("flight.speed_mps must be >= 0")
    return Flight(
        frames=frames,
        frame_rate_hz=rate,
        speed_mps=speed,
        yaw=yaw,
        waypoints_enu=waypoints,
    )


def _actor(raw: dict, index: int) -> ActorSpec:
    kind = str(raw.get("kind", "drone"))
    if kind not in {"drone", "bird", "vehicle"}:
        raise ValueError(f"actor kind must be drone, bird, or vehicle, got {kind!r}")
    name = str(raw.get("name", f"{kind}_{index}"))
    scale = raw.get("scale_m", [1.0, 1.0, 1.0])
    return ActorSpec(
        name=name,
        kind=kind,
        asset=str(raw.get("asset", "Sphere" if kind == "bird" else "Cube")),
        segmentation_id=int(raw.get("segmentation_id", 10 + index)),
        speed_mps=float(raw.get("speed_mps", 5.0)),
        scale_m=np.asarray(scale, dtype=np.float64).reshape(3),
        waypoints_enu=_points(raw.get("waypoints_enu"), f"actors[{index}].waypoints_enu"),
    )


def _wires(raw: dict) -> WireSpec:
    spans = []
    for index, item in enumerate(raw.get("spans") or []):
        diameter = float(item.get("diameter_m", 0.02))
        sag = float(item.get("sag_m", 1.0))
        spans.append(
            WireSpan(
                name=str(item.get("name", f"span_{index}")),
                kind=str(item.get("kind", "power_line")),
                start_enu=_vec3(item.get("start_enu"), f"wires.spans[{index}].start_enu"),
                end_enu=_vec3(item.get("end_enu"), f"wires.spans[{index}].end_enu"),
                sag_m=sag,
                diameter_m=diameter,
                samples=int(item.get("samples", 32)),
            )
        )
    spec = WireSpec(
        segmentation_id=int(raw.get("segmentation_id", 250)),
        asset=str(raw.get("asset", "Cube")),
        spans=spans,
    )
    _validate_diameters(spec)
    return spec


def _time_window(raw: dict) -> TimeWindow:
    start = _parse_time(raw.get("start", "2026-06-21T14:00:00Z"))
    if "end" in raw and raw["end"]:
        end = _parse_time(raw["end"])
    else:
        seconds = float(raw.get("duration_s", 0.0))
        end = start + timedelta(seconds=seconds)
    if end < start:
        raise ValueError("time.end is before time.start")
    return TimeWindow(start=start, end=end)


def _location(raw: dict) -> Location:
    latitude = float(raw.get("latitude_deg", 41.8781))
    longitude = float(raw.get("longitude_deg", -87.6298))
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude_deg must be in [-90, 90], got {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude_deg must be in [-180, 180], got {longitude}")
    return Location(
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=float(raw.get("altitude_m", 0.0)),
        name=str(raw.get("name", "")),
    )


def _weather(raw: dict) -> Weather:
    values = {}
    for name in Weather().as_dict():
        value = float(raw.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"weather.{name} must be in [0, 1], got {value}")
        values[name] = value
    return Weather(**values)


def _glare_mode(raw: dict) -> str:
    mode = str(raw.get("mode", "auto"))
    if mode not in {"auto", "simulator", "synthetic"}:
        raise ValueError(f"glare.mode must be auto, simulator, or synthetic, got {mode!r}")
    return mode


def _capture(raw: dict) -> CaptureSpec:
    factor = int(raw.get("supersample", 1))
    if factor < 1:
        raise ValueError("capture.supersample must be >= 1")
    return CaptureSpec(
        supersample=factor,
        pause_step=bool(raw.get("pause_step", True)),
        record_wire_visibility=bool(raw.get("record_wire_visibility", True)),
    )


def _randomization(raw: dict) -> Randomization:
    def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
        value = raw.get(key, list(default))
        if len(value) != 2:
            raise ValueError(f"randomization.{key} must be [low, high]")
        low, high = float(value[0]), float(value[1])
        if high < low:
            raise ValueError(f"randomization.{key} high is below low")
        return low, high

    return Randomization(
        enabled=bool(raw.get("enabled", False)),
        altitude_m=pair("altitude_m", (ALTITUDE_MIN_M, ALTITUDE_MAX_M)),
        speed_mps=pair("speed_mps", (4.0, 18.0)),
        fog=pair("fog", (0.0, 0.35)),
        hour_utc=pair("hour_utc", (7.0, 18.0)),
        wire_diameter_m=pair("wire_diameter_m", (DIAMETER_MIN_M, DIAMETER_MAX_M)),
        wire_sag_m=pair("wire_sag_m", (0.4, 3.0)),
        lateral_offset_m=pair("lateral_offset_m", (-15.0, 15.0)),
    )


def _validate_flight_altitude(waypoints: np.ndarray) -> None:
    altitude = np.asarray(waypoints, dtype=np.float64)[:, 2]
    if np.any(altitude < ALTITUDE_MIN_M) or np.any(altitude > ALTITUDE_MAX_M):
        raise ValueError(
            f"flight altitudes must be in [{ALTITUDE_MIN_M:.0f}, {ALTITUDE_MAX_M:.0f}] m, "
            f"got {altitude.min():.2f} to {altitude.max():.2f}"
        )


def _validate_diameters(wires: WireSpec) -> None:
    for span in wires.spans:
        if not DIAMETER_MIN_M <= span.diameter_m <= DIAMETER_MAX_M:
            raise ValueError(
                f"wire {span.name} diameter_m must be in "
                f"[{DIAMETER_MIN_M}, {DIAMETER_MAX_M}] m, got {span.diameter_m}"
            )
        if span.sag_m < 0:
            raise ValueError(f"wire {span.name} sag_m must be >= 0")
        if span.samples < 2:
            raise ValueError(f"wire {span.name} samples must be >= 2")


def _validate_ids(actors: list[ActorSpec], wires: WireSpec) -> None:
    ids = [actor.segmentation_id for actor in actors]
    ids.append(wires.segmentation_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"segmentation ids must be unique, got {ids}")
    for object_id in ids:
        if not 0 <= int(object_id) <= 255:
            raise ValueError(f"segmentation id {object_id} is outside 0..255")


def _shift_altitude(waypoints: np.ndarray, altitude: float) -> np.ndarray:
    pts = np.asarray(waypoints, dtype=np.float64).copy()
    pts[:, 2] += altitude - float(pts[0, 2])
    return pts


def _fit_altitude(waypoints: np.ndarray) -> np.ndarray:
    """Slide a profile into the 30–300 m band. Reject one that is taller than the band."""
    pts = np.asarray(waypoints, dtype=np.float64).copy()
    low = float(pts[:, 2].min())
    high = float(pts[:, 2].max())
    if high - low > ALTITUDE_MAX_M - ALTITUDE_MIN_M:
        raise ValueError(
            f"flight altitude profile spans {high - low:.1f} m, which cannot fit in "
            f"[{ALTITUDE_MIN_M:.0f}, {ALTITUDE_MAX_M:.0f}]"
        )
    if low < ALTITUDE_MIN_M:
        pts[:, 2] += ALTITUDE_MIN_M - low
    if float(pts[:, 2].max()) > ALTITUDE_MAX_M:
        pts[:, 2] -= float(pts[:, 2].max()) - ALTITUDE_MAX_M
    _validate_flight_altitude(pts)
    return pts


def _replace_hour(moment: datetime, hour: float) -> datetime:
    hour = float(hour) % 24.0
    whole = int(hour)
    minutes = int(round((hour - whole) * 60.0))
    if minutes == 60:
        whole += 1
        minutes = 0
    return moment.replace(hour=whole % 24, minute=minutes, second=0, microsecond=0)


def _parse_time(value: object) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _points(value, name: str) -> np.ndarray:
    if not value or len(value) < 1:
        raise ValueError(f"{name} needs at least one XYZ waypoint")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must be a list of [x, y, z]")
    return array


def _vec3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    if array.shape != (3,):
        raise ValueError(f"{name} must have 3 numbers")
    return array


def sun_yaw_rad(sun_world_enu: np.ndarray) -> float:
    """Horizontal yaw toward the sun. Used by tests of the toward_sun mode."""
    horizontal = np.array([sun_world_enu[0], sun_world_enu[1], 0.0], dtype=np.float64)
    return yaw_rad_from_enu_direction(horizontal)
