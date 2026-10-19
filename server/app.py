"""FastAPI workbench.

``POST /api/v1/infer`` runs the feedforward model. ``GET /api/v1/occupancy``
reads the latest volume, or a baked episode volume, and calls
:func:`aerosplat.models.collision.query_collision`. ``GET /api/v1/health``
reports the GPU and the last stage timings. Episode routes serve the Phase 1
canonical recording so the UI can scrub without a forward pass.
"""

from __future__ import annotations

import base64
import io
import json
import math
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from aerosplat.models.collision import query_collision
from aerosplat.models.gaussians import GaussianLayout
from aerosplat.models.occupancy_head import OccupancyVolume

from .codec import (
    frame_infer_blob,
    gaussian_layout_names,
    pack_gaussians,
    pack_occupancy,
)
from .config import Settings, load_settings
from .engine import Engine, gpu_status, sample_from_cameras
from .episodes import (
    Episode,
    camera_glare_png,
    camera_image_png,
    cloud_bytes,
    discover_episodes,
    frame_payload,
    load_volume,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    engine = Engine(settings=cfg)
    episodes = discover_episodes(cfg.episode_dirs)
    app = FastAPI(
        title="AeroSplat workbench",
        version="0.1.0",
        summary="Debug API for 4D Gaussians, glare, and occupancy.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = cfg
    app.state.engine = engine
    app.state.episodes = episodes
    app.state.latest_volume = None

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        last = engine.last_total_ms
        return {
            "gpu": gpu_status(),
            "latency_ms": engine.last_times.as_dict(),
            "throughput_fps": None if last is None or last <= 0 else 1000.0 / last,
            "inferences": engine.inferences,
            "model": {
                "loaded": engine.trainer is not None,
                "name": cfg.model_name,
                "device": None if engine.device is None else str(engine.device),
                "init": engine.init_kind,
                "checkpoint": None if cfg.checkpoint is None else str(cfg.checkpoint),
                "checkpoint_error": engine.checkpoint_error,
            },
            "episodes": sorted(episodes),
        }

    @app.get("/api/v1/schema")
    def schema() -> dict[str, Any]:
        return {
            "gaussian_magic": "AGS1",
            "occupancy_magic": "AOC1",
            "channels": gaussian_layout_names(),
            "gaussian_dim": GaussianLayout.DIM,
            "sh_degree": GaussianLayout.SH_DEGREE,
            "sh_coeffs": GaussianLayout.SH_COEFFS,
            "slices": {
                "xyz": [GaussianLayout.XYZ.start, GaussianLayout.XYZ.stop],
                "scale": [GaussianLayout.SCALE.start, GaussianLayout.SCALE.stop],
                "quat": [GaussianLayout.QUAT.start, GaussianLayout.QUAT.stop],
                "opacity": [GaussianLayout.OPACITY.start, GaussianLayout.OPACITY.stop],
                "sh": [GaussianLayout.SH.start, GaussianLayout.SH.stop],
                "velocity": [GaussianLayout.VELOCITY.start, GaussianLayout.VELOCITY.stop],
            },
            "note": (
                "SH is coefficient-major then RGB. Velocity is world m/s. "
                "valid and wire_prob are side channels, not part of the 62."
            ),
        }

    @app.get("/api/v1/episodes")
    def list_episodes() -> list[dict[str, Any]]:
        rows = []
        for episode in episodes.values():
            rows.append(
                {
                    "id": episode.id,
                    "samples": episode.samples,
                    "timestamps": episode.timestamps(),
                    "title": episode.meta.get("title", episode.id),
                    "description": episode.meta.get("description", ""),
                    "has_clouds": (episode.root / "clouds").is_dir(),
                    "source_sun": episode.meta.get("source_sun"),
                    "corridor": episode.meta.get("corridor"),
                    "corridor_radius_m": episode.meta.get("corridor_radius_m"),
                }
            )
        return rows

    @app.get("/api/v1/episodes/{episode_id}")
    def episode_detail(episode_id: str) -> dict[str, Any]:
        episode = _episode(episodes, episode_id)
        return {
            "id": episode.id,
            "samples": episode.samples,
            "timestamps": episode.timestamps(),
            "meta": episode.meta,
        }

    @app.get("/api/v1/episodes/{episode_id}/frames/{index}")
    def episode_frame(episode_id: str, index: int) -> dict[str, Any]:
        episode = _episode(episodes, episode_id)
        _check_index(episode, index)
        return frame_payload(episode, index)

    @app.get("/api/v1/episodes/{episode_id}/frames/{index}/cameras/{camera}/image")
    def episode_image(episode_id: str, index: int, camera: str) -> Response:
        episode = _episode(episodes, episode_id)
        _check_index(episode, index)
        try:
            png = camera_image_png(episode, index, camera)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no camera {camera}") from exc
        return Response(content=png, media_type="image/png")

    @app.get("/api/v1/episodes/{episode_id}/frames/{index}/cameras/{camera}/glare")
    def episode_glare(episode_id: str, index: int, camera: str) -> Response:
        episode = _episode(episodes, episode_id)
        _check_index(episode, index)
        png = camera_glare_png(episode, index, camera)
        if png is None:
            raise HTTPException(status_code=404, detail="no baked glare for this camera")
        return Response(content=png, media_type="image/png")

    @app.get("/api/v1/episodes/{episode_id}/clouds/{index}")
    def episode_cloud(episode_id: str, index: int) -> Response:
        episode = _episode(episodes, episode_id)
        _check_index(episode, index)
        blob = cloud_bytes(episode, index)
        if blob is None:
            raise HTTPException(status_code=404, detail="this frame has no baked gaussian stream")
        return Response(content=blob, media_type="application/vnd.aerosplat.gaussians")

    @app.get("/api/v1/occupancy")
    def occupancy(
        episode: str | None = None,
        index: int = 0,
        radius: float | None = None,
        corridor: str | None = None,
        min_prob: float = 0.5,
        format: str = "json",
    ) -> Response:
        volume, default_corridor, default_radius = _select_volume(
            episodes, episode, index, app.state.latest_volume
        )
        points = _parse_corridor(corridor) if corridor else default_corridor
        used_radius = float(default_radius if radius is None else radius)
        report = None
        if points is not None:
            if used_radius < 0:
                raise HTTPException(status_code=400, detail="radius must be >= 0")
            trajectory = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
            try:
                queried = query_collision(volume, trajectory, used_radius)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            report = queried.item()
        payload = _occupancy_payload(volume, report, min_prob)
        if format == "binary":
            return Response(
                content=payload["occupancy_bytes"],
                media_type="application/vnd.aerosplat.occupancy",
            )
        payload["collision"] = None if report is None else _clean_report(report)
        return JSONResponse(_jsonable(payload))

    @app.post("/api/v1/infer")
    async def infer(request: Request) -> Response:
        records, sun, latitude, longitude, fmt = await _read_infer_request(request)
        try:
            sample = sample_from_cameras(records)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = engine.run(sample, sun, latitude, longitude)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
        app.state.latest_volume = result.volume
        meta, gaussians = _infer_parts(engine, result)
        if fmt == "binary":
            body = frame_infer_blob(json.dumps(_jsonable(meta)).encode("utf-8"), gaussians)
            return Response(content=body, media_type="application/vnd.aerosplat.infer")
        meta["gaussian_b64"] = base64.b64encode(gaussians).decode("ascii")
        return JSONResponse(_jsonable(meta))

    dist = cfg.repo_root / "web" / "dist"
    if dist.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def _episode(episodes: dict[str, Episode], episode_id: str) -> Episode:
    episode = episodes.get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail=f"unknown episode {episode_id!r}")
    return episode


def _check_index(episode: Episode, index: int) -> None:
    if index < 0 or index >= episode.samples:
        raise HTTPException(
            status_code=404, detail=f"frame {index} is outside 0..{episode.samples - 1}"
        )


def _select_volume(
    episodes: dict[str, Episode],
    episode_id: str | None,
    index: int,
    latest: OccupancyVolume | None,
) -> tuple[OccupancyVolume, np.ndarray | None, float]:
    if episode_id:
        episode = _episode(episodes, episode_id)
        _check_index(episode, index)
        volume = load_volume(episode, index)
        if volume is None:
            raise HTTPException(status_code=404, detail="no baked occupancy for this frame")
        corridor = episode.meta.get("corridor")
        radius = float(episode.meta.get("corridor_radius_m", 0.5))
        points = np.asarray(corridor, dtype=np.float32) if corridor else None
        return volume, points, radius
    if latest is None:
        raise HTTPException(
            status_code=404,
            detail="no inference yet; pass episode and index, or POST /infer",
        )
    return latest, None, 0.5


def _parse_corridor(text: str) -> np.ndarray:
    rows = []
    for piece in text.split(";"):
        if not piece.strip():
            continue
        coords = [float(part) for part in piece.split(",")]
        if len(coords) != 3:
            raise HTTPException(
                status_code=400, detail="corridor points are x,y,z separated by ';'"
            )
        rows.append(coords)
    if len(rows) < 1:
        raise HTTPException(status_code=400, detail="corridor needs at least one point")
    return np.asarray(rows, dtype=np.float32)


def _occupancy_payload(
    volume: OccupancyVolume, report: dict | None, min_prob: float
) -> dict[str, Any]:
    if not 0.0 <= min_prob < 1.0:
        raise HTTPException(status_code=400, detail="min_prob must lie in [0, 1)")
    occ = torch.sigmoid(volume.occupancy_logits[0]).numpy()
    wire = torch.sigmoid(volume.wire_logits[0]).numpy()
    direction = volume.wire_direction[0].permute(1, 2, 3, 0).numpy()
    offset = volume.wire_offset[0].permute(1, 2, 3, 0).numpy()
    origin = volume.origin[0].numpy()
    voxel = float(volume.voxel_size.reshape(-1)[0])
    blob = pack_occupancy(
        origin,
        voxel,
        occ,
        wire,
        direction,
        offset,
        min_prob=min_prob,
        collision=report,
    )
    return {
        "origin": origin.tolist(),
        "voxel_size": voxel,
        "dims": [int(v) for v in occ.shape],
        "min_prob": min_prob,
        "collision": report,
        "occupancy_b64": base64.b64encode(blob).decode("ascii"),
        "occupancy_bytes": blob,
    }


def _infer_parts(engine: Engine, result) -> tuple[dict[str, Any], bytes]:
    packed = result.cloud.packed.numpy()
    valid = result.cloud.valid.numpy()
    wire = result.cloud.wire_prob.numpy()
    gaussians = pack_gaussians(
        packed,
        valid,
        wire,
        timestamp=0.0,
        dt=0.0,
        source_sun=result.sun_world,
    )
    occ_payload = _occupancy_payload(result.volume, None, min_prob=0.5)
    glare = {}
    for name, layer in result.glare.items():
        image = layer[0].clamp(0, 1).numpy()
        png = _png_chw(image)
        glare[name] = {
            "png_b64": base64.b64encode(png).decode("ascii"),
            "mean": [float(v) for v in image.reshape(3, -1).mean(axis=1)],
            "max": float(image.max()),
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
        }
    meta = {
        "batch": int(packed.shape[0]),
        "count": int(packed.shape[1]),
        "valid_count": int(valid.sum()),
        "channels": GaussianLayout.DIM,
        "layout": gaussian_layout_names(),
        "sun_world": result.sun_world.tolist(),
        "sun_source": result.sun_source,
        "latency_ms": result.times.as_dict(),
        "throughput_fps": 1000.0 / result.times.total_ms if result.times.total_ms > 0 else None,
        "model": {
            "name": engine.settings.model_name,
            "init": engine.init_kind,
            "checkpoint": None
            if engine.settings.checkpoint is None
            else str(engine.settings.checkpoint),
            "checkpoint_error": engine.checkpoint_error,
        },
        "network_hw": result.network_hw,
        "glare": glare,
        "camera_order": list(result.cloud.camera_order),
        "occupancy_b64": occ_payload["occupancy_b64"],
        "occupancy": {
            "origin": occ_payload["origin"],
            "voxel_size": occ_payload["voxel_size"],
            "dims": occ_payload["dims"],
        },
        "binary_layout": "uint32 meta_len, utf-8 json without gaussian_b64, then AGS1 bytes",
    }
    return meta, gaussians


async def _read_infer_request(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw = form.get("manifest")
        if raw is None:
            raise HTTPException(status_code=400, detail="multipart infer needs a manifest field")
        try:
            manifest = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="manifest is not JSON") from exc
        records = []
        for camera in manifest.get("cameras", []):
            name = camera["name"]
            upload = form.get(name)
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=400, detail=f"missing file for camera {name}")
            data = await upload.read()
            records.append({**camera, "image": _decode_image_bytes(data)})
        fmt = str(manifest.get("format") or request.query_params.get("format") or "json")
        return (
            records,
            manifest.get("sun_world"),
            manifest.get("latitude_deg"),
            manifest.get("longitude_deg"),
            fmt,
        )
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="body is not JSON") from exc
    records = []
    for camera in payload.get("cameras", []):
        encoded = camera.get("image_png_base64")
        if not encoded:
            raise HTTPException(
                status_code=400, detail=f"camera {camera.get('name')} needs image_png_base64"
            )
        records.append({**camera, "image": _decode_image_b64(encoded)})
    fmt = str(payload.get("format") or request.query_params.get("format") or "json")
    return (
        records,
        payload.get("sun_world"),
        payload.get("latitude_deg"),
        payload.get("longitude_deg"),
        fmt,
    )


def _decode_image_b64(encoded: str) -> np.ndarray:
    if "," in encoded and encoded.strip().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_png_base64 is not base64") from exc
    return _decode_image_bytes(raw)


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image is not a readable PNG or JPEG") from exc
    return np.transpose(array, (2, 0, 1))


def _png_chw(image: np.ndarray) -> bytes:
    hwc = np.transpose(np.clip(image, 0.0, 1.0), (1, 2, 0))
    picture = Image.fromarray((hwc * 255.0).round().astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _clean_report(report: dict[str, float]) -> dict[str, float | None]:
    cleaned: dict[str, float | None] = {}
    for key, value in report.items():
        number = float(value)
        cleaned[key] = None if not math.isfinite(number) else number
    return cleaned


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if key != "occupancy_bytes"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        number = value.item()
        return None if isinstance(number, float) and not math.isfinite(number) else number
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


app = create_app()
