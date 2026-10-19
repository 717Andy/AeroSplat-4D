"""Lazy model runner. The checkpoint is optional.

A missing file, or a state dict that does not match ``model_name``, falls
back to a randomly initialized network of that config. The forward is the
same sequence as :class:`aerosplat.train.SplatTrainer`: lifter, Gaussian
heads, glare, occupancy. The rasterizer stays out of this process.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from torch import Tensor

from aerosplat.geometry.sun import resolve_sun_world
from aerosplat.models.factory import ModelConfig, build_lifter
from aerosplat.models.gaussians import GaussianCloud
from aerosplat.models.occupancy_head import OccupancyHead, OccupancyVolume
from aerosplat.models.pretrain import batch_from_sample, prepare_rig
from aerosplat.models.relight import RelightHead, glare_for_cameras
from aerosplat.models.splats import FeedforwardSplatter, SplatConfig, SplatPredictor
from aerosplat.schema import CameraFrame, MultiCameraSample
from aerosplat.train import SplatTrainer

from .config import Settings


@dataclass
class StageTimes:
    prepare_ms: float = 0.0
    lifter_ms: float = 0.0
    gaussians_ms: float = 0.0
    glare_ms: float = 0.0
    occupancy_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "prepare_ms": self.prepare_ms,
            "lifter_ms": self.lifter_ms,
            "gaussians_ms": self.gaussians_ms,
            "glare_ms": self.glare_ms,
            "occupancy_ms": self.occupancy_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class InferenceResult:
    cloud: GaussianCloud
    glare: dict[str, Tensor]
    volume: OccupancyVolume
    sun_world: np.ndarray
    sun_source: str
    times: StageTimes
    network_hw: dict[str, list[int]]


@dataclass
class Engine:
    settings: Settings
    trainer: SplatTrainer | None = None
    device: torch.device | None = None
    init_kind: str | None = None
    checkpoint_error: str | None = None
    model_config: ModelConfig | None = None
    last_times: StageTimes = field(default_factory=StageTimes)
    last_total_ms: float | None = None
    inferences: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def ensure(self) -> SplatTrainer:
        with self._lock:
            if self.trainer is None:
                self._load()
            assert self.trainer is not None
            return self.trainer

    def _load(self) -> None:
        config_path = (
            self.settings.repo_root / "configs" / "model" / f"{self.settings.model_name}.yaml"
        )
        if not config_path.is_file():
            raise FileNotFoundError(f"no model config at {config_path}")
        model_cfg, splat_cfg = _configs_from_yaml(config_path)
        self.model_config = model_cfg
        splatter = FeedforwardSplatter(
            build_lifter(model_cfg), SplatPredictor(model_cfg.embed_dim, splat_cfg)
        )
        trainer = SplatTrainer(
            splatter,
            RelightHead(feature_dim=model_cfg.embed_dim),
            OccupancyHead(),
        )
        self.init_kind = "random"
        self.checkpoint_error = None
        checkpoint = self.settings.checkpoint
        if checkpoint is not None:
            if not checkpoint.is_file():
                self.checkpoint_error = f"checkpoint not found: {checkpoint}"
            else:
                try:
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    state = (
                        payload["model"]
                        if isinstance(payload, dict) and "model" in payload
                        else payload
                    )
                    trainer.load_state_dict(state)
                    self.init_kind = "checkpoint"
                except Exception as exc:  # shape mismatches are expected for the tiny fallback
                    self.checkpoint_error = f"{type(exc).__name__}: {exc}"
                    self.init_kind = "random"
        device_name = self.settings.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        trainer.to(device)
        trainer.eval()
        self.device = device
        self.trainer = trainer

    def run(
        self,
        sample: MultiCameraSample,
        sun_world: list[float] | None,
        latitude: float | None,
        longitude: float | None,
    ) -> InferenceResult:
        with self._lock:
            if self.trainer is None:
                self._load()
            return self._run_locked(sample, sun_world, latitude, longitude)

    def _run_locked(
        self,
        sample: MultiCameraSample,
        sun_world: list[float] | None,
        latitude: float | None,
        longitude: float | None,
    ) -> InferenceResult:
        trainer = self.trainer
        assert trainer is not None and self.model_config is not None and self.device is not None
        started = time.perf_counter()
        prepared = prepare_rig(sample, self.model_config.img_size, self.model_config.patch_size)
        images, cameras = batch_from_sample(prepared)
        prepare_done = time.perf_counter()
        first = next(iter(prepared.cameras.values()))
        if sun_world is not None:
            sun = torch.tensor(sun_world, dtype=torch.float32).reshape(1, 3)
            sun = sun / sun.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            source = "request"
        else:
            sun, source = resolve_sun_world(
                float(sample.timestamp),
                latitude,
                longitude,
                first.image,
                first.K,
                first.R,
            )
            if sun.ndim == 1:
                sun = sun.reshape(1, 3)
        images = {name: tensor.to(self.device) for name, tensor in images.items()}
        cameras = {
            name: {key: value.to(self.device) for key, value in pose.items()}
            for name, pose in cameras.items()
        }
        sun = sun.to(device=self.device, dtype=torch.float32)
        origin = torch.tensor(self.settings.occ_origin, dtype=torch.float32, device=self.device)
        origin = origin.view(1, 3).expand(sun.shape[0], 3)
        with torch.inference_mode():
            lifter_start = time.perf_counter()
            lifted = trainer.splatter.lifter(images, cameras, match="hybrid")
            lifter_done = time.perf_counter()
            cloud = trainer.splatter.predictor(lifted, images, cameras)
            gaussian_done = time.perf_counter()
            feature_maps = {name: view.features.float() for name, view in lifted.cameras.items()}
            image_hw = {
                name: (int(image.shape[-2]), int(image.shape[-1])) for name, image in images.items()
            }
            glare = glare_for_cameras(
                trainer.relight,
                feature_maps,
                cloud,
                cameras,
                sun.float(),
                image_hw,
            )
            glare_done = time.perf_counter()
            volume = trainer.occupancy.forward_cloud(
                cloud,
                origin,
                self.settings.voxel_size,
                self.settings.grid_size,
            )
            occ_done = time.perf_counter()
        times = StageTimes(
            prepare_ms=_ms(started, prepare_done),
            lifter_ms=_ms(lifter_start, lifter_done),
            gaussians_ms=_ms(lifter_done, gaussian_done),
            glare_ms=_ms(gaussian_done, glare_done),
            occupancy_ms=_ms(glare_done, occ_done),
            total_ms=_ms(started, occ_done),
        )
        self.last_times = times
        self.last_total_ms = times.total_ms
        self.inferences += 1
        cloud_cpu = _cloud_to_cpu(cloud)
        glare_cpu = {name: tensor.detach().to("cpu").float() for name, tensor in glare.items()}
        volume_cpu = OccupancyVolume(
            occupancy_logits=volume.occupancy_logits.detach().to("cpu").float(),
            wire_logits=volume.wire_logits.detach().to("cpu").float(),
            wire_direction=volume.wire_direction.detach().to("cpu").float(),
            wire_offset=volume.wire_offset.detach().to("cpu").float(),
            origin=volume.origin.detach().to("cpu").float(),
            voxel_size=volume.voxel_size.detach().to("cpu").float(),
        )
        return InferenceResult(
            cloud=cloud_cpu,
            glare=glare_cpu,
            volume=volume_cpu,
            sun_world=sun.detach().to("cpu").numpy().reshape(-1)[:3].astype(np.float32),
            sun_source=source,
            times=times,
            network_hw={name: [int(hw[0]), int(hw[1])] for name, hw in image_hw.items()},
        )


def gpu_status() -> dict:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "cpu",
            "vram_total_bytes": None,
            "vram_used_bytes": None,
            "process_allocated_bytes": None,
        }
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_total_bytes": int(total),
        "vram_used_bytes": int(total - free),
        "process_allocated_bytes": int(torch.cuda.memory_allocated(0)),
    }


def _ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def _configs_from_yaml(path) -> tuple[ModelConfig, SplatConfig]:
    data = yaml.safe_load(path.read_text()) or {}
    indices = data.get("out_indices") or (1,)
    model = ModelConfig(
        name=str(data.get("name", "vit_tiny")),
        img_size=int(data.get("img_size", 32)),
        patch_size=int(data.get("patch_size", 8)),
        embed_dim=int(data.get("embed_dim", 32)),
        depth=int(data.get("depth", 2)),
        num_heads=int(data.get("num_heads", 4)),
        mlp_ratio=float(data.get("mlp_ratio", 2.0)),
        out_indices=tuple(int(i) for i in indices),
        cross_view_layers=int(data.get("cross_view_layers", 1)),
        pose_encoding=str(data.get("pose_encoding", "plucker")),
        pose_zero_init=bool(data.get("pose_zero_init", False)),
        num_frequencies=int(data.get("num_frequencies", 4)),
        normalize=str(data.get("normalize", "none")),
        gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
        init_values=float(data.get("init_values", 1.0)),
        depth_planes=int(data.get("depth_planes", 8)),
        depth_near=float(data.get("depth_near", 0.4)),
        depth_far=float(data.get("depth_far", 12.0)),
        depth_spacing=str(data.get("depth_spacing", "linear")),
        lift_stride=int(data.get("lift_stride", 4)),
        temperature=float(data.get("temperature", 0.02)),
        pretrained=None,
        pretrained_allow_download=False,
    )
    raw = data.get("splats") or {}
    splat = SplatConfig(
        slots_per_cell=int(raw.get("slots_per_cell", 4)),
        max_extra_gaussians=None
        if raw.get("max_extra_gaussians") in (None, "null")
        else int(raw["max_extra_gaussians"]),
        edge_threshold=float(raw.get("edge_threshold", 0.4)),
        offset_radius_m=float(raw.get("offset_radius_m", 0.5)),
        along_radius_m=float(raw.get("along_radius_m", 0.4)),
        min_scale_m=float(raw.get("min_scale_m", 1e-4)),
        max_scale_m=float(raw.get("max_scale_m", 2.0)),
        wire_transverse_max_m=float(raw.get("wire_transverse_max_m", 0.03)),
        wire_long_max_m=float(raw.get("wire_long_max_m", 1.0)),
        base_scale_m=float(raw.get("base_scale_m", 0.2)),
        extra_thin_scale_m=float(raw.get("extra_thin_scale_m", 0.01)),
        extra_long_scale_m=float(raw.get("extra_long_scale_m", 0.25)),
        opacity_init=float(raw.get("opacity_init", 0.1)),
    )
    return model, splat


def _cloud_to_cpu(cloud: GaussianCloud) -> GaussianCloud:
    return GaussianCloud(
        packed=cloud.packed.detach().to("cpu").float(),
        valid=cloud.valid.detach().to("cpu"),
        wire_prob=cloud.wire_prob.detach().to("cpu").float(),
        slot_id=cloud.slot_id.detach().to("cpu"),
        camera_index=cloud.camera_index.detach().to("cpu"),
        row=cloud.row.detach().to("cpu"),
        col=cloud.col.detach().to("cpu"),
        subslot=cloud.subslot.detach().to("cpu"),
        uv=cloud.uv.detach().to("cpu").float(),
        camera_order=tuple(cloud.camera_order),
        anchor_valid=cloud.anchor_valid.detach().to("cpu"),
    )


def sample_from_cameras(records: list[dict]) -> MultiCameraSample:
    """Build one sample from decoded RGB images and K, R, t."""
    cameras: dict[str, CameraFrame] = {}
    stamps = []
    for record in records:
        image = record["image"]
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(image))
        else:
            tensor = image
        frame = CameraFrame(
            name=str(record["name"]),
            image=tensor,
            K=record["K"],
            R=record["R"],
            t=record["t"],
            timestamp=float(record.get("timestamp", 0.0)),
        )
        cameras[frame.name] = frame
        stamps.append(frame.timestamp)
    if not cameras:
        raise ValueError("infer needs at least one camera")
    return MultiCameraSample(
        scene_id=str(records[0].get("scene_id", "infer")),
        sample_index=0,
        timestamp=float(min(stamps)),
        cameras=cameras,
    )
