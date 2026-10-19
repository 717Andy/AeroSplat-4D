"""Build a TensorRT engine from an AeroSplat ONNX file.

The engine is not portable. Build it on the GPU that will run it: a Jetson
Orin and a desktop RTX do not load each other's engines. FP16 is the build
this script requests by default. INT8 is a calibrator stub: it yields batches
and a cache file, and it is only attached when ``--int8`` is set and TensorRT
is installed. CI does not install TensorRT and does not build an engine.

The ONNX file is the network only. The rasterizer stays a separate CUDA call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class Int8Calibrator:
    """Batches for a TensorRT entropy calibrator.

    This class does not import TensorRT. ``as_trt`` wraps it when the build
    runs on a machine that has the package. Each batch is a float32 array
    named by the ONNX input it feeds.
    """

    def __init__(self, batches: list[dict[str, np.ndarray]], cache_file: Path) -> None:
        if not batches:
            raise ValueError("INT8 calibration needs at least one batch")
        self.batches = list(batches)
        self.cache_file = Path(cache_file)
        self._index = 0

    def __len__(self) -> int:
        return len(self.batches)

    def reset(self) -> None:
        self._index = 0

    def next_batch(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self.batches):
            return None
        batch = self.batches[self._index]
        self._index += 1
        return batch

    def read_cache(self) -> bytes | None:
        if not self.cache_file.is_file():
            return None
        return self.cache_file.read_bytes()

    def write_cache(self, blob: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(blob)


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    precision: str = "fp16",
    workspace_gb: float = 2.0,
    calibrator: Int8Calibrator | None = None,
) -> Path:
    """Compile ``onnx_path`` with TensorRT. Import happens here, not at module import."""
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"precision must be fp32, fp16, or int8, got {precision!r}")
    if precision == "int8" and calibrator is None:
        raise ValueError("int8 requires an Int8Calibrator")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not installed. Build this engine on the Jetson Orin or the "
            "RTX machine that will run it. The ONNX file itself is portable."
        ) from exc
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    onnx_path = Path(onnx_path)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this GPU does not advertise fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this GPU does not advertise fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _trt_calibrator(trt, calibrator)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)
    return engine_path


def _trt_calibrator(trt, calibrator: Int8Calibrator):
    """Adapt :class:`Int8Calibrator` to ``IInt8EntropyCalibrator2``."""

    class _Adapter(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._source = calibrator
            self._source.reset()
            self._device = None

        def get_batch_size(self) -> int:
            first = next(iter(self._source.batches[0].values()))
            return int(first.shape[0])

        def get_batch(self, names):
            batch = self._source.next_batch()
            if batch is None:
                return None
            import torch

            pointers = []
            for name in names:
                array = np.ascontiguousarray(batch[name].astype(np.float32))
                tensor = torch.from_numpy(array).cuda()
                pointers.append(int(tensor.data_ptr()))
                self._device = tensor
            return pointers

        def read_calibration_cache(self):
            return self._source.read_cache()

        def write_calibration_cache(self, cache):
            self._source.write_cache(bytes(cache))

    return _Adapter()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory of .npy batches for INT8. Each file is one input array.",
    )
    args = parser.parse_args(argv)
    calibrator = None
    if args.precision == "int8":
        if args.calibration_dir is None:
            raise SystemExit("--calibration-dir is required for int8")
        batches = []
        for path in sorted(args.calibration_dir.glob("*.npy")):
            batches.append({path.stem: np.load(path)})
        calibrator = Int8Calibrator(batches, args.engine.with_suffix(".calibration"))
    path = build_engine(
        args.onnx,
        args.engine,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        calibrator=calibrator,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
