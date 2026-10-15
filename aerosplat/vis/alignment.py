"""Project depth to world points and check that cameras agree.

Two checks:

* Self-reprojection. Unproject a depth map and project it back. Error above a
  small fraction of a pixel means K, R, t, or the depth kind is inconsistent.
* Cross-camera depth. Points from camera A that land on a valid pixel of
  camera B should agree with B's planar depth when both cameras see the same
  surface. Cameras that do not overlap report zero overlap instead of a fake
  error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosplat.geometry.camera import project_points, reprojection_error, unproject_depth
from aerosplat.schema import CameraFrame, MultiCameraSample


@dataclass
class PairStat:
    source: str
    target: str
    overlap: int
    median_abs_m: float
    inlier_fraction: float


@dataclass
class AlignmentReport:
    reprojection_max_px: dict[str, float]
    reprojection_median_px: dict[str, float]
    pairs: list[PairStat] = field(default_factory=list)
    points: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "reprojection_max_px": self.reprojection_max_px,
            "reprojection_median_px": self.reprojection_median_px,
            "pairs": [
                {
                    "source": pair.source,
                    "target": pair.target,
                    "overlap": pair.overlap,
                    "median_abs_m": pair.median_abs_m,
                    "inlier_fraction": pair.inlier_fraction,
                }
                for pair in self.pairs
            ],
        }


def check_alignment(
    sample: MultiCameraSample,
    *,
    stride: int = 1,
    min_depth: float = 0.05,
    max_depth: float = 40.0,
    agreement_m: float = 0.25,
) -> AlignmentReport:
    reproj_max: dict[str, float] = {}
    reproj_med: dict[str, float] = {}
    points: dict[str, np.ndarray] = {}
    frames = list(sample.cameras.values())
    for frame in frames:
        if frame.depth is None:
            continue
        depth = frame.depth.detach().cpu().numpy()
        K = frame.K.detach().cpu().numpy()
        R = frame.R.detach().cpu().numpy()
        t = frame.t.detach().cpu().numpy()
        err = reprojection_error(
            depth, K, R, t, min_depth=min_depth, max_depth=max_depth, stride=stride
        )
        reproj_max[frame.name] = float(err.max()) if err.size else float("nan")
        reproj_med[frame.name] = float(np.median(err)) if err.size else float("nan")
        points[frame.name] = unproject_depth(
            depth, K, R, t, min_depth=min_depth, max_depth=max_depth, stride=stride
        )
    pairs: list[PairStat] = []
    for source in frames:
        for target in frames:
            if source.name == target.name or source.depth is None or target.depth is None:
                continue
            pairs.append(
                _pair(
                    source,
                    target,
                    stride=stride,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    agreement_m=agreement_m,
                )
            )
    return AlignmentReport(reproj_max, reproj_med, pairs, points)


def _pair(
    source: CameraFrame,
    target: CameraFrame,
    *,
    stride: int,
    min_depth: float,
    max_depth: float,
    agreement_m: float,
) -> PairStat:
    src_depth = source.depth.detach().cpu().numpy()
    tgt_depth = target.depth.detach().cpu().numpy()
    K_s = source.K.detach().cpu().numpy()
    R_s = source.R.detach().cpu().numpy()
    t_s = source.t.detach().cpu().numpy()
    K_t = target.K.detach().cpu().numpy()
    R_t = target.R.detach().cpu().numpy()
    t_t = target.t.detach().cpu().numpy()
    pts = unproject_depth(
        src_depth, K_s, R_s, t_s, min_depth=min_depth, max_depth=max_depth, stride=stride
    )
    if pts.shape[0] == 0:
        return PairStat(source.name, target.name, 0, float("nan"), 0.0)
    uv, z = project_points(pts, K_t, R_t, t_t)
    ui = np.rint(uv[:, 0]).astype(np.int64)
    vi = np.rint(uv[:, 1]).astype(np.int64)
    height, width = tgt_depth.shape
    inside = (z > min_depth) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    if not np.any(inside):
        return PairStat(source.name, target.name, 0, float("nan"), 0.0)
    sampled = tgt_depth[vi[inside], ui[inside]].astype(np.float64)
    z = z[inside]
    valid = np.isfinite(sampled) & (sampled > min_depth) & (sampled <= max_depth)
    if not np.any(valid):
        return PairStat(source.name, target.name, 0, float("nan"), 0.0)
    err = np.abs(sampled[valid] - z[valid])
    inliers = err < agreement_m
    return PairStat(
        source.name,
        target.name,
        int(err.size),
        float(np.median(err)),
        float(np.mean(inliers)),
    )


def save_report(report: AlignmentReport, directory: Path, sample: MultiCameraSample) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "alignment.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    names = []
    chunks = []
    for name, pts in report.points.items():
        if pts.size == 0:
            continue
        names.append(np.full((pts.shape[0],), name))
        chunks.append(pts)
    if chunks:
        stacked = np.concatenate(chunks, axis=0)
        labels = np.concatenate(names, axis=0)
        rng = np.random.default_rng(0)
        if stacked.shape[0] > 8000:
            keep = rng.choice(stacked.shape[0], size=8000, replace=False)
            stacked = stacked[keep]
            labels = labels[keep]
        np.savez_compressed(directory / "points.npz", points=stacked, camera=labels)
        _save_figure(directory / "alignment.png", stacked, labels, sample)


def _save_figure(
    path: Path, points: np.ndarray, labels: np.ndarray, sample: MultiCameraSample
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cameras = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (ix, iy, title) in zip(axes, ((0, 1, "top-down X-Y"), (0, 2, "side X-Z")), strict=True):
        for index, name in enumerate(cameras):
            mask = labels == name
            ax.scatter(
                points[mask, ix],
                points[mask, iy],
                s=2,
                c=[cmap(index % 10)],
                label=name,
                linewidths=0,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("east (m)" if ix == 0 else "")
        ax.grid(True, linewidth=0.3, alpha=0.4)
    axes[0].legend(markerscale=4, frameon=False)
    fig.suptitle(f"{sample.scene_id} t={sample.timestamp:.3f}s")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def format_report(report: AlignmentReport) -> str:
    lines = ["self-reprojection (pixels):"]
    for name in report.reprojection_max_px:
        lines.append(
            f"  {name:12s}  median {report.reprojection_median_px[name]:.4e}"
            f"  max {report.reprojection_max_px[name]:.4e}"
        )
    lines.append("cross-camera depth agreement:")
    for pair in report.pairs:
        med = "n/a" if pair.median_abs_m != pair.median_abs_m else f"{pair.median_abs_m:.3f} m"
        lines.append(
            f"  {pair.source:12s} -> {pair.target:12s}  "
            f"overlap {pair.overlap:5d}  median {med}  "
            f"inliers {pair.inlier_fraction:.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check multi-camera depth alignment.")
    parser.add_argument("--root", type=Path, default=None, help="Canonical dataset root.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--save", type=Path, default=None, help="Directory for the PNG, NPZ, and JSON."
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    from aerosplat.config import load_config
    from aerosplat.data.dataset import build_dataset
    from aerosplat.data.fixture import default_fixture_root, generate_fixture

    if args.root is not None:
        overrides = [f"data.root={args.root}", *args.overrides]
    else:
        overrides = list(args.overrides)
    cfg = load_config(overrides)
    root = Path(str(cfg.data.root))
    if str(cfg.data.source) == "canonical" and not (root / "dataset.json").is_file():
        if args.root is None:
            generate_fixture(default_fixture_root())
            root = default_fixture_root()
        else:
            raise SystemExit(f"{root} has no dataset.json")
    # Rebuild in case we just generated the default fixture into a path the
    # config already points at.
    cfg = load_config(overrides)
    dataset = build_dataset(cfg)
    sample = dataset[args.index]
    report = check_alignment(sample, stride=args.stride)
    print(format_report(report))
    if args.save is not None:
        save_report(report, args.save, sample)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
