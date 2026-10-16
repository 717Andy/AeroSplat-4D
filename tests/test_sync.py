"""Asymmetric frame rates: nearest image vs interpolated pose."""

import numpy as np
import pytest

from aerosplat.data.errors import DataError
from aerosplat.data.records import FrameRecord, SyncConfig
from aerosplat.data.sync import nearest_index, synchronize
from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.coordinates import c2w_to_w2c


def _frame(index, time, center):
    R, t = c2w_to_w2c(np.eye(3), np.asarray(center, dtype=np.float64))
    return FrameRecord(
        frame_index=index,
        timestamp=time,
        image_path=f"img_{index}.png",
        depth_path=None,
        K=np.eye(3),
        R=R,
        t=t,
    )


def _streams():
    # Identity rotation: camera center is the stored center. Front at 10 Hz,
    # side at 10 Hz offset by 20 ms, down at 5 Hz.
    front = [_frame(i, t, [t, 0, 0]) for i, t in enumerate((0.0, 0.1, 0.2))]
    side = [_frame(i, t, [t, 1, 0]) for i, t in enumerate((0.02, 0.12, 0.22))]
    down = [_frame(i, t, [t, 0, 1]) for i, t in enumerate((0.0, 0.2))]
    return {"front": front, "side": side, "down": down}


def test_nearest_index_tie_keeps_the_earlier_frame():
    times = np.array([0.0, 0.2])
    assert nearest_index(times, 0.1) == 0


def test_interpolate_policy_keeps_capture_pose_and_query_pose_apart():
    stamps = synchronize(
        "s",
        _streams(),
        SyncConfig(
            policy="interpolate_pose",
            reference_camera="front",
            max_image_dt=0.15,
            camera_order=("front", "side", "down"),
        ),
    )
    assert [s.timestamp for s in stamps] == [0.0, 0.1, 0.2]
    mid = stamps[1]
    side = mid.cameras["side"]
    # Nearest side image is 0.12 (dt 0.02) rather than 0.02 (dt 0.08).
    assert side.frame.timestamp == pytest.approx(0.12)
    assert side.sync == "interpolated"
    assert np.allclose(camera_center(side.frame.R, side.frame.t), [0.12, 1.0, 0.0])
    assert np.allclose(camera_center(side.R_query, side.t_query), [0.1, 1.0, 0.0])
    # Down images are 0.0 and 0.2, equally far from 0.1. The earlier one is used
    # for the image; the pose is the midpoint.
    down = mid.cameras["down"]
    assert down.frame.timestamp == 0.0
    assert np.allclose(camera_center(down.R_query, down.t_query), [0.1, 0.0, 1.0])


def test_tight_tolerance_drops_the_slow_camera():
    stamps = synchronize(
        "s",
        _streams(),
        SyncConfig(
            policy="nearest",
            reference_camera="front",
            max_image_dt=0.05,
            require_all_cameras=True,
        ),
    )
    # t=0.1: down's nearest frame is 0.1 s away, which is over the limit.
    assert [s.timestamp for s in stamps] == [0.0, 0.2]
    assert stamps[0].cameras["side"].sync == "nearest"
    assert stamps[0].cameras["side"].frame.timestamp == pytest.approx(0.02)


def test_query_before_the_stream_is_clamped():
    stamps = synchronize(
        "s",
        _streams(),
        SyncConfig(policy="interpolate_pose", reference_camera="front", max_image_dt=0.05),
    )
    side = stamps[0].cameras["side"]
    assert side.sync == "clamped"
    assert np.allclose(camera_center(side.R_query, side.t_query), [0.02, 1.0, 0.0])


def test_missing_reference_camera_errors():
    with pytest.raises(DataError):
        synchronize("s", _streams(), SyncConfig(reference_camera="nose"))
