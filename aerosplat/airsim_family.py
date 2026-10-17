"""Shared AirSim-family simulator profiles.

Colosseum and Cosys-AirSim speak the same vehicle RPC (``MultirotorClient``,
``ImageRequest``, ``simGetImages``, ``simGetVehiclePose``) and the same body
frame (X forward, Y right, Z down). The differences that change a recording
are collected here so the capture writer and the raw-log parser share one
table.

ImageType integers 0–9 match upstream AirSim. Cosys-AirSim adds Lighting (10)
and Annotation (11). Annotation requests also carry a layer name; this package
does not record that type.
"""

from __future__ import annotations

from dataclasses import dataclass

from aerosplat.data.errors import DataError

# ImageCaptureBase::ImageType. Values 0–9 are shared. 10 and 11 are Cosys-AirSim.
SCENE = 0
DEPTH_PLANAR = 1
DEPTH_PERSPECTIVE = 2
DEPTH_VIS = 3
DISPARITY_NORMALIZED = 4
SEGMENTATION = 5
SURFACE_NORMALS = 6
INFRARED = 7
OPTICAL_FLOW = 8
OPTICAL_FLOW_VIS = 9
LIGHTING = 10
ANNOTATION = 11

# DepthPlanar is camera-plane Z in meters. DepthPerspective is Euclidean range.
METRIC_DEPTH_KINDS = {
    DEPTH_PLANAR: "planar_z",
    DEPTH_PERSPECTIVE: "euclidean",
}

# These integers are easy to mistake for metric depth. They are not.
NON_METRIC_DEPTH = {
    DEPTH_VIS: (
        "DepthVis is a visualization (white at about 100 m, black at 0). "
        "Metric depth is DepthPlanar (ImageType 1, planar Z meters) or "
        "DepthPerspective (ImageType 2, Euclidean range)."
    ),
    DISPARITY_NORMALIZED: (
        "DisparityNormalized is (Xl - Xr) / Xmax in [0, 1]. It is not a depth in meters."
    ),
}

SIMULATORS = ("colosseum", "cosys")


@dataclass(frozen=True)
class SimulatorProfile:
    """One AirSim-family simulator.

    ``python_modules`` is the import list for the live client, in try order.
    Cosys-AirSim is imported as ``cosysairsim`` and is not aliased to ``airsim``,
    so a leftover Colosseum install cannot answer a Cosys capture.
    """

    name: str
    settings_version: float
    python_modules: tuple[str, ...]
    install_url: str
    archived: bool
    flip_scene_rgb: bool
    flip_depth: bool
    flip_segmentation: bool
    segmentation_id_space: str
    segmentation_note: str


_COLOSSEUM = SimulatorProfile(
    name="colosseum",
    settings_version=1.2,
    python_modules=("airsim", "colosseum"),
    install_url="https://github.com/CodexLabsLLC/Colosseum",
    archived=True,
    flip_scene_rgb=False,
    flip_depth=False,
    flip_segmentation=False,
    segmentation_id_space="mesh_object",
    segmentation_note=(
        "ImageType 5 is an RGB palette of mesh-name object IDs assigned with "
        "simSetSegmentationObjectID. The palette is a color index for those mesh "
        "names. Cosys-AirSim uses the same ImageType integer for per-mesh instance "
        "colors; the two id spaces are different."
    ),
)

_COSYS = SimulatorProfile(
    name="cosys",
    settings_version=2.0,
    python_modules=("cosysairsim",),
    install_url="https://github.com/Cosys-Lab/Cosys-AirSim",
    archived=False,
    # Cosys documents np.flipud on uncompressed Scene RGB from simGetImages.
    # The segmentation example does not flip, and float depth is row-major
    # planar Z / perspective range with no flip in the official depth path.
    flip_scene_rgb=True,
    flip_depth=False,
    flip_segmentation=False,
    segmentation_id_space="instance",
    segmentation_note=(
        "ImageType 5 is instance segmentation. InitialInstanceSegmentation true "
        "assigns a color index per mesh component at startup. Read the palette "
        "with simGetSegmentationColorMap, the mesh list with "
        "simListInstanceSegmentationObjects, and poses with "
        "simListInstanceSegmentationPoses(ned=True). ned=False returns Unreal "
        "coordinates, as does simGetObjectPose when ned is false. "
        "simSetSegmentationObjectID does not check id collisions. Infrared "
        "(ImageType 7) is grey (id, id, id). Annotation (ImageType 11) needs a "
        "layer name and is not recorded."
    ),
)

_PROFILES = {profile.name: profile for profile in (_COLOSSEUM, _COSYS)}


def get_profile(name: str) -> SimulatorProfile:
    """Return the profile for ``colosseum`` or ``cosys``."""
    key = str(name).strip().lower()
    profile = _PROFILES.get(key)
    if profile is None:
        raise DataError(
            f"unknown AirSim-family simulator {name!r}. Expected one of {list(SIMULATORS)}."
        )
    return profile


def resolve_simulator(requested: str | None, settings_version: float) -> SimulatorProfile:
    """Pick a profile, checking it against ``SettingsVersion`` when both are known.

    Cosys-AirSim refuses settings older than 2.0. Colosseum / AirSim 1.x files
    use 1.2. Passing the wrong name is an error rather than a silent load.
    """
    if requested is None:
        if settings_version >= 2.0:
            return get_profile("cosys")
        if settings_version > 0:
            return get_profile("colosseum")
        raise DataError(
            "settings.json has no SettingsVersion. Pass simulator='colosseum' or 'cosys'."
        )
    profile = get_profile(requested)
    if profile.name == "cosys" and settings_version < 2.0:
        raise DataError(
            f"Cosys-AirSim requires SettingsVersion >= 2.0 (file has {settings_version}). "
            "Colosseum 1.2 settings are a different schema."
        )
    if profile.name == "colosseum" and settings_version >= 2.0:
        raise DataError(
            f"Colosseum settings use SettingsVersion 1.2 (file has {settings_version}). "
            "Use simulator='cosys' for a Cosys-AirSim file."
        )
    return profile


def depth_kind_from_image_type(name: str) -> str:
    """Map a config image-type name onto ``planar_z`` or ``euclidean``."""
    key = str(name).strip().lower()
    table = {
        "depth_planar": "planar_z",
        "planar_z": "planar_z",
        "scene_depth": "planar_z",
        "depth_perspective": "euclidean",
        "euclidean": "euclidean",
        "perspective": "euclidean",
    }
    if key in table:
        return table[key]
    if key in {"depth_vis", "depthvis", "disparity", "disparity_normalized"}:
        raise DataError(
            f"image type {name!r} is not metric depth. "
            "Use depth_planar (ImageType 1) or depth_perspective (ImageType 2)."
        )
    raise DataError(
        f"unknown depth image type {name!r}. Expected depth_planar or depth_perspective."
    )
