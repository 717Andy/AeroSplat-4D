"""Load composed YAML configs.

Hydra is used for composition (data and rig groups). Training entry points in
later phases should call :func:`load_config` rather than parsing YAML ad hoc.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from aerosplat.paths import find_configs_dir


def load_config(
    overrides: list[str] | None = None,
    config_name: str = "config",
) -> DictConfig:
    """Compose ``configs/<config_name>.yaml`` with Hydra-style overrides.

    Example: ``load_config(["data=tartanair", "data.root=/data/P000"])``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = find_configs_dir()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
