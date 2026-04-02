"""Preset profile registry.

Presets are organized by modality. Each modality module provides a PRESETS dict
mapping short names to factory functions. The registry combines them using
``modality/name`` keys (e.g., ``mammography/learning_raw``).

For backward compatibility, presets from the *default* modality can also be
looked up by short name alone (e.g., ``learning_raw``).
"""

from __future__ import annotations

from typing import Callable, Dict, List

from ..config import PipelineConfig
from ..exceptions import InvalidProfileError
from . import mammography

# ── Registry ─────────────────────────────────────────────────────────

_MODALITY_PRESETS: Dict[str, Dict[str, Callable[[], PipelineConfig]]] = {}

DEFAULT_MODALITY = "mammography"


def register_modality(modality: str, presets: Dict[str, Callable[[], PipelineConfig]]) -> None:
    """Register preset profiles for a modality."""
    _MODALITY_PRESETS[modality] = presets


def _build_full_registry() -> Dict[str, Callable[[], PipelineConfig]]:
    """Build a flat registry with ``modality/name`` keys."""
    registry: Dict[str, Callable[[], PipelineConfig]] = {}
    for modality, presets in _MODALITY_PRESETS.items():
        for name, factory in presets.items():
            registry[f"{modality}/{name}"] = factory
    return registry


def get_preset(name: str) -> PipelineConfig:
    """Return a PipelineConfig for the given preset name.

    Accepts both qualified names (``mammography/learning_raw``) and
    short names (``learning_raw`` — resolved against DEFAULT_MODALITY).
    """
    full_registry = _build_full_registry()

    # Try exact match first (qualified name)
    if name in full_registry:
        return full_registry[name]()

    # Try short name against default modality
    qualified = f"{DEFAULT_MODALITY}/{name}"
    if qualified in full_registry:
        return full_registry[qualified]()

    available = list(full_registry.keys())
    raise InvalidProfileError(
        f"Unknown profile '{name}'. Available: {available}"
    )


def list_presets(modality: str | None = None) -> List[str]:
    """Return available preset names.

    If *modality* is given, return only that modality's presets (short names).
    Otherwise return all qualified names.
    """
    if modality is not None:
        presets = _MODALITY_PRESETS.get(modality, {})
        return list(presets.keys())
    return list(_build_full_registry().keys())


def list_modalities() -> List[str]:
    """Return registered modality names."""
    return list(_MODALITY_PRESETS.keys())


# ── Auto-register built-in modalities ────────────────────────────────

register_modality(mammography.MODALITY, mammography.PRESETS)
