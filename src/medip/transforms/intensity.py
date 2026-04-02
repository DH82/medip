"""Intensity transforms: Modality LUT and VOI LUT / windowing.

DICOM grayscale pipeline order (PS3.4 N.2.1.1):
  stored pixels -> Modality LUT -> VOI LUT -> Presentation LUT

This module implements Modality and VOI steps.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import VOIPolicy
from ..io.metadata import DicomMetadata

logger = logging.getLogger(__name__)


# ── Modality LUT / Rescale ───────────────────────────────────────────


def apply_modality_transform(
    pixels: np.ndarray,
    meta: DicomMetadata,
    apply: bool = True,
) -> np.ndarray:
    """Apply Modality LUT (Rescale Slope/Intercept or LUT Sequence).

    Uses pydicom.pixels.apply_modality_lut when available,
    falls back to manual slope/intercept.
    """
    if not apply:
        return pixels.astype(np.float32)

    # Try pydicom's standard implementation first
    if meta.has_modality_lut_sequence:
        try:
            import pydicom
            ds = pydicom.dcmread(meta.file_path, stop_before_pixels=True)
            from pydicom.pixel_data_handlers.util import apply_modality_lut
            return apply_modality_lut(pixels, ds).astype(np.float32)
        except Exception:
            logger.warning("Modality LUT Sequence apply failed, using slope/intercept")

    return (pixels.astype(np.float64) * meta.rescale_slope + meta.rescale_intercept).astype(np.float32)


# ── VOI LUT / Windowing ─────────────────────────────────────────────


def apply_voi_transform(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: VOIPolicy = VOIPolicy.AUTO,
    voi_index: int = 0,
    output_max: float = 4095.0,
) -> np.ndarray:
    """Apply VOI LUT or windowing based on policy.

    Parameters
    ----------
    pixels : array
        Input after modality transform.
    meta : DicomMetadata
    policy : VOIPolicy
        NONE   - skip, return as-is
        AUTO   - prefer VOI LUT Sequence, fallback to window
        LUT    - use VOI LUT Sequence only
        WINDOW - use Window Center/Width only
        LEGACY_LINEAR  - manual linear window (original MakePNG behavior)
        LEGACY_SIGMOID - manual sigmoid window (original MakePNG behavior)
    voi_index : int
        Which window/LUT entry to use (for multi-window DICOM).
    output_max : float
        Maximum output value for legacy modes.
    """
    if policy == VOIPolicy.NONE:
        return pixels

    if policy == VOIPolicy.AUTO:
        if meta.has_voi_lut_sequence:
            return _apply_voi_lut_sequence(pixels, meta, voi_index)
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index)
        logger.debug("No VOI LUT or window found, returning as-is")
        return pixels

    if policy == VOIPolicy.LUT:
        if meta.has_voi_lut_sequence:
            return _apply_voi_lut_sequence(pixels, meta, voi_index)
        logger.warning("VOI LUT Sequence requested but not found, returning as-is")
        return pixels

    if policy == VOIPolicy.WINDOW:
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index)
        logger.warning("Window requested but WC/WW not found, returning as-is")
        return pixels

    if policy == VOIPolicy.LEGACY_LINEAR:
        return _legacy_linear(pixels, meta, output_max)

    if policy == VOIPolicy.LEGACY_SIGMOID:
        return _legacy_sigmoid(pixels, meta, output_max)

    return pixels


def _apply_voi_lut_sequence(
    pixels: np.ndarray, meta: DicomMetadata, voi_index: int
) -> np.ndarray:
    """Apply VOI LUT Sequence using pydicom."""
    try:
        import pydicom
        ds = pydicom.dcmread(meta.file_path, stop_before_pixels=True)
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        return apply_voi_lut(pixels, ds, index=voi_index).astype(np.float32)
    except Exception:
        logger.warning("VOI LUT Sequence apply failed, trying window fallback")
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index)
        return pixels


def _apply_window(
    pixels: np.ndarray, meta: DicomMetadata, voi_index: int
) -> np.ndarray:
    """Apply windowing using pydicom's apply_windowing."""
    try:
        import pydicom
        ds = pydicom.dcmread(meta.file_path, stop_before_pixels=True)
        from pydicom.pixel_data_handlers.util import apply_windowing
        return apply_windowing(pixels, ds, index=voi_index).astype(np.float32)
    except Exception:
        logger.warning("pydicom windowing failed, using manual linear")
        return _legacy_linear(pixels, meta, output_max=pixels.max())


def _legacy_linear(
    pixels: np.ndarray, meta: DicomMetadata, output_max: float
) -> np.ndarray:
    """Manual linear window — reproduces original MakePNG.py behavior."""
    if not meta.has_window:
        return pixels

    wc = meta.window_center[0]
    ww = meta.window_width[0]
    min_val = wc - ww / 2.0
    max_val = wc + ww / 2.0
    clipped = np.clip(pixels, min_val, max_val)
    norm = (clipped - min_val) / (max_val - min_val + 1e-8)
    return (np.clip(norm, 0, 1) * output_max).astype(np.float32)


def _legacy_sigmoid(
    pixels: np.ndarray, meta: DicomMetadata, output_max: float
) -> np.ndarray:
    """Manual sigmoid window — reproduces original MakePNG.py behavior."""
    if not meta.has_window:
        return pixels

    wc = meta.window_center[0]
    ww = meta.window_width[0]
    sigmoid_val = 1.0 / (1.0 + np.exp(-4.0 * (pixels - wc) / (ww + 1e-8)))
    return (sigmoid_val * output_max).astype(np.float32)
