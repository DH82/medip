"""Intensity transforms: Modality LUT and VOI LUT / windowing.

DICOM grayscale pipeline order (PS3.4 N.2.1.1):
  stored pixels -> Modality LUT -> VOI LUT -> Presentation LUT

This module implements Modality and VOI steps.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..config import VOIPolicy
from ..io.metadata import DicomMetadata

logger = logging.getLogger(__name__)


def _pydicom_luts():
    """Return (apply_modality_lut, apply_voi_lut, apply_windowing) from pydicom.

    ``pydicom.pixels`` is the home of these since pydicom 3; the old
    ``pydicom.pixel_data_handlers`` path still works but warns on every import
    and is removed in pydicom 4.
    """
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut, apply_windowing
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import (  # type: ignore[no-redef]
            apply_modality_lut,
            apply_voi_lut,
            apply_windowing,
        )
    return apply_modality_lut, apply_voi_lut, apply_windowing


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
            from ..io.dicom_reader import dcmread
            ds = dcmread(meta.file_path, stop_before_pixels=True)
            apply_modality_lut, _, _ = _pydicom_luts()
            return apply_modality_lut(pixels, ds).astype(np.float32)
        except Exception:
            logger.warning("Modality LUT Sequence apply failed, using slope/intercept")

    return (pixels.astype(np.float64) * meta.rescale_slope + meta.rescale_intercept).astype(np.float32)


# ── VOI LUT / Windowing ─────────────────────────────────────────────


ValueRange = Tuple[float, float]


def stored_value_range(meta: DicomMetadata) -> ValueRange:
    """Range of stored pixel values implied by BitsStored / PixelRepresentation."""
    bits = max(int(meta.bits_stored), 1)
    if meta.pixel_representation == 1:
        return float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1)
    return 0.0, float(2 ** bits - 1)


def rescaled_value_range(meta: DicomMetadata) -> ValueRange:
    """Stored range through Rescale Slope/Intercept.

    This is also the output range of pydicom's ``apply_windowing``, which
    derives its y-range the same way.
    """
    lo, hi = stored_value_range(meta)
    lo, hi = lo * meta.rescale_slope + meta.rescale_intercept, hi * meta.rescale_slope + meta.rescale_intercept
    return (min(lo, hi), max(lo, hi))


def modality_output_range(meta: DicomMetadata, applied: bool = True) -> Optional[ValueRange]:
    """Fixed range of the Modality LUT output, or None when it is not known.

    With a Modality LUT Sequence the range depends on the table, so None is
    returned and callers fall back to per-image behavior.
    """
    if not applied:
        return stored_value_range(meta)
    if meta.has_modality_lut_sequence:
        return None
    return rescaled_value_range(meta)


def apply_voi_transform(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: VOIPolicy = VOIPolicy.AUTO,
    voi_index: int = 0,
    output_max: float = 4095.0,
    valid_mask: Optional[np.ndarray] = None,
    window_output_max: Optional[float] = None,
) -> np.ndarray:
    """Apply VOI LUT or windowing based on policy. See :func:`apply_voi_transform_ranged`."""
    return apply_voi_transform_ranged(
        pixels, meta, policy, voi_index, output_max, valid_mask, window_output_max
    )[0]


def apply_voi_transform_ranged(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: VOIPolicy = VOIPolicy.AUTO,
    voi_index: int = 0,
    output_max: float = 4095.0,
    valid_mask: Optional[np.ndarray] = None,
    window_output_max: Optional[float] = None,
    input_range: Optional[ValueRange] = None,
) -> Tuple[np.ndarray, Optional[ValueRange]]:
    """Apply VOI LUT or windowing; also return the fixed range of the output.

    The range is what the transform *can* produce, not the image's min/max, so
    MONOCHROME1 can be inverted the same way on every image afterwards.

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
        MIN_MAX_WINDOW - window spanning the image min/max (DCMTK --min-max-window)
        LEGACY_LINEAR  - manual linear window (original MakePNG behavior)
        LEGACY_SIGMOID - manual sigmoid window (original MakePNG behavior)
    voi_index : int
        Which window/LUT entry to use (for multi-window DICOM).
    output_max : float
        Maximum output value for min-max and legacy modes.
    valid_mask : array or None
        Boolean mask (True = meaningful pixel) used by MIN_MAX_WINDOW to keep
        Pixel Padding regions out of the min/max estimate. Ignored otherwise.
    window_output_max : float or None
        Rescale the window / VOI LUT output from its fixed range (pydicom's
        y-range for a window, ``[0, 2**bits - 1]`` for a LUT) onto
        ``[0, window_output_max]``. The factor depends only on the header, so
        the mapping stays comparable across images — unlike a per-image
        min/max stretch. None (default) leaves the range as-is.
    input_range : (lo, hi) or None
        Fixed range of *pixels*; returned unchanged when no VOI is applied.

    Returns
    -------
    (pixels, (lo, hi) or None)
    """
    if policy == VOIPolicy.NONE:
        return pixels, input_range

    if policy == VOIPolicy.AUTO:
        if meta.has_voi_lut_sequence:
            return _apply_voi_lut_sequence(pixels, meta, voi_index, window_output_max)
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index, window_output_max)
        logger.debug("No VOI LUT or window found, returning as-is")
        return pixels, input_range

    if policy == VOIPolicy.LUT:
        if meta.has_voi_lut_sequence:
            return _apply_voi_lut_sequence(pixels, meta, voi_index, window_output_max)
        logger.warning("VOI LUT Sequence requested but not found, returning as-is")
        return pixels, input_range

    if policy == VOIPolicy.WINDOW:
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index, window_output_max)
        logger.warning("Window requested but WC/WW not found, returning as-is")
        return pixels, input_range

    if policy == VOIPolicy.MIN_MAX_WINDOW:
        return _min_max_window(pixels, output_max, valid_mask), (0.0, float(output_max))

    if policy in (VOIPolicy.LEGACY_LINEAR, VOIPolicy.LEGACY_SIGMOID):
        if not meta.has_window:
            return pixels, input_range
        fn = _legacy_linear if policy == VOIPolicy.LEGACY_LINEAR else _legacy_sigmoid
        return fn(pixels, meta, output_max), (0.0, float(output_max))

    return pixels, input_range


def _rescale_onto(
    values: np.ndarray, value_range: ValueRange, output_max: Optional[float]
) -> Tuple[np.ndarray, ValueRange]:
    """Map a fixed *value_range* onto [0, output_max] (no-op when output_max is None)."""
    if output_max is None:
        return values, value_range
    lo, hi = value_range
    if hi <= lo:
        return values, value_range
    return ((values - lo) * (output_max / (hi - lo))).astype(np.float32), (0.0, float(output_max))


def _voi_lut_range(meta: DicomMetadata, voi_index: int) -> ValueRange:
    """Output range of a VOI LUT Sequence item: [0, 2**bits - 1] from LUTDescriptor."""
    try:
        item = meta.voi_lut_sequence[min(voi_index, len(meta.voi_lut_sequence) - 1)]
        bits = int(item.LUTDescriptor[2])
    except Exception:
        bits = int(meta.bits_stored)
    return 0.0, float(2 ** bits - 1)


def _apply_voi_lut_sequence(
    pixels: np.ndarray, meta: DicomMetadata, voi_index: int,
    output_max: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[ValueRange]]:
    """Apply VOI LUT Sequence using pydicom."""
    try:
        from ..io.dicom_reader import dcmread
        ds = dcmread(meta.file_path, stop_before_pixels=True)
        _, apply_voi_lut, _ = _pydicom_luts()
        out = apply_voi_lut(pixels, ds, index=voi_index).astype(np.float32)
    except Exception:
        logger.warning("VOI LUT Sequence apply failed, trying window fallback")
        if meta.has_window:
            return _apply_window(pixels, meta, voi_index, output_max)
        return pixels, None
    return _rescale_onto(out, _voi_lut_range(meta, voi_index), output_max)


def _apply_window(
    pixels: np.ndarray,
    meta: DicomMetadata,
    voi_index: int,
    output_max: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[ValueRange]]:
    """Apply windowing using pydicom's apply_windowing.

    ``voi_index`` selects the window when the DICOM carries several (some
    mammography detectors ship three). pydicom maps into its y-range — the
    stored range through Rescale Slope/Intercept, ``[0, 2**BitsStored - 1]``
    for a plain unsigned image; pass ``output_max`` to rescale that onto a
    different ceiling, e.g. the full 16-bit range for PNG16 export.
    """
    try:
        from ..io.dicom_reader import dcmread
        ds = dcmread(meta.file_path, stop_before_pixels=True)
        _, _, apply_windowing = _pydicom_luts()
        windowed = apply_windowing(pixels, ds, index=voi_index).astype(np.float32)
    except Exception:
        logger.warning("pydicom windowing failed, using manual linear")
        ceiling = output_max or float(pixels.max())
        return _legacy_linear(pixels, meta, output_max=ceiling), (0.0, float(ceiling))

    return _rescale_onto(windowed, rescaled_value_range(meta), output_max)


def _min_max_window(
    pixels: np.ndarray,
    output_max: float,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Linear window spanning the image's own min/max.

    Reproduces DCMTK ``dcmj2pnm --min-max-window``: the VOI window is derived
    from the image itself (center = (min+max)/2, width = max-min) rather than
    from Window Center/Width tags, so the full stored dynamic range is mapped
    onto the output range. Used by the png16_minmax presets.

    Note this is a per-image transform: absolute pixel values are no longer
    comparable across images. Prefer ``VOIPolicy.NONE`` when cross-image
    calibration matters.
    """
    if valid_mask is not None and valid_mask.shape == pixels.shape and valid_mask.any():
        sample = pixels[valid_mask]
    else:
        sample = pixels

    if sample.size == 0:
        return pixels

    min_val = float(sample.min())
    max_val = float(sample.max())

    if max_val <= min_val:
        logger.warning("min-max window: constant image (value %.3f), returning zeros", min_val)
        return np.zeros_like(pixels, dtype=np.float32)

    norm = (np.clip(pixels, min_val, max_val) - min_val) / (max_val - min_val)
    return (norm * output_max).astype(np.float32)


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
