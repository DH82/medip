"""Photometric Interpretation handling.

MONOCHROME1: 0 = white (high signal), max = black
MONOCHROME2: 0 = black, max = white (most common)

For training, MONOCHROME1 images should typically be inverted to MONOCHROME2
so pixel intensity consistently maps to tissue density.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..config import PhotometricPolicy
from ..io.metadata import DicomMetadata

logger = logging.getLogger(__name__)


def apply_photometric(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: PhotometricPolicy = PhotometricPolicy.AUTO,
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Handle photometric interpretation.

    AUTO: invert MONOCHROME1 to MONOCHROME2.
    KEEP: leave as-is.
    INVERT_MONO1: explicitly invert only if MONOCHROME1.

    ``value_range`` is the fixed (lo, hi) range the pixels can take at this
    point of the pipeline; inversion maps lo <-> hi, so every image is inverted
    the same way. Without it the image's own maximum is used, which makes the
    result depend on each image's content.
    """
    if policy == PhotometricPolicy.KEEP:
        return pixels

    if policy in (PhotometricPolicy.AUTO, PhotometricPolicy.INVERT_MONO1):
        if meta.is_monochrome1:
            logger.debug("Inverting MONOCHROME1 -> MONOCHROME2")
            if value_range is not None:
                lo, hi = value_range
                return ((lo + hi) - pixels).astype(np.float32)
            return _invert(pixels)

    return pixels


def _invert(pixels: np.ndarray) -> np.ndarray:
    """Invert grayscale: max - pixel (fallback when no fixed range is known)."""
    p_max = pixels.max()
    if p_max == 0:
        return pixels
    return (p_max - pixels).astype(pixels.dtype)
