"""Photometric Interpretation handling.

MONOCHROME1: 0 = white (high signal), max = black
MONOCHROME2: 0 = black, max = white (most common)

For training, MONOCHROME1 images should typically be inverted to MONOCHROME2
so pixel intensity consistently maps to tissue density.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import PhotometricPolicy
from ..io.metadata import DicomMetadata

logger = logging.getLogger(__name__)


def apply_photometric(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: PhotometricPolicy = PhotometricPolicy.AUTO,
) -> np.ndarray:
    """Handle photometric interpretation.

    AUTO: invert MONOCHROME1 to MONOCHROME2.
    KEEP: leave as-is.
    INVERT_MONO1: explicitly invert only if MONOCHROME1.
    """
    if policy == PhotometricPolicy.KEEP:
        return pixels

    if policy in (PhotometricPolicy.AUTO, PhotometricPolicy.INVERT_MONO1):
        if meta.is_monochrome1:
            logger.debug("Inverting MONOCHROME1 -> MONOCHROME2")
            return _invert(pixels)

    return pixels


def _invert(pixels: np.ndarray) -> np.ndarray:
    """Invert grayscale: max - pixel."""
    p_max = pixels.max()
    if p_max == 0:
        return pixels
    return (p_max - pixels).astype(pixels.dtype)
