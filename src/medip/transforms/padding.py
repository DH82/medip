"""Pixel Padding Value handling.

DICOM Pixel Padding Value (0028,0120) and Pixel Padding Range Limit (0028,0121)
identify pixels that are not meaningful image data (e.g., collimator regions).

This transform should be applied BEFORE intensity normalization so padding
pixels don't skew statistics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import PaddingPolicy
from ..io.metadata import DicomMetadata

logger = logging.getLogger(__name__)


def apply_padding_mask(
    pixels: np.ndarray,
    meta: DicomMetadata,
    policy: PaddingPolicy = PaddingPolicy.MASK,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Handle Pixel Padding Value.

    Parameters
    ----------
    pixels : array (H, W) float32
    meta : DicomMetadata
    policy : PaddingPolicy
        NONE - do nothing
        MASK - return a boolean mask where True = valid pixel
        CLIP - set padding pixels to 0

    Returns
    -------
    (processed_pixels, padding_mask)
        padding_mask is None if policy is NONE or no padding value exists.
        When MASK, padding_mask is bool array (True = valid).
        When CLIP, padding pixels are zeroed and mask is returned for reference.
    """
    if policy == PaddingPolicy.NONE or not meta.has_pixel_padding:
        return pixels, None

    pad_val = meta.pixel_padding_value
    pad_limit = meta.pixel_padding_range_limit

    if pad_limit is not None:
        lo, hi = sorted([pad_val, pad_limit])
        padding_mask = ~((pixels >= lo) & (pixels <= hi))
    else:
        padding_mask = pixels != pad_val

    logger.debug(
        "Padding mask: %d / %d pixels masked",
        (~padding_mask).sum(),
        padding_mask.size,
    )

    if policy == PaddingPolicy.CLIP:
        out = pixels.copy()
        out[~padding_mask] = 0.0
        return out, padding_mask

    # MASK policy: return unchanged pixels + mask
    return pixels, padding_mask
