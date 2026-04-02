"""Foreground mask extraction (experimental / research module).

These heuristics (Otsu + morphology + largest contour) are NOT part of the
DICOM standard pipeline. They are useful for research but should not be
treated as authoritative.

Originally developed for mammography breast segmentation, but applicable
to other modalities where foreground/background separation is needed.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from ..config import MaskPolicy, ROIConfig

logger = logging.getLogger(__name__)


def extract_mask(
    pixels: np.ndarray,
    config: ROIConfig,
    padding_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract a binary mask from the image.

    Parameters
    ----------
    pixels : (H, W) float32
    config : ROIConfig
    padding_mask : optional bool array from padding step

    Returns
    -------
    mask : (H, W) uint8, 0/1, or None if mask_policy is NONE
    """
    if config.mask_policy == MaskPolicy.NONE:
        return None

    if config.mask_policy == MaskPolicy.OTSU:
        mask = _otsu_mask(pixels, config)
    elif config.mask_policy == MaskPolicy.LARGEST_COMPONENT:
        mask = _largest_component_mask(pixels, config)
    elif config.mask_policy == MaskPolicy.CUSTOM:
        logger.warning("Custom mask policy selected but no custom function provided")
        return None
    else:
        return None

    # Combine with padding mask if available
    if padding_mask is not None:
        mask = mask & padding_mask.astype(np.uint8)

    _log_mask_quality(mask, pixels.shape)
    return mask


def _otsu_mask(pixels: np.ndarray, config: ROIConfig) -> np.ndarray:
    """Otsu-based mask with morphological cleanup.

    Reproduces extract_robust_mask from original MakePNG.py but with
    configurable parameters.
    """
    _min, _max = pixels.min(), pixels.max()

    if _max > _min:
        arr_norm = (pixels - _min) / (_max - _min)
    else:
        return np.zeros(pixels.shape, dtype=np.uint8)

    arr_blur = cv2.GaussianBlur(arr_norm.astype(np.float32), (5, 5), 0)
    src_uint8 = (arr_blur * 255).astype(np.uint8)

    otsu_val, _ = cv2.threshold(
        src_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = otsu_val * config.otsu_sensitivity
    mask = (src_uint8 > threshold).astype(np.uint8)

    # Morphological cleanup
    k_open = config.morph_open_kernel
    k_close = config.morph_close_kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

    return _keep_largest_contour(mask, config.min_contour_area)


def _largest_component_mask(pixels: np.ndarray, config: ROIConfig) -> np.ndarray:
    """Simple threshold + largest connected component."""
    _min, _max = pixels.min(), pixels.max()
    if _max <= _min:
        return np.zeros(pixels.shape, dtype=np.uint8)

    arr_norm = ((pixels - _min) / (_max - _min) * 255).astype(np.uint8)
    _, binary = cv2.threshold(arr_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (binary > 0).astype(np.uint8)
    return _keep_largest_contour(mask, config.min_contour_area)


def _keep_largest_contour(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only the largest contour above min_area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            cv2.drawContours(final_mask, [largest], -1, 1, thickness=cv2.FILLED)

    return final_mask


def _log_mask_quality(mask: np.ndarray, image_shape: tuple) -> None:
    """Log mask coverage statistics."""
    total = mask.size
    foreground = mask.sum()
    ratio = foreground / total if total > 0 else 0
    logger.debug(
        "Mask: %d/%d pixels (%.1f%%) foreground, image shape %s",
        foreground, total, ratio * 100, image_shape,
    )
