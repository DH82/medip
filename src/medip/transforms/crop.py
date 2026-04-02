"""ROI cropping transforms."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..config import CropPolicy, ROIConfig

logger = logging.getLogger(__name__)


def crop_to_roi(
    pixels: np.ndarray,
    mask: Optional[np.ndarray],
    config: ROIConfig,
    pixel_spacing: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """Crop image and mask to region of interest.

    Parameters
    ----------
    pixels : (H, W) float32
    mask : (H, W) uint8 or None
    config : ROIConfig
    pixel_spacing : (row, col) in mm for margin calculation

    Returns
    -------
    (cropped_pixels, cropped_mask, crop_info)
    crop_info contains: x, y, w, h of the crop bounding box
    """
    crop_info = {"x": 0, "y": 0, "w": pixels.shape[1], "h": pixels.shape[0], "applied": False}

    if config.crop_policy == CropPolicy.NONE:
        return pixels, mask, crop_info

    if mask is None or mask.max() == 0:
        if config.fallback_to_full:
            logger.debug("No valid mask for cropping, returning full image")
            return pixels, mask, crop_info
        return pixels, mask, crop_info

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return pixels, mask, crop_info

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Apply margin
    margin_px = _compute_margin_px(config, pixel_spacing)
    if margin_px > 0:
        x = max(0, x - margin_px)
        y = max(0, y - margin_px)
        w = min(pixels.shape[1] - x, w + 2 * margin_px)
        h = min(pixels.shape[0] - y, h + 2 * margin_px)

    cropped_pixels = pixels[y:y + h, x:x + w]
    cropped_mask = mask[y:y + h, x:x + w] if mask is not None else None
    crop_info = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "applied": True}

    logger.debug("Cropped to bbox: x=%d, y=%d, w=%d, h=%d", x, y, w, h)
    return cropped_pixels, cropped_mask, crop_info


def _compute_margin_px(
    config: ROIConfig,
    pixel_spacing: Optional[Tuple[float, float]],
) -> int:
    """Compute margin in pixels from config."""
    if config.crop_margin_px > 0:
        return config.crop_margin_px

    if config.crop_margin_mm > 0 and pixel_spacing is not None:
        avg_spacing = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
        if avg_spacing > 0:
            return int(round(config.crop_margin_mm / avg_spacing))

    return 0
