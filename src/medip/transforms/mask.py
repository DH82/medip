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

from ..config import MaskPolicy, MaskSource, ROIConfig

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

    # Combine with padding mask if available. This happens BEFORE component
    # selection: Pixel Padding regions are not image data, so they must not
    # bridge separate objects into one component or inflate its area. Selecting
    # first lets background noise connect burned-in glyphs into a blob that
    # clears min_contour_area, only for the padding intersection to strip the
    # noise away again and leave the glyphs behind.
    if padding_mask is not None:
        mask = mask & padding_mask.astype(np.uint8)

    mask = _select_components(mask, config)

    if config.mask_to_bbox:
        mask = _expand_to_bbox(mask)

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

    return mask


def _largest_component_mask(pixels: np.ndarray, config: ROIConfig) -> np.ndarray:
    """Simple threshold + largest connected component."""
    _min, _max = pixels.min(), pixels.max()
    if _max <= _min:
        return np.zeros(pixels.shape, dtype=np.uint8)

    arr_norm = ((pixels - _min) / (_max - _min) * 255).astype(np.uint8)
    _, binary = cv2.threshold(arr_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.uint8)


def _expand_to_bbox(mask: np.ndarray) -> np.ndarray:
    """Replace the mask with the smallest rectangle that contains it.

    A rectangle can only ever add pixels, so nothing the segmentation found is
    lost to a ragged boundary — at the cost of taking in whatever else falls
    inside those bounds. With a single-component mask that is the strip of
    background hugging the breast; with scattered components it can be most of
    the image, so check what survives component selection before enabling this.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask

    boxed = np.zeros_like(mask)
    boxed[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return boxed


def _select_components(mask: np.ndarray, config: ROIConfig) -> np.ndarray:
    """Reduce the thresholded mask to the contours worth keeping.

    ``keep_all_components`` decides between two very different behaviors:

    - False (default): only the single largest contour survives. Simple, but on
      an MLO mammogram the pectoral muscle and the breast are often separate
      components, and whichever is smaller — frequently the breast itself — is
      discarded along with the background.
    - True: every contour of at least ``min_contour_area`` survives. Set
      ``min_contour_area`` above the size of any burned-in annotation you want
      dropped, but below the smallest tissue island you want kept.
    """
    if not config.keep_all_components:
        return _keep_largest_contour(mask, config.min_contour_area)

    # Select on the component's pixel count, not cv2.contourArea. contourArea is
    # the area enclosed by the outline, so a thin L-shaped registration bracket
    # scores large while covering few pixels — and filling that contour would
    # restore the burned-in label sitting inside it.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    final_mask = np.zeros_like(mask)
    kept = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= config.min_contour_area:
            final_mask[labels == index] = 1
            kept += 1
    logger.debug("Kept %d/%d components of >= %d px", kept, count - 1, config.min_contour_area)
    return final_mask


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
