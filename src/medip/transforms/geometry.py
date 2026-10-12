"""Geometry transforms: resampling to target spacing.

Uses SimpleITK for resampling as it properly handles DICOM geometry.
Pixel Spacing in DICOM is (row_spacing, col_spacing) in mm.
SimpleITK spacing is (x, y, z) = (col_spacing, row_spacing, slice_spacing).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..config import GeometryConfig, Interpolator, SpacingPolicy

logger = logging.getLogger(__name__)

_SITK_INTERPOLATORS = {
    Interpolator.BSPLINE: "sitkBSpline",
    Interpolator.LINEAR: "sitkLinear",
    Interpolator.NEAREST: "sitkNearestNeighbor",
}


def resample_image(
    pixels: np.ndarray,
    current_spacing: Optional[Tuple[float, float]],
    config: GeometryConfig,
    is_mask: bool = False,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Resample a 2D image to target spacing.

    Parameters
    ----------
    pixels : (H, W) float32
    current_spacing : (row_spacing, col_spacing) in mm, or None
    config : GeometryConfig
    is_mask : bool
        If True, use mask_interpolator (nearest) to avoid interpolation artifacts.

    Returns
    -------
    (resampled_pixels, new_spacing)
    """
    if config.spacing_policy == SpacingPolicy.KEEP:
        return pixels, current_spacing

    if config.spacing_policy == SpacingPolicy.NONE:
        return pixels, None

    # AUTO mode: resample to target_spacing only when PixelSpacing is available
    if config.spacing_policy == SpacingPolicy.AUTO:
        if current_spacing is None:
            logger.debug("AUTO spacing: no PixelSpacing in DICOM, keeping original")
            return pixels, current_spacing
        # Fall through to TARGET logic below

    # TARGET mode
    if config.target_spacing is None:
        logger.warning("spacing_policy=target but target_spacing is None, keeping original")
        return pixels, current_spacing

    if current_spacing is None:
        logger.warning("No spacing info available, cannot resample — keeping original")
        return pixels, current_spacing

    target_row, target_col = config.target_spacing
    cur_row, cur_col = current_spacing

    # Skip if already at target
    if abs(cur_row - target_row) < 1e-6 and abs(cur_col - target_col) < 1e-6:
        return pixels, current_spacing

    try:
        import SimpleITK as sitk
    except ImportError:
        logger.error("SimpleITK not installed. Install with: pip install \"medip[itk]\"")
        return pixels, current_spacing

    interp_key = config.mask_interpolator if is_mask else config.image_interpolator
    sitk_interp = getattr(sitk, _SITK_INTERPOLATORS[interp_key])

    # Build SimpleITK image: (H, W) -> single-slice 3D
    arr_3d = pixels[np.newaxis, ...].astype(np.float32)
    image = sitk.GetImageFromArray(arr_3d)
    # SITK spacing = (col, row, slice)
    image.SetSpacing((cur_col, cur_row, 1.0))

    orig_size = image.GetSize()  # (W, H, 1)
    new_w = max(1, int(round(orig_size[0] * (cur_col / target_col))))
    new_h = max(1, int(round(orig_size[1] * (cur_row / target_row))))

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((target_col, target_row, 1.0))
    resampler.SetSize((new_w, new_h, 1))
    resampler.SetInterpolator(sitk_interp)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(image)
    out_arr = sitk.GetArrayFromImage(resampled)

    # (1, H, W) -> (H, W)
    if out_arr.ndim == 3:
        out_arr = out_arr[0]

    new_spacing = (target_row, target_col)
    logger.debug(
        "Resampled (%d,%d) @ (%.4f,%.4f) -> (%d,%d) @ (%.4f,%.4f)",
        pixels.shape[0], pixels.shape[1], cur_row, cur_col,
        out_arr.shape[0], out_arr.shape[1], target_row, target_col,
    )

    return out_arr.astype(np.float32), new_spacing
