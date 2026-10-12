"""Export transforms: save processed images in various formats.

Supported formats:
  - npy     : NumPy float32 array (lossless, training default)
  - npz     : Compressed NumPy array
  - png16   : 16-bit grayscale PNG (lossless, good balance)
  - png8    : 8-bit grayscale PNG (legacy, lossy)
  - tiff16  : 16-bit TIFF (lossless, widely compatible)
  - jpeg    : 8-bit JPEG (lossy, visualization only)
  - dicom   : Secondary Capture DICOM (preserves metadata)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import ExportConfig, ExportDtype, ExportFormat
from ..exceptions import ExportError

logger = logging.getLogger(__name__)

# Extension mapping
_FORMAT_EXT = {
    ExportFormat.NPY: ".npy",
    ExportFormat.NPZ: ".npz",
    ExportFormat.PNG16: ".png",
    ExportFormat.PNG8: ".png",
    ExportFormat.TIFF16: ".tiff",
    ExportFormat.JPEG: ".jpg",
    ExportFormat.DICOM: ".dcm",
}


# Extensions export_image is allowed to strip from a caller-supplied path.
# Anything else (e.g. the trailing ".498" of a dotted DICOM UID filename) is
# part of the name, not an extension.
_KNOWN_EXT = tuple(
    sorted(
        set(_FORMAT_EXT.values()) | {".dcm", ".dicom", ".png", ".jpeg"},
        key=len,
        reverse=True,
    )
)


def _resolve_output_path(output_path: Path, fmt: ExportFormat) -> Path:
    """Give *output_path* the extension for *fmt*.

    Only a recognized image extension is replaced. Names that merely contain
    dots — DICOM UIDs, most commonly — keep every component, so files like
    ``1.2.826.0.1.3680043.8.498.12345`` no longer collapse onto each other.
    """
    ext = _FORMAT_EXT[fmt]
    name = output_path.name
    for known in _KNOWN_EXT:
        if name.lower().endswith(known) and len(name) > len(known):
            name = name[: -len(known)]
            break
    return output_path.with_name(name + ext)


def export_image(
    pixels: np.ndarray,
    output_path: str | Path,
    config: ExportConfig,
    mask: Optional[np.ndarray] = None,
    metadata_dict: Optional[Dict[str, Any]] = None,
    pixel_spacing: Optional[tuple] = None,
) -> Path:
    """Export processed image to the configured format.

    The mask and metadata sidecars are written first and the image last, through
    a temporary file renamed into place. So an image file only ever exists
    complete, with its sidecars beside it — which is what lets
    ``runtime.skip_existing`` trust it when an interrupted run resumes.

    Returns the path to the saved primary image file.
    """
    output_path = Path(output_path)
    primary_path = _resolve_output_path(output_path, config.format)
    ext = _FORMAT_EXT[config.format]
    base_name = primary_path.name[: -len(ext)]

    primary_path.parent.mkdir(parents=True, exist_ok=True)

    # Cast to target dtype
    out_pixels = _cast_dtype(pixels, config)

    # Save mask if requested
    if config.save_mask and mask is not None:
        _export_mask(mask, primary_path.with_name(f"{base_name}_mask.png"))

    # Save metadata sidecar
    if config.save_metadata_json and metadata_dict is not None:
        _export_metadata_json(metadata_dict, primary_path.with_name(base_name + ".json"))

    # Image last, atomically. The temporary name keeps the real extension
    # because the writers pick the format from it.
    partial_path = primary_path.with_name(f".{base_name}.partial{ext}")
    try:
        _EXPORTERS[config.format](out_pixels, partial_path, config, pixel_spacing)
        os.replace(partial_path, primary_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    logger.debug("Saved: %s (%s)", primary_path, config.format.value)

    return primary_path


# ── dtype casting ────────────────────────────────────────────────────


def _cast_dtype(pixels: np.ndarray, config: ExportConfig) -> np.ndarray:
    if config.dtype == ExportDtype.FLOAT32:
        return pixels.astype(np.float32)

    if config.dtype == ExportDtype.UINT16:
        # Preserve raw pixel values when they already fit in [0, 65535].
        # This avoids the per-image min-max rescaling that would destroy
        # cross-image calibration (important for training on raw DICOM).
        # Values outside the range are clipped with a warning.
        p_min = float(pixels.min()) if pixels.size else 0.0
        p_max = float(pixels.max()) if pixels.size else 0.0
        if p_min < 0 or p_max > 65535:
            logger.warning(
                "UINT16 cast: pixel range [%.3f, %.3f] exceeds [0, 65535]; clipping.",
                p_min,
                p_max,
            )
        clipped = np.clip(pixels, 0, 65535)
        if np.issubdtype(pixels.dtype, np.floating):
            clipped = np.rint(clipped)
        return clipped.astype(np.uint16)

    if config.dtype == ExportDtype.UINT8:
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            normalized = (pixels - p_min) / (p_max - p_min)
            return (normalized * 255).astype(np.uint8)
        return np.zeros_like(pixels, dtype=np.uint8)

    return pixels


# ── Format exporters ─────────────────────────────────────────────────


def _export_npy(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    np.save(str(path), pixels)


def _export_npz(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    save_dict = {"image": pixels}
    if spacing is not None:
        save_dict["spacing"] = np.array(spacing)
    np.savez_compressed(str(path), **save_dict)


def _export_png16(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    import cv2

    if pixels.dtype != np.uint16:
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            img = ((pixels - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
        else:
            img = np.zeros_like(pixels, dtype=np.uint16)
    else:
        img = pixels

    if not cv2.imwrite(str(path), img):
        raise ExportError(f"Failed to write PNG16: {path}")


def _export_png8(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    import cv2

    if pixels.dtype != np.uint8:
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            img = ((pixels - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(pixels, dtype=np.uint8)
    else:
        img = pixels

    if not cv2.imwrite(str(path), img):
        raise ExportError(f"Failed to write PNG8: {path}")


def _export_tiff16(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    import cv2

    if pixels.dtype != np.uint16:
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            img = ((pixels - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
        else:
            img = np.zeros_like(pixels, dtype=np.uint16)
    else:
        img = pixels

    if not cv2.imwrite(str(path), img):
        raise ExportError(f"Failed to write TIFF16: {path}")


def _export_jpeg(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    import cv2

    if pixels.dtype != np.uint8:
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            img = ((pixels - p_min) / (p_max - p_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(pixels, dtype=np.uint8)
    else:
        img = pixels

    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise ExportError(f"Failed to write JPEG: {path}")


def _export_dicom(
    pixels: np.ndarray, path: Path, config: ExportConfig, spacing: Optional[tuple]
) -> None:
    """Export as Secondary Capture DICOM."""
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = FileDataset(str(path), {}, preamble=b"\x00" * 128)

    # File Meta
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.Modality = "OT"  # Other
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = pixels.shape[:2]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    if spacing is not None:
        ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]

    # Determine bit depth from dtype
    if pixels.dtype == np.uint16:
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.astype(np.uint16).tobytes()
    elif pixels.dtype == np.uint8:
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pixels.astype(np.uint8).tobytes()
    else:
        # float32 -> store as uint16 in DICOM
        p_min, p_max = pixels.min(), pixels.max()
        if p_max > p_min:
            img16 = ((pixels - p_min) / (p_max - p_min) * 65535).astype(np.uint16)
        else:
            img16 = np.zeros_like(pixels, dtype=np.uint16)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = (p_max - p_min) / 65535.0 if p_max > p_min else 1.0
        ds.RescaleIntercept = float(p_min)
        ds.PixelData = img16.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


# ── Helpers ──────────────────────────────────────────────────────────


def _export_mask(mask: np.ndarray, path: Path) -> None:
    import cv2

    mask_img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask_img)
    logger.debug("Saved mask: %s", path)


def _export_metadata_json(metadata: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug("Saved metadata: %s", path)


# ── Dispatcher ───────────────────────────────────────────────────────

_EXPORTERS = {
    ExportFormat.NPY: _export_npy,
    ExportFormat.NPZ: _export_npz,
    ExportFormat.PNG16: _export_png16,
    ExportFormat.PNG8: _export_png8,
    ExportFormat.TIFF16: _export_tiff16,
    ExportFormat.JPEG: _export_jpeg,
    ExportFormat.DICOM: _export_dicom,
}
