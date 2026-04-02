"""Unified DICOM pixel reader.

Authoritative decode: pydicom handles both metadata and pixel decoding.
SimpleITK is used only for resampling in the geometry transform.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from ..exceptions import DicomReadError, UnsupportedTransferSyntaxError
from .metadata import DicomMetadata, extract_metadata

logger = logging.getLogger(__name__)

# Compressed transfer syntaxes that need extra decoders
_COMPRESSED_TS = {
    "1.2.840.10008.1.2.4.50",   # JPEG Baseline
    "1.2.840.10008.1.2.4.51",   # JPEG Extended
    "1.2.840.10008.1.2.4.57",   # JPEG Lossless
    "1.2.840.10008.1.2.4.70",   # JPEG Lossless SV1
    "1.2.840.10008.1.2.4.80",   # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.81",   # JPEG-LS Near-lossless
    "1.2.840.10008.1.2.4.90",   # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.91",   # JPEG 2000
    "1.2.840.10008.1.2.4.201",  # HTJ2K Lossless
    "1.2.840.10008.1.2.4.202",  # HTJ2K Lossy
    "1.2.840.10008.1.2.4.203",  # HTJ2K RPCL
}


def _decode_pixels(ds: Dataset) -> np.ndarray:
    """Decode pixel data from pydicom Dataset.

    Returns raw pixel array WITHOUT any rescale/LUT applied.
    Shape: (H, W) for single-frame grayscale.
    """
    try:
        arr = ds.pixel_array
    except AttributeError as e:
        raise DicomReadError(f"No pixel data in DICOM: {e}") from e
    except Exception as e:
        # Check if it is a compressed syntax issue
        ts_uid = ""
        if hasattr(ds, "file_meta"):
            ts_uid = str(ds.file_meta.get("TransferSyntaxUID", ""))
        if ts_uid in _COMPRESSED_TS:
            ts_name = ds.file_meta.get("TransferSyntaxUID", "")
            raise UnsupportedTransferSyntaxError(
                ts_uid, str(getattr(ts_name, "name", ts_uid))
            ) from e
        raise DicomReadError(f"Failed to decode pixel data: {e}") from e

    # Handle multi-frame: take first frame
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        # (H, W, channels) — take first channel for grayscale
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise DicomReadError(
            f"Expected 2D grayscale image, got shape {arr.shape}"
        )

    return arr.astype(np.float32)


def read_dicom(path: str | Path) -> Tuple[np.ndarray, DicomMetadata]:
    """Read a DICOM file: returns (pixel_array, metadata).

    pixel_array is raw float32 (H, W), no transforms applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path))
    metadata = extract_metadata(ds, stop_before_pixels=False)
    pixels = _decode_pixels(ds)

    return pixels, metadata


def read_dicom_pixels(path: str | Path) -> np.ndarray:
    """Read only the pixel array from a DICOM file."""
    pixels, _ = read_dicom(path)
    return pixels
