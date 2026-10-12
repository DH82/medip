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


# Some older detectors (LORAD, and other pre-2005 exports) write the File Meta
# group without the 128-byte preamble and the "DICM" prefix, which pydicom
# rejects by default. The pixel data and vendor tags in those files are intact,
# so force reading recovers them — but force=True also makes pydicom try to
# parse genuinely non-DICOM files, which is why this stays opt-in. Enable it per
# process with ``set_force_read(True)``; every read in medip honors it.
_FORCE_READ = False


def set_force_read(enabled: bool) -> None:
    """Enable/disable ``force=True`` for every DICOM read in this process."""
    global _FORCE_READ
    _FORCE_READ = bool(enabled)


def get_force_read() -> bool:
    return _FORCE_READ


def dcmread(path: str | Path, **kwargs):
    """``pydicom.dcmread`` honoring the process-wide force-read setting."""
    kwargs.setdefault("force", _FORCE_READ)
    return pydicom.dcmread(str(path), **kwargs)


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

    arr = _keep_bits_stored(arr, ds)
    return arr.astype(np.float32)


def _keep_bits_stored(arr: np.ndarray, ds: Dataset) -> np.ndarray:
    """Drop the bits above BitsStored, as the DICOM standard requires.

    Some detectors put other data in the unused high bits of each 16-bit word
    (seen on LORAD exports: a strip of pixels reading 4096+ with
    BitsStored = 12). pydicom 3 masks these bits off and pydicom 2.x does not,
    so without this step the same file decodes differently depending on the
    installed pydicom — and a single stray value moves a min-max window.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return arr
    bits_stored = int(ds.get("BitsStored", 0) or 0)
    if bits_stored <= 0 or bits_stored >= arr.dtype.itemsize * 8:
        return arr

    mask = (1 << bits_stored) - 1
    if int(ds.get("PixelRepresentation", 0) or 0) == 0:
        return arr & np.array(mask, dtype=arr.dtype)

    # Signed: keep the low bits, then sign-extend from bit (BitsStored - 1).
    sign = 1 << (bits_stored - 1)
    low = arr.astype(np.int64) & mask
    return ((low ^ sign) - sign).astype(arr.dtype)


def read_dicom(path: str | Path) -> Tuple[np.ndarray, DicomMetadata]:
    """Read a DICOM file: returns (pixel_array, metadata).

    pixel_array is raw float32 (H, W), no transforms applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = dcmread(path)
    metadata = extract_metadata(ds, stop_before_pixels=False)
    pixels = _decode_pixels(ds)

    return pixels, metadata


def read_dicom_pixels(path: str | Path) -> np.ndarray:
    """Read only the pixel array from a DICOM file."""
    pixels, _ = read_dicom(path)
    return pixels
