"""DicomMetadata: authoritative DICOM tag extraction.

All pixel-interpretation metadata is extracted from pydicom Dataset
in one place so downstream transforms never need to re-read tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DicomSequence


def _parse_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (MultiValue, list, tuple)):
        return float(val[0]) if len(val) > 0 else None
    return float(val)


def _parse_float_list(val: Any) -> Optional[List[float]]:
    if val is None:
        return None
    if isinstance(val, (MultiValue, list, tuple)):
        return [float(v) for v in val]
    return [float(val)]


def _parse_spacing(ds: Dataset) -> Optional[Tuple[float, float]]:
    """Extract pixel spacing as (row_spacing, col_spacing) in mm.

    Priority: PixelSpacing > ImagerPixelSpacing.
    DICOM standard: PixelSpacing is [row_spacing, col_spacing].
    """
    ps = ds.get("PixelSpacing", None)
    if ps is None:
        ps = ds.get("ImagerPixelSpacing", None)
    if ps is None:
        return None
    vals = [float(v) for v in ps]
    if len(vals) >= 2:
        return (vals[0], vals[1])
    return None


@dataclass
class DicomMetadata:
    """Structured DICOM metadata for pixel interpretation."""

    # ── Source ────────────────────────────────────────────────────────
    file_path: Optional[str] = None

    # ── Pixel Description ────────────────────────────────────────────
    bits_allocated: int = 16
    bits_stored: int = 12
    high_bit: int = 11
    pixel_representation: int = 0  # 0=unsigned, 1=signed
    samples_per_pixel: int = 1

    # ── Photometric ──────────────────────────────────────────────────
    photometric_interpretation: str = "MONOCHROME2"

    # ── Presentation ─────────────────────────────────────────────────
    presentation_intent_type: Optional[str] = None  # FOR PROCESSING | FOR PRESENTATION

    # ── Modality LUT / Rescale ───────────────────────────────────────
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    modality_lut_sequence: Optional[List[Dataset]] = field(default=None, repr=False)

    # ── VOI LUT / Window ─────────────────────────────────────────────
    window_center: Optional[List[float]] = None
    window_width: Optional[List[float]] = None
    voi_lut_function: str = "LINEAR"  # LINEAR | LINEAR_EXACT | SIGMOID
    voi_lut_sequence: Optional[List[Dataset]] = field(default=None, repr=False)

    # ── Pixel Padding ────────────────────────────────────────────────
    pixel_padding_value: Optional[int] = None
    pixel_padding_range_limit: Optional[int] = None

    # ── Geometry ─────────────────────────────────────────────────────
    pixel_spacing: Optional[Tuple[float, float]] = None  # (row, col) mm
    imager_pixel_spacing: Optional[Tuple[float, float]] = None

    # ── Laterality / View ────────────────────────────────────────────
    image_laterality: Optional[str] = None  # L | R
    view_position: Optional[str] = None  # CC | MLO | ...

    # ── Transfer Syntax ──────────────────────────────────────────────
    transfer_syntax_uid: Optional[str] = None

    # ── Image dimensions (from pixel data) ───────────────────────────
    rows: int = 0
    columns: int = 0

    @property
    def is_monochrome1(self) -> bool:
        return self.photometric_interpretation == "MONOCHROME1"

    @property
    def is_for_processing(self) -> bool:
        return self.presentation_intent_type == "FOR PROCESSING"

    @property
    def is_for_presentation(self) -> bool:
        return self.presentation_intent_type == "FOR PRESENTATION"

    @property
    def has_window(self) -> bool:
        return self.window_center is not None and self.window_width is not None

    @property
    def has_voi_lut_sequence(self) -> bool:
        return self.voi_lut_sequence is not None and len(self.voi_lut_sequence) > 0

    @property
    def has_modality_lut_sequence(self) -> bool:
        return self.modality_lut_sequence is not None and len(self.modality_lut_sequence) > 0

    @property
    def has_pixel_padding(self) -> bool:
        return self.pixel_padding_value is not None

    @property
    def effective_spacing(self) -> Optional[Tuple[float, float]]:
        """Return best available spacing: PixelSpacing > ImagerPixelSpacing."""
        return self.pixel_spacing or self.imager_pixel_spacing

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for metadata sidecar JSON."""
        return {
            "file_path": self.file_path,
            "bits_allocated": self.bits_allocated,
            "bits_stored": self.bits_stored,
            "high_bit": self.high_bit,
            "pixel_representation": self.pixel_representation,
            "photometric_interpretation": self.photometric_interpretation,
            "presentation_intent_type": self.presentation_intent_type,
            "rescale_slope": self.rescale_slope,
            "rescale_intercept": self.rescale_intercept,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "voi_lut_function": self.voi_lut_function,
            "pixel_padding_value": self.pixel_padding_value,
            "pixel_padding_range_limit": self.pixel_padding_range_limit,
            "pixel_spacing": list(self.pixel_spacing) if self.pixel_spacing else None,
            "imager_pixel_spacing": list(self.imager_pixel_spacing) if self.imager_pixel_spacing else None,
            "image_laterality": self.image_laterality,
            "view_position": self.view_position,
            "transfer_syntax_uid": self.transfer_syntax_uid,
            "rows": self.rows,
            "columns": self.columns,
        }


def extract_metadata(ds_or_path, stop_before_pixels: bool = True) -> DicomMetadata:
    """Extract DicomMetadata from a pydicom Dataset or file path.

    Parameters
    ----------
    ds_or_path : Dataset | str | Path
        pydicom Dataset or path to DICOM file.
    stop_before_pixels : bool
        If reading from path, skip pixel data for speed (default True).
    """
    if isinstance(ds_or_path, (str, Path)):
        ds = pydicom.dcmread(str(ds_or_path), stop_before_pixels=stop_before_pixels)
        file_path = str(ds_or_path)
    else:
        ds = ds_or_path
        file_path = getattr(ds, "filename", None)

    # Parse window center / width as lists (multi-window support)
    wc_raw = ds.get("WindowCenter", None)
    ww_raw = ds.get("WindowWidth", None)
    wc = _parse_float_list(wc_raw)
    ww = _parse_float_list(ww_raw)

    # Pixel spacing
    ps = _parse_spacing(ds)
    ips_raw = ds.get("ImagerPixelSpacing", None)
    ips = None
    if ips_raw is not None:
        ips_vals = [float(v) for v in ips_raw]
        if len(ips_vals) >= 2:
            ips = (ips_vals[0], ips_vals[1])

    # VOI LUT Sequence
    voi_lut_seq = ds.get("VOILUTSequence", None)
    if isinstance(voi_lut_seq, DicomSequence):
        voi_lut_seq = list(voi_lut_seq)

    # Modality LUT Sequence
    mod_lut_seq = ds.get("ModalityLUTSequence", None)
    if isinstance(mod_lut_seq, DicomSequence):
        mod_lut_seq = list(mod_lut_seq)

    return DicomMetadata(
        file_path=file_path,
        bits_allocated=int(ds.get("BitsAllocated", 16)),
        bits_stored=int(ds.get("BitsStored", 12)),
        high_bit=int(ds.get("HighBit", 11)),
        pixel_representation=int(ds.get("PixelRepresentation", 0)),
        samples_per_pixel=int(ds.get("SamplesPerPixel", 1)),
        photometric_interpretation=str(ds.get("PhotometricInterpretation", "MONOCHROME2")),
        presentation_intent_type=str(ds.get("PresentationIntentType", "")).strip() or None,
        rescale_slope=float(ds.get("RescaleSlope", 1.0)),
        rescale_intercept=float(ds.get("RescaleIntercept", 0.0)),
        modality_lut_sequence=mod_lut_seq,
        window_center=wc,
        window_width=ww,
        voi_lut_function=str(ds.get("VOILUTFunction", "LINEAR")).upper(),
        voi_lut_sequence=voi_lut_seq,
        pixel_padding_value=ds.get("PixelPaddingValue", None),
        pixel_padding_range_limit=ds.get("PixelPaddingRangeLimit", None),
        pixel_spacing=ps,
        imager_pixel_spacing=ips,
        image_laterality=str(ds.get("ImageLaterality", "")).strip() or None,
        view_position=str(ds.get("ViewPosition", "")).strip() or None,
        transfer_syntax_uid=str(ds.file_meta.get("TransferSyntaxUID", "")) if hasattr(ds, "file_meta") else None,
        rows=int(ds.get("Rows", 0)),
        columns=int(ds.get("Columns", 0)),
    )
