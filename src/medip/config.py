"""Configuration dataclasses for medip pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── DICOM policy enums ──────────────────────────────────────────────

class VOIPolicy(str, Enum):
    NONE = "none"
    AUTO = "auto"
    LUT = "lut"
    WINDOW = "window"
    LEGACY_LINEAR = "legacy_linear"
    LEGACY_SIGMOID = "legacy_sigmoid"


class PhotometricPolicy(str, Enum):
    AUTO = "auto"
    KEEP = "keep"
    INVERT_MONO1 = "invert_mono1"


class PaddingPolicy(str, Enum):
    NONE = "none"
    MASK = "mask"
    CLIP = "clip"


class PresentationIntentPolicy(str, Enum):
    AUTO = "auto"
    FOR_PROCESSING = "for_processing"
    FOR_PRESENTATION = "for_presentation"
    IGNORE = "ignore"


class SpacingPolicy(str, Enum):
    AUTO = "auto"
    KEEP = "keep"
    TARGET = "target"
    NONE = "none"


class Interpolator(str, Enum):
    BSPLINE = "bspline"
    LINEAR = "linear"
    NEAREST = "nearest"


class MaskPolicy(str, Enum):
    NONE = "none"
    OTSU = "otsu"
    LARGEST_COMPONENT = "largest_component"
    CUSTOM = "custom"


class CropPolicy(str, Enum):
    NONE = "none"
    BBOX = "bbox"
    TIGHT_BBOX = "tight_bbox"
    FIXED_MARGIN = "fixed_margin"


class ExportFormat(str, Enum):
    NPY = "npy"
    NPZ = "npz"
    PNG16 = "png16"
    PNG8 = "png8"
    TIFF16 = "tiff16"
    JPEG = "jpeg"
    DICOM = "dicom"
    NIFTI = "nifti"


class ExportDtype(str, Enum):
    FLOAT32 = "float32"
    UINT16 = "uint16"
    UINT8 = "uint8"


# ── Config dataclasses ───────────────────────────────────────────────

@dataclass
class DicomConfig:
    apply_modality_lut: bool = True
    voi_policy: VOIPolicy = VOIPolicy.AUTO
    voi_index: int = 0
    photometric_policy: PhotometricPolicy = PhotometricPolicy.AUTO
    padding_policy: PaddingPolicy = PaddingPolicy.MASK
    presentation_intent_policy: PresentationIntentPolicy = PresentationIntentPolicy.AUTO
    compressed_policy: str = "auto"


@dataclass
class GeometryConfig:
    spacing_policy: SpacingPolicy = SpacingPolicy.AUTO
    target_spacing: Optional[Tuple[float, float]] = (0.07, 0.07)  # (row, col) in mm
    image_interpolator: Interpolator = Interpolator.BSPLINE
    mask_interpolator: Interpolator = Interpolator.NEAREST


@dataclass
class ROIConfig:
    mask_policy: MaskPolicy = MaskPolicy.NONE
    crop_policy: CropPolicy = CropPolicy.NONE
    crop_margin_mm: float = 0.0
    crop_margin_px: int = 0
    otsu_sensitivity: float = 0.2
    morph_open_kernel: int = 7
    morph_close_kernel: int = 15
    min_contour_area: int = 2000
    fallback_to_full: bool = True


@dataclass
class ExportConfig:
    format: ExportFormat = ExportFormat.NPY
    dtype: ExportDtype = ExportDtype.FLOAT32
    outputs: List[str] = field(default_factory=lambda: ["image"])
    save_mask: bool = False
    save_metadata_json: bool = True
    png8_output_max: float = 4095.0  # legacy compat: scale factor before 8-bit


@dataclass
class RuntimeConfig:
    num_workers: int = 8
    fail_fast: bool = False
    dry_run: bool = False
    save_report: bool = False


@dataclass
class PipelineConfig:
    """Top-level config combining all sub-configs."""

    profile: str = "learning_raw"
    dicom: DicomConfig = field(default_factory=DicomConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
