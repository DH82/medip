"""Configuration dataclasses for medip pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ── DICOM policy enums ──────────────────────────────────────────────

class VOIPolicy(str, Enum):
    NONE = "none"
    AUTO = "auto"
    LUT = "lut"
    WINDOW = "window"
    MIN_MAX_WINDOW = "min_max_window"
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


class MaskSource(str, Enum):
    """Which version of the pixels the mask is computed from."""

    POST_VOI = "post_voi"  # after windowing — what gets exported
    PRE_VOI = "pre_voi"    # after Modality LUT and photometric, before windowing


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


class OutputLayout(str, Enum):
    TREE = "tree"  # mirror the input folder structure
    FLAT = "flat"  # one folder; the relative path is joined into the name with "__"


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
    keep_all_components: bool = False  # True: keep every component >= min_contour_area
    mask_source: MaskSource = MaskSource.POST_VOI
    mask_to_bbox: bool = False  # expand the mask to its smallest enclosing rectangle
    fallback_to_full: bool = True


@dataclass
class ExportConfig:
    format: ExportFormat = ExportFormat.NPY
    dtype: ExportDtype = ExportDtype.FLOAT32
    outputs: List[str] = field(default_factory=lambda: ["image"])
    save_mask: bool = False
    save_metadata_json: bool = True
    png8_output_max: float = 4095.0  # legacy compat: scale factor before 8-bit
    # Polynomial applied to the pixels just before export, highest degree first
    # (numpy.polyval order). Used to map one detector's intensities onto another's
    # so a mixed-vendor cohort shares a scale. Must be monotonic over the output
    # range or it will reorder tissue brightness; fit_vendor_curve.py checks that.
    intensity_curve: Optional[List[float]] = None
    # Monotonic lookup table as {"x": [...], "y": [...]}, applied by linear
    # interpolation before export. Use this instead of intensity_curve when the two
    # detectors' intensity distributions differ in shape and not just scale: a
    # polynomial through those quantiles stops being monotonic, while a table
    # through the same points is monotonic by construction.
    intensity_lut: Optional[Dict[str, List[float]]] = None
    voi_output_max: Optional[float] = None  # overrides png8_output_max for VOI legacy/min-max modes


@dataclass
class RuntimeConfig:
    num_workers: int = 8
    fail_fast: bool = False
    dry_run: bool = False
    save_report: bool = False
    output_layout: OutputLayout = OutputLayout.TREE
    # Skip inputs whose output file already exists, so an interrupted run resumes.
    skip_existing: bool = False
    # Read DICOMs with pydicom force=True: recovers files missing the 128-byte
    # preamble / "DICM" prefix, but also makes pydicom attempt non-DICOM files.
    force_read: bool = False
    # Text file caching the list of input DICOMs (one path per line). Reused when
    # it exists, which skips the directory walk on slow network mounts.
    scan_cache: Optional[str] = None


@dataclass
class VendorOverride:
    """Settings applied only to images from matching manufacturers.

    ``manufacturer`` holds case-insensitive substrings of the DICOM Manufacturer
    tag; the first override with a match wins. The section dicts use the same
    keys as the YAML ``dicom`` / ``geometry`` / ``roi`` / ``export`` sections.
    """

    manufacturer: List[str]
    dicom: Dict[str, Any] = field(default_factory=dict)
    geometry: Dict[str, Any] = field(default_factory=dict)
    roi: Dict[str, Any] = field(default_factory=dict)
    export: Dict[str, Any] = field(default_factory=dict)

    def matches(self, manufacturer: Optional[str]) -> bool:
        name = (manufacturer or "").upper()
        return any(pattern.upper() in name for pattern in self.manufacturer)


@dataclass
class PipelineConfig:
    """Top-level config combining all sub-configs."""

    profile: str = "custom"  # presets set their own name; see presets.DEFAULT_PROFILE
    dicom: DicomConfig = field(default_factory=DicomConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vendor_overrides: List[VendorOverride] = field(default_factory=list)
