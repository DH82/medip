"""Load PipelineConfig from YAML files with profile + override pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import (
    CropPolicy,
    DicomConfig,
    ExportConfig,
    ExportDtype,
    ExportFormat,
    GeometryConfig,
    Interpolator,
    MaskPolicy,
    PaddingPolicy,
    PhotometricPolicy,
    PipelineConfig,
    PresentationIntentPolicy,
    ROIConfig,
    RuntimeConfig,
    SpacingPolicy,
    VOIPolicy,
)
from ..presets import get_preset


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file.

    The YAML can specify a `profile` key to start from a preset,
    then override individual fields.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for config files: pip install pyyaml")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return config_from_dict(raw)


def config_from_dict(raw: Dict[str, Any]) -> PipelineConfig:
    """Build PipelineConfig from a dict (e.g., parsed YAML)."""
    profile_name = raw.get("profile", "learning_raw")
    cfg = get_preset(profile_name)

    # Override dicom section
    if "dicom" in raw:
        d = raw["dicom"]
        if "apply_modality_lut" in d:
            cfg.dicom.apply_modality_lut = bool(d["apply_modality_lut"])
        if "voi_policy" in d:
            cfg.dicom.voi_policy = VOIPolicy(d["voi_policy"])
        if "voi_index" in d:
            cfg.dicom.voi_index = int(d["voi_index"])
        if "photometric_policy" in d:
            cfg.dicom.photometric_policy = PhotometricPolicy(d["photometric_policy"])
        if "padding_policy" in d:
            cfg.dicom.padding_policy = PaddingPolicy(d["padding_policy"])
        if "presentation_intent_policy" in d:
            cfg.dicom.presentation_intent_policy = PresentationIntentPolicy(d["presentation_intent_policy"])
        if "compressed_policy" in d:
            cfg.dicom.compressed_policy = d["compressed_policy"]

    # Override geometry section
    if "geometry" in raw:
        g = raw["geometry"]
        if "spacing_policy" in g:
            cfg.geometry.spacing_policy = SpacingPolicy(g["spacing_policy"])
        if "target_spacing" in g:
            ts = g["target_spacing"]
            cfg.geometry.target_spacing = tuple(ts) if ts else None
        if "image_interpolator" in g:
            cfg.geometry.image_interpolator = Interpolator(g["image_interpolator"])
        if "mask_interpolator" in g:
            cfg.geometry.mask_interpolator = Interpolator(g["mask_interpolator"])

    # Override roi section
    if "roi" in raw:
        r = raw["roi"]
        if "mask_policy" in r:
            cfg.roi.mask_policy = MaskPolicy(r["mask_policy"])
        if "crop_policy" in r:
            cfg.roi.crop_policy = CropPolicy(r["crop_policy"])
        if "crop_margin_mm" in r:
            cfg.roi.crop_margin_mm = float(r["crop_margin_mm"])
        if "crop_margin_px" in r:
            cfg.roi.crop_margin_px = int(r["crop_margin_px"])
        if "otsu_sensitivity" in r:
            cfg.roi.otsu_sensitivity = float(r["otsu_sensitivity"])
        if "morph_open_kernel" in r:
            cfg.roi.morph_open_kernel = int(r["morph_open_kernel"])
        if "morph_close_kernel" in r:
            cfg.roi.morph_close_kernel = int(r["morph_close_kernel"])
        if "min_contour_area" in r:
            cfg.roi.min_contour_area = int(r["min_contour_area"])
        if "fallback_to_full" in r:
            cfg.roi.fallback_to_full = bool(r["fallback_to_full"])

    # Override export section
    if "export" in raw:
        e = raw["export"]
        if "format" in e:
            cfg.export.format = ExportFormat(e["format"])
        if "dtype" in e:
            cfg.export.dtype = ExportDtype(e["dtype"])
        if "outputs" in e:
            cfg.export.outputs = list(e["outputs"])
        if "save_mask" in e:
            cfg.export.save_mask = bool(e["save_mask"])
        if "save_metadata_json" in e:
            cfg.export.save_metadata_json = bool(e["save_metadata_json"])
        if "png8_output_max" in e:
            cfg.export.png8_output_max = float(e["png8_output_max"])

    # Override runtime section
    if "runtime" in raw:
        rt = raw["runtime"]
        if "num_workers" in rt:
            cfg.runtime.num_workers = int(rt["num_workers"])
        if "fail_fast" in rt:
            cfg.runtime.fail_fast = bool(rt["fail_fast"])
        if "dry_run" in rt:
            cfg.runtime.dry_run = bool(rt["dry_run"])
        if "save_report" in rt:
            cfg.runtime.save_report = bool(rt["save_report"])

    return cfg
