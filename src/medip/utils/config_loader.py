"""Load PipelineConfig from YAML files with profile + override pattern."""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

from ..config import (
    CropPolicy,
    DicomConfig,
    ExportConfig,
    ExportDtype,
    ExportFormat,
    GeometryConfig,
    Interpolator,
    MaskPolicy,
    MaskSource,
    OutputLayout,
    PaddingPolicy,
    PhotometricPolicy,
    PipelineConfig,
    PresentationIntentPolicy,
    ROIConfig,
    RuntimeConfig,
    SpacingPolicy,
    VendorOverride,
    VOIPolicy,
)
from ..presets import DEFAULT_PROFILE, get_preset

_SECTIONS = {
    "dicom": DicomConfig,
    "geometry": GeometryConfig,
    "roi": ROIConfig,
    "export": ExportConfig,
    "runtime": RuntimeConfig,
}

# Sections a vendor override may change. Runtime settings apply to the whole run.
_OVERRIDE_SECTIONS = ("dicom", "geometry", "roi", "export")


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
        try:
            raw = yaml.load(f, Loader=_unique_key_loader(yaml)) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: {e}") from None

    try:
        return config_from_dict(raw)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"{path}: {e}") from None


def _unique_key_loader(yaml):
    """SafeLoader that rejects repeated keys.

    Plain safe_load keeps the last of two identical keys, so uncommenting an
    alternative in a config without commenting out the active line would
    silently pick whichever came second.
    """

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None,
                    f"duplicate key '{key}' — comment out one of the two lines",
                    key_node.start_mark,
                )
            seen.add(key)
        return loader.construct_mapping(node, deep)

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return UniqueKeyLoader


def _check_section(name: str, values: Any, cls: type) -> Dict[str, Any]:
    """Return *values* as a dict, rejecting misspelled options."""
    values = values or {}
    if not isinstance(values, dict):
        raise ValueError(f"'{name}' must be a mapping of options")
    allowed = {f.name for f in fields(cls)}
    bad = sorted(set(values) - allowed)
    if bad:
        raise KeyError(f"unknown option(s) {bad} in '{name}'; expected one of {sorted(allowed)}")
    return values


def config_from_dict(raw: Dict[str, Any]) -> PipelineConfig:
    """Build PipelineConfig from a dict (e.g., parsed YAML)."""
    known = {"profile", "vendor_overrides", *_SECTIONS}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise KeyError(f"unknown section(s) {unknown}; expected {sorted(known)}")

    cfg = get_preset(raw.get("profile", DEFAULT_PROFILE))
    for name, cls in _SECTIONS.items():
        if name in raw:
            _APPLY[name](getattr(cfg, name), _check_section(name, raw[name], cls))

    if "vendor_overrides" in raw:
        cfg.vendor_overrides = _parse_vendor_overrides(raw["vendor_overrides"])
    return cfg


def _parse_vendor_overrides(raw: Any) -> list:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("'vendor_overrides' must be a list of entries")

    overrides = []
    for i, entry in enumerate(raw):
        where = f"vendor_overrides[{i}]"
        if not isinstance(entry, dict) or "manufacturer" not in entry:
            raise ValueError(f"{where} needs a 'manufacturer' list")
        bad = sorted(set(entry) - {"manufacturer", *_OVERRIDE_SECTIONS})
        if bad:
            raise KeyError(
                f"unknown key(s) {bad} in {where}; expected manufacturer, {', '.join(_OVERRIDE_SECTIONS)}"
            )
        names = entry["manufacturer"]
        names = [names] if isinstance(names, str) else [str(n) for n in names]
        sections = {
            name: _check_section(f"{where}.{name}", entry.get(name), _SECTIONS[name])
            for name in _OVERRIDE_SECTIONS
        }
        overrides.append(VendorOverride(manufacturer=names, **sections))

    # Parse every override once now so a bad value fails at load time, not on
    # the first matching image halfway through a run.
    probe = get_preset(DEFAULT_PROFILE)
    for override in overrides:
        apply_vendor_override(probe, override)
    return overrides


def apply_vendor_override(cfg: PipelineConfig, override: VendorOverride) -> PipelineConfig:
    """Return a copy of *cfg* with *override*'s settings applied."""
    out = copy.deepcopy(cfg)
    out.vendor_overrides = []
    for name in _OVERRIDE_SECTIONS:
        values = getattr(override, name)
        if values:
            _APPLY[name](getattr(out, name), values)
    return out


# ── Per-section parsing ──────────────────────────────────────────────


def _apply_dicom(c: DicomConfig, d: Dict[str, Any]) -> None:
    if "apply_modality_lut" in d:
        c.apply_modality_lut = bool(d["apply_modality_lut"])
    if "voi_policy" in d:
        c.voi_policy = VOIPolicy(d["voi_policy"])
    if "voi_index" in d:
        c.voi_index = int(d["voi_index"])
    if "photometric_policy" in d:
        c.photometric_policy = PhotometricPolicy(d["photometric_policy"])
    if "padding_policy" in d:
        c.padding_policy = PaddingPolicy(d["padding_policy"])
    if "presentation_intent_policy" in d:
        c.presentation_intent_policy = PresentationIntentPolicy(d["presentation_intent_policy"])
    if "compressed_policy" in d:
        c.compressed_policy = d["compressed_policy"]


def _apply_geometry(c: GeometryConfig, g: Dict[str, Any]) -> None:
    if "spacing_policy" in g:
        c.spacing_policy = SpacingPolicy(g["spacing_policy"])
    if "target_spacing" in g:
        ts = g["target_spacing"]
        c.target_spacing = tuple(ts) if ts else None
    if "image_interpolator" in g:
        c.image_interpolator = Interpolator(g["image_interpolator"])
    if "mask_interpolator" in g:
        c.mask_interpolator = Interpolator(g["mask_interpolator"])


def _apply_roi(c: ROIConfig, r: Dict[str, Any]) -> None:
    if "mask_policy" in r:
        c.mask_policy = MaskPolicy(r["mask_policy"])
    if "crop_policy" in r:
        c.crop_policy = CropPolicy(r["crop_policy"])
    if "crop_margin_mm" in r:
        c.crop_margin_mm = float(r["crop_margin_mm"])
    if "crop_margin_px" in r:
        c.crop_margin_px = int(r["crop_margin_px"])
    if "otsu_sensitivity" in r:
        c.otsu_sensitivity = float(r["otsu_sensitivity"])
    if "morph_open_kernel" in r:
        c.morph_open_kernel = int(r["morph_open_kernel"])
    if "morph_close_kernel" in r:
        c.morph_close_kernel = int(r["morph_close_kernel"])
    if "min_contour_area" in r:
        c.min_contour_area = int(r["min_contour_area"])
    if "keep_all_components" in r:
        c.keep_all_components = bool(r["keep_all_components"])
    if "mask_source" in r:
        c.mask_source = MaskSource(r["mask_source"])
    if "mask_to_bbox" in r:
        c.mask_to_bbox = bool(r["mask_to_bbox"])
    if "fallback_to_full" in r:
        c.fallback_to_full = bool(r["fallback_to_full"])


def _apply_export(c: ExportConfig, e: Dict[str, Any]) -> None:
    if "format" in e:
        c.format = ExportFormat(e["format"])
    if "dtype" in e:
        c.dtype = ExportDtype(e["dtype"])
    if "outputs" in e:
        c.outputs = list(e["outputs"])
    if "save_mask" in e:
        c.save_mask = bool(e["save_mask"])
    if "save_metadata_json" in e:
        c.save_metadata_json = bool(e["save_metadata_json"])
    if "png8_output_max" in e:
        c.png8_output_max = float(e["png8_output_max"])
    if "intensity_lut" in e:
        lut = e["intensity_lut"]
        c.intensity_lut = (
            {"x": [float(v) for v in lut["x"]], "y": [float(v) for v in lut["y"]]}
            if lut else None
        )
    if "intensity_curve" in e:
        ic = e["intensity_curve"]
        c.intensity_curve = [float(v) for v in ic] if ic else None
    if "voi_output_max" in e:
        vom = e["voi_output_max"]
        c.voi_output_max = float(vom) if vom is not None else None


def _apply_runtime(c: RuntimeConfig, rt: Dict[str, Any]) -> None:
    if "num_workers" in rt:
        c.num_workers = int(rt["num_workers"])
    if "fail_fast" in rt:
        c.fail_fast = bool(rt["fail_fast"])
    if "dry_run" in rt:
        c.dry_run = bool(rt["dry_run"])
    if "save_report" in rt:
        c.save_report = bool(rt["save_report"])
    if "output_layout" in rt:
        c.output_layout = OutputLayout(rt["output_layout"])
    if "skip_existing" in rt:
        c.skip_existing = bool(rt["skip_existing"])
    if "force_read" in rt:
        c.force_read = bool(rt["force_read"])
    if "scan_cache" in rt:
        c.scan_cache = str(rt["scan_cache"]) if rt["scan_cache"] else None


_APPLY = {
    "dicom": _apply_dicom,
    "geometry": _apply_geometry,
    "roi": _apply_roi,
    "export": _apply_export,
    "runtime": _apply_runtime,
}
