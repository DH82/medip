"""Base preprocessing pipeline.

Implements the standard DICOM grayscale transform chain:
  decode -> padding -> modality -> VOI -> photometric -> geometry -> mask -> crop -> export

``MammographyPipeline`` inherits from BasePipeline and overrides
``_resolve_voi_policy`` to honor Presentation Intent Type.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

from ..config import (
    MaskSource,
    OutputLayout,
    PipelineConfig,
    PresentationIntentPolicy,
    VendorOverride,
    VOIPolicy,
)
from ..exceptions import DicomReadError
from ..io.dicom_reader import read_dicom, set_force_read
from ..io.metadata import DicomMetadata, extract_metadata
from ..presets import DEFAULT_PROFILE, get_preset
from ..transforms.crop import crop_to_roi
from ..transforms.export import _resolve_output_path, export_image
from ..transforms.geometry import resample_image
from ..transforms.intensity import (
    apply_modality_transform,
    apply_voi_transform_ranged,
    modality_output_range,
)
from ..transforms.mask import extract_mask
from ..transforms.padding import apply_padding_mask
from ..transforms.photometric import apply_photometric

logger = logging.getLogger(__name__)

DICOM_SUFFIXES = (".dcm", ".dicom")

# Joins the parts of a relative input path into one name for the flat layout:
# patient/date/uid.dcm -> patient__date__uid.png
FLAT_PATH_JOIN = "__"


def _progress(iterable, **kwargs):
    """Wrap an iterable with tqdm when available."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


class BasePipeline:
    """Profile-driven DICOM preprocessing pipeline.

    Subclass and override ``_resolve_voi_policy`` or other hooks
    to add modality-specific behavior.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, profile: Optional[str] = None):
        if config is not None:
            self.config = config
        elif profile is not None:
            self.config = get_preset(profile)
        else:
            self.config = get_preset(DEFAULT_PROFILE)
        self._override_configs: Dict[int, PipelineConfig] = {}

    def process_single(
        self,
        dcm_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Process a single DICOM file through the full pipeline.

        Parameters
        ----------
        dcm_path : path to DICOM file
        output_path : output file path (extension overridden by config format)

        Returns
        -------
        dict with keys: output_path, metadata, crop_info, spacing,
        vendor_override (the matching override's manufacturer list, or None)
        """
        dcm_path = Path(dcm_path)
        set_force_read(self.config.runtime.force_read)

        # 1. Read DICOM (pydicom-authoritative decode)
        pixels, meta = read_dicom(dcm_path)

        # Per-vendor settings, chosen from the Manufacturer tag
        cfg, override = self._config_for(meta)

        # 2. Resolve VOI policy (subclass hook)
        voi_policy = self._resolve_voi_policy(meta, cfg)

        # 3. Pixel Padding (BEFORE intensity transforms)
        pixels, padding_mask = apply_padding_mask(
            pixels, meta, cfg.dicom.padding_policy
        )

        # 4. Modality LUT / Rescale
        pixels = apply_modality_transform(
            pixels, meta, apply=cfg.dicom.apply_modality_lut
        )
        value_range = modality_output_range(meta, cfg.dicom.apply_modality_lut)

        # Snapshot for the mask before windowing. A narrow vendor window clips and
        # saturates large parts of the histogram, which is exactly what Otsu reads,
        # so a mask taken after it can be far worse than one taken before. It is
        # brought to MONOCHROME2 here because the image itself is only inverted
        # after the VOI step.
        mask_pixels = (
            apply_photometric(pixels.copy(), meta, cfg.dicom.photometric_policy, value_range)
            if cfg.roi.mask_source == MaskSource.PRE_VOI else None
        )

        # 5. VOI LUT / Windowing, on the values the Window Center/Width and VOI
        # LUT tags were written for (DICOM PS3.4 N.2: Modality -> VOI -> Presentation)
        voi_output_max = cfg.export.voi_output_max
        if voi_output_max is None:
            voi_output_max = cfg.export.png8_output_max
        pixels, value_range = apply_voi_transform_ranged(
            pixels, meta,
            policy=voi_policy,
            voi_index=cfg.dicom.voi_index,
            output_max=voi_output_max,
            valid_mask=padding_mask,
            window_output_max=cfg.export.voi_output_max,
            input_range=value_range,
        )

        # 6. Photometric Interpretation: MONOCHROME1 -> MONOCHROME2, inverted
        # within the fixed output range of the previous step
        pixels = apply_photometric(pixels, meta, cfg.dicom.photometric_policy, value_range)

        # 7. Geometry / Resample
        current_spacing = meta.effective_spacing
        pixels, new_spacing = resample_image(pixels, current_spacing, cfg.geometry)

        # The mask snapshot has to follow the image through geometry so the two
        # stay the same shape.
        if mask_pixels is not None:
            mask_pixels, _ = resample_image(mask_pixels, current_spacing, cfg.geometry)

        # Resample padding mask too if it exists
        if padding_mask is not None:
            padding_mask_float = padding_mask.astype(np.float32)
            padding_mask_float, _ = resample_image(
                padding_mask_float, current_spacing, cfg.geometry, is_mask=True
            )
            padding_mask = (padding_mask_float > 0.5).astype(np.uint8)

        # 8. Mask extraction
        mask = extract_mask(
            pixels if mask_pixels is None else mask_pixels,
            cfg.roi,
            padding_mask=padding_mask,
        )

        # 9. Apply mask to image (zero out background) if mask exists
        if mask is not None and cfg.roi.mask_policy.value != "none":
            pixels = pixels * mask

        # 10. Crop
        pixels, mask, crop_info = crop_to_roi(
            pixels, mask, cfg.roi, pixel_spacing=new_spacing
        )

        # 11. Vendor intensity harmonization
        if cfg.export.intensity_lut:
            lut = cfg.export.intensity_lut
            pixels = np.interp(
                pixels,
                np.asarray(lut["x"], dtype=np.float64),
                np.asarray(lut["y"], dtype=np.float64),
            ).astype(np.float32)
        elif cfg.export.intensity_curve:
            ceiling = cfg.export.voi_output_max or 65535.0
            pixels = np.clip(
                np.polyval(np.asarray(cfg.export.intensity_curve, dtype=np.float64), pixels),
                0.0, ceiling,
            ).astype(np.float32)

        # 12. Export
        override_name = "/".join(override.manufacturer) if override else None
        result = {
            "metadata": meta,
            "crop_info": crop_info,
            "spacing": new_spacing,
            "shape": pixels.shape,
            "vendor_override": override_name,
        }

        if output_path is not None:
            metadata_dict = self._build_sidecar(
                meta, crop_info, new_spacing, cfg, pixels, voi_policy=voi_policy
            )
            metadata_dict["applied_transforms"]["vendor_override"] = override_name
            saved_path = export_image(
                pixels, output_path, cfg.export,
                mask=mask,
                metadata_dict=metadata_dict,
                pixel_spacing=new_spacing,
            )
            result["output_path"] = str(saved_path)
        else:
            result["pixels"] = pixels
            if mask is not None:
                result["mask"] = mask

        return result

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process all DICOM files under a directory (searched recursively).

        ``pattern`` is an optional glob (e.g. ``"**/*.dcm"``); by default every
        ``.dcm`` / ``.dicom`` file is found with ``os.walk``, which is much faster
        than globbing on network mounts. Output naming follows
        ``runtime.output_layout``; ``runtime.skip_existing`` resumes a run.

        Returns a summary dict: total, success_count, skipped, failures
        (list of (path, error)) and vendor_overrides ({manufacturer: count}).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rt = self.config.runtime

        dcm_files = self._find_inputs(input_dir, pattern)
        summary: Dict[str, Any] = {
            "success_count": 0,
            "skipped": 0,
            "failures": [],
            "total": len(dcm_files),
            "vendor_overrides": {},
        }
        if not dcm_files:
            logger.info("No DICOM files found in %s", input_dir)
            return summary

        jobs = [
            (dcm_path, self.output_stem(dcm_path, input_dir, output_dir, rt.output_layout))
            for dcm_path in dcm_files
        ]
        jobs = self._drop_collisions(jobs, summary["failures"])
        if rt.skip_existing:
            jobs, summary["skipped"] = self._drop_existing(jobs)
            if summary["skipped"]:
                logger.info("Skipping %d already converted file(s)", summary["skipped"])

        failures: List[Tuple[str, str]] = summary["failures"]
        counts: Dict[str, int] = summary["vendor_overrides"]

        def record(override_name: Optional[str]) -> None:
            summary["success_count"] += 1
            if override_name:
                counts[override_name] = counts.get(override_name, 0) + 1

        if rt.num_workers <= 1:
            for dcm_path, out_path in _progress(jobs, desc="Processing"):
                try:
                    record(self.process_single(dcm_path, out_path)["vendor_override"])
                except Exception as e:
                    failures.append((str(dcm_path), str(e)))
                    if rt.fail_fast:
                        raise
        elif jobs:
            with concurrent.futures.ProcessPoolExecutor(max_workers=rt.num_workers) as pool:
                # Workers are started by the submits below, so this is the window
                # in which they must not re-import the caller's script.
                with _workers_skip_main(self.__class__):
                    future_map = {
                        pool.submit(_worker, self.__class__, self.config, dcm_p, out_p): dcm_p
                        for dcm_p, out_p in jobs
                    }
                for future in _progress(
                    concurrent.futures.as_completed(future_map),
                    total=len(future_map),
                    desc="Processing",
                ):
                    dcm_p = future_map[future]
                    try:
                        record(future.result())
                    except Exception as e:
                        failures.append((str(dcm_p), str(e)))
                        if rt.fail_fast:
                            # Without this the pool would finish every queued
                            # file before the exception got out of the with-block.
                            for pending in future_map:
                                pending.cancel()
                            raise

        if failures:
            log_path = output_dir / "failures.log"
            with open(log_path, "w") as f:
                for path, reason in failures:
                    f.write(f"{path}\t{reason}\n")
            summary["failure_log"] = str(log_path)

        logger.info(
            "Done: %d/%d succeeded, %d skipped, %d failed",
            summary["success_count"], len(dcm_files), summary["skipped"], len(failures),
        )
        return summary

    def inspect(self, dcm_path: str | Path) -> Dict[str, Any]:
        """Inspect a DICOM file and return structured metadata."""
        meta = extract_metadata(dcm_path)
        return meta.to_dict()

    # ── Hooks (override in subclasses) ──────────────────────────────

    def _resolve_voi_policy(
        self, meta: DicomMetadata, cfg: PipelineConfig
    ) -> VOIPolicy:
        """Resolve effective VOI policy. Override for modality-specific logic."""
        return cfg.dicom.voi_policy

    # ── Internal helpers ─────────────────────────────────────────────

    def _config_for(self, meta: DicomMetadata) -> Tuple[PipelineConfig, Optional[VendorOverride]]:
        """Return the config for this image: the base one, or a vendor override."""
        for index, override in enumerate(self.config.vendor_overrides):
            if override.matches(meta.manufacturer):
                if index not in self._override_configs:
                    from ..utils.config_loader import apply_vendor_override

                    self._override_configs[index] = apply_vendor_override(self.config, override)
                return self._override_configs[index], override
        return self.config, None

    def _find_inputs(self, input_dir: Path, pattern: Optional[str]) -> List[Path]:
        """List input DICOMs, through ``runtime.scan_cache`` when it is set.

        The cache records which input directory and search it was made for. A
        cache made for anything else is ignored and rewritten, so one cache
        path cannot silently feed another cohort's file list into this run.
        Paths are stored relative to the input directory.
        """
        cache = Path(self.config.runtime.scan_cache) if self.config.runtime.scan_cache else None
        header = f"# medip scan of {input_dir.resolve()} pattern={pattern or '*'}"
        if cache is not None and cache.is_file():
            lines = cache.read_text().splitlines()
            if lines and lines[0] == header:
                files = [input_dir / line for line in lines[1:] if line.strip()]
                logger.info("Reusing cached scan: %d files from %s", len(files), cache)
                return files
            logger.warning(
                "Scan cache %s was made for a different input (%s); rescanning",
                cache, lines[0][2:] if lines else "empty file",
            )

        files = find_dicom_files(input_dir, pattern)

        if cache is not None and files:
            cache.parent.mkdir(parents=True, exist_ok=True)
            body = "".join(f"{p.relative_to(input_dir)}\n" for p in files)
            cache.write_text(f"{header}\n{body}")
            logger.info("Scan cached to %s", cache)
        return files

    def _drop_collisions(
        self, jobs: List[Tuple[Path, Path]], failures: List[Tuple[str, str]]
    ) -> List[Tuple[Path, Path]]:
        """Keep the first input for each output file; record the rest as failures.

        Two inputs can map to one output name: ``a.dcm`` and ``a.dicom`` in the
        same folder, or ``a/b__c.dcm`` and ``a__b/c.dcm`` in the flat layout.
        Converting both would make the second overwrite the first.
        """
        fmt = self.config.export.format
        owner: Dict[Path, Path] = {}
        kept = []
        for dcm_path, out_stem in jobs:
            target = _resolve_output_path(out_stem, fmt)
            if target in owner:
                failures.append((str(dcm_path), f"output name {target.name} already taken by {owner[target]}"))
                continue
            owner[target] = dcm_path
            kept.append((dcm_path, out_stem))
        if len(kept) < len(jobs):
            logger.warning("%d input(s) skipped: their output name collides with another input", len(jobs) - len(kept))
        return kept

    def _drop_existing(self, jobs: List[Tuple[Path, Path]]) -> Tuple[List[Tuple[Path, Path]], int]:
        """Remove jobs whose output file exists.

        Each output directory is listed once instead of stat-ing every file: on a
        network mount a stat is a full round trip, which adds up to hours over a
        few hundred thousand images.
        """
        listings: Dict[Path, set] = {}
        remaining = []
        fmt = self.config.export.format
        for dcm_path, out_stem in jobs:
            target = _resolve_output_path(out_stem, fmt)
            if target.parent not in listings:
                try:
                    listings[target.parent] = set(os.listdir(target.parent))
                except FileNotFoundError:
                    listings[target.parent] = set()
            if target.name not in listings[target.parent]:
                remaining.append((dcm_path, out_stem))
        return remaining, len(jobs) - len(remaining)

    @staticmethod
    def output_stem(
        dcm_path: Path, input_dir: Path, output_dir: Path,
        layout: OutputLayout = OutputLayout.TREE,
    ) -> Path:
        """Output path without extension for *dcm_path*.

        Only a DICOM extension is stripped, so the dots inside UID file names
        survive: ``1.2.840.113619.2.66.403.dcm`` -> ``1.2.840.113619.2.66.403``.
        """
        rel = dcm_path.relative_to(input_dir)
        if rel.suffix.lower() in DICOM_SUFFIXES:
            rel = rel.with_suffix("")
        if layout == OutputLayout.FLAT:
            return output_dir / FLAT_PATH_JOIN.join(rel.parts)
        return output_dir / rel

    @staticmethod
    def _build_sidecar(
        meta: DicomMetadata,
        crop_info: Dict,
        spacing: Optional[Tuple[float, float]],
        cfg: PipelineConfig,
        pixels: np.ndarray,
        voi_policy: Optional[VOIPolicy] = None,
    ) -> Dict[str, Any]:
        effective_voi = voi_policy if voi_policy is not None else cfg.dicom.voi_policy
        return {
            **meta.to_dict(),
            "applied_transforms": {
                "modality_lut": cfg.dicom.apply_modality_lut,
                "voi_policy": effective_voi.value,
                "photometric_policy": cfg.dicom.photometric_policy.value,
                "padding_policy": cfg.dicom.padding_policy.value,
                "spacing_policy": cfg.geometry.spacing_policy.value,
                "mask_policy": cfg.roi.mask_policy.value,
                "crop_policy": cfg.roi.crop_policy.value,
            },
            "crop_bbox": crop_info,
            "output_spacing": list(spacing) if spacing else None,
            "output_dtype": str(pixels.dtype),
            "output_shape": list(pixels.shape),
            "profile": cfg.profile,
        }


def find_dicom_files(input_dir: str | Path, pattern: Optional[str] = None) -> List[Path]:
    """Every ``.dcm`` / ``.dicom`` file (any case) under *input_dir*, sorted.

    ``os.walk`` is used instead of a glob because it is much faster on network
    mounts. Pass a glob *pattern* (e.g. ``"**/*.IMA"``) to search differently.
    """
    input_dir = Path(input_dir)
    if pattern:
        return sorted(p for p in input_dir.glob(pattern) if p.is_file())
    files: List[Path] = []
    for root, dirs, names in os.walk(input_dir):
        dirs.sort()
        files.extend(
            Path(root) / name for name in sorted(names)
            if name.lower().endswith(DICOM_SUFFIXES)
        )
    return files


@contextlib.contextmanager
def _workers_skip_main(pipeline_cls: type):
    """Start worker processes without re-importing the caller's ``__main__``.

    Under the "spawn" and "forkserver" start methods (the default on macOS,
    Windows, and on Linux from Python 3.14) every worker re-imports the script
    that called us. That breaks two ordinary ways of calling ``medip.run``:
    code piped into ``python -`` (there is no file to import), and a script
    without an ``if __name__ == "__main__":`` guard (each worker would run the
    whole script again). The workers only need medip itself, so while they
    start, ``__main__`` is made to look like an interactive session, which
    multiprocessing does not re-import.

    Skipped when the pipeline class lives outside medip: a subclass defined in
    the caller's script has to be importable from ``__main__`` in the worker.
    """
    main = sys.modules.get("__main__")
    if main is None or not pipeline_cls.__module__.startswith("medip."):
        yield
        return

    saved = {key: main.__dict__[key] for key in ("__file__", "__spec__") if key in main.__dict__}
    main.__dict__.pop("__file__", None)
    main.__spec__ = None
    try:
        yield
    finally:
        main.__dict__.pop("__spec__", None)
        main.__dict__.update(saved)


def _worker(pipeline_cls: type, config: PipelineConfig, dcm_path: Path, out_path: Path) -> Optional[str]:
    """Worker function for multiprocessing. Returns the vendor override used."""
    pipeline = pipeline_cls(config=config)
    return pipeline.process_single(dcm_path, out_path)["vendor_override"]
