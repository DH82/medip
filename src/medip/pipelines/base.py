"""Base preprocessing pipeline.

Implements the standard DICOM grayscale transform chain:
  decode -> padding -> modality -> photometric -> VOI -> geometry -> mask -> crop -> export

Modality-specific pipelines inherit from BasePipeline and override
hooks like ``_resolve_voi_policy`` to customize behavior.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..config import PipelineConfig, PresentationIntentPolicy, VOIPolicy
from ..exceptions import DicomReadError
from ..io.dicom_reader import read_dicom
from ..io.metadata import DicomMetadata, extract_metadata
from ..presets import get_preset
from ..transforms.crop import crop_to_roi
from ..transforms.export import export_image
from ..transforms.geometry import resample_image
from ..transforms.intensity import apply_modality_transform, apply_voi_transform
from ..transforms.mask import extract_mask
from ..transforms.padding import apply_padding_mask
from ..transforms.photometric import apply_photometric

logger = logging.getLogger(__name__)


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
            self.config = get_preset("learning_raw")

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
        dict with keys: output_path, metadata, crop_info, spacing
        """
        dcm_path = Path(dcm_path)
        cfg = self.config

        # 1. Read DICOM (pydicom-authoritative decode)
        pixels, meta = read_dicom(dcm_path)

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

        # 5. Photometric Interpretation (MONOCHROME1 inversion)
        pixels = apply_photometric(pixels, meta, cfg.dicom.photometric_policy)

        # 6. VOI LUT / Windowing
        pixels = apply_voi_transform(
            pixels, meta,
            policy=voi_policy,
            voi_index=cfg.dicom.voi_index,
            output_max=cfg.export.png8_output_max,
        )

        # 7. Geometry / Resample
        current_spacing = meta.effective_spacing
        pixels, new_spacing = resample_image(pixels, current_spacing, cfg.geometry)

        # Resample padding mask too if it exists
        if padding_mask is not None:
            padding_mask_float = padding_mask.astype(np.float32)
            padding_mask_float, _ = resample_image(
                padding_mask_float, current_spacing, cfg.geometry, is_mask=True
            )
            padding_mask = (padding_mask_float > 0.5).astype(np.uint8)

        # 8. Mask extraction
        mask = extract_mask(pixels, cfg.roi, padding_mask=padding_mask)

        # 9. Apply mask to image (zero out background) if mask exists
        if mask is not None and cfg.roi.mask_policy.value != "none":
            pixels = pixels * mask

        # 10. Crop
        pixels, mask, crop_info = crop_to_roi(
            pixels, mask, cfg.roi, pixel_spacing=new_spacing
        )

        # 11. Export
        result = {
            "metadata": meta,
            "crop_info": crop_info,
            "spacing": new_spacing,
            "shape": pixels.shape,
        }

        if output_path is not None:
            metadata_dict = self._build_sidecar(meta, crop_info, new_spacing, cfg, pixels)
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
        pattern: str = "**/*.dcm",
    ) -> Dict[str, Any]:
        """Process all DICOM files in a directory.

        Returns summary dict with success_count, failures, total.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dcm_files = sorted(input_dir.glob(pattern))
        if not dcm_files:
            dcm_files = sorted(
                p for p in input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() == ".dcm"
            )

        if not dcm_files:
            logger.info("No DICOM files found in %s", input_dir)
            return {"success_count": 0, "failures": [], "total": 0}

        num_workers = self.config.runtime.num_workers
        success_count = 0
        failures: List[Tuple[str, str]] = []

        args_list = [
            (dcm_path, self._build_output_path(dcm_path, input_dir, output_dir))
            for dcm_path in dcm_files
        ]

        if num_workers <= 1:
            for dcm_path, out_path in tqdm(args_list, desc="Processing"):
                try:
                    self.process_single(dcm_path, out_path)
                    success_count += 1
                except Exception as e:
                    failures.append((str(dcm_path), str(e)))
                    if self.config.runtime.fail_fast:
                        raise
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
                future_map = {
                    pool.submit(_worker, self.__class__, self.config, dcm_p, out_p): dcm_p
                    for dcm_p, out_p in args_list
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_map),
                    total=len(future_map),
                    desc="Processing",
                ):
                    dcm_p = future_map[future]
                    try:
                        future.result()
                        success_count += 1
                    except Exception as e:
                        failures.append((str(dcm_p), str(e)))
                        if self.config.runtime.fail_fast:
                            raise

        summary = {
            "success_count": success_count,
            "failures": failures,
            "total": len(dcm_files),
        }

        logger.info(
            "Done: %d/%d succeeded, %d failed",
            success_count, len(dcm_files), len(failures),
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

    @staticmethod
    def _build_output_path(
        dcm_path: Path, input_dir: Path, output_dir: Path
    ) -> Path:
        rel = dcm_path.relative_to(input_dir)
        return output_dir / rel.with_suffix("")

    @staticmethod
    def _build_sidecar(
        meta: DicomMetadata,
        crop_info: Dict,
        spacing: Optional[Tuple[float, float]],
        cfg: PipelineConfig,
        pixels: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            **meta.to_dict(),
            "applied_transforms": {
                "modality_lut": cfg.dicom.apply_modality_lut,
                "voi_policy": cfg.dicom.voi_policy.value,
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


def _worker(pipeline_cls: type, config: PipelineConfig, dcm_path: Path, out_path: Path) -> None:
    """Worker function for multiprocessing."""
    pipeline = pipeline_cls(config=config)
    pipeline.process_single(dcm_path, out_path)
