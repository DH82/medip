"""One-call entry point: ``medip.run(input, output)``.

With no ``config`` the run follows ``default.yaml`` shipped next to this file,
which spells out every option. The CLI's ``medip run`` goes through here too.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config import PipelineConfig
from .pipelines.mammography import MammographyPipeline
from .transforms.export import _resolve_output_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")

ConfigLike = Union[None, str, Path, PipelineConfig]


def load_default_config() -> PipelineConfig:
    """Return the configuration in ``default.yaml``."""
    from .utils.config_loader import load_config

    return load_config(DEFAULT_CONFIG_PATH)


def write_default_config(path: Union[str, Path], overwrite: bool = False) -> Path:
    """Copy ``default.yaml`` to *path* as a starting point for customization."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_CONFIG_PATH, path)
    return path


def resolve_config(config: ConfigLike = None) -> PipelineConfig:
    """Turn the ``config`` argument of :func:`run` into a PipelineConfig.

    - ``None``: ``default.yaml``
    - a ``PipelineConfig``: used as-is
    - a path to an existing file, or a name ending in ``.yaml``/``.yml``: that YAML
    - anything else: a preset name (e.g. ``"learning_raw"``)
    """
    if config is None:
        return load_default_config()
    if isinstance(config, PipelineConfig):
        return config

    from .utils.config_loader import load_config

    path = Path(config)
    if path.is_file() or path.suffix.lower() in (".yaml", ".yml"):
        return load_config(path)

    from .presets import get_preset

    return get_preset(str(config))


def run(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: ConfigLike = None,
) -> Dict[str, Any]:
    """Preprocess a DICOM file or a directory of DICOMs.

    Parameters
    ----------
    input_path : a DICOM file, or a directory searched recursively for ``.dcm``/``.dicom``
    output_dir : output directory; ``runtime.output_layout`` decides whether a
        directory input keeps its sub-folders (``tree``) or is flattened (``flat``)
    config : ``None`` for ``default.yaml``, a YAML path, a preset name,
        or a ``PipelineConfig``

    Returns
    -------
    dict with ``total``, ``success_count``, ``skipped``, ``failures`` (list of
    (path, error)), ``vendor_overrides`` ({manufacturer: images}) and ``config``
    (the PipelineConfig that was used).

    Example
    -------
    >>> import medip
    >>> medip.run("dicom_dir/", "png_dir/")                        # default.yaml
    >>> medip.run("dicom_dir/", "png_dir/", "my.yaml")            # your own YAML
    """
    cfg = resolve_config(config)
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    if cfg.runtime.dry_run:
        print(describe_config(cfg, input_path, output_dir))
        return {"total": 0, "success_count": 0, "failures": [], "config": cfg, "dry_run": True}

    pipeline = MammographyPipeline(config=cfg)

    if input_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {
            "total": 1, "success_count": 0, "skipped": 0, "failures": [], "vendor_overrides": {},
        }
        out_stem = pipeline.output_stem(input_path, input_path.parent, output_dir)
        if cfg.runtime.skip_existing and _resolve_output_path(out_stem, cfg.export.format).exists():
            summary["skipped"] = 1
        else:
            try:
                result = pipeline.process_single(input_path, out_stem)
                summary["success_count"] = 1
                summary["output_path"] = result.get("output_path")
                if result.get("vendor_override"):
                    summary["vendor_overrides"][result["vendor_override"]] = 1
            except Exception as e:
                if cfg.runtime.fail_fast:
                    raise
                summary["failures"].append((str(input_path), str(e)))
    else:
        summary = pipeline.process_directory(input_path, output_dir)

    if cfg.runtime.save_report:
        report_path = output_dir / "processing_report.json"
        report = {
            "profile": cfg.profile,
            "total": summary["total"],
            "success": summary["success_count"],
            "skipped": summary["skipped"],
            "vendor_overrides": summary["vendor_overrides"],
            "failed": len(summary["failures"]),
            "failures": [{"file": f, "error": e} for f, e in summary["failures"]],
        }
        report_path.write_text(json.dumps(report, indent=2))
        summary["report_path"] = str(report_path)

    summary["config"] = cfg
    return summary


def describe_config(cfg: PipelineConfig, input_path: Path, output_dir: Path) -> str:
    """Human-readable summary of what a run would do."""
    return "\n".join([
        f"[DRY RUN] Profile: {cfg.profile}",
        f"  Input:   {input_path}",
        f"  Output:  {output_dir}",
        f"  Format:  {cfg.export.format.value}",
        f"  DType:   {cfg.export.dtype.value}",
        f"  VOI:     {cfg.dicom.voi_policy.value}",
        f"  Spacing: {cfg.geometry.spacing_policy.value}",
        f"  Mask:    {cfg.roi.mask_policy.value}",
        f"  Crop:    {cfg.roi.crop_policy.value}",
        f"  Layout:  {cfg.runtime.output_layout.value}"
        + ("  (skip existing)" if cfg.runtime.skip_existing else ""),
        f"  Workers: {cfg.runtime.num_workers}",
        *(f"  Vendor override: {'/'.join(o.manufacturer)}" for o in cfg.vendor_overrides),
    ])
