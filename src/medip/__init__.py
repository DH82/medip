"""medip: DICOM mammography preprocessing toolkit.

Profile-driven pipeline. The default configuration (``default.yaml`` in this
package, same as preset ``mammography/png16_minmax_masked``) writes uint16 PNGs
with a min-max window and an Otsu background mask.

Usage:
    import medip

    medip.run("dicom_dir/", "png_dir/")                    # default.yaml
    medip.write_default_config("my.yaml")                  # copy every option, edit
    medip.run("dicom_dir/", "png_dir/", "my.yaml")         # your own YAML

    # Lower level
    from medip import MammographyPipeline

    pipeline = MammographyPipeline()          # default: png16_minmax_masked
    result = pipeline.process_single("input.dcm", "output/image")  # -> output/image.png

    # Another preset, or a preset with overrides
    from medip import get_preset
    config = get_preset("learning_raw")
    config.runtime.num_workers = 16
    pipeline = MammographyPipeline(config=config)
"""

__version__ = "0.2.0"

from .api import (
    DEFAULT_CONFIG_PATH,
    load_default_config,
    run,
    write_default_config,
)
from .config import PipelineConfig
from .pipelines.base import BasePipeline
from .pipelines.mammography import MammographyPipeline
from .presets import DEFAULT_PROFILE, get_preset, list_presets, list_modalities

__all__ = [
    "run",
    "load_default_config",
    "write_default_config",
    "DEFAULT_CONFIG_PATH",
    "BasePipeline",
    "MammographyPipeline",
    "PipelineConfig",
    "DEFAULT_PROFILE",
    "get_preset",
    "list_presets",
    "list_modalities",
    "__version__",
]
