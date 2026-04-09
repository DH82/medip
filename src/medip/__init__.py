"""medip: DICOM-standard medical image preprocessing toolkit.

Supports modality-specific pipelines with profile-driven configuration.
Currently supported: Mammography.

Usage:
    from medip import MammographyPipeline

    pipeline = MammographyPipeline(profile="learning_raw")
    result = pipeline.process_single("input.dcm", "output")

    # Or with custom config
    from medip import get_preset
    config = get_preset("mammography/learning_raw")
    pipeline = MammographyPipeline(config=config)
"""

__version__ = "0.1.1"

from .config import PipelineConfig
from .pipelines.base import BasePipeline
from .pipelines.mammography import MammographyPipeline
from .presets import get_preset, list_presets, list_modalities

__all__ = [
    "BasePipeline",
    "MammographyPipeline",
    "PipelineConfig",
    "get_preset",
    "list_presets",
    "list_modalities",
    "__version__",
]
