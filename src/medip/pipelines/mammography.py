"""Mammography preprocessing pipeline.

Extends BasePipeline with mammography-specific VOI policy resolution
based on Presentation Intent Type (FOR PROCESSING / FOR PRESENTATION).
"""

from __future__ import annotations

import logging

from ..config import PipelineConfig, PresentationIntentPolicy, VOIPolicy
from ..io.metadata import DicomMetadata
from .base import BasePipeline

logger = logging.getLogger(__name__)


class MammographyPipeline(BasePipeline):
    """Mammography-specific pipeline.

    Adds Presentation Intent Type awareness:
    - FOR PROCESSING images skip VOI by default (raw for AI training)
    - FOR PRESENTATION images apply VOI normally
    """

    def _resolve_voi_policy(
        self, meta: DicomMetadata, cfg: PipelineConfig
    ) -> VOIPolicy:
        """Resolve VOI policy based on Presentation Intent Type."""
        intent_policy = cfg.dicom.presentation_intent_policy
        base_voi = cfg.dicom.voi_policy

        if intent_policy == PresentationIntentPolicy.AUTO:
            if meta.is_for_processing and base_voi == VOIPolicy.AUTO:
                logger.debug("FOR PROCESSING detected, skipping VOI")
                return VOIPolicy.NONE
            return base_voi

        if intent_policy == PresentationIntentPolicy.FOR_PROCESSING:
            return VOIPolicy.NONE

        return base_voi
