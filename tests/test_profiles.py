"""Tests for profile presets and config."""

import pytest

from medip.config import (
    ExportDtype,
    ExportFormat,
    MaskPolicy,
    PipelineConfig,
    SpacingPolicy,
    VOIPolicy,
)
from medip.presets import get_preset, list_presets, list_modalities
from medip.exceptions import InvalidProfileError


class TestPresets:
    def test_mammography_modality_registered(self):
        assert "mammography" in list_modalities()

    def test_all_presets_exist(self):
        names = list_presets(modality="mammography")
        assert "learning_raw" in names
        assert "learning_presentation" in names
        assert "clinical_display" in names

    def test_qualified_name_lookup(self):
        cfg = get_preset("mammography/learning_raw")
        assert isinstance(cfg, PipelineConfig)
        assert cfg.profile == "mammography/learning_raw"

    def test_short_name_lookup(self):
        cfg = get_preset("learning_raw")
        assert isinstance(cfg, PipelineConfig)
        assert cfg.profile == "mammography/learning_raw"

    def test_learning_raw_defaults(self):
        cfg = get_preset("learning_raw")
        assert cfg.dicom.voi_policy == VOIPolicy.NONE
        assert cfg.geometry.spacing_policy == SpacingPolicy.KEEP
        assert cfg.roi.mask_policy == MaskPolicy.NONE
        assert cfg.export.format == ExportFormat.NPY
        assert cfg.export.dtype == ExportDtype.FLOAT32

    def test_clinical_display_defaults(self):
        cfg = get_preset("clinical_display")
        assert cfg.dicom.voi_policy == VOIPolicy.AUTO
        assert cfg.export.format == ExportFormat.PNG8

    def test_learning_raw_uint16_defaults(self):
        cfg = get_preset("learning_raw_uint16")
        assert cfg.dicom.voi_policy == VOIPolicy.NONE
        assert cfg.geometry.spacing_policy == SpacingPolicy.KEEP
        assert cfg.roi.mask_policy == MaskPolicy.NONE
        assert cfg.export.format == ExportFormat.NPY
        assert cfg.export.dtype == ExportDtype.UINT16

    def test_learning_presentation_defaults(self):
        cfg = get_preset("learning_presentation")
        assert cfg.dicom.voi_policy == VOIPolicy.AUTO
        assert cfg.export.format == ExportFormat.PNG16
        assert cfg.export.dtype == ExportDtype.UINT16

    def test_invalid_profile(self):
        with pytest.raises(InvalidProfileError):
            get_preset("nonexistent")

    def test_list_all_presets(self):
        all_presets = list_presets()
        assert all(
            name.startswith("mammography/") for name in all_presets
        )

    @pytest.mark.parametrize("name", list_presets(modality="mammography"))
    def test_all_mammography_presets_return_pipeline_config(self, name):
        cfg = get_preset(name)
        assert isinstance(cfg, PipelineConfig)
