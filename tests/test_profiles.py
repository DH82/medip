"""Tests for profile presets and config."""

import pytest

from medip.config import (
    CropPolicy,
    ExportDtype,
    ExportFormat,
    MaskPolicy,
    PipelineConfig,
    PresentationIntentPolicy,
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
        assert "png16_minmax" in names
        assert "png16_minmax_masked" in names
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

    def test_png16_minmax_defaults(self):
        cfg = get_preset("png16_minmax")
        assert cfg.dicom.voi_policy == VOIPolicy.MIN_MAX_WINDOW
        assert cfg.dicom.presentation_intent_policy == PresentationIntentPolicy.IGNORE
        assert cfg.geometry.spacing_policy == SpacingPolicy.KEEP
        assert cfg.roi.crop_policy == CropPolicy.NONE
        assert cfg.export.format == ExportFormat.PNG16
        assert cfg.export.dtype == ExportDtype.UINT16
        assert cfg.export.voi_output_max == 65535.0

    def test_png16_minmax_masked_defaults(self):
        cfg = get_preset("png16_minmax_masked")
        assert cfg.dicom.voi_policy == VOIPolicy.MIN_MAX_WINDOW
        assert cfg.roi.mask_policy == MaskPolicy.OTSU
        # the FOV must survive: cropping would rescale the breast under a
        # fixed-size resize downstream
        assert cfg.roi.crop_policy == CropPolicy.NONE
        assert cfg.export.format == ExportFormat.PNG16
        assert cfg.export.voi_output_max == 65535.0

    def test_png16_minmax_variants_differ_only_in_mask(self):
        plain = get_preset("png16_minmax")
        masked = get_preset("png16_minmax_masked")
        assert plain.roi.mask_policy == MaskPolicy.NONE
        assert masked.roi.mask_policy == MaskPolicy.OTSU
        assert plain.dicom == masked.dicom
        assert plain.geometry == masked.geometry
        assert plain.export.format == masked.export.format
        assert plain.export.voi_output_max == masked.export.voi_output_max

    def test_learning_presentation_defaults(self):
        cfg = get_preset("learning_presentation")
        assert cfg.dicom.voi_policy == VOIPolicy.AUTO
        assert cfg.export.format == ExportFormat.PNG16
        assert cfg.export.dtype == ExportDtype.UINT16

    def test_default_profile_is_png16_minmax_masked(self):
        from medip import DEFAULT_PROFILE, MammographyPipeline
        from medip.utils.config_loader import config_from_dict

        assert DEFAULT_PROFILE == "mammography/png16_minmax_masked"
        assert MammographyPipeline().config.profile == DEFAULT_PROFILE
        assert config_from_dict({}).profile == DEFAULT_PROFILE

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


class TestMaskComponents:
    """_select_components: largest-only vs every component above min area."""

    def _two_blobs(self):
        import numpy as np
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:110, 10:110] = 1     # 100x100 = 10,000 px
        mask[150:190, 150:190] = 1   # 40x40   =  1,600 px
        return mask

    def test_largest_only_drops_the_smaller_blob(self):
        from medip.config import ROIConfig
        from medip.transforms.mask import _select_components

        cfg = ROIConfig(min_contour_area=500, keep_all_components=False)
        out = _select_components(self._two_blobs(), cfg)
        assert out[50, 50] == 1
        assert out[170, 170] == 0

    def test_keep_all_keeps_both_above_min_area(self):
        from medip.config import ROIConfig
        from medip.transforms.mask import _select_components

        cfg = ROIConfig(min_contour_area=500, keep_all_components=True)
        out = _select_components(self._two_blobs(), cfg)
        assert out[50, 50] == 1
        assert out[170, 170] == 1

    def test_keep_all_still_honours_min_area(self):
        from medip.config import ROIConfig
        from medip.transforms.mask import _select_components

        # min area above the smaller blob: it must still be dropped, which is how
        # burned-in annotation is kept out when keep_all_components is on
        cfg = ROIConfig(min_contour_area=5000, keep_all_components=True)
        out = _select_components(self._two_blobs(), cfg)
        assert out[50, 50] == 1
        assert out[170, 170] == 0
