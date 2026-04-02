"""Tests for DicomMetadata extraction."""

import pytest
import numpy as np

from medip.io.metadata import DicomMetadata, extract_metadata


class TestDicomMetadata:
    """Unit tests for DicomMetadata dataclass."""

    def test_defaults(self):
        meta = DicomMetadata()
        assert meta.bits_allocated == 16
        assert meta.bits_stored == 12
        assert meta.photometric_interpretation == "MONOCHROME2"
        assert not meta.is_monochrome1
        assert meta.rescale_slope == 1.0
        assert meta.rescale_intercept == 0.0

    def test_monochrome1_detection(self):
        meta = DicomMetadata(photometric_interpretation="MONOCHROME1")
        assert meta.is_monochrome1

    def test_presentation_intent(self):
        meta = DicomMetadata(presentation_intent_type="FOR PROCESSING")
        assert meta.is_for_processing
        assert not meta.is_for_presentation

        meta2 = DicomMetadata(presentation_intent_type="FOR PRESENTATION")
        assert meta2.is_for_presentation
        assert not meta2.is_for_processing

    def test_has_window(self):
        meta = DicomMetadata(window_center=[2048.0], window_width=[4096.0])
        assert meta.has_window

        meta_no = DicomMetadata()
        assert not meta_no.has_window

    def test_has_pixel_padding(self):
        meta = DicomMetadata(pixel_padding_value=0)
        assert meta.has_pixel_padding

        meta_none = DicomMetadata()
        assert not meta_none.has_pixel_padding

    def test_effective_spacing(self):
        meta = DicomMetadata(
            pixel_spacing=(0.1, 0.1),
            imager_pixel_spacing=(0.15, 0.15),
        )
        assert meta.effective_spacing == (0.1, 0.1)

        meta2 = DicomMetadata(imager_pixel_spacing=(0.15, 0.15))
        assert meta2.effective_spacing == (0.15, 0.15)

    def test_to_dict(self):
        meta = DicomMetadata(
            bits_stored=14,
            window_center=[2048.0],
            pixel_spacing=(0.07, 0.07),
        )
        d = meta.to_dict()
        assert d["bits_stored"] == 14
        assert d["window_center"] == [2048.0]
        assert d["pixel_spacing"] == [0.07, 0.07]
