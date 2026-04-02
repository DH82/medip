"""Tests for intensity transforms."""

import numpy as np
import pytest

from medip.config import VOIPolicy
from medip.io.metadata import DicomMetadata
from medip.transforms.intensity import (
    apply_modality_transform,
    apply_voi_transform,
    _legacy_linear,
    _legacy_sigmoid,
)


class TestModalityTransform:
    def test_slope_intercept(self):
        pixels = np.array([[100, 200], [300, 400]], dtype=np.float32)
        meta = DicomMetadata(rescale_slope=2.0, rescale_intercept=-100.0)
        result = apply_modality_transform(pixels, meta, apply=True)
        expected = pixels * 2.0 + (-100.0)
        np.testing.assert_allclose(result, expected)

    def test_no_apply(self):
        pixels = np.array([[100, 200]], dtype=np.float32)
        meta = DicomMetadata(rescale_slope=2.0, rescale_intercept=-100.0)
        result = apply_modality_transform(pixels, meta, apply=False)
        np.testing.assert_allclose(result, pixels)


class TestVOITransform:
    def test_none_policy(self):
        pixels = np.array([[100, 200]], dtype=np.float32)
        meta = DicomMetadata(window_center=[150.0], window_width=[100.0])
        result = apply_voi_transform(pixels, meta, policy=VOIPolicy.NONE)
        np.testing.assert_allclose(result, pixels)


class TestLegacyLinear:
    def test_basic_windowing(self):
        pixels = np.array([[0, 500, 1000, 1500, 2000]], dtype=np.float32)
        meta = DicomMetadata(window_center=[1000.0], window_width=[1000.0])
        result = _legacy_linear(pixels, meta, output_max=4095.0)

        # min_val=500, max_val=1500
        assert result[0, 0] == 0.0  # 0 clipped to 500 -> 0
        assert result[0, 1] == 0.0  # 500 -> 0
        assert abs(result[0, 2] - 2047.5) < 1.0  # 1000 -> midpoint
        assert abs(result[0, 3] - 4095.0) < 1.0  # 1500 -> max
        assert abs(result[0, 4] - 4095.0) < 1.0  # 2000 clipped to 1500 -> max

    def test_no_window(self):
        pixels = np.array([[100]], dtype=np.float32)
        meta = DicomMetadata()
        result = _legacy_linear(pixels, meta, output_max=4095.0)
        np.testing.assert_allclose(result, pixels)


class TestLegacySigmoid:
    def test_midpoint(self):
        meta = DicomMetadata(window_center=[1000.0], window_width=[500.0])
        pixels = np.array([[1000.0]], dtype=np.float32)
        result = _legacy_sigmoid(pixels, meta, output_max=4095.0)
        # At center, sigmoid = 0.5
        assert abs(result[0, 0] - 2047.5) < 1.0

    def test_no_window(self):
        pixels = np.array([[100]], dtype=np.float32)
        meta = DicomMetadata()
        result = _legacy_sigmoid(pixels, meta, output_max=4095.0)
        np.testing.assert_allclose(result, pixels)
