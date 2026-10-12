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
    _min_max_window,
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


class TestMinMaxWindow:
    def test_spans_image_range(self):
        pixels = np.array([[500, 1000, 1500]], dtype=np.float32)
        result = _min_max_window(pixels, output_max=65535.0)
        np.testing.assert_allclose(result, [[0.0, 32767.5, 65535.0]], rtol=1e-5)

    def test_constant_image(self):
        pixels = np.full((2, 2), 700.0, dtype=np.float32)
        result = _min_max_window(pixels, output_max=65535.0)
        np.testing.assert_allclose(result, np.zeros((2, 2)))

    def test_valid_mask_excludes_padding(self):
        # -2000 is a padding value; it must not stretch the window
        pixels = np.array([[-2000, 500, 1000, 1500]], dtype=np.float32)
        valid = np.array([[False, True, True, True]])
        result = _min_max_window(pixels, output_max=65535.0, valid_mask=valid)
        # Window is [500, 1500]; the padding pixel clips to the low end
        np.testing.assert_allclose(result, [[0.0, 0.0, 32767.5, 65535.0]], rtol=1e-5)

    def test_policy_dispatch_ignores_window_tags(self):
        pixels = np.array([[500, 1000, 1500]], dtype=np.float32)
        # WC/WW are present but MIN_MAX_WINDOW must ignore them
        meta = DicomMetadata(window_center=[100.0], window_width=[10.0])
        result = apply_voi_transform(
            pixels, meta, policy=VOIPolicy.MIN_MAX_WINDOW, output_max=65535.0
        )
        np.testing.assert_allclose(result, [[0.0, 32767.5, 65535.0]], rtol=1e-5)


class TestWindowOutputMax:
    """_apply_window rescale: a fixed BitsStored factor, not a per-image stretch."""

    @staticmethod
    def _patch_pydicom(monkeypatch, windowed=None, apply_windowing=None, apply_voi_lut=None):
        """Stub pydicom's LUT helpers, whichever pydicom version provides them."""
        import pydicom
        from medip.transforms import intensity

        monkeypatch.setattr(pydicom, "dcmread", lambda *a, **k: object())
        windowing = apply_windowing or (lambda pixels, ds, index=0: windowed)
        voi_lut = apply_voi_lut or (lambda pixels, ds, index=0: windowed)
        monkeypatch.setattr(intensity, "_pydicom_luts", lambda: (None, voi_lut, windowing))

    def test_rescale_is_a_fixed_factor(self, monkeypatch):
        from medip.transforms.intensity import _apply_window

        windowed = np.array([[0.0, 1024.0, 3346.0]], dtype=np.float32)
        self._patch_pydicom(monkeypatch, windowed)
        meta = DicomMetadata(bits_stored=12, window_center=[2646.0], window_width=[750.0])

        out, value_range = _apply_window(windowed, meta, 0, output_max=65535.0)
        assert value_range == (0.0, 65535.0)

        factor = 65535.0 / (2 ** 12 - 1)  # 16.0037, not 16
        np.testing.assert_allclose(out, windowed * factor, rtol=1e-6)
        # the image max must NOT be pinned to the ceiling — that would be a
        # per-image stretch and would break cross-image comparability
        assert out.max() < 65535.0

    def test_none_leaves_pydicom_range(self, monkeypatch):
        from medip.transforms.intensity import _apply_window

        windowed = np.array([[0.0, 3346.0]], dtype=np.float32)
        self._patch_pydicom(monkeypatch, windowed)
        meta = DicomMetadata(bits_stored=12, window_center=[2646.0], window_width=[750.0])

        out, value_range = _apply_window(windowed, meta, 0, output_max=None)
        np.testing.assert_allclose(out, windowed)
        assert value_range == (0.0, 4095.0)

    def test_voi_index_selects_the_window(self, monkeypatch):
        from medip.transforms.intensity import apply_voi_transform

        seen = {}

        def spy(pixels, ds, index=0):
            seen["index"] = index
            return pixels

        self._patch_pydicom(monkeypatch, apply_windowing=spy)

        meta = DicomMetadata(
            bits_stored=12,
            window_center=[2646.0, 2670.0, 2604.0],
            window_width=[750.0, 600.0, 900.0],
        )
        apply_voi_transform(
            np.zeros((2, 2), dtype=np.float32), meta,
            policy=VOIPolicy.WINDOW, voi_index=0,
        )
        assert seen["index"] == 0


    def test_voi_lut_sequence_honors_output_max(self, monkeypatch):
        """A VOI LUT Sequence is rescaled onto voi_output_max like a window."""
        from types import SimpleNamespace
        from medip.transforms.intensity import apply_voi_transform_ranged

        lut_out = np.array([[0.0, 2047.0, 4095.0]], dtype=np.float32)
        self._patch_pydicom(monkeypatch, apply_voi_lut=lambda pixels, ds, index=0: lut_out)
        meta = DicomMetadata(
            bits_stored=14,
            voi_lut_sequence=[SimpleNamespace(LUTDescriptor=[4096, 0, 12])],
        )
        out, value_range = apply_voi_transform_ranged(
            lut_out, meta, policy=VOIPolicy.AUTO, window_output_max=65535.0,
        )
        np.testing.assert_allclose(out, lut_out * (65535.0 / 4095.0), rtol=1e-6)
        assert value_range == (0.0, 65535.0)

    def test_window_range_follows_rescale(self, monkeypatch):
        """pydicom's window y-range is the stored range through slope/intercept."""
        from medip.transforms.intensity import _apply_window

        windowed = np.array([[-1024.0, 3071.0]], dtype=np.float32)
        self._patch_pydicom(monkeypatch, windowed)
        meta = DicomMetadata(bits_stored=12, rescale_intercept=-1024.0,
                             window_center=[0.0], window_width=[100.0])
        out, value_range = _apply_window(windowed, meta, 0, output_max=65535.0)
        np.testing.assert_allclose(out, [[0.0, 65535.0]], rtol=1e-6)


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
