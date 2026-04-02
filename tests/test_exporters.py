"""Tests for export transforms."""

import numpy as np
import pytest
from pathlib import Path

from medip.config import ExportConfig, ExportDtype, ExportFormat
from medip.transforms.export import export_image, _cast_dtype


class TestCastDtype:
    def test_float32(self):
        pixels = np.array([[100, 200]], dtype=np.float32)
        config = ExportConfig(dtype=ExportDtype.FLOAT32)
        result = _cast_dtype(pixels, config)
        assert result.dtype == np.float32

    def test_uint16(self):
        pixels = np.array([[0, 100, 200]], dtype=np.float32)
        config = ExportConfig(dtype=ExportDtype.UINT16)
        result = _cast_dtype(pixels, config)
        assert result.dtype == np.uint16
        assert result.min() == 0
        assert result.max() == 65535

    def test_uint8(self):
        pixels = np.array([[0, 100, 200]], dtype=np.float32)
        config = ExportConfig(dtype=ExportDtype.UINT8)
        result = _cast_dtype(pixels, config)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_constant_image(self):
        pixels = np.full((10, 10), 42.0, dtype=np.float32)
        config = ExportConfig(dtype=ExportDtype.UINT16)
        result = _cast_dtype(pixels, config)
        assert result.dtype == np.uint16
        assert np.all(result == 0)


class TestExportImage:
    def test_npy_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32)
        config = ExportConfig(
            format=ExportFormat.NPY,
            dtype=ExportDtype.FLOAT32,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.npy", config)
        assert out.exists()
        loaded = np.load(str(out))
        np.testing.assert_allclose(loaded, pixels)

    def test_npz_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32)
        config = ExportConfig(
            format=ExportFormat.NPZ,
            dtype=ExportDtype.FLOAT32,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.npz", config)
        assert out.exists()
        loaded = np.load(str(out))
        np.testing.assert_allclose(loaded["image"], pixels)

    def test_png8_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32) * 255
        config = ExportConfig(
            format=ExportFormat.PNG8,
            dtype=ExportDtype.UINT8,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.png", config)
        assert out.exists()
        assert out.suffix == ".png"

    def test_png16_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32) * 4095
        config = ExportConfig(
            format=ExportFormat.PNG16,
            dtype=ExportDtype.UINT16,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.png", config)
        assert out.exists()

    def test_tiff16_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32) * 4095
        config = ExportConfig(
            format=ExportFormat.TIFF16,
            dtype=ExportDtype.UINT16,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.tiff", config)
        assert out.exists()

    def test_jpeg_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32) * 255
        config = ExportConfig(
            format=ExportFormat.JPEG,
            dtype=ExportDtype.UINT8,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.jpg", config)
        assert out.exists()

    def test_metadata_sidecar(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32)
        config = ExportConfig(
            format=ExportFormat.NPY,
            dtype=ExportDtype.FLOAT32,
            save_metadata_json=True,
        )
        metadata_dict = {"bits_stored": 12, "spacing": [0.07, 0.07]}
        out = export_image(pixels, tmp_path / "test.npy", config, metadata_dict=metadata_dict)
        json_path = out.with_suffix(".json")
        assert json_path.exists()

    def test_mask_export(self, tmp_path):
        pixels = np.random.rand(64, 64).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.uint8)
        config = ExportConfig(
            format=ExportFormat.NPY,
            dtype=ExportDtype.FLOAT32,
            save_mask=True,
            save_metadata_json=False,
        )
        out = export_image(pixels, tmp_path / "test.npy", config, mask=mask)
        mask_path = out.parent / f"{out.stem}_mask.png"
        assert mask_path.exists()
