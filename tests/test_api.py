"""Tests for medip.run() and the shipped default.yaml."""

import pathlib
import re
from dataclasses import asdict, fields

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import medip
import medip.config
from medip.api import DEFAULT_CONFIG_PATH, resolve_config
from medip.config import DicomConfig, ExportConfig, GeometryConfig, ROIConfig, RuntimeConfig
from medip.presets import DEFAULT_PROFILE, get_preset
from medip.utils.config_loader import config_from_dict, load_config

SECTIONS = {
    "dicom": DicomConfig,
    "geometry": GeometryConfig,
    "roi": ROIConfig,
    "export": ExportConfig,
    "runtime": RuntimeConfig,
}

# Fields that exist on the dataclasses but have no effect anywhere in the
# pipeline, so default.yaml deliberately leaves them out.
UNUSED_FIELDS = {"compressed_policy", "outputs", "fallback_to_full"}


def _write_dicom(path, rows=160, cols=120, manufacturer="HOLOGIC, Inc.", img=None,
                 photometric="MONOCHROME2", window=(1500, 1000)):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MG"
    ds.Manufacturer = manufacturer
    ds.WindowCenter, ds.WindowWidth = window
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0

    if img is None:
        img = np.full((rows, cols), 50, dtype=np.uint16)
        img[20:140, 0:80] = 3000  # "breast", larger than min_contour_area
    ds.Rows, ds.Columns = img.shape
    ds.PixelData = img.tobytes()
    _save(ds, path)


def _save(ds, path):
    """Write a DICOM file on both pydicom 2.x and 3.x."""
    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)  # pydicom 2.x


class TestDefaultYaml:
    def test_matches_default_preset(self):
        assert asdict(load_config(DEFAULT_CONFIG_PATH)) == asdict(get_preset(DEFAULT_PROFILE))

    def test_lists_every_option(self):
        text = DEFAULT_CONFIG_PATH.read_text()
        for section, cls in SECTIONS.items():
            for f in fields(cls):
                if f.name in UNUSED_FIELDS:
                    continue
                assert re.search(rf"^  {f.name}:", text, re.M), f"{section}.{f.name} missing"

    def test_every_commented_alternative_is_valid(self):
        """Uncommenting any '# key: value' line must give a loadable config."""
        text = DEFAULT_CONFIG_PATH.read_text()
        section = None
        checked = 0
        for line in text.splitlines():
            top = re.match(r"^(\w+):\s*$", line)
            if top:
                section = top.group(1)
                continue
            alt = re.match(r"^  # (\w+): (.+?)(\s+#.*)?$", line)
            if not alt or section is None:
                continue
            key, value = alt.group(1), alt.group(2)
            if value == "":
                continue
            import yaml
            config_from_dict({section: {key: yaml.safe_load(value)}})
            checked += 1
        assert checked > 30

    def test_vendor_override_example_loads(self, tmp_path):
        """The commented GE example must load once uncommented."""
        text = DEFAULT_CONFIG_PATH.read_text()
        head, example = text.split("# vendor_overrides:\n")
        enabled = head.replace("vendor_overrides: []", "# vendor_overrides: []") + "vendor_overrides:\n" + "".join(
            line[2:] + "\n" for line in example.splitlines()
        )
        p = tmp_path / "ge.yaml"
        p.write_text(enabled)
        (override,) = load_config(p).vendor_overrides
        assert override.matches("GE MEDICAL SYSTEMS") and not override.matches("HOLOGIC, Inc.")
        assert override.dicom == {"voi_policy": "window", "voi_index": 0}

    def test_run_without_config_uses_default_yaml(self):
        assert asdict(resolve_config(None)) == asdict(load_config(DEFAULT_CONFIG_PATH))


class TestLoaderValidation:
    def test_duplicate_key_rejected(self, tmp_path):
        p = tmp_path / "dup.yaml"
        p.write_text("dicom:\n  voi_policy: min_max_window\n  voi_policy: none\n")
        with pytest.raises(Exception, match="duplicate key 'voi_policy'"):
            load_config(p)

    def test_unknown_option_rejected(self, tmp_path):
        p = tmp_path / "typo.yaml"
        p.write_text("roi:\n  otsu_sensitivty: 0.4\n")
        with pytest.raises(ValueError, match="otsu_sensitivty"):
            load_config(p)

    def test_unknown_section_rejected(self):
        with pytest.raises(KeyError, match="exprot"):
            config_from_dict({"exprot": {}})

    def test_bad_vendor_override_value_fails_at_load(self):
        with pytest.raises(ValueError):
            config_from_dict({"vendor_overrides": [{"manufacturer": ["GE"], "dicom": {"voi_policy": "nope"}}]})

    def test_vendor_override_rejects_runtime(self):
        with pytest.raises(KeyError, match="runtime"):
            config_from_dict({"vendor_overrides": [{"manufacturer": ["GE"], "runtime": {}}]})

    def test_empty_section_allowed(self):
        assert config_from_dict({"dicom": None}).profile == DEFAULT_PROFILE


class TestRun:
    def test_directory_with_default_config(self, tmp_path):
        src = tmp_path / "dicom" / "patient1"
        src.mkdir(parents=True)
        _write_dicom(src / "a.dcm")
        _write_dicom(src / "b.dcm")

        summary = medip.run(tmp_path / "dicom", tmp_path / "out")

        assert summary["total"] == 2 and summary["success_count"] == 2
        png = tmp_path / "out" / "patient1" / "a.png"
        assert png.exists() and (tmp_path / "out" / "patient1" / "a.json").exists()
        import cv2
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        assert img.dtype == np.uint16 and img.max() == 65535

    def test_single_file_with_custom_yaml(self, tmp_path):
        dcm = tmp_path / "x.dcm"
        _write_dicom(dcm)
        cfg = medip.write_default_config(tmp_path / "my.yaml")
        cfg.write_text(cfg.read_text().replace(
            "  format: png16 ", "  # format: png16 ").replace(
            "  # format: npy ", "  format: npy "))

        summary = medip.run(dcm, tmp_path / "out", config=cfg)

        assert summary["success_count"] == 1
        assert (tmp_path / "out" / "x.npy").exists()

    def test_flat_layout_and_skip_existing(self, tmp_path):
        src = tmp_path / "dicom" / "p1" / "20200101"
        src.mkdir(parents=True)
        _write_dicom(src / "1.2.840.113619.2.66.403.dcm")
        _write_dicom(src / "1.2.840.113619.2.66.405.dcm")
        cfg = medip.load_default_config()
        cfg.runtime.output_layout = medip.config.OutputLayout.FLAT
        cfg.runtime.skip_existing = True
        cfg.runtime.scan_cache = str(tmp_path / "scan.txt")

        first = medip.run(tmp_path / "dicom", tmp_path / "out", config=cfg)
        names = sorted(p.name for p in (tmp_path / "out").glob("*.png"))
        assert names == [
            "p1__20200101__1.2.840.113619.2.66.403.png",
            "p1__20200101__1.2.840.113619.2.66.405.png",
        ]
        assert first["success_count"] == 2 and (tmp_path / "scan.txt").exists()

        (tmp_path / "out" / names[0]).unlink()
        second = medip.run(tmp_path / "dicom", tmp_path / "out", config=cfg)
        assert second["skipped"] == 1 and second["success_count"] == 1

    def test_vendor_override_applies_per_file(self, tmp_path):
        src = tmp_path / "dicom"
        src.mkdir()
        _write_dicom(src / "ge.dcm", manufacturer="GE MEDICAL SYSTEMS")
        _write_dicom(src / "hol.dcm")
        cfg = config_from_dict({
            "vendor_overrides": [{"manufacturer": ["GE MEDICAL"], "dicom": {"voi_policy": "window"}}],
            "runtime": {"num_workers": 1},
        })

        summary = medip.run(src, tmp_path / "out", config=cfg)

        assert summary["vendor_overrides"] == {"GE MEDICAL": 1}
        import json
        ge = json.loads((tmp_path / "out" / "ge.json").read_text())
        hol = json.loads((tmp_path / "out" / "hol.json").read_text())
        assert ge["applied_transforms"]["voi_policy"] == "window"
        assert ge["applied_transforms"]["vendor_override"] == "GE MEDICAL"
        assert hol["applied_transforms"]["voi_policy"] == "min_max_window"
        assert hol["applied_transforms"]["vendor_override"] is None

    def test_yaml_as_third_positional_argument(self, tmp_path):
        dcm = tmp_path / "x.dcm"
        _write_dicom(dcm)
        yaml_path = tmp_path / "my.yaml"
        yaml_path.write_text("export:\n  format: npy\n  dtype: float32\n")

        summary = medip.run(dcm, tmp_path / "out", yaml_path)

        assert summary["success_count"] == 1
        assert (tmp_path / "out" / "x.npy").exists()

    def test_missing_yaml_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            medip.run(tmp_path, tmp_path / "out", tmp_path / "nope.yaml")

    @pytest.mark.parametrize("how", ["script", "stdin"])
    def test_parallel_run_needs_no_main_guard(self, tmp_path, how):
        """Workers must not re-import the caller: no guard, or no file at all."""
        import os
        import subprocess
        import sys

        src = tmp_path / "dicom"
        src.mkdir()
        for name in ("a", "b", "c"):
            _write_dicom(src / f"{name}.dcm")
        code = (
            "import medip, sys\n"
            "cfg = medip.load_default_config()\n"
            "cfg.runtime.num_workers = 2\n"
            f"s = medip.run({str(src)!r}, {str(tmp_path / 'out')!r}, cfg)\n"
            "main = sys.modules['__main__']\n"
            "print('RESULT', s['success_count'], getattr(main, '__file__', None))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(pathlib.Path(medip.__file__).parents[1])}
        if how == "script":
            script = tmp_path / "noguard.py"
            script.write_text(code)
            proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                                  env=env, cwd=tmp_path, timeout=120)
            expected_file = str(script)
        else:
            proc = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True,
                                  env=env, cwd=tmp_path, timeout=120)
            expected_file = "<stdin>"

        assert proc.returncode == 0, proc.stderr[-2000:]
        # every image converted, and __main__.__file__ restored afterwards
        assert f"RESULT 3 {expected_file}" in proc.stdout

    def test_preset_name(self, tmp_path):
        assert resolve_config("learning_raw").profile == "mammography/learning_raw"

    def test_write_default_config_refuses_overwrite(self, tmp_path):
        p = medip.write_default_config(tmp_path / "c.yaml")
        with pytest.raises(FileExistsError):
            medip.write_default_config(p)


class TestReviewFixes:
    """Regressions for issues found in the pre-release review."""

    def test_bits_above_bits_stored_are_dropped(self, tmp_path):
        img = np.full((160, 120), 1000, dtype=np.uint16)
        img[0, :10] = 4096 + 96  # stray bit 12 set, BitsStored = 12
        _write_dicom(tmp_path / "x.dcm", img=img)
        from medip.io.dicom_reader import read_dicom

        pixels, _ = read_dicom(tmp_path / "x.dcm")
        assert pixels.max() == 1000 and pixels[0, 0] == 96

    @pytest.mark.parametrize("profile", ["learning_presentation", "png16_minmax", "learning_raw"])
    def test_monochrome1_matches_monochrome2(self, tmp_path, profile):
        """The same anatomy stored either way must come out identical."""
        m2 = np.tile(np.linspace(500, 3000, 120), (160, 1)).astype(np.uint16)
        _write_dicom(tmp_path / "m2.dcm", img=m2, window=(1500, 1000))
        _write_dicom(tmp_path / "m1.dcm", img=(4095 - m2).astype(np.uint16),
                     photometric="MONOCHROME1", window=(4096 - 1500, 1000))
        # 4096 - c, not 4095 - c: the DICOM linear window is centered on c - 0.5
        pipeline = medip.MammographyPipeline(profile=profile)

        a = pipeline.process_single(tmp_path / "m2.dcm")["pixels"]
        b = pipeline.process_single(tmp_path / "m1.dcm")["pixels"]
        np.testing.assert_allclose(a, b, atol=1)

    def test_monochrome1_raw_inversion_does_not_depend_on_image_max(self, tmp_path):
        base = np.full((160, 120), 1000, dtype=np.uint16)
        brighter = base.copy()
        brighter[0, 0] = 3000  # one bright pixel must not shift every other value
        _write_dicom(tmp_path / "a.dcm", img=base, photometric="MONOCHROME1")
        _write_dicom(tmp_path / "b.dcm", img=brighter, photometric="MONOCHROME1")
        pipeline = medip.MammographyPipeline(profile="learning_raw")

        a = pipeline.process_single(tmp_path / "a.dcm")["pixels"]
        b = pipeline.process_single(tmp_path / "b.dcm")["pixels"]
        assert a[5, 5] == b[5, 5] == 4095 - 1000

    def test_interrupted_write_leaves_no_output(self, tmp_path, monkeypatch):
        """A crash while writing must not leave a file that skip_existing trusts."""
        import cv2
        from medip.transforms import export

        src = tmp_path / "dicom"
        src.mkdir()
        _write_dicom(src / "a.dcm")
        cfg = config_from_dict({"runtime": {"skip_existing": True, "num_workers": 1}})

        def crash(path, img, *args):
            pathlib.Path(path).write_bytes(b"\x89PNG half")
            raise KeyboardInterrupt

        monkeypatch.setattr(cv2, "imwrite", crash)
        with pytest.raises(KeyboardInterrupt):
            export.export_image(np.zeros((4, 4), np.float32), tmp_path / "out" / "a", cfg.export)
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == []
        monkeypatch.undo()

        summary = medip.run(src, tmp_path / "out", cfg)
        assert summary["skipped"] == 0 and summary["success_count"] == 1

    def test_scan_cache_from_another_input_is_not_reused(self, tmp_path):
        for name in ("A", "B"):
            (tmp_path / name).mkdir()
            _write_dicom(tmp_path / name / f"{name}.dcm")
        cache = tmp_path / "scan.txt"
        cfg = config_from_dict({"runtime": {"scan_cache": str(cache), "num_workers": 1}})

        medip.run(tmp_path / "A", tmp_path / "outA", cfg)
        summary = medip.run(tmp_path / "B", tmp_path / "outB", cfg)

        assert summary["success_count"] == 1 and not summary["failures"]
        assert (tmp_path / "outB" / "B.png").exists()
        assert str((tmp_path / "B").resolve()) in cache.read_text().splitlines()[0]

    def test_output_name_collision_is_reported(self, tmp_path):
        src = tmp_path / "dicom"
        src.mkdir()
        _write_dicom(src / "a.dcm")
        _write_dicom(src / "a.dicom")
        cfg = config_from_dict({"runtime": {"num_workers": 1}})

        summary = medip.run(src, tmp_path / "out", cfg)

        assert summary["success_count"] == 1
        (failure,) = summary["failures"]
        assert failure[0].endswith("a.dicom") and "already taken" in failure[1]

    def test_validate_finds_the_same_files_as_run(self, tmp_path, capsys):
        from medip.cli import main

        _write_dicom(tmp_path / "a.dcm")
        _write_dicom(tmp_path / "b.DICOM")
        main(["validate", str(tmp_path)])
        assert "2 files scanned" in capsys.readouterr().out
