"""Tests for CLI interface."""

from medip.pipelines import base as base_module
import pytest
from medip.cli import main


class TestCLI:
    def test_presets_command(self, capsys):
        main(["presets"])
        captured = capsys.readouterr()
        assert "learning_raw" in captured.out
        assert "mammography" in captured.out

    def test_presets_modality_filter(self, capsys):
        main(["presets", "--modality", "mammography"])
        captured = capsys.readouterr()
        assert "mammography" in captured.out
        assert "clinical_display" in captured.out

    def test_no_command_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_run_dry_run(self, capsys, tmp_path):
        main([
            "run",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--profile", "learning_raw",
            "--dry-run",
        ])
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "learning_raw" in captured.out

    def test_run_dry_run_with_modality(self, capsys, tmp_path):
        main([
            "run",
            "--input", str(tmp_path),
            "--output", str(tmp_path / "out"),
            "--modality", "mammography",
            "--dry-run",
        ])
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "mammography" in captured.out

    def test_progress_falls_back_without_tqdm(self, monkeypatch):
        monkeypatch.setattr(base_module, "_tqdm", None)
        values = [1, 2, 3]
        assert list(base_module._progress(values, desc="Processing")) == values
