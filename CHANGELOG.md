# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-12

### Changed (breaking)
- **New default: `mammography/png16_minmax_masked`** (was `learning_raw`): uint16 PNG, min-max window, Otsu background mask, original resolution, no crop. Used by `medip.run()`, `medip run` without `--config`/`--profile`, `MammographyPipeline()` and YAML files without `profile:`. Exposed as `medip.DEFAULT_PROFILE`; pass `--profile learning_raw` for the old behavior
- Scope narrowed to mammography: removed the NIfTI exporter (`ExportFormat.NIFTI`, the `[nifti]` extra), the `--modality` flag of `medip run` / `medip presets`, and the docs for planned CT / chest X-ray / pathology pipelines
- `examples/` renamed to `configs/`
- `pyyaml` is a core dependency (needed for `default.yaml`)
- `process_directory` finds inputs with `os.walk` (`.dcm` and `.dicom`, any case) instead of a glob, and strips only DICOM extensions from output names, so UID file names keep every dotted component

### Added
- `medip.run(input, output, config=None)`: one-call API. With no `config` it follows `src/medip/default.yaml`, which lists every option with the default active and the alternatives commented out; `config` also takes a YAML path, a preset name or a `PipelineConfig`
- `medip.write_default_config(path)` / `medip init-config [path]`: copy `default.yaml` for editing
- YAML configs fail on duplicate keys (two alternatives uncommented at once) and on unknown sections/options, instead of silently taking the last value or ignoring a typo
- Presets `mammography/png16_minmax` (`dcmj2pnm +on2 --min-max-window` in Python) and `mammography/png16_minmax_masked` (the same plus an Otsu mask that removes burned-in view text, which vendor FOR PRESENTATION images carry even when `BurnedInAnnotation = NO`)
- `VOIPolicy.MIN_MAX_WINDOW`: linear VOI window spanning the image's own min/max. Honors the Pixel Padding mask so padding regions stay out of the min/max estimate
- `ExportConfig.voi_output_max`: output ceiling for min-max, legacy and window VOI. For `window`, pydicom's `[0, 2**BitsStored - 1]` output is rescaled onto it by a fixed factor, which keeps a VOI window comparable across images. Unset leaves pydicom's range untouched
- `vendor_overrides`: per-manufacturer changes to the dicom / geometry / roi / export settings, matched on the `Manufacturer` tag. The applied override is recorded in the metadata JSON and the run summary
- Large-cohort runtime options: `output_layout: flat` (one folder, source path joined with `__`), `skip_existing` (resume; each output folder is listed once instead of stat-ing every file, which on an NFS mount turned an 88-minute pre-scan of 387k files into seconds), `scan_cache` (reuse the input file list), `force_read` (pydicom `force=True` for files without the `DICM` preamble). CLI: `--layout`, `--skip-existing`, `--scan-cache`, `--force-read`. Example: `configs/large_cohort.yaml`
- Failures of a directory run are written to `<output>/failures.log`
- Parallel runs (`num_workers > 1`) no longer re-import the caller's script in each worker, so `medip.run` works without an `if __name__ == "__main__":` guard and from code piped into `python -` under the "spawn" / "forkserver" start methods (the default from Python 3.14 on Linux, and on macOS / Windows)
- Mask options `mask_source` (compute the mask before or after the VOI window), `mask_to_bbox` (expand the mask to its bounding rectangle) and `keep_all_components`
- `ExportConfig.intensity_curve` / `intensity_lut`: polynomial or monotonic lookup table applied before export, to bring one detector's brightness onto another's
- `Manufacturer` and `ManufacturerModelName` in `DicomMetadata`, the metadata JSON and `medip inspect`

### Changed
- `png16_minmax_masked` keeps `otsu_sensitivity=0.2`: at 0.4, 48% of 160 images lost 10% or more of their breast area, for a background reduction of 0.4% -> 0.1% of the mask
- `pydicom.pixels` is used for the Modality LUT / VOI LUT / windowing helpers, falling back to `pydicom.pixel_data_handlers.util` on pydicom < 3. The old path warns on every import and is removed in pydicom 4

### Fixed
- Bits above BitsStored are dropped when decoding, as the DICOM standard requires. pydicom 3 does this and pydicom 2.x does not, so the same file decoded differently by installed version; on LORAD exports with stray high bits a single value moved the min-max window of the whole image. Output is now identical on pydicom 2.4 and 3.x
- MONOCHROME1 is inverted after the VOI step (DICOM PS3.4 N.2) instead of before it, so Window Center/Width and VOI LUTs apply to the values they were written for. Inversion uses the fixed output range of that step rather than each image's maximum, so `learning_raw` values of MONOCHROME1 images stay comparable across images. MONOCHROME2 images and the min-max window are unaffected
- A VOI LUT Sequence (`voi_policy: auto / lut`) is rescaled onto `voi_output_max` like a window, instead of staying in its native 0..2^bits-1 range. The window rescale follows pydicom's y-range, including Rescale Slope/Intercept
- Images are written to a temporary file and renamed into place after their sidecars, so an interrupted run cannot leave a partial image that `skip_existing` would treat as done
- `scan_cache` records the input directory and search it was made for, and stores paths relative to it; a cache made for another input is rewritten instead of crashing or feeding in the wrong file list
- `fail_fast` with `num_workers > 1` cancels the queued files on the first failure instead of converting them all first
- Two inputs that map to one output name (`a.dcm` / `a.dicom`, or flat-layout clashes) are reported as failures instead of overwriting each other
- `medip validate` finds the same files as `medip run` (`.dcm` and `.dicom`, any case)
- Tests run on pydicom 2.4 as well as 3.x
- Export no longer truncates dotted filenames. Paths named after DICOM UIDs (`1.2.826.0.1.3680043.8.498.12345`) lost everything after the last dot via `Path.with_suffix`, so images from one series silently overwrote each other. Only recognized image extensions are replaced now; the metadata sidecar path was fixed the same way
- The metadata sidecar records the *effective* VOI policy (after `_resolve_voi_policy`) instead of the configured one, so FOR PROCESSING images no longer report a VOI that was skipped

## [0.1.1] - 2026-04-09

### Added
- New preset `mammography/learning_raw_uint16`: same pipeline as `learning_raw` but exports uint16 npy for ~50% smaller training files
- CLI help lists the new `learning_raw_uint16` profile

### Changed
- `ExportDtype.UINT16` cast now preserves raw pixel values via clip-to-[0, 65535] instead of per-image min-max rescaling. This keeps cross-image calibration intact (critical for training on raw DICOM). Out-of-range values are clipped with a warning.

## [0.1.0] - 2025-04-02

### Added
- Initial release of medip
- Modality-extensible pipeline architecture: `BasePipeline` + `MammographyPipeline`
- Profile-driven pipeline with modality-scoped presets: `mammography/learning_raw`, `mammography/learning_presentation`, `mammography/clinical_display`
- DICOM-standard transform chain: padding -> modality LUT -> photometric -> VOI -> geometry -> mask -> crop -> export
- `DicomMetadata` dataclass with 20+ DICOM tags for pixel interpretation
- pydicom-based authoritative pixel decoding
- 8 export formats: npy, npz, png8, png16, tiff16, jpeg, dicom (Secondary Capture), nifti
- MONOCHROME1 auto-inversion
- Pixel Padding Value/Range masking
- FOR PROCESSING / FOR PRESENTATION auto-detection (mammography)
- Multi-window DICOM support (window index selection)
- `SpacingPolicy.AUTO`: resample to target spacing (default 0.07mm) when PixelSpacing exists in DICOM header
- SimpleITK-based resampling with configurable interpolators
- Otsu + morphology mask extraction (research/experimental)
- Bounding box crop with mm/px margin support
- Metadata JSON sidecar export
- CLI commands: `run`, `inspect`, `dump-metadata`, `validate`, `presets`
- CLI `--modality` flag for pipeline selection
- YAML config file support with profile + override pattern
- Parallel processing with configurable worker count
- Optional dependencies: `[cli]`, `[jpeg]`, `[itk]`, `[nifti]`, `[all]`, `[dev]`
