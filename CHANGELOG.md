# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
