# medip

DICOM-standard medical image preprocessing toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why medip?

Medical image preprocessing is not just image conversion.
The DICOM standard clearly distinguishes **Modality LUT**, **VOI LUT**, **Photometric Interpretation**, **Pixel Padding**, and more. Ignoring these distinctions degrades training data quality.

`medip` enforces DICOM compliance while letting you **control the entire preprocessing policy with a single profile**.

### Supported Modalities

| Modality | Status | Pipeline |
|----------|--------|----------|
| **Mammography** | Supported | `MammographyPipeline` |
| Chest X-ray | Planned | — |
| CT | Planned | — |
| Pathology (WSI) | Planned | — |

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Modality-extensible architecture** | Modality-specific pipelines extend a common `BasePipeline` |
| **Training-first defaults** | Default output is `float32 npy`; 8-bit PNG is a visualization option |
| **Profile + Override** | Pick a preset, then override only what you need |
| **Standard core, experimental extras** | Heuristics like Otsu mask and morphology are isolated under `roi` options |

---

## Installation

```bash
# Minimal (pydicom + numpy + opencv)
pip install medip

# CLI + YAML config support
pip install medip[cli]

# Compressed DICOM (JPEG, JPEG2000) decoding
pip install medip[jpeg]

# SimpleITK-based resampling
pip install medip[itk]

# NIfTI output
pip install medip[nifti]

# Everything
pip install medip[all]

# Development (includes tests)
pip install medip[dev]
```

---

## Quick Start

### Python API

```python
from medip import MammographyPipeline

# 1. Training default (float32, no VOI, original resolution)
pipeline = MammographyPipeline(profile="learning_raw")
result = pipeline.process_single("input.dcm", "output/image")
# -> output/image.npy + output/image.json (metadata sidecar)

# 2. Batch directory processing
summary = pipeline.process_directory("dicom_dir/", "output_dir/")
print(f"{summary['success_count']}/{summary['total']} succeeded")

# 3. Inspect DICOM metadata
info = pipeline.inspect("input.dcm")
print(info["photometric_interpretation"])  # MONOCHROME1 or MONOCHROME2
print(info["presentation_intent_type"])    # FOR PROCESSING or FOR PRESENTATION
```

### CLI

```bash
# Single file preprocessing
medip run --input scan.dcm --output results/ --profile learning_raw

# Specify modality
medip run --input scan.dcm --output results/ --modality mammography

# Batch directory processing
medip run --input dicom_dir/ --output output_dir/ --profile learning_presentation

# Qualified profile name
medip run --input scan.dcm --output results/ --profile mammography/learning_raw

# Run with YAML config file
medip run --input dicom_dir/ --output output_dir/ --config my_config.yaml

# Dry run (preview without processing)
medip run --input dicom_dir/ --output output_dir/ --dry-run

# Inspect a DICOM file
medip inspect scan.dcm

# Export metadata as JSON
medip dump-metadata scan.dcm --output meta.json

# Validate DICOM files in a directory
medip validate dicom_dir/

# List available presets
medip presets

# List presets for a specific modality
medip presets --modality mammography
```

---

## Preset Profiles

### Mammography

| Profile | Purpose | VOI | Spacing | Mask/Crop | Output |
|---------|---------|-----|---------|-----------|--------|
| `learning_raw` | **Training default** | None | Original | None | float32 npy |
| `learning_presentation` | Presentation image training | Auto | Original | None | uint16 png |
| `clinical_display` | Visualization / review | Auto | 0.07mm | Otsu+Crop | uint8 png |

### Profile Selection Guide

```
Building training data?
├── FOR PROCESSING images  → learning_raw (recommended)
└── FOR PRESENTATION images → learning_presentation

Visualization / review?
└── clinical_display
```

---

## Preprocessing Pipeline

```
DICOM File
  │
  ▼
[1] Pixel Decode (pydicom)
  │   - Decompress (JPEG, JPEG2000, etc.)
  │   - Extract raw pixel array
  │
  ▼
[2] Pixel Padding
  │   - Mask Pixel Padding Value/Range
  │   - Applied before intensity to prevent statistical distortion
  │
  ▼
[3] Modality LUT / Rescale
  │   - Apply Rescale Slope/Intercept
  │   - Or apply Modality LUT Sequence
  │
  ▼
[4] Photometric Interpretation
  │   - Invert MONOCHROME1 → MONOCHROME2 (auto)
  │
  ▼
[5] VOI LUT / Windowing (optional)
  │   - none: Preserve raw values for training
  │   - auto: Prefer VOI LUT Sequence, fallback to Window
  │
  ▼
[6] Geometry / Resample (optional)
  │   - auto: Resample to target spacing when PixelSpacing exists (default 0.07mm)
  │   - keep: Preserve original resolution
  │   - target: Resample to specified spacing (SimpleITK)
  │
  ▼
[7] Mask Extraction (optional)
  │   - Otsu + morphology + largest contour
  │
  ▼
[8] Crop (optional)
  │   - Bounding box crop with optional margin
  │
  ▼
[9] Export
      - npy, npz, png8, png16, tiff16, jpeg, dicom, nifti
      - metadata.json sidecar (optional)
```

Steps [1]-[9] are implemented in `BasePipeline`. Modality-specific pipelines override individual steps as needed.

Example: `MammographyPipeline` automatically resolves VOI policy based on **Presentation Intent Type** (FOR PROCESSING → skip VOI).

---

## Output Formats

| Format | Extension | Bits | Lossy | Purpose |
|--------|-----------|------|-------|---------|
| `npy` | `.npy` | float32 | No | **Training default**, direct Python load |
| `npz` | `.npz` | float32 | No | Compressed storage, can include spacing |
| `png16` | `.png` | uint16 | No | Training, image viewer compatible |
| `png8` | `.png` | uint8 | Yes | Visualization |
| `tiff16` | `.tiff` | uint16 | No | ImageJ/Fiji compatible |
| `jpeg` | `.jpg` | uint8 | Yes | Visualization only |
| `dicom` | `.dcm` | uint16 | No | Secondary Capture, PACS compatible |
| `nifti` | `.nii.gz` | float32 | No | Neuroimaging tools (FreeSurfer, FSL) |

---

## YAML Configuration

Start from a profile, then override only the fields you need.

```yaml
# my_config.yaml
profile: learning_raw               # or mammography/learning_raw

dicom:
  voi_policy: none
  photometric_policy: auto
  padding_policy: mask
  presentation_intent_policy: auto

geometry:
  spacing_policy: auto               # auto: resample to 0.07mm when PixelSpacing exists
  target_spacing: [0.07, 0.07]       # target spacing for auto/target mode (mm)

roi:
  mask_policy: none
  crop_policy: none

export:
  format: png16
  dtype: uint16
  save_metadata_json: true

runtime:
  num_workers: 8
```

See the `examples/` directory for per-preset example configs.

---

## Architecture

### Modality-Extensible Structure

```
BasePipeline (common DICOM processing chain)
  ├── MammographyPipeline  (Presentation Intent-based VOI policy)
  ├── ChestXrayPipeline    (planned)
  ├── CTPipeline           (planned)
  └── PathologyPipeline    (planned)
```

To add a new modality:
1. Create a pipeline class under `pipelines/` inheriting from `BasePipeline`
2. Create a presets module under `presets/`
3. Register it with `register_modality()` in `presets/__init__.py`

### Project Structure

```
src/medip/
    __init__.py              # Package entry point
    cli.py                   # CLI (medip command)
    config.py                # Config dataclasses + enums
    exceptions.py            # Custom exceptions

    presets/
        __init__.py          # Preset registry (register_modality, get_preset)
        mammography.py       # Mammography presets

    io/
        dicom_reader.py      # pydicom-based pixel decoding
        metadata.py          # DicomMetadata dataclass

    transforms/
        intensity.py         # Modality LUT + VOI LUT
        photometric.py       # MONOCHROME1/2 handling
        padding.py           # Pixel Padding Value masking
        geometry.py          # Resampling (SimpleITK)
        mask.py              # Otsu/morphology-based masks
        crop.py              # ROI cropping
        export.py            # 8 output formats

    pipelines/
        base.py              # BasePipeline (common processing chain)
        mammography.py       # MammographyPipeline

    utils/
        config_loader.py     # YAML config loading
        logging.py           # Logging setup

tests/
    test_metadata.py
    test_intensity.py
    test_profiles.py
    test_exporters.py
    test_cli.py
```

---

## DICOM Tag Reference

Tags read and interpreted by `medip`.

### Pixel Description
| Tag | Description |
|-----|-------------|
| `BitsAllocated` | Bits allocated per pixel (typically 16) |
| `BitsStored` | Actual bits used (10, 12, 14, etc.) |
| `HighBit` | Most significant bit position |
| `PixelRepresentation` | 0=unsigned, 1=signed |

### Intensity
| Tag | Description |
|-----|-------------|
| `RescaleSlope` / `RescaleIntercept` | Modality LUT linear transform |
| `ModalityLUTSequence` | Non-linear Modality LUT (rare) |
| `WindowCenter` / `WindowWidth` | VOI windowing parameters |
| `VOILUTFunction` | `LINEAR`, `LINEAR_EXACT`, `SIGMOID` |
| `VOILUTSequence` | Table-based VOI LUT |

### Presentation / Mammography
| Tag | Description |
|-----|-------------|
| `PhotometricInterpretation` | `MONOCHROME1` (brighter = lower density) / `MONOCHROME2` (brighter = higher density) |
| `PresentationIntentType` | `FOR PROCESSING` (raw) / `FOR PRESENTATION` (vendor post-processed) |
| `PixelPaddingValue` | Non-meaningful pixel value (e.g., collimator regions) |
| `PixelPaddingRangeLimit` | End of padding value range |

### Geometry
| Tag | Description |
|-----|-------------|
| `PixelSpacing` | `[row_spacing, col_spacing]` (mm) |
| `ImagerPixelSpacing` | Detector-level spacing |
| `ImageLaterality` | `L` / `R` |
| `ViewPosition` | `CC`, `MLO`, etc. |

---

## FAQ

### Q: Cannot read compressed DICOM
```
UnsupportedTransferSyntaxError: ... Install optional decoders
```
**A:** JPEG/JPEG2000 compressed DICOM requires additional decoders:
```bash
pip install medip[jpeg]
```

### Q: What happens when Window Center/Width is missing?
**A:** With `voi_policy=auto`, if WC/WW is absent, VOI transform is skipped and the Modality LUT output is passed through as-is. A warning is logged.

### Q: Are MONOCHROME1 images automatically inverted?
**A:** With `photometric_policy=auto` (default), MONOCHROME1 is automatically inverted to MONOCHROME2. Set `photometric_policy=keep` to disable this.

### Q: What is the difference between FOR PROCESSING and FOR PRESENTATION?
**A:**
- **FOR PROCESSING**: Raw image without vendor post-processing. Recommended for CAD/AI training.
- **FOR PRESENTATION**: Image with vendor post-processing (edge enhancement, contrast adjustment, etc.) applied. Used for clinical reading.

In `MammographyPipeline` with `presentation_intent_policy=auto`, VOI is automatically skipped for FOR PROCESSING images.

### Q: Pixel Spacing order is confusing
**A:** Per the DICOM standard, `PixelSpacing` is `[row_spacing, col_spacing]`:
- `row_spacing` = distance between adjacent rows (vertical direction)
- `col_spacing` = distance between adjacent columns (horizontal direction)

`medip` maintains this order consistently throughout.

### Q: How do I add a new modality?
**A:**
```python
# 1. pipelines/my_modality.py
from medip.pipelines.base import BasePipeline

class MyModalityPipeline(BasePipeline):
    def _resolve_voi_policy(self, meta, cfg):
        # Modality-specific VOI policy logic
        ...

# 2. presets/my_modality.py
from medip.config import PipelineConfig, ...

MODALITY = "my_modality"
PRESETS = {
    "default": lambda: PipelineConfig(profile="my_modality/default", ...),
}

# 3. Register in presets/__init__.py
from . import my_modality
register_modality(my_modality.MODALITY, my_modality.PRESETS)
```

---

## License

MIT License. See [LICENSE](LICENSE).
