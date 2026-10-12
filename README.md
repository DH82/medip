# medip

Mammography DICOM preprocessing. By default it writes **16-bit PNGs with a min-max window and the
background masked out**: the `dcmj2pnm +on2 --min-max-window` (DCMTK) conversion done in Python, plus
removal of burned-in view labels.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/medip.svg)](https://pypi.org/project/medip/)
[![Downloads](https://static.pepy.tech/badge/medip)](https://pepy.tech/project/medip)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/DH82/medip/blob/main/LICENSE)

---

## 1. Install

```bash
pip install medip
```

Optional extras (quote them so the shell leaves the brackets alone):

```bash
pip install "medip[cli,jpeg]"    # recommended: progress bars + compressed DICOM support
```

| Extra | Contents | When you need it |
| --- | --- | --- |
| (none) | pydicom, numpy, opencv, pyyaml | always |
| `cli` | tqdm (progress bars) | recommended |
| `jpeg` | JPEG / JPEG 2000 decoders | compressed DICOMs (most PACS exports) |
| `itk` | SimpleITK | only for resampling (`spacing_policy: auto / target`) |
| `dev` | all of the above + pytest | development |

---

## 2. Quick start

### Python

```python
import medip

medip.run("/data/dicom", "/data/png16")              # default settings (default.yaml)
medip.run("/data/dicom", "/data/png16", "my.yaml")   # your own YAML
```

No `if __name__ == "__main__":` guard is needed, even with several workers; this also works from a
notebook or from code piped into `python -`.

- A directory input is searched recursively for `.dcm` / `.dicom` files and its folder structure is
  mirrored: `dicom/p1/s1/a.dcm` → `png16/p1/s1/a.png` + `a.json` (metadata).
- A single DICOM file works too.
- Returns `{"total", "success_count", "skipped", "failures": [(path, error), ...], "vendor_overrides", "config"}`.
  For a directory input, failures are also written to `<output>/failures.log`, and inputs whose
  output names would collide (e.g. `a.dcm` and `a.dicom`) are reported there instead of overwriting each other.

### Command line

```bash
medip run --input /data/dicom --output /data/png16              # default.yaml (same as above)
medip run --input /data/dicom --output out/ --config my.yaml     # your settings
medip run --input /data/dicom --output out/ --profile learning_raw   # another preset
medip run --input /data/dicom --output out/ --dry-run            # show the settings only

medip init-config my.yaml   # copy default.yaml to edit
medip inspect scan.dcm      # DICOM tags that drive preprocessing
medip validate /data/dicom  # report missing tags
medip presets               # list presets and their settings
```

### Reading the output

```python
import cv2
img = cv2.imread("out/scan.png", cv2.IMREAD_UNCHANGED)   # uint16, 0..65535
```

Without `IMREAD_UNCHANGED`, OpenCV reduces the image to 8 bits.

---

## 3. Changing settings — `default.yaml`

[`default.yaml`](https://github.com/DH82/medip/blob/main/src/medip/default.yaml) ships inside the package and lists **every option**.
`medip init-config my.yaml` writes a copy you can edit; `medip.DEFAULT_CONFIG_PATH` is where the
installed file lives.

- An uncommented line is the value in use; a `# key: value` line is an alternative.
- To change an option, comment out the active line and uncomment the one you want:

```yaml
export:
  # format: png16                           # 16-bit PNG (.png)        <- commented out
  # format: png8                            # 8-bit PNG
  format: npy                               # NumPy array (.npy)       <- uncommented
```

Editing a copy is recommended:

```python
import medip
medip.write_default_config("my.yaml")                  # = medip init-config my.yaml
# ... edit my.yaml ...
medip.run("/data/dicom", "/data/out", "my.yaml")
```

The third argument (`config`) accepts a YAML path, a preset name (`"learning_raw"`) or a `PipelineConfig`.
On the command line the order is `--config` > `--profile` > `default.yaml`.

**Mistakes are reported, not ignored.** Two active lines for the same key (`duplicate key`) or a
misspelled option (`unknown option`) stop the run before anything is written.

### What the defaults do (= preset `png16_minmax_masked`)

| Step | Setting | Why |
| --- | --- | --- |
| Modality LUT | applied (Rescale Slope/Intercept) | DICOM standard |
| Photometric | MONOCHROME1 inverted (after the VOI step, as DICOM specifies) | brighter = denser for every image |
| Pixel Padding | excluded from the min/max | collimator regions must not stretch the window |
| VOI | **min-max window** → 0..65535 | same as `dcmj2pnm --min-max-window` |
| Presentation Intent | ignored (window always applied) | consistent output for every image |
| Resolution | original | resizing is left to the consumer |
| Mask | **Otsu background mask**, expanded to its bounding rectangle | removes burned-in text (below) |
| Crop | none | the breast keeps its scale under a fixed-size resize |
| Output | uint16 PNG + metadata JSON | same as `dcmj2pnm +on2` |

**Why the mask.** Vendor "FOR PRESENTATION" mammograms (e.g. Hologic) often have the view label
(`LCC`, `RMLO`, ...) burned into the pixels while the header says `BurnedInAnnotation = NO`. The text is
brighter than any tissue, so it dominates anything that looks at image maxima or at left/right
differences, and it leaks the view/laterality into the pixels. The mask sets everything outside the
breast to 0 and leaves breast pixels unchanged.

Mask details that were measured, not guessed:

- `otsu_sensitivity: 0.2`. At 0.4, 48% of 160 images lost 10% or more of their breast area (skin line
  and subcutaneous fat), for almost no gain in background removal (0.4% vs 0.1% of the mask).
- `mask_to_bbox: true`. A rectangle can only add pixels, so a ragged contour never cuts tissue; what it
  takes in is background next to the breast, which is already near zero.
- Only the largest component is kept, so tissue disconnected from the breast body (sometimes the
  inferior corner of an MLO view) is dropped with the background. Set `mask_policy: none` if you need
  every pixel.

To drop the mask use `mask_policy: none` (or `--profile png16_minmax`); for output bit-identical to
DCMTK, also set `padding_policy: none`.

---

## 4. Large cohorts

For hundreds of thousands of images on a network mount, save this as `large_cohort.yaml`
(also in the repository as [`configs/large_cohort.yaml`](https://github.com/DH82/medip/blob/main/configs/large_cohort.yaml)):

```yaml
profile: png16_minmax_masked

export:
  save_metadata_json: false      # images only

runtime:
  num_workers: 8
  output_layout: flat
  skip_existing: true
  scan_cache: scan_cache.txt     # rescanned automatically for a different input
  # force_read: true             # old exports without the "DICM" preamble

vendor_overrides:
  - manufacturer: ["GE MEDICAL", "GE HEALTHCARE"]
    dicom:
      voi_policy: window
      voi_index: 0
    roi:
      mask_source: pre_voi
```

```bash
medip run --input /data/dicom --output /data/png16 --config large_cohort.yaml
```

```python
medip.run("/data/dicom", "/data/png16", "large_cohort.yaml")
```

| Option | Effect |
| --- | --- |
| `runtime.output_layout: flat` | one output folder; `patient/date/uid.dcm` → `patient__date__uid.png` |
| `runtime.skip_existing: true` | existing outputs are skipped, so **re-running the same command resumes** |
| `runtime.scan_cache: <file>` | the input file list is cached; later runs on the same input skip the directory walk |
| `runtime.force_read: true` | reads old exports that lack the `DICM` preamble (also tries non-DICOM files) |

The same switches exist on the command line: `--layout flat --skip-existing --scan-cache scan.txt --force-read`.

Existing outputs are found by listing each output folder once instead of checking every file, which
matters on network storage where each check is a round trip. Images are written to a temporary file
and renamed into place last, after their JSON / mask, so an interrupted run never leaves a partial
image that a resumed run would skip. (Outputs written by medip before 0.2.0 have no such guarantee.)

### Per-manufacturer settings

`vendor_overrides` changes settings for images whose `Manufacturer` tag matches (case-insensitive
substring; first match wins). The file above uses it for GE: the vendor's first reading window
(`voi_policy: window`) instead of min-max, with the mask taken from the image before windowing
(`mask_source: pre_voi`).

The trade-off, measured on 40 GE and 40 Hologic images: under min-max no GE tissue is lost, but
the median GE tissue value is about 3x the Hologic one (Hologic/GE ratio 0.33). The vendor window
brings the two much closer (ratio 1.35) but floors about 18% of the GE tissue to 0 and halves the
number of grey levels. Pick the one that suits your use. The same example is in `default.yaml`, commented out.
The run summary and each metadata JSON record which override was applied.

---

## 5. Other presets

| Preset | Use | VOI | Resolution | Mask / crop | Output |
| --- | --- | --- | --- | --- | --- |
| **`png16_minmax_masked`** | **default** (= `default.yaml`) | min-max | original | Otsu / none | uint16 PNG |
| `png16_minmax` | DCMTK-equivalent, no mask | min-max | original | none | uint16 PNG |
| `learning_raw` | training a new model (raw values) | none | original | none | float32 npy |
| `learning_raw_uint16` | same, half the disk space | none | original | none | uint16 npy |
| `learning_presentation` | FOR PRESENTATION training data | vendor window | original | none | uint16 PNG |
| `clinical_display` | visual review | vendor window | 0.07 mm | Otsu / bbox | uint8 PNG |

The min-max window differs per image, so pixel values are not comparable across images. When that
matters (e.g. training from scratch), `learning_raw` keeps the calibrated values.

A YAML file can start from any preset and change only a few keys (more examples in
[configs/](https://github.com/DH82/medip/blob/main/configs)).

```yaml
profile: learning_raw
runtime:
  num_workers: 16
```

---

## 6. Processing steps

```text
DICOM
 [1] decode pixels (pydicom; compressed DICOM needs [jpeg])
 [2] Pixel Padding mask
 [3] Modality LUT (Rescale Slope/Intercept)
 [4] VOI: none | auto | window | lut | min_max_window
 [5] Photometric (MONOCHROME1 -> MONOCHROME2, within the fixed output range of [4])
 [6] resample (optional, SimpleITK)
 [7] mask (optional: Otsu + morphology + largest contour)
 [8] crop (optional)
 [9] intensity curve / LUT (optional)
[10] save: png16 | png8 | npy | npz | tiff16 | jpeg | dicom  (+ metadata JSON)
```

---

## 7. Source layout

```text
medip/
├── configs/                # example YAML files (other presets, large cohorts)
├── src/medip/
│   ├── default.yaml        # default configuration, every option listed
│   ├── api.py              # medip.run()
│   ├── cli.py              # the medip command
│   ├── config.py           # settings dataclasses / enums
│   ├── presets/            # preset definitions
│   ├── pipelines/          # step order (base.py) + mammography rules (mammography.py)
│   ├── transforms/         # one module per step
│   ├── io/                 # DICOM reading, metadata
│   └── utils/              # YAML loading, logging
└── tests/
```

---

## 8. FAQ

**Compressed DICOMs fail to decode** (`UnsupportedTransferSyntaxError`)
→ `pip install "medip[jpeg]"`

**Old DICOMs without the preamble fail** (some early LORAD exports)
→ `force_read: true` or `--force-read`. It also makes pydicom try non-DICOM files, so keep it off otherwise.

**Are FOR PROCESSING images converted?**
→ Yes; the default ignores Presentation Intent and windows every image. With
`presentation_intent_policy: auto` and `voi_policy: auto`, FOR PROCESSING images skip the window.

**How do I check the mask did not cut tissue?**
→ Set `save_mask: true`; `<name>_mask.png` is written next to each image.

**Pixel Spacing order?**
→ `[row spacing, column spacing]` in mm, as in the DICOM standard, everywhere in medip.

---

## 9. Development

```bash
git clone https://github.com/DH82/medip.git
cd medip
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](https://github.com/DH82/medip/blob/main/LICENSE).
