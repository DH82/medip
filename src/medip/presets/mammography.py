"""Mammography preset profiles.

Each preset returns a PipelineConfig with defaults tuned for
mammography DICOM (MONOCHROME1/2, Presentation Intent, etc.).

``png16_minmax_masked`` is the default profile (see presets.DEFAULT_PROFILE).
"""

from __future__ import annotations

from ..config import (
    CropPolicy,
    DicomConfig,
    ExportConfig,
    ExportDtype,
    ExportFormat,
    GeometryConfig,
    Interpolator,
    MaskPolicy,
    PaddingPolicy,
    PhotometricPolicy,
    PipelineConfig,
    PresentationIntentPolicy,
    ROIConfig,
    SpacingPolicy,
    VOIPolicy,
)

MODALITY = "mammography"


def learning_raw() -> PipelineConfig:
    """Training default: preserve raw pixel values after modality transform.

    - Modality LUT applied (Rescale Slope/Intercept or Modality LUT Sequence)
    - NO VOI/windowing (keeps full dynamic range for model input)
    - NO geometry resample (keeps original resolution)
    - NO mask/crop (delivers full FOV)
    - Output: float32 npy with metadata sidecar
    """
    return PipelineConfig(
        profile="mammography/learning_raw",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.NONE,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.MASK,
            presentation_intent_policy=PresentationIntentPolicy.AUTO,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.KEEP,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.NONE,
            crop_policy=CropPolicy.NONE,
        ),
        export=ExportConfig(
            format=ExportFormat.NPY,
            dtype=ExportDtype.FLOAT32,
            save_mask=False,
            save_metadata_json=True,
        ),
    )


def learning_raw_uint16() -> PipelineConfig:
    """Training variant of ``learning_raw`` that exports uint16 npy.

    Identical pipeline to ``learning_raw`` except the image is saved as a
    uint16 NumPy array instead of float32. Raw post-modality-LUT pixel
    values for mammography typically fit in [0, 65535], so the cast is
    lossless and halves the on-disk size relative to float32.

    - Modality LUT applied (Rescale Slope/Intercept or Modality LUT Sequence)
    - NO VOI/windowing (keeps full dynamic range for model input)
    - NO geometry resample (keeps original resolution)
    - NO mask/crop (delivers full FOV)
    - Output: uint16 npy with metadata sidecar
    """
    return PipelineConfig(
        profile="mammography/learning_raw_uint16",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.NONE,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.MASK,
            presentation_intent_policy=PresentationIntentPolicy.AUTO,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.KEEP,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.NONE,
            crop_policy=CropPolicy.NONE,
        ),
        export=ExportConfig(
            format=ExportFormat.NPY,
            dtype=ExportDtype.UINT16,
            save_mask=False,
            save_metadata_json=True,
        ),
    )


def learning_presentation() -> PipelineConfig:
    """Training on presentation images: applies VOI windowing.

    - Modality LUT + VOI LUT/windowing applied
    - Suitable when training data is FOR PRESENTATION
    - Output: uint16 PNG to balance precision and file size
    """
    return PipelineConfig(
        profile="mammography/learning_presentation",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.AUTO,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.MASK,
            presentation_intent_policy=PresentationIntentPolicy.FOR_PRESENTATION,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.KEEP,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.NONE,
            crop_policy=CropPolicy.NONE,
        ),
        export=ExportConfig(
            format=ExportFormat.PNG16,
            dtype=ExportDtype.UINT16,
            save_mask=False,
            save_metadata_json=True,
        ),
    )


def png16_minmax() -> PipelineConfig:
    """16-bit PNG with a min-max window, reproducing ``dcmj2pnm +on2 --min-max-window``.

    The DCMTK conversion done in Python:

    - Modality LUT applied, MONOCHROME1 auto-inverted
    - VOI window spans the image's own min/max, mapped onto [0, 65535]
    - Original resolution kept
    - NO mask/crop
    - Output: uint16 PNG readable with ``cv2.imread(p, cv2.IMREAD_UNCHANGED)``

    The VOI is applied regardless of Presentation Intent Type. The min-max
    window is per-image, so pixel values are NOT comparable across images —
    use ``learning_raw`` when you need cross-image calibration.

    Pixel Padding is kept out of the min/max estimate (``padding_policy=MASK``);
    DCMTK includes it. Set ``padding_policy: none`` for a bit-identical result.
    """
    return PipelineConfig(
        profile="mammography/png16_minmax",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.MIN_MAX_WINDOW,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.MASK,
            presentation_intent_policy=PresentationIntentPolicy.IGNORE,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.KEEP,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.NONE,
            crop_policy=CropPolicy.NONE,
        ),
        export=ExportConfig(
            format=ExportFormat.PNG16,
            dtype=ExportDtype.UINT16,
            save_mask=False,
            save_metadata_json=True,
            voi_output_max=65535.0,
        ),
    )


def png16_minmax_masked() -> PipelineConfig:
    """``png16_minmax`` plus a background mask that removes burned-in annotation.

    This is the default profile.

    Vendor "FOR PRESENTATION" mammograms often carry burned-in view/laterality
    text (LCC, RCC, LMLO, RMLO) in the background, sometimes while the header
    still claims ``BurnedInAnnotation = NO``. The text is typically brighter
    than any tissue, so it dominates anything that looks at maxima or at
    left/right differences, and it leaks view/laterality labels into the
    pixels. This preset zeroes everything outside the breast.

    Differs from ``png16_minmax`` only in the mask:

    - ``mask_policy=OTSU``: Otsu + morphology, largest contour only, expanded
      to its bounding rectangle
    - ``crop_policy=NONE``: the field of view is preserved, so the breast keeps
      the same scale when a model resizes every image to a fixed size.

    Two consequences worth knowing:

    - Keeping only the largest contour also drops tissue disconnected from the
      breast body (commonly the inferior corner of an MLO view).
    - Masking runs after the VOI step, so the annotation still participates in
      the min/max window. On saturated mammograms the effect is under a few
      percent, since breast tissue reaches the same ceiling as the text.
    """
    return PipelineConfig(
        profile="mammography/png16_minmax_masked",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.MIN_MAX_WINDOW,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.MASK,
            presentation_intent_policy=PresentationIntentPolicy.IGNORE,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.KEEP,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.OTSU,
            # 0.2 (the global default). This was briefly raised to 0.4 to stop
            # noisy background bridging into the breast contour, but 0.4 cuts
            # into the breast itself: measured over 160 images, it discarded 10%
            # or more of the plausible tissue area on 48% of them, shaving the
            # skin line and subcutaneous fat off the breast edge. The background
            # it bought back is marginal — 0.1% of the mask at 0.4 versus 0.4% at
            # 0.2 — so the trade was not worth it.
            otsu_sensitivity=0.2,
            # Take the mask out to its bounding rectangle. The segmentation only
            # has to find the breast, not trace it: a rectangle can add pixels but
            # never removes any, so a ragged or slightly-too-tight contour stops
            # costing tissue. What it takes in is the background beside the
            # breast, which is already near zero.
            mask_to_bbox=True,
            crop_policy=CropPolicy.NONE,
        ),
        export=ExportConfig(
            format=ExportFormat.PNG16,
            dtype=ExportDtype.UINT16,
            save_mask=False,
            save_metadata_json=True,
            voi_output_max=65535.0,
        ),
    )


def clinical_display() -> PipelineConfig:
    """Viewer/visualization: full VOI + resample for display.

    NOT recommended as training default.
    """
    return PipelineConfig(
        profile="mammography/clinical_display",
        dicom=DicomConfig(
            apply_modality_lut=True,
            voi_policy=VOIPolicy.AUTO,
            photometric_policy=PhotometricPolicy.AUTO,
            padding_policy=PaddingPolicy.CLIP,
            presentation_intent_policy=PresentationIntentPolicy.IGNORE,
        ),
        geometry=GeometryConfig(
            spacing_policy=SpacingPolicy.TARGET,
            target_spacing=(0.07, 0.07),
            image_interpolator=Interpolator.BSPLINE,
        ),
        roi=ROIConfig(
            mask_policy=MaskPolicy.OTSU,
            crop_policy=CropPolicy.BBOX,
        ),
        export=ExportConfig(
            format=ExportFormat.PNG8,
            dtype=ExportDtype.UINT8,
            save_mask=False,
            save_metadata_json=False,
        ),
    )


# The first entry is the default (presets.DEFAULT_PROFILE).
PRESETS = {
    "png16_minmax_masked": png16_minmax_masked,
    "png16_minmax": png16_minmax,
    "learning_raw": learning_raw,
    "learning_raw_uint16": learning_raw_uint16,
    "learning_presentation": learning_presentation,
    "clinical_display": clinical_display,
}
