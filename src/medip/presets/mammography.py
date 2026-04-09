"""Mammography preset profiles.

Each preset returns a PipelineConfig with defaults tuned for
mammography DICOM (MONOCHROME1/2, Presentation Intent, etc.).
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


PRESETS = {
    "learning_raw": learning_raw,
    "learning_raw_uint16": learning_raw_uint16,
    "learning_presentation": learning_presentation,
    "clinical_display": clinical_display,
}
