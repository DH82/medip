"""medip custom exceptions."""


class MedpreprocError(Exception):
    """Base exception for medip."""


class DicomReadError(MedpreprocError):
    """Failed to read or decode a DICOM file."""


class UnsupportedTransferSyntaxError(DicomReadError):
    """Compressed transfer syntax requires optional decoder plugin."""

    def __init__(self, ts_uid: str, ts_name: str = ""):
        self.ts_uid = ts_uid
        self.ts_name = ts_name
        msg = (
            f"Unsupported transfer syntax: {ts_name} ({ts_uid}). "
            "Install optional decoders: pip install medip[jpeg]"
        )
        super().__init__(msg)


class InvalidProfileError(MedpreprocError):
    """Unknown or misconfigured profile name."""


class ExportError(MedpreprocError):
    """Failed to export processed image."""


class MetadataError(MedpreprocError):
    """Missing or inconsistent DICOM metadata."""
