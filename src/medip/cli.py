"""medip CLI.

Commands:
  medip run          --input <path> --output <path> [--profile <name>] [--config <yaml>]
  medip inspect      <file.dcm>
  medip dump-metadata <file.dcm> [--output <json>]
  medip validate     <input_dir>
  medip presets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BANNER = r"""
                     _ _
  _ __ ___   ___  __| (_)_ __
 | '_ ` _ \ / _ \/ _` | | '_ \
 | | | | | |  __/ (_| | | |_) |
 |_| |_| |_|\___|\__,_|_| .__/
                         |_|
"""

_EPILOG = """\
Quick Start:
  medip run --input scan.dcm --output results/ --profile learning_raw
  medip inspect scan.dcm
  medip presets

Modalities:
  mammography (default)   Mammography DICOM preprocessing

Profiles (mammography):
  learning_raw            Training default (float32, no VOI, full resolution)
  learning_presentation   Presentation image training (uint16 PNG, auto VOI)
  clinical_display        Visualization (uint8 PNG, resampled, cropped)

Documentation: https://github.com/DH82/medip
"""

_RUN_EPILOG = """\
Examples:
  # Single file with default profile (learning_raw)
  medip run --input scan.dcm --output results/

  # Directory batch processing
  medip run --input dicom_dir/ --output output_dir/ --profile learning_presentation

  # Qualified profile name (modality/name)
  medip run --input scan.dcm --output results/ --profile mammography/learning_raw

  # Custom YAML config
  medip run --input dicom_dir/ --output output_dir/ --config my_config.yaml

  # Preview without processing
  medip run --input dicom_dir/ --output output_dir/ --dry-run

  # Parallel processing with 16 workers
  medip run --input dicom_dir/ --output output_dir/ --num-workers 16

Config Priority: --config file > --profile flag > default (learning_raw)
"""

_INSPECT_EPILOG = """\
Displays key DICOM tags for preprocessing decisions:
  - Photometric Interpretation (MONOCHROME1/2)
  - Presentation Intent Type (FOR PROCESSING / FOR PRESENTATION)
  - Window Center/Width, VOI LUT Function
  - Pixel Spacing, Pixel Padding Value
  - Bits Stored, Transfer Syntax

Example:
  medip inspect mammogram.dcm
"""

_VALIDATE_EPILOG = """\
Checks each DICOM file for common preprocessing issues:
  - Missing Window Center / Window Width / VOI LUT
  - Missing Pixel Spacing
  - Missing Presentation Intent Type
  - Unreadable files

Example:
  medip validate dicom_dir/
  medip validate dicom_dir/ --pattern "**/*.dcm"
"""

# ── Pipeline registry ────────────────────────────────────────────────

_PIPELINE_MAP = {}


def _get_pipeline_cls(modality: str):
    """Lazy-load pipeline class for a modality."""
    if not _PIPELINE_MAP:
        from .pipelines.mammography import MammographyPipeline
        from .pipelines.base import BasePipeline
        _PIPELINE_MAP["mammography"] = MammographyPipeline
        _PIPELINE_MAP["_default"] = BasePipeline
    return _PIPELINE_MAP.get(modality, _PIPELINE_MAP["_default"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="medip",
        description="DICOM-standard medical image preprocessing toolkit",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: INFO)",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Use 'medip <command> --help' for command-specific help",
    )

    # ── run ───────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run the preprocessing pipeline on DICOM file(s)",
        description="Process DICOM files through the profile-driven transform chain.\n"
                    "Supports single files and directory batch processing.",
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single .dcm file or directory containing DICOM files",
    )
    run_parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory (created automatically if it does not exist)",
    )
    run_parser.add_argument(
        "--profile", "-p", default=None,
        help="Preset profile name (e.g., learning_raw, mammography/clinical_display). "
             "Run 'medip presets' to see all available profiles",
    )
    run_parser.add_argument(
        "--modality", "-m", default="mammography",
        help="Target modality for pipeline selection (default: mammography)",
    )
    run_parser.add_argument(
        "--config", "-c", default=None, metavar="YAML",
        help="YAML config file for fine-grained control. "
             "Overrides --profile. See examples/ for templates",
    )
    run_parser.add_argument(
        "--num-workers", "-j", type=int, default=None, metavar="N",
        help="Number of parallel worker processes (default: 8)",
    )
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and exit without processing",
    )
    run_parser.add_argument(
        "--save-report", action="store_true",
        help="Save processing_report.json to output directory",
    )

    # ── inspect ───────────────────────────────────────────────────
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect key DICOM tags of a file",
        description="Display preprocessing-relevant DICOM tags in a readable format.",
        epilog=_INSPECT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inspect_parser.add_argument(
        "file", help="Path to the DICOM file to inspect",
    )

    # ── dump-metadata ─────────────────────────────────────────────
    dump_parser = subparsers.add_parser(
        "dump-metadata",
        help="Export all preprocessing metadata as JSON",
        description="Extract all DICOM tags used by medip and output as JSON.\n"
                    "Useful for auditing, debugging, or building metadata databases.",
    )
    dump_parser.add_argument(
        "file", help="Path to the DICOM file",
    )
    dump_parser.add_argument(
        "--output", "-o", default=None, metavar="JSON",
        help="Output JSON file path (default: print to stdout)",
    )

    # ── validate ──────────────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate",
        help="Check DICOM files for common preprocessing issues",
        description="Scan a directory of DICOM files and report missing or problematic tags.",
        epilog=_VALIDATE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_parser.add_argument(
        "directory", help="Directory containing DICOM files to validate",
    )
    validate_parser.add_argument(
        "--pattern", default="**/*.dcm", metavar="GLOB",
        help="Glob pattern for finding DICOM files (default: **/*.dcm)",
    )

    # ── presets ────────────────────────────────────────────────────
    presets_parser = subparsers.add_parser(
        "presets",
        help="Show all available preset profiles with their settings",
        description="Display the configuration of each built-in preset profile.\n"
                    "Use these as starting points, then override via YAML config.",
    )
    presets_parser.add_argument(
        "--modality", default=None,
        help="Show presets for a specific modality only",
    )

    args = parser.parse_args(argv)

    # Setup logging
    from .utils.logging import setup_logging
    setup_logging(args.log_level)

    if args.command is None:
        print(_BANNER)
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "dump-metadata":
        _cmd_dump_metadata(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "presets":
        _cmd_presets(args)


def _cmd_run(args):
    from .presets import get_preset

    # Build config: config file > profile > default
    if args.config:
        from .utils.config_loader import load_config
        config = load_config(args.config)
    elif args.profile:
        config = get_preset(args.profile)
    else:
        config = get_preset("learning_raw")

    # CLI overrides
    if args.num_workers is not None:
        config.runtime.num_workers = args.num_workers
    if args.dry_run:
        config.runtime.dry_run = True
    if args.save_report:
        config.runtime.save_report = True

    if config.runtime.dry_run:
        print(f"[DRY RUN] Profile: {config.profile}")
        print(f"  Modality: {args.modality}")
        print(f"  Input:   {args.input}")
        print(f"  Output:  {args.output}")
        print(f"  Format:  {config.export.format.value}")
        print(f"  DType:   {config.export.dtype.value}")
        print(f"  VOI:     {config.dicom.voi_policy.value}")
        print(f"  Spacing: {config.geometry.spacing_policy.value}")
        print(f"  Mask:    {config.roi.mask_policy.value}")
        print(f"  Crop:    {config.roi.crop_policy.value}")
        print(f"  Workers: {config.runtime.num_workers}")
        return

    pipeline_cls = _get_pipeline_cls(args.modality)
    pipeline = pipeline_cls(config=config)
    input_path = Path(args.input)

    if input_path.is_file():
        out_path = Path(args.output) / input_path.stem
        result = pipeline.process_single(input_path, out_path)
        print(f"[Done] Saved: {result.get('output_path', 'N/A')}")
    else:
        summary = pipeline.process_directory(input_path, args.output)
        print(f"\n[Done] {summary['success_count']}/{summary['total']} succeeded")
        if summary["failures"]:
            print(f"[Failed: {len(summary['failures'])}]")
            for path, err in summary["failures"][:20]:
                print(f"  - {path}: {err}")

        if config.runtime.save_report:
            report_path = Path(args.output) / "processing_report.json"
            report = {
                "profile": config.profile,
                "modality": args.modality,
                "total": summary["total"],
                "success": summary["success_count"],
                "failed": len(summary["failures"]),
                "failures": [{"file": f, "error": e} for f, e in summary["failures"]],
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved: {report_path}")


def _cmd_inspect(args):
    from .io.metadata import extract_metadata

    meta = extract_metadata(args.file)
    d = meta.to_dict()

    print(f"\n  File: {args.file}")
    print(f"  {'=' * 56}")

    print(f"\n  {'Pixel Description'}")
    print(f"  {'-' * 40}")
    _print_field("Bits Stored", f"{d['bits_stored']}-bit (allocated {d['bits_allocated']})")
    _print_field("Pixel Representation", "unsigned" if d.get("pixel_representation", 0) == 0 else "signed")
    _print_field("Size", f"{d['rows']} x {d['columns']}")

    print(f"\n  {'Intensity'}")
    print(f"  {'-' * 40}")
    _print_field("Rescale Slope", d["rescale_slope"])
    _print_field("Rescale Intercept", d["rescale_intercept"])
    _print_field("Window Center", d["window_center"] or "(not set)")
    _print_field("Window Width", d["window_width"] or "(not set)")
    _print_field("VOI LUT Function", d["voi_lut_function"])

    print(f"\n  {'Modality-specific'}")
    print(f"  {'-' * 40}")
    _print_field("Photometric", d["photometric_interpretation"])
    _print_field("Presentation Intent", d["presentation_intent_type"] or "(not set)")
    _print_field("Pixel Padding", d["pixel_padding_value"] if d["pixel_padding_value"] is not None else "(none)")
    _print_field("Laterality", d["image_laterality"] or "(not set)")
    _print_field("View Position", d["view_position"] or "(not set)")

    print(f"\n  {'Geometry'}")
    print(f"  {'-' * 40}")
    _print_field("Pixel Spacing", f"{d['pixel_spacing']} mm" if d["pixel_spacing"] else "(not set)")
    _print_field("Imager Spacing", f"{d.get('imager_pixel_spacing')} mm" if d.get("imager_pixel_spacing") else "(not set)")

    print(f"\n  {'Transfer'}")
    print(f"  {'-' * 40}")
    _print_field("Transfer Syntax", d["transfer_syntax_uid"] or "(not set)")
    print()


def _cmd_dump_metadata(args):
    from .io.metadata import extract_metadata

    meta = extract_metadata(args.file)
    out = json.dumps(meta.to_dict(), indent=2, default=str)

    if args.output:
        Path(args.output).write_text(out)
        print(f"Metadata saved: {args.output}")
    else:
        print(out)


def _cmd_validate(args):
    from .io.metadata import extract_metadata

    directory = Path(args.directory)
    dcm_files = sorted(directory.rglob(args.pattern))

    if not dcm_files:
        dcm_files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() == ".dcm"
        )

    if not dcm_files:
        print(f"No DICOM files found in {directory}")
        return

    ok_count = 0
    issues = []

    for f in dcm_files:
        try:
            meta = extract_metadata(f)
            warnings = []
            if not meta.has_window and not meta.has_voi_lut_sequence:
                warnings.append("no WC/WW or VOI LUT")
            if meta.pixel_spacing is None and meta.imager_pixel_spacing is None:
                warnings.append("no pixel spacing")
            if meta.presentation_intent_type is None:
                warnings.append("no presentation intent type")

            if warnings:
                issues.append((str(f), warnings))
            else:
                ok_count += 1
        except Exception as e:
            issues.append((str(f), [f"READ ERROR: {e}"]))

    print(f"\nValidation Summary: {len(dcm_files)} files scanned")
    print(f"  OK:     {ok_count}")
    print(f"  Issues: {len(issues)}")

    if issues:
        print(f"\nDetails:")
        for path, warns in issues[:30]:
            print(f"  {path}")
            for w in warns:
                print(f"    - {w}")

    if len(issues) > 30:
        print(f"\n  ... and {len(issues) - 30} more (use --log-level DEBUG for full output)")
    print()


def _cmd_presets(args):
    from .presets import list_modalities, list_presets, get_preset

    modalities = list_modalities()
    filter_modality = getattr(args, "modality", None)

    if filter_modality:
        modalities = [m for m in modalities if m == filter_modality]

    print("\nAvailable Preset Profiles")
    print("=" * 60)

    for modality in modalities:
        preset_names = list_presets(modality=modality)
        print(f"\n  [{modality}]")
        print(f"  {'-' * 56}")

        for name in preset_names:
            qualified = f"{modality}/{name}"
            cfg = get_preset(qualified)
            print(f"\n    {qualified}")
            print(f"    {'.' * 52}")
            print(f"      VOI policy:        {cfg.dicom.voi_policy.value}")
            print(f"      Photometric:       {cfg.dicom.photometric_policy.value}")
            print(f"      Modality LUT:      {'yes' if cfg.dicom.apply_modality_lut else 'no'}")
            print(f"      Padding:           {cfg.dicom.padding_policy.value}")
            print(f"      Intent policy:     {cfg.dicom.presentation_intent_policy.value}")
            print(f"      Spacing:           {cfg.geometry.spacing_policy.value}", end="")
            if cfg.geometry.target_spacing:
                print(f" -> {cfg.geometry.target_spacing} mm")
            else:
                print()
            print(f"      Mask:              {cfg.roi.mask_policy.value}")
            print(f"      Crop:              {cfg.roi.crop_policy.value}")
            print(f"      Export format:     {cfg.export.format.value}")
            print(f"      Export dtype:      {cfg.export.dtype.value}")
            print(f"      Metadata sidecar:  {'yes' if cfg.export.save_metadata_json else 'no'}")

    print(f"\nUsage: medip run --input <path> --output <path> --profile <name>")
    print(f"Override: medip run --input <path> --output <path> --config config.yaml\n")


def _print_field(label: str, value) -> None:
    print(f"    {label:<22} {value}")


if __name__ == "__main__":
    main()
