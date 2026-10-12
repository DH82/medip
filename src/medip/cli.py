"""medip CLI.

Commands:
  medip run          --input <path> --output <path> [--profile <name>] [--config <yaml>]
  medip init-config  [path.yaml]
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

from . import __version__

_BANNER = r"""
                     _ _
  _ __ ___   ___  __| (_)_ __
 | '_ ` _ \ / _ \/ _` | | '_ \
 | | | | | |  __/ (_| | | |_) |
 |_| |_| |_|\___|\__,_|_| .__/
                         |_|
"""

_EPILOG = """\
Quick Start (default config = default.yaml: uint16 PNG, min-max window, background mask):
  medip run --input dicom_dir/ --output png_dir/
  medip init-config my.yaml        # copy of default.yaml with every option, to edit
  medip run --input dicom_dir/ --output png_dir/ --config my.yaml
  medip inspect scan.dcm
  medip presets

Profiles (mammography):
  png16_minmax_masked     [default] uint16 PNG, min-max window, Otsu background mask
  png16_minmax            Same without the mask (`dcmj2pnm +on2 --min-max-window`)
  learning_raw            Training (float32 npy, no VOI, full resolution)
  learning_raw_uint16     Training with uint16 npy (lossless cast, half disk size)
  learning_presentation   Presentation image training (uint16 PNG, auto VOI)
  clinical_display        Visualization (uint8 PNG, resampled, cropped)

Documentation: https://github.com/DH82/medip
"""

_RUN_EPILOG = """\
Examples:
  # Default config (default.yaml = png16_minmax_masked): DICOM -> uint16 PNG
  medip run --input dicom_dir/ --output png_dir/

  # Another profile
  medip run --input dicom_dir/ --output npy_dir/ --profile learning_raw

  # Customized config (start from: medip init-config my.yaml)
  medip run --input dicom_dir/ --output png_dir/ --config my.yaml

  # Preview without processing
  medip run --input dicom_dir/ --output png_dir/ --dry-run

  # Parallel processing with 16 workers
  medip run --input dicom_dir/ --output png_dir/ --num-workers 16

  # Large cohort: one flat folder, resumable (re-run the same command after an interruption)
  medip run --input dicom_dir/ --output png_dir/ --layout flat --skip-existing --scan-cache scan.txt

Config Priority: --config file > --profile flag > default.yaml
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
  medip validate dicom_dir/ --pattern "**/*.IMA"
"""

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="medip",
        description="DICOM mammography preprocessing toolkit (default: uint16 PNG, min-max window)",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"%(prog)s {__version__}",
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
        help="Preset profile name (default: settings in default.yaml). "
             "Run 'medip presets' to see all available profiles",
    )
    run_parser.add_argument(
        "--config", "-c", default=None, metavar="YAML",
        help="YAML config file for fine-grained control. "
             "Overrides --profile. 'medip init-config' writes a template",
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
    run_parser.add_argument(
        "--layout", choices=["tree", "flat"], default=None,
        help="Output layout: 'tree' mirrors the input folders, 'flat' writes one folder "
             "named after the source path (a/b/c.dcm -> a__b__c.png). Overrides the config",
    )
    run_parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip inputs whose output already exists (resume an interrupted run)",
    )
    run_parser.add_argument(
        "--force-read", action="store_true",
        help="Read DICOMs with pydicom force=True (files missing the 'DICM' preamble)",
    )
    run_parser.add_argument(
        "--scan-cache", default=None, metavar="FILE",
        help="Cache the list of input files here and reuse it on later runs",
    )

    # ── init-config ───────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init-config",
        help="Write a copy of the default config (every option listed) to edit",
        description="Copy medip's default.yaml, where every option is listed and the\n"
                    "non-default values are commented out, so it can be customized.",
    )
    init_parser.add_argument(
        "path", nargs="?", default="medip_config.yaml",
        help="Where to write the config (default: medip_config.yaml)",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file",
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
        "--pattern", default=None, metavar="GLOB",
        help="Glob pattern for finding DICOM files (default: every .dcm / .dicom, as 'medip run')",
    )

    # ── presets ────────────────────────────────────────────────────
    subparsers.add_parser(
        "presets",
        help="Show all available preset profiles with their settings",
        description="Display the configuration of each built-in preset profile.\n"
                    "Use these as starting points, then override via YAML config.",
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
    elif args.command == "init-config":
        _cmd_init_config(args)
    elif args.command == "presets":
        _cmd_presets(args)


def _cmd_run(args):
    from .api import resolve_config, run
    from .config import OutputLayout

    from .exceptions import InvalidProfileError

    # Build config: --config file > --profile > default.yaml
    try:
        config = resolve_config(args.config or args.profile)
    except (ValueError, KeyError, FileNotFoundError, InvalidProfileError) as e:
        print(f"ERROR: invalid config: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides
    if args.num_workers is not None:
        config.runtime.num_workers = args.num_workers
    if args.dry_run:
        config.runtime.dry_run = True
    if args.save_report:
        config.runtime.save_report = True
    if args.layout:
        config.runtime.output_layout = OutputLayout(args.layout)
    if args.skip_existing:
        config.runtime.skip_existing = True
    if args.force_read:
        config.runtime.force_read = True
    if args.scan_cache:
        config.runtime.scan_cache = args.scan_cache

    summary = run(args.input, args.output, config=config)
    if summary.get("dry_run"):
        return

    print(f"\n[Done] {summary['success_count']}/{summary['total']} succeeded"
          + (f", {summary['skipped']} skipped (already converted)" if summary.get("skipped") else ""))
    if summary.get("output_path"):
        print(f"Saved: {summary['output_path']}")
    for name, count in sorted(summary.get("vendor_overrides", {}).items()):
        print(f"  vendor override '{name}': {count}")
    if summary["failures"]:
        print(f"[Failed: {len(summary['failures'])}]")
        for path, err in summary["failures"][:20]:
            print(f"  - {path}: {err}")
        if summary.get("failure_log"):
            print(f"Full list: {summary['failure_log']}")
    if summary.get("report_path"):
        print(f"Report saved: {summary['report_path']}")


def _cmd_init_config(args):
    from .api import write_default_config

    try:
        path = write_default_config(args.path, overwrite=args.force)
    except FileExistsError:
        print(f"ERROR: {args.path} already exists (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {path}")
    print(f"Edit it, then: medip run --input <dicom_dir> --output <out_dir> --config {path}")


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
    _print_field("Manufacturer", d.get("manufacturer") or "(not set)")
    _print_field("Model", d.get("manufacturer_model_name") or "(not set)")

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
    from .pipelines.base import find_dicom_files

    directory = Path(args.directory)
    dcm_files = find_dicom_files(directory, args.pattern)

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
    from .presets import DEFAULT_PROFILE, get_preset, list_modalities, list_presets

    print("\nAvailable Preset Profiles")
    print("=" * 60)

    for modality in list_modalities():
        preset_names = list_presets(modality=modality)
        print(f"\n  [{modality}]")
        print(f"  {'-' * 56}")

        for name in preset_names:
            qualified = f"{modality}/{name}"
            default_mark = "  (default)" if qualified == DEFAULT_PROFILE else ""
            print(f"\n    {qualified}{default_mark}")
            print(f"    {'.' * 52}")
            cfg = get_preset(qualified)
            print(f"      VOI policy:        {cfg.dicom.voi_policy.value}")
            print(f"      Photometric:       {cfg.dicom.photometric_policy.value}")
            print(f"      Modality LUT:      {'yes' if cfg.dicom.apply_modality_lut else 'no'}")
            print(f"      Padding:           {cfg.dicom.padding_policy.value}")
            print(f"      Intent policy:     {cfg.dicom.presentation_intent_policy.value}")
            print(f"      Spacing:           {cfg.geometry.spacing_policy.value}", end="")
            if cfg.geometry.spacing_policy.value in ("auto", "target") and cfg.geometry.target_spacing:
                print(f" -> {cfg.geometry.target_spacing} mm")
            else:
                print()
            print(f"      Mask:              {cfg.roi.mask_policy.value}")
            print(f"      Crop:              {cfg.roi.crop_policy.value}")
            print(f"      Export format:     {cfg.export.format.value}")
            print(f"      Export dtype:      {cfg.export.dtype.value}")
            print(f"      Metadata sidecar:  {'yes' if cfg.export.save_metadata_json else 'no'}")

    print(f"\nUsage: medip run --input <path> --output <path> [--profile <name>]")
    print(f"Override: medip run --input <path> --output <path> --config config.yaml\n")


def _print_field(label: str, value) -> None:
    print(f"    {label:<22} {value}")


if __name__ == "__main__":
    main()
