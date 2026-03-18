"""Analyze command for wmk CLI.

This module provides the analyze command that analyzes ONNX models
for runtime support across NPU execution providers.

Usage:
    wmk analyze --model MODEL --ep EP --device DEVICE [OPTIONS]

Examples:
    wmk analyze --model model.onnx --ep QNNExecutionProvider --device NPU
    wmk analyze --model model.onnx --ep qnn --device NPU
    wmk analyze --model model.onnx --ep ov --device GPU --information
    wmk analyze --model model.onnx --ep vitis --device GPU --output results.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ..utils import cli as cli_utils
from ..utils.constants import normalize_ep_name
from ..utils.logging import configure_logging


logger = logging.getLogger(__name__)


@click.command(name="analyze")  # type: ignore[misc]
@cli_utils.model_option(required=True)
@cli_utils.ep_option(
    required=False, optional_message="If not specified, analyzes all supported EPs"
)
@cli_utils.device_option(
    required=False, optional_message="If not specified, uses NPU as default", default="NPU"
)
@cli_utils.verbosity_options
@click.option(  # type: ignore[misc]
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save JSON output to file (default: console display)",
)
@click.option(  # type: ignore[misc]
    "--information/--no-information",
    default=True,
    help="Include detailed recommendations in output (default: enabled)",
)
@click.option(  # type: ignore[misc]
    "--htp-metadata",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to HTP metadata JSON file for enhanced pattern extraction",
)
@click.option(  # type: ignore[misc]
    "--run-unknown-op/--no-run-unknown-op",
    default=True,
    help="Run unknown operators on local machine if possible (default: enabled)",
)
@click.option(  # type: ignore[misc]
    "--save-node",
    multiple=True,
    type=click.Choice(["gray", "black"], case_sensitive=False),
    help="Save specific node types for further analysis. Can be specified multiple times "
    "(e.g., --save-node gray --save-node black).",
)
def analyze(
    model: Path,
    ep: str | None,
    device: str | None,
    output: Path | None,
    information: bool,
    verbose: bool,
    quiet: bool,
    htp_metadata: Path | None,
    run_unknown_op: bool,
    save_node: tuple[str, ...],
) -> None:
    r"""Analyze ONNX model for runtime support.

    Analyze ONNX model to determine runtime support status for the specified
    execution provider and device. Performs static analysis to detect patterns
    and check operator compatibility.

    Exit Codes:

        0: Success - execution provider supports model

        1: Partial support - some unsupported operators

        2: Error - invalid input or analysis failure

    Examples:
    Analyze all supported EPs with default device:

        wmk analyze --model model.onnx

    Check QNN NPU support (full name):

        wmk analyze --model model.onnx --ep QNNExecutionProvider --device NPU

    Check QNN NPU support (using alias):

        wmk analyze --model model.onnx --ep qnn --device NPU

    Check Intel OpenVINO GPU support with recommendations (using alias):

        wmk analyze --model model.onnx --ep ov --device GPU --information

    Analyze all EPs and save results to file:

        wmk analyze --model model.onnx --output results.json

    Use HTP metadata for enhanced pattern extraction:

        wmk analyze --model model.onnx
            --ep OpenVINOExecutionProvider --driver GPU --information --htp-metadata metadata.json
    """
    # Configure logging
    configure_logging(verbose=verbose, quiet=quiet)

    try:
        # Import core components
        logger.debug("Importing static analyzer components...")
        from ..analyze import ONNXStaticAnalyzer, __version__

        logger.info("Using analyzer version: %s", __version__)

        # Validate model file
        if not model.exists():
            logger.error("ONNX model file not found: %s", model)
            sys.exit(2)

        logger.debug("Model path: %s", model)
        logger.debug("Execution provider: %s", ep)
        logger.debug("Device: %s", device)
        logger.debug("Information: %s", information)
        if htp_metadata:
            logger.debug("HTP metadata path: %s", htp_metadata)

        # Normalize EP name (convert aliases to full names)
        ep_normalized = normalize_ep_name(ep)
        if ep != ep_normalized:
            logger.debug("EP alias '%s' normalized to '%s'", ep, ep_normalized)

        # Run static analysis using ONNXStaticAnalyzer
        logger.info("Running static analysis...")
        analyzer = ONNXStaticAnalyzer()
        save_node_types = set(save_node)
        result = analyzer.analyze(
            model_path=model,
            ep=ep_normalized,
            device=device,
            enable_information=information,
            htp_metadata_path=str(htp_metadata) if htp_metadata else None,
            run_unknown_op=run_unknown_op,
            save_node_types=save_node_types,
        )

        logger.info(
            "Analysis complete: Model is %s",
            "fully supported" if result.is_fully_supported() else "partially supported",
        )

        # Serialize to JSON
        json_output = result.to_json()

        # Parse JSON for console display
        import json

        from ..analyze.console_writer import (
            display_analysis_results,
        )
        from ..analyze.models.output import AnalysisOutput

        data = json.loads(json_output)
        analysis = AnalysisOutput.model_validate(data)

        # Save JSON to file if output path specified
        if output:
            output.write_text(json_output, encoding="utf-8")
            logger.info("JSON results saved to: %s", output)

        # Always display friendly console output
        display_analysis_results(analysis, verbose=verbose)

        # Determine exit code based on support level
        unsupported_ops = result.get_unsupported_operators()
        is_model_supported = result.is_fully_supported()
        if is_model_supported:
            # Full support
            logger.info("✓ Model is fully supported")
            sys.exit(0)
        else:
            # Partial or no support
            logger.warning("⚠ Model has %d unsupported operators", len(unsupported_ops))
            if verbose:
                for op_name in unsupported_ops[:5]:  # Show first 5
                    logger.warning("  - %s", op_name)
                if len(unsupported_ops) > 5:
                    logger.warning("  ... and %d more", len(unsupported_ops) - 5)
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(2)

    except Exception as e:
        logger.error("Analysis failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(2)


# Register the command
# This will be auto-discovered by the CLI framework
# Export only the command for CLI discovery
__all__ = ["analyze"]


if __name__ == "__main__":
    analyze()
