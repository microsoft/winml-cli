# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: RUF001
# RUF001: info emoji used intentionally in user-facing log messages
"""Export Configuration Generator for HTP Exporter.

Automatically generates optimal ONNX export configurations based on:
- Model type/architecture
- Task type (classification, generation, etc.)
- Target deployment (QNN, CPU, etc.)
- Input specifications

Universal design - no hardcoded model-specific logic, uses pattern matching.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


logger = logging.getLogger(__name__)


@dataclass
class ExportConfigTemplate:
    """Template for ONNX export configuration."""

    # Core ONNX export parameters
    opset_version: int = 17
    do_constant_folding: bool = True
    verbose: bool = False
    dynamo: bool = False  # Force legacy TorchScript

    # Input/output names (auto-generated if not provided)
    input_names: list[str] | None = None
    output_names: list[str] | None = None

    # Dynamic axes (None = static batch, recommended for QNN)
    dynamic_axes: dict[str, dict[int, str]] | None = None

    # Input specifications
    input_specs: dict[str, dict[str, Any]] | None = None

    # Target deployment
    target_deployment: Literal["qnn", "cpu", "universal"] = "qnn"

    # Model-specific hints
    model_type: str | None = None
    task: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, removing None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}

    def to_json(self, output_path: str | Path) -> None:
        """Save configuration to JSON file."""
        with Path(output_path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("✅ Export config saved to: %s", output_path)


class ExportConfigGenerator:
    """Generate optimal export configurations based on model and deployment target.

    Uses universal pattern-matching approach (no hardcoded logic).
    """

    @staticmethod
    def generate(
        model_name_or_path: str,
        target_deployment: Literal["qnn", "cpu", "universal"] = "qnn",
        task: str | None = None,
        batch_size: int = 1,
        input_shape: tuple | None = None,
        **overrides: Any,
    ) -> ExportConfigTemplate:
        """Generate export configuration automatically.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            target_deployment: Target deployment platform
            task: Task type (auto-detected if not provided)
            batch_size: Batch size (1 recommended for QNN)
            input_shape: Input tensor shape (auto-detected if not provided)
            **overrides: Manual overrides for any config parameter

        Returns:
            ExportConfigTemplate with optimal settings
        """
        logger.info("Generating export config for: %s", model_name_or_path)
        logger.info("Target deployment: %s", target_deployment)

        # Get model info
        model_info = ExportConfigGenerator._get_model_info(model_name_or_path)
        model_type = model_info.get("model_type")
        detected_task = task or model_info.get("task")

        logger.info("Model type: %s", model_type)
        logger.info("Task: %s", detected_task)

        # Get input specifications
        input_specs = ExportConfigGenerator._generate_input_specs(
            model_type=model_type,
            task=detected_task,
            batch_size=batch_size,
            input_shape=input_shape,
        )

        # Get input/output names
        input_names = list(input_specs.keys()) if input_specs else None
        output_names = ExportConfigGenerator._get_output_names(model_type, detected_task)

        # Create base config
        config = ExportConfigTemplate(
            input_names=input_names,
            output_names=output_names,
            input_specs=input_specs,
            model_type=model_type,
            task=detected_task,
            target_deployment=target_deployment,
        )

        # Apply deployment-specific optimizations
        if target_deployment == "qnn":
            # QNN requires static batch (already default)
            config.dynamic_axes = None
            config.dynamo = False
            logger.info("✅ QNN-optimized config: static batch, dynamo=False")
        elif target_deployment == "cpu":
            # CPU can use dynamic batch
            if input_names and detected_task in [
                "text-classification",
                "text-generation",
            ]:
                config.dynamic_axes = {
                    name: {0: "batch_size", 1: "sequence_length"} for name in input_names
                }
                logger.info("✅ CPU config: dynamic batch enabled")
        elif target_deployment == "universal":
            # Universal - user decides
            config.dynamic_axes = None  # Default static
            logger.info("ℹ️  Universal config: static batch (change via overrides if needed)")

        # Apply user overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info("Override applied: %s = %s", key, value)

        return config

    @staticmethod
    def _get_model_info(model_name_or_path: str) -> dict[str, Any]:
        """Get model type and task info using Optimum.

        Uses universal detection - no hardcoded logic.
        """
        try:
            from optimum.exporters import TasksManager
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name_or_path)
            model_type = config.model_type

            # Auto-detect task
            try:
                supported_tasks = TasksManager.get_supported_tasks_for_model_type(
                    model_type,
                    exporter="onnx",
                    library_name="transformers",
                )
                task = (
                    next(iter(supported_tasks.keys())) if supported_tasks else "feature-extraction"
                )
            except Exception:
                task = "feature-extraction"

            return {
                "model_type": model_type,
                "task": task,
                "config": config,
            }

        except Exception as e:
            logger.warning("Could not load model info: %s", e)
            return {"model_type": "unknown", "task": "feature-extraction"}

    @staticmethod
    def _generate_input_specs(
        model_type: str | None,
        task: str | None,
        batch_size: int,
        input_shape: tuple | None,
    ) -> dict[str, dict[str, Any]]:
        """Generate input specifications using pattern matching.

        Uses InputSpecGenerator patterns (universal approach).
        """
        try:
            from ...inference.onnx_config.input_generator import (
                InputSpecGenerator,
            )

            # Get input names for this model type
            input_names = InputSpecGenerator.get_input_names(
                model_type or "unknown",
                task or "feature-extraction",
                config=None,
            )

            # Generate input specs
            specs = {}
            for input_name in input_names:
                if input_name in ["input_ids", "attention_mask", "token_type_ids"]:
                    # Text input
                    seq_length = input_shape[1] if input_shape and len(input_shape) > 1 else 128
                    specs[input_name] = {
                        "shape": [batch_size, seq_length],
                        "dtype": "int",
                        "range": [0, 30522] if input_name == "input_ids" else None,
                    }
                elif input_name == "pixel_values":
                    # Vision input
                    if input_shape:
                        specs[input_name] = {
                            "shape": list(input_shape),
                            "dtype": "float",
                        }
                    else:
                        specs[input_name] = {
                            "shape": [batch_size, 3, 224, 224],
                            "dtype": "float",
                        }
                else:
                    # Generic input (use default shape)
                    specs[input_name] = {
                        "shape": [batch_size, 128],  # Generic sequence
                        "dtype": "float",
                    }

            # Remove None ranges
            for spec in specs.values():
                if spec.get("range") is None:
                    spec.pop("range", None)

            return specs

        except Exception as e:
            logger.warning("Could not generate input specs: %s", e)
            return {}

    @staticmethod
    def _get_output_names(
        model_type: str | None,
        task: str | None,
    ) -> list[str]:
        """Get output names using pattern matching.

        Uses InputSpecGenerator patterns (universal approach).
        """
        try:
            from ...inference.onnx_config.patterns import TASK_TO_OUTPUTS

            if task and task in TASK_TO_OUTPUTS:
                return TASK_TO_OUTPUTS[task]

            # Default outputs
            if task and "classification" in task:
                return ["logits"]
            if task and "generation" in task:
                return ["last_hidden_state"]
            return ["output"]

        except Exception as e:
            logger.warning("Could not determine output names: %s", e)
            return ["output"]

    @staticmethod
    def generate_for_qnn(model_name_or_path: str, **kwargs: Any) -> ExportConfigTemplate:
        """Convenience method for QNN-optimized export config."""
        return ExportConfigGenerator.generate(model_name_or_path, target_deployment="qnn", **kwargs)


# CLI-friendly functions


def generate_config_cli(
    model: str,
    output: str = "export_config.json",
    target: str = "qnn",
    task: str | None = None,
    batch_size: int = 1,
) -> None:
    """CLI entrypoint for config generation.

    Usage:
        from winml.modelkit.export.htp.config_generator import generate_config_cli
        generate_config_cli("prajjwal1/bert-tiny", "bert_config.json", "qnn")
    """
    config = ExportConfigGenerator.generate(
        model_name_or_path=model,
        target_deployment=target,
        task=task,
        batch_size=batch_size,
    )

    config.to_json(output)

    print(f"\n✅ Generated export config for {model}")
    print(f"   Target: {target}")
    print(f"   Config saved to: {output}")
    print("\nUsage:")
    print(f"   modelexport export --model {model} --output model.onnx \\")
    print(f"     --export-config {output}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python export_config_generator.py <model_name_or_path> [output_file] [target]"
        )
        print("Example: python export_config_generator.py prajjwal1/bert-tiny bert_config.json qnn")
        sys.exit(1)

    model = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "export_config.json"
    target = sys.argv[3] if len(sys.argv) > 3 else "qnn"

    generate_config_cli(model, output, target)
