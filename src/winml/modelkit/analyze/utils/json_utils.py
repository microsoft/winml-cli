# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""JSON utility functions for schema validation."""

import json
from pathlib import Path
from typing import Any


def validate_json_schema(data: dict[str, Any], schema_path: Path) -> bool:
    """Validate JSON data against a JSON schema file.

    Args:
        data: JSON data to validate
        schema_path: Path to JSON schema file

    Returns:
        True if validation succeeds

    Raises:
        jsonschema.ValidationError: If validation fails
        FileNotFoundError: If schema file not found
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    from jsonschema import validate

    validate(instance=data, schema=schema)
    return True
