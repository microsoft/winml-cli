"""JSON utility functions for schema validation."""

import json
from pathlib import Path
from typing import Any

from jsonschema import validate


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

    validate(instance=data, schema=schema)
    return True


def load_json_file(file_path: Path) -> dict[str, Any]:
    """Load and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data as dictionary

    Raises:
        FileNotFoundError: If file not found
        json.JSONDecodeError: If JSON is malformed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def save_json_file(data: dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to serialize
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")
