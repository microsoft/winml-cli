"""Generate optimum_architectures.py from optimum-onnx source.

Parses model_configs.py to extract all @register_tasks_manager_onnx
registrations and produces a static reference file.

Usage:
    uv run python tests/assets/generate_optimum_architectures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMUM_ONNX_ROOT = Path(__file__).resolve().parents[2] / ".." / "external" / "optimum-onnx"
MODEL_CONFIGS_PATH = OPTIMUM_ONNX_ROOT / "optimum" / "exporters" / "onnx" / "model_configs.py"
OUTPUT_PATH = Path(__file__).resolve().parent / "optimum_architectures.py"

# Task constant definitions mirroring model_configs.py
COMMON_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "multiple-choice",
    "question-answering",
    "text-classification",
    "token-classification",
]

COMMON_TEXT_GENERATION_TASKS = [
    "feature-extraction",
    "feature-extraction-with-past",
    "text-generation",
    "text-generation-with-past",
]

COMMON_TEXT2TEXT_GENERATION_TASKS = [
    *COMMON_TEXT_GENERATION_TASKS,
    "text2text-generation",
    "text2text-generation-with-past",
]

TASK_CONSTANTS: dict[str, list[str]] = {
    "COMMON_TEXT_TASKS": COMMON_TEXT_TASKS,
    "COMMON_TEXT_GENERATION_TASKS": COMMON_TEXT_GENERATION_TASKS,
    "COMMON_TEXT2TEXT_GENERATION_TASKS": COMMON_TEXT2TEXT_GENERATION_TASKS,
}

# Category inference from OnnxConfig base class hierarchy
# Maps base class name -> category
BASE_CLASS_CATEGORY: dict[str, str] = {
    "TextEncoderOnnxConfig": "text_encoder",
    "TextDecoderOnnxConfig": "text_decoder",
    "TextDecoderWithPositionIdsOnnxConfig": "text_decoder",
    "TextSeq2SeqOnnxConfig": "seq2seq",
    "TextAndVisionOnnxConfig": "multimodal",
    "VisionOnnxConfig": "vision",
    "AudioOnnxConfig": "audio",
    "AudioToTextOnnxConfig": "audio",
    "OnnxSeq2SeqConfigWithPast": "seq2seq",
    "OnnxConfigWithPast": "other",
    "OnnxConfig": "other",
    "EncoderDecoderBaseOnnxConfig": "seq2seq",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _find_balanced_parens(source: str, start: int) -> str:
    """Extract content between balanced parentheses starting at ``(`` at *start*.

    Returns the content between (and), not including the outer parens.
    """
    if source[start] != "(":
        msg = f"Expected '(' at position {start}, got '{source[start]}'"
        raise ValueError(msg)

    depth = 0
    i = start
    while i < len(source):
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start + 1 : i]
        i += 1

    msg = "Unbalanced parentheses"
    raise ValueError(msg)


def _resolve_tasks(task_expr: str) -> list[str]:
    """Resolve a task expression from the decorator into a list of tasks.

    Handles patterns like:
      *COMMON_TEXT_TASKS
      *["feature-extraction", "fill-mask"]
      *[*COMMON_TEXT_GENERATION_TASKS, "text-classification"]
      "text-classification"  (bare string outside of *)
    """
    tasks: list[str] = []

    # Replace constant references with their values
    for const_name, const_val in TASK_CONSTANTS.items():
        if const_name in task_expr:
            task_expr = task_expr.replace(f"*{const_name}", ", ".join(f'"{t}"' for t in const_val))

    # Extract all quoted strings
    found = re.findall(r'"([^"]+)"', task_expr)
    tasks.extend(found)

    return tasks


def _resolve_base_category(
    parent_class: str,
    class_parents: dict[str, str],
) -> str:
    """Walk the inheritance chain to find the base OnnxConfig category."""
    visited: set[str] = set()
    current = parent_class

    while current and current not in visited:
        if current in BASE_CLASS_CATEGORY:
            return BASE_CLASS_CATEGORY[current]
        visited.add(current)
        current = class_parents.get(current, "")

    return "other"


# ---------------------------------------------------------------------------
# Main parsing
# ---------------------------------------------------------------------------


def parse_model_configs(source: str) -> list[dict[str, str | list[str]]]:
    """Parse model_configs.py source and extract all registrations.

    Uses a two-pass approach:
      1. Find every ``@register_tasks_manager_onnx(`` with balanced-paren
         extraction for the decorator arguments.
      2. From the position after the decorator, find the ``class`` statement
         and extract class name + parent with balanced-paren extraction.

    Returns:
        List of dicts with keys: arch_name, tasks, config_class, parent_class, library.
    """
    results: list[dict[str, str | list[str]]] = []

    decorator_prefix = "@register_tasks_manager_onnx("
    search_start = 0

    while True:
        idx = source.find(decorator_prefix, search_start)
        if idx == -1:
            break

        paren_start = idx + len(decorator_prefix) - 1  # points at '('
        try:
            args_str = _find_balanced_parens(source, paren_start)
        except ValueError:
            search_start = idx + 1
            continue

        # Move past the decorator to find the class definition
        after_decorator = paren_start + len(args_str) + 2  # +2 for ( and )
        class_match = re.search(
            r"class\s+(\w+)\s*\(",
            source[after_decorator : after_decorator + 500],
        )
        if not class_match:
            search_start = after_decorator
            continue

        config_class = class_match.group(1)
        class_paren_start = after_decorator + class_match.end() - 1  # points at '('
        try:
            parent_str = _find_balanced_parens(source, class_paren_start)
        except ValueError:
            search_start = after_decorator
            continue

        # Clean up parent string: handle conditional expressions
        parent_class = parent_str.strip()
        if " if " in parent_class:
            # e.g. "TextDecoderWithPositionIdsOnnxConfig if ... else ..."
            parent_class = parent_class.split(" if ")[0].strip()

        # Remove newlines and compress whitespace from args
        args_str = re.sub(r"\s+", " ", args_str)

        # Extract library_name if present
        library = "transformers"
        lib_match = re.search(r'library_name\s*=\s*"([^"]+)"', args_str)
        if lib_match:
            library = lib_match.group(1)
            args_str = re.sub(r',?\s*library_name\s*=\s*"[^"]+"', "", args_str)

        # Extract architecture name (first quoted string)
        arch_match = re.search(r'"([^"]+)"', args_str)
        if not arch_match:
            search_start = after_decorator
            continue
        arch_name = arch_match.group(1)

        # Everything after the arch name is the task specification
        task_str = args_str[args_str.index(arch_match.group(0)) + len(arch_match.group(0)) :]
        task_str = task_str.strip().lstrip(",").strip()

        tasks = _resolve_tasks(task_str)

        results.append(
            {
                "arch_name": arch_name,
                "tasks": sorted(set(tasks)),
                "config_class": config_class,
                "parent_class": parent_class,
                "library": library,
            }
        )

        search_start = after_decorator

    return results


def build_class_hierarchy(source: str) -> dict[str, str]:
    """Build a mapping of ClassName -> ParentClassName from the source.

    Handles multi-line class definitions and conditional parent expressions.
    """
    hierarchy: dict[str, str] = {}

    # Find all class definitions
    for match in re.finditer(r"class\s+(\w+)\s*\(", source):
        cls_name = match.group(1)
        paren_start = match.end() - 1  # points at '('
        try:
            parent_str = _find_balanced_parens(source, paren_start)
        except ValueError:
            continue

        parent = parent_str.strip()
        # Handle conditional parents like: X if condition else Y
        if " if " in parent:
            parent = parent.split(" if ")[0].strip()

        hierarchy[cls_name] = parent

    return hierarchy


def generate_output(
    registrations: list[dict[str, str | list[str]]],
    class_parents: dict[str, str],
) -> str:
    """Generate the output Python file content.

    For architectures with duplicate keys (same arch name registered under
    different libraries, e.g. ``clip`` for both transformers and
    sentence_transformers), a ``_<library>`` suffix is appended.
    """
    lines: list[str] = []

    lines.append('"""Optimum-ONNX supported architectures - generated reference.')
    lines.append("")
    lines.append("Regenerate: uv run python tests/assets/generate_optimum_architectures.py")
    lines.append("Source: optimum/exporters/onnx/model_configs.py")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("")
    lines.append("")
    lines.append("@dataclass(frozen=True)")
    lines.append("class ArchitectureInfo:")
    lines.append('    """Describes a single optimum-onnx architecture registration."""')
    lines.append("")
    lines.append("    tasks: tuple[str, ...]")
    lines.append("    config_class: str")
    lines.append("    category: str")
    lines.append(
        "    # text_encoder | text_decoder | seq2seq | vision"
        " | multimodal | audio | diffusers | other"
    )
    lines.append("    library: str  # transformers | diffusers | timm | sentence_transformers")
    lines.append("")
    lines.append("")

    # Detect duplicate arch_name values so we can disambiguate
    name_counts: dict[str, int] = {}
    for reg in registrations:
        name = reg["arch_name"]
        name_counts[name] = name_counts.get(name, 0) + 1  # type: ignore[arg-type]

    seen_names: dict[str, int] = {}

    lines.append("OPTIMUM_ARCHITECTURES: dict[str, ArchitectureInfo] = {")

    for reg in registrations:
        arch_name: str = reg["arch_name"]  # type: ignore[assignment]
        tasks: list[str] = reg["tasks"]  # type: ignore[assignment]
        config_class: str = reg["config_class"]  # type: ignore[assignment]
        parent_class: str = reg["parent_class"]  # type: ignore[assignment]
        library: str = reg["library"]  # type: ignore[assignment]

        # Disambiguate duplicate arch names by appending library suffix
        key = arch_name
        if name_counts.get(arch_name, 0) > 1:
            occurrence = seen_names.get(arch_name, 0)
            seen_names[arch_name] = occurrence + 1
            if occurrence > 0:
                # Second+ occurrence gets a suffix
                key = f"{arch_name}:{library}"
            else:
                seen_names[arch_name] = 1

        # Determine category
        category = _resolve_base_category(parent_class, class_parents)

        # Override category for diffusers library
        if library == "diffusers":
            category = "diffusers"

        # Format tasks tuple - use multi-line if it would exceed 100 chars
        tasks_inline = ", ".join(f'"{t}"' for t in tasks)
        if len(tasks) == 1:
            tasks_inline += ","

        single_line = f"        tasks=({tasks_inline}),"
        lines.append(f'    "{key}": ArchitectureInfo(')
        if len(single_line) <= 100:
            lines.append(single_line)
        else:
            lines.append("        tasks=(")
            lines.extend(f'            "{t}",' for t in tasks)
            lines.append("        ),")
        lines.append(f'        config_class="{config_class}",')
        lines.append(f'        category="{category}",')
        lines.append(f'        library="{library}",')
        lines.append("    ),")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point for generating the optimum architectures file."""
    if not MODEL_CONFIGS_PATH.exists():
        print(f"ERROR: Cannot find model_configs.py at {MODEL_CONFIGS_PATH}")
        sys.exit(1)

    source = MODEL_CONFIGS_PATH.read_text(encoding="utf-8")

    registrations = parse_model_configs(source)
    class_parents = build_class_hierarchy(source)

    output = generate_output(registrations, class_parents)

    OUTPUT_PATH.write_text(output, encoding="utf-8")

    # Summary
    arch_count = len(registrations)
    unique_names = len({r["arch_name"] for r in registrations})
    libraries = sorted({r["library"] for r in registrations})
    print(f"Generated {OUTPUT_PATH}")
    print(f"  Registrations: {arch_count}")
    print(f"  Unique arch names: {unique_names}")
    print(f"  Libraries: {', '.join(libraries)}")


if __name__ == "__main__":
    main()
