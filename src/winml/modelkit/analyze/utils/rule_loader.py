# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rule database loader for JSON rule files."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..models.ihv_type import IHVType
from ..models.information import Information
from ..models.runtime_checks import RuntimeCheckRule


logger = logging.getLogger(__name__)

#: Environment variable for the runtime check rules directory.
#: Holds a single directory path; it is not split on ``os.pathsep``.
WINMLCLI_RULES_DIR_ENV = "WINMLCLI_RULES_DIR"

#: Environment variable for the runtime debug rule directory.
#: Holds a single directory path; it is not split on ``os.pathsep``.
WINMLCLI_RULES_DIR_FOR_DEBUG_ENV = "WINMLCLI_RULES_DIR_FOR_DEBUG"

# Directory containing this module file. Relative env-var entries are resolved from here.
_RULE_LOADER_DIR: Path = Path(__file__).resolve().parent

# Default runtime_check_rules directory (relative to the analyze package).
_DEFAULT_RUNTIME_RULES_DIR: Path = (
    Path(__file__).resolve().parent.parent / "rules" / "runtime_check_rules"
)


def _resolve_env_rules_dir_entry(entry: str) -> Path:
    """Resolve a WINMLCLI_RULES_DIR entry into an absolute directory path.

    Absolute paths are used directly. Relative paths are interpreted relative
    to this module file's directory.
    """
    entry_path = Path(entry).expanduser()
    if entry_path.is_absolute():
        return entry_path.resolve()
    return (_RULE_LOADER_DIR / entry_path).resolve()


def _get_env_rules_dir(env_name: str) -> Path | None:
    """Resolve the single directory configured in ``env_name``.

    The value is treated as one directory path and is intentionally not split
    on ``os.pathsep`` -- only a single rules directory is supported. Returns
    ``None`` when the env var is unset or blank.
    """
    env_val = os.environ.get(env_name, "").strip()
    if not env_val:
        return None
    return _resolve_env_rules_dir_entry(env_val)


def get_runtime_rules_search_dirs() -> list[Path]:
    """Return the directory to search for runtime rule artifacts.

    Selection behavior:
        1. If :data:`WINMLCLI_RULES_DIR` is set, use only that directory.
            Absolute paths are used directly; a relative path is resolved
            relative to this module file directory.
        2. If :data:`WINMLCLI_RULES_DIR` is unset/empty, use the embedded default
            directory (``src/winml/modelkit/analyze/rules/runtime_check_rules/``).

    Returns:
        Single-element list with the selected directory (the embedded default
        when the env var is unset). The directory may not exist; callers filter.
    """
    env_dir = _get_env_rules_dir(WINMLCLI_RULES_DIR_ENV)
    if env_dir is not None:
        return [env_dir]
    return [_DEFAULT_RUNTIME_RULES_DIR]


def get_runtime_rules_debug_search_dirs() -> list[Path]:
    """Return the debug-rule directory from the env var only.

    Unlike :func:`get_runtime_rules_search_dirs`, this intentionally has no
    embedded default fallback: an empty list is returned when
    :data:`WINMLCLI_RULES_DIR_FOR_DEBUG` is unset.
    """
    env_dir = _get_env_rules_dir(WINMLCLI_RULES_DIR_FOR_DEBUG_ENV)
    return [env_dir] if env_dir is not None else []


def resolve_rule_parquet_path(parquet_filename: str, for_debug: bool = False) -> Path:
    """Resolve preferred parquet runtime-rule path from ``<EP>_<DEVICE>/`` subdirs.

    Args:
        parquet_filename: Bare file name, e.g.
            ``Split_QNNExecutionProvider_NPU_ai.onnx_opset13.parquet``

    Returns:
        Preferred candidate Path in search order. Existence is not checked here.
    """

    def _infer_ep_device_subdir(filename: str) -> str | None:
        # Filename layout:
        # <op>_<ep_name>_<DEVICE>_<domain>_opset<ver>[_qdq].parquet
        match = re.match(
            r"^.+_(?P<ep>[^_]+)_(?P<device>CPU|GPU|NPU)_.+_opset\d+(?:_qdq)?\.parquet$",
            filename,
        )
        if not match:
            return None
        return f"{match.group('ep')}_{match.group('device')}"

    ep_device_subdir = _infer_ep_device_subdir(parquet_filename)
    relative_path = (
        Path(ep_device_subdir) / parquet_filename
        if ep_device_subdir is not None
        else Path(parquet_filename)
    )

    if for_debug:
        debug_dirs = get_runtime_rules_debug_search_dirs()
        if debug_dirs:
            return debug_dirs[0] / relative_path

    search_dirs = get_runtime_rules_search_dirs()
    if search_dirs:
        return search_dirs[0] / relative_path

    return relative_path


class RuleLoader:
    """Loads and manages JSON rule database.

    Attributes:
        rules_dir: Path to rules directory
        runtime_rules: Cached runtime rules by IHV type
    """

    def __init__(self, rules_dir: Path | None = None) -> None:
        """Initialize rule loader.

        Args:
            rules_dir: Path to rules directory (default: src/analyze/rules/)
        """
        if rules_dir is None:
            # Default to rules directory relative to this file
            module_dir = Path(__file__).parent.parent
            rules_dir = module_dir / "rules"

        self.rules_dir = Path(rules_dir)
        self.runtime_rules: dict[str, list[RuntimeCheckRule]] = {}

    def get_runtime_rules_dir(self) -> Path:
        """Get the path to runtime check rules directory.

        Returns:
            Path to runtime_check_rules directory
        """
        return self.rules_dir / "runtime_check_rules"

    def load_runtime_rules(
        self, ihv_type: IHVType | None = None
    ) -> dict[str, list[RuntimeCheckRule]]:
        """Load runtime support rules from JSON files.

        Args:
            ihv_type: Load rules for specific IHV only (None = all IHVs)

        Returns:
            Dictionary mapping IHV type to list of RuntimeRule instances

        Raises:
            FileNotFoundError: If rule file not found and rules required
        """
        runtime_rules_dir = self.rules_dir / "runtime_check_rules"

        # Determine which IHVs to load
        ihvs_to_load = [ihv_type.value] if ihv_type else [ihv.value for ihv in IHVType]

        loaded_rules: dict[str, list[RuntimeCheckRule]] = {}

        for ihv in ihvs_to_load:
            # Map IHV to filename prefix
            prefix_map = {
                "QC": "qc",
                "Intel": "intel",
                "AMD": "amd",
                "NVIDIA": "nvidia",
            }

            # Find files matching the prefix pattern (e.g., qc_*.json)
            prefix = prefix_map.get(ihv, ihv.lower())
            matching_files = list(runtime_rules_dir.glob(f"{prefix}_*.json"))

            if not matching_files:
                logger.warning("No rule files found for %s with prefix %s, skipping", ihv, prefix)
                loaded_rules[ihv] = []
                continue

            # Load all matching files for this IHV
            all_rules = []
            for rule_file in matching_files:
                try:
                    rules_data = json.loads(rule_file.read_text(encoding="utf-8"))

                    # Parse each rule
                    for rule_dict in rules_data:
                        try:
                            rule = RuntimeCheckRule(**rule_dict)
                            all_rules.append(rule)
                        except Exception as e:
                            logger.error("Failed to parse rule in %s: %s", rule_file, e)
                            continue

                    logger.info("Loaded %d rules from %s", len(rules_data), rule_file)

                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in %s: %s", rule_file, e)
                except Exception as e:
                    logger.error("Error loading %s: %s", rule_file, e)

            loaded_rules[ihv] = all_rules
            logger.info("Total %d rules loaded for %s", len(all_rules), ihv)

        # Cache loaded rules
        self.runtime_rules.update(loaded_rules)

        return loaded_rules

    def load_information_rules(self, ihv_type: IHVType | None = None) -> list[Information]:
        """Load information generation rules from merged and legacy locations.

        Preferred source is merged pattern configuration under
        ``modelkit/pattern/rules/*.json`` in the top-level ``InformationRules``
        section. Legacy ``analyze/rules/information_rules/*_information.json``
        files are still supported for backward compatibility and are merged in.

        Args:
            ihv_type: Optional IHV type for per-IHV rule loading.

        Returns:
            List of enabled Information instances.
        """
        all_informations: list[Information] = []
        seen_information_keys: set[tuple[str | None, str | None, str]] = set()

        # 1) Preferred source: merged pattern configuration files.
        pattern_rule_files = self._get_pattern_rule_files_for_information(ihv_type)
        if pattern_rule_files:
            logger.info(
                "Loading merged information rules from %d pattern rule file(s)",
                len(pattern_rule_files),
            )

        for rule_file in pattern_rule_files:
            try:
                config_data = json.loads(rule_file.read_text(encoding="utf-8"))
                informations_data = config_data.get("InformationRules", [])

                if not isinstance(informations_data, list):
                    informations_data = [informations_data]

                self._extend_information_from_dicts(
                    all_informations=all_informations,
                    seen_information_keys=seen_information_keys,
                    informations_data=informations_data,
                    source_file=rule_file,
                )
                logger.info(
                    "Loaded %d merged information rule(s) from %s",
                    len(informations_data),
                    rule_file,
                )
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in %s: %s", rule_file, e)
            except Exception as e:
                logger.error("Error loading %s: %s", rule_file, e)

        # 2) Backward compatibility source: *_information.json files.
        information_rules_dir = self.rules_dir / "information_rules"
        if not information_rules_dir.exists():
            if all_informations:
                logger.info(
                    "Loaded %d information rule(s) from merged pattern rules; "
                    "legacy information_rules directory not found: %s",
                    len(all_informations),
                    information_rules_dir,
                )
                return all_informations
            logger.warning("Information rules directory not found: %s", information_rules_dir)
            return []

        if ihv_type is not None:
            ihv_lowercase = ihv_type.value.lower()
            legacy_files_to_load = [
                information_rules_dir / "default_information.json",
                information_rules_dir / f"{ihv_lowercase}_information.json",
            ]
            legacy_files_to_load = [f for f in legacy_files_to_load if f.exists()]
            logger.info(
                "Loading legacy information rules for IHV %s from %d file(s)",
                ihv_type.value,
                len(legacy_files_to_load),
            )
        else:
            legacy_files_to_load = list(information_rules_dir.glob("*_information.json"))
            logger.info(
                "Loading all legacy information rules from %d file(s)",
                len(legacy_files_to_load),
            )

        for rule_file in legacy_files_to_load:
            try:
                informations_data = json.loads(rule_file.read_text(encoding="utf-8"))
                if not isinstance(informations_data, list):
                    informations_data = [informations_data]

                self._extend_information_from_dicts(
                    all_informations=all_informations,
                    seen_information_keys=seen_information_keys,
                    informations_data=informations_data,
                    source_file=rule_file,
                )

                logger.info(
                    "Loaded %d legacy information rule(s) from %s",
                    len(informations_data),
                    rule_file,
                )
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in %s: %s", rule_file, e)
            except Exception as e:
                logger.error("Error loading %s: %s", rule_file, e)

        logger.info("Total %d information rules loaded", len(all_informations))
        return all_informations

    def _get_pattern_rule_files_for_information(self, ihv_type: IHVType | None) -> list[Path]:
        """Resolve merged pattern-rule files that may contain InformationRules.

        Only applies when rules_dir follows the default source layout:
        ``.../modelkit/analyze/rules``.
        """
        # Detect source-tree layout used by this package.
        if self.rules_dir.name != "rules" or self.rules_dir.parent.name != "analyze":
            return []

        pattern_rules_dir = self.rules_dir.parent.parent / "pattern" / "rules"
        if not pattern_rules_dir.exists():
            return []

        if ihv_type is None:
            files = [
                f
                for f in pattern_rules_dir.glob("*.json")
                if not f.name.endswith(".schema.json")
            ]
            return sorted(files)

        ihv_to_file_stem = {
            IHVType.QC: "qnn",
            IHVType.INTEL: "openvino",
            IHVType.AMD: "quark",
            IHVType.NVIDIA: "nvidia",
            IHVType.MICROSOFT: "microsoft",
            IHVType.UNKNOWN: None,
        }

        files = [pattern_rules_dir / "default.json"]
        ihv_stem = ihv_to_file_stem.get(ihv_type)
        if ihv_stem:
            files.append(pattern_rules_dir / f"{ihv_stem}.json")

        return [f for f in files if f.exists()]

    def _extend_information_from_dicts(
        self,
        *,
        all_informations: list[Information],
        seen_information_keys: set[tuple[str | None, str | None, str]],
        informations_data: list[Any],
        source_file: Path,
    ) -> None:
        """Parse and append Information dicts, filtering disabled entries and duplicates."""
        for info_dict in informations_data:
            try:
                if not isinstance(info_dict, dict):
                    continue

                if not info_dict.get("enabled", True):
                    continue

                normalized_info = dict(info_dict)

                if normalized_info.get("actions"):
                    normalized_info["actions"] = [
                        action
                        for action in normalized_info["actions"]
                        if action.get("enabled", True)
                    ]

                information = Information(**normalized_info)
                dedupe_key = (
                    normalized_info.get("Information_id"),
                    information.pattern_id,
                    information.explanation,
                )
                if dedupe_key in seen_information_keys:
                    continue

                seen_information_keys.add(dedupe_key)
                all_informations.append(information)
            except Exception as e:
                info_id = (
                    info_dict.get("Information_id", "unknown")
                    if isinstance(info_dict, dict)
                    else "unknown"
                )
                logger.error(
                    "Failed to parse information %s in %s: %s",
                    info_id,
                    source_file,
                    e,
                )
                continue

    def get_rules_for_pattern(self, pattern_id: str, ihv_type: IHVType) -> list[RuntimeCheckRule]:
        """Get all rules matching specific pattern and IHV.

        Args:
            pattern_id: Pattern identifier (e.g., "OP/ai.onnx/Conv")
            ihv_type: Target IHV type

        Returns:
            List of matching RuntimeRule instances
        """
        if ihv_type not in self.runtime_rules:
            # Rules not loaded yet, load them
            self.load_runtime_rules(ihv_type)

        ihv_rules = self.runtime_rules.get(ihv_type.value, [])

        # Filter by pattern_id
        return [rule for rule in ihv_rules if rule.pattern_id == pattern_id]
