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

from ..models.ihv_type import IHVType
from ..models.information import Information
from ..models.runtime_checks import RuntimeCheckRule


logger = logging.getLogger(__name__)

#: Environment variable for additional runtime check rules directories.
#: Use ``os.pathsep`` (`;` on Windows, `:` on Unix) to separate multiple paths.
WINMLCLI_RULES_DIR_ENV = "WINMLCLI_RULES_DIR"

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


def get_runtime_rules_search_dirs() -> list[Path]:
    """Return ordered list of directories to search for runtime rule artifacts.

    The search order is:
        1. Any extra directories listed in the :data:`WINMLCLI_RULES_DIR` env var
            (separated by ``os.pathsep``). Absolute paths are used directly;
            relative paths are resolved relative to this module file directory.
      2. Default embedded directory (``src/winml/modelkit/analyze/rules/runtime_check_rules/``)

    Returns:
        List of directory Paths (may include non-existent ones; callers filter).
    """
    dirs: list[Path] = []
    env_val = os.environ.get(WINMLCLI_RULES_DIR_ENV, "").strip()
    if env_val:
        for entry in env_val.split(os.pathsep):
            entry = entry.strip()
            if entry:
                dirs.append(_resolve_env_rules_dir_entry(entry))
    dirs.append(_DEFAULT_RUNTIME_RULES_DIR)
    return dirs


def resolve_rule_parquet_path(parquet_filename: str) -> Path:
    """Resolve a parquet runtime-rule artifact by searching known directories.

    Args:
        parquet_filename: Bare file name, e.g.
            ``Split_QNNExecutionProvider_NPU_ai.onnx_opset13.parquet``

    Returns:
        Resolved Path to the parquet file if found. If not found, returns the
        path under the first search directory to preserve deterministic debug output.
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

    search_dirs = get_runtime_rules_search_dirs()
    ep_device_subdir = _infer_ep_device_subdir(parquet_filename)

    for search_dir in search_dirs:
        candidate = search_dir / parquet_filename
        if candidate.exists():
            return candidate

        if ep_device_subdir is not None:
            candidate_in_subdir = search_dir / ep_device_subdir / parquet_filename
            if candidate_in_subdir.exists():
                return candidate_in_subdir

        # Backward-compatible fallback for any one-level nested layout.
        nested_matches = sorted(search_dir.glob(f"*/{parquet_filename}"))
        if nested_matches:
            return nested_matches[0]

    if search_dirs:
        return search_dirs[0] / parquet_filename

    return _DEFAULT_RUNTIME_RULES_DIR / parquet_filename


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
            prefix = prefix_map[ihv]
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
        """Load information generation rules from JSON files.

        Only loads files with suffix "_information.json" to distinguish from
        other data files in the information_rules directory.

        Args:
            ihv_type: Optional IHV type for per-IHV rule loading.
                     If provided, loads default_information.json + {ihv_lowercase}_information.json.
                     If None, loads all *_information.json files (backward compatibility).

        Returns:
            List of Information instances (only enabled ones)

        Examples:
            >>> loader = RuleLoader()
            >>> # Load QC-specific rules (default + qc)
            >>> rules = loader.load_information_rules(ihv_type=IHVType.QC)
            >>> # Load all rules (backward compatibility)
            >>> rules = loader.load_information_rules(ihv_type=None)
        """
        information_rules_dir = self.rules_dir / "information_rules"

        if not information_rules_dir.exists():
            logger.warning("Information rules directory not found: %s", information_rules_dir)
            return []

        all_informations: list[Information] = []

        # Determine which files to load
        if ihv_type is not None:
            # Per-IHV loading: default + ihv-specific
            ihv_lowercase = ihv_type.value.lower()
            files_to_load = [
                information_rules_dir / "default_information.json",
                information_rules_dir / f"{ihv_lowercase}_information.json",
            ]
            # Filter only existing files
            files_to_load = [f for f in files_to_load if f.exists()]
            logger.info(
                "Loading information rules for IHV %s from %d files",
                ihv_type.value,
                len(files_to_load),
            )
        else:
            # Load all *_information.json files (backward compatibility)
            files_to_load = list(information_rules_dir.glob("*_information.json"))
            logger.info("Loading all information rules from %d files", len(files_to_load))

        # Load each file
        for rule_file in files_to_load:
            try:
                informations_data = json.loads(rule_file.read_text(encoding="utf-8"))

                # Ensure it's a list
                if not isinstance(informations_data, list):
                    informations_data = [informations_data]

                # Parse each information
                for info_dict in informations_data:
                    try:
                        # Filter enabled actions only
                        if info_dict.get("actions"):
                            info_dict["actions"] = [
                                action
                                for action in info_dict["actions"]
                                if action.get("enabled", True)
                            ]

                        # Create Information instance
                        information = Information(**info_dict)
                        all_informations.append(information)
                    except Exception as e:
                        logger.error(
                            "Failed to parse information %s in %s: %s",
                            info_dict.get("Information_id", "unknown"),
                            rule_file,
                            e,
                        )
                        continue

                logger.info(
                    "Loaded %d information rules from %s", len(informations_data), rule_file
                )

            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in %s: %s", rule_file, e)
            except Exception as e:
                logger.error("Error loading %s: %s", rule_file, e)

        logger.info("Total %d information rules loaded", len(all_informations))
        return all_informations

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
