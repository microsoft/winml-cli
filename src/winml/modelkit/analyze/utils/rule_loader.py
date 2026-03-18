"""Rule database loader for JSON rule files."""

import json
import logging
from pathlib import Path

from ..models.ihv_type import IHVType
from ..models.information import Information
from ..models.runtime_checks import RuntimeCheckRule


logger = logging.getLogger(__name__)


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
