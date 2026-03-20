# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern Configuration - Config-driven pattern management.

This module provides a unified configuration system for both HTP (Hierarchy Tag Propagation)
patterns and skeleton-based patterns, with support for IHV-specific customization.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from winml.modelkit.pattern.base import Pattern
    from winml.modelkit.pattern.models import SubgraphPattern

logger = logging.getLogger(__name__)


@dataclass
class PatternAlternative:
    """Configuration for a pattern alternative (rewrite recommendation).

    Alternatives can be used in two ways:
    1. For Information generation: pattern_to_id references a pattern for display
    2. For Pattern rewriting: pattern_class + module specify the target Pattern class

    Attributes:
        pattern_to_id: Target pattern identifier (e.g., "OP/com.microsoft/Gelu")
        priority: Priority for recommendation (1=highest priority)
        reason: Optional explanation for why this alternative is recommended
        pattern_class: Optional Python class name for rewrite target (e.g., "SingleGeluPattern")
        module: Optional fully-qualified module path for pattern_class
                (e.g., "modelkit.pattern.gelu_patterns")
    """

    pattern_to_id: str
    priority: int
    reason: str | None = None
    pattern_class: str | None = None
    module: str | None = None


@dataclass
class PatternConfig:
    """Configuration for a single skeleton pattern.

    Attributes:
        pattern_id: Pattern identifier (e.g., "SUBGRAPH/Gelu1")
        pattern_class: Python class name for the pattern (e.g., "Gelu1Pattern")
        module: Fully-qualified module path
        enabled: Whether this pattern is active
        description: Optional human-readable pattern description
        alternatives: List of alternative pattern recommendations
    """

    pattern_id: str
    pattern_class: str
    module: str
    enabled: bool
    description: str | None = None
    alternatives: list[PatternAlternative] = field(default_factory=list)

    def load_pattern(self) -> Pattern:
        """Dynamically load and instantiate the pattern class.

        Returns:
            Pattern instance

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If pattern class not found in module
        """
        # Support both physical (modelkit.) and namespace (winml.modelkit.) paths.
        # JSON configs use physical names; installed wheel uses namespace.
        modules_to_try = [self.module]
        if self.module.startswith("modelkit."):
            modules_to_try.append("winml." + self.module)
        elif self.module.startswith("winml.modelkit."):
            modules_to_try.append(self.module.removeprefix("winml."))

        for mod_name in modules_to_try:
            try:
                mod = importlib.import_module(mod_name)
                pattern_cls = getattr(mod, self.pattern_class)
            except (ImportError, AttributeError):
                continue
            # Instantiation errors should propagate, not be silently caught
            return pattern_cls()

        msg = f"Failed to load pattern {self.pattern_class} from {self.module}"
        logger.error(msg)
        raise ImportError(msg)


class UnifiedPatternConfig:
    """Unified configuration manager for both HTP and skeleton patterns.

    This class loads unified pattern configurations from JSON files and provides
    access to both HTP pattern metadata and skeleton pattern definitions.

    Configuration files are located in modelkit/pattern/rules/:
    - default.json: Default unified configuration for all IHVs
    - qnn.json: Qualcomm QNN-specific unified configuration
    - openvino.json: Intel OpenVINO-specific unified configuration
    - quark.json: AMD Quark-specific unified configuration

    Each config file contains two sections:
    - HTPPatternRules: Pattern metadata for hierarchy tag propagation
    - SkeletonPatternRules: Skeleton pattern registry for topology matching

    Configuration Inheritance:
    - default.json contains shared patterns used by all IHVs
    - IHV-specific files contain only IHV-specific customizations
    - At runtime, UnifiedPatternConfig merges default + IHV configs
    - Merge strategy: IHV config overrides default config for same pattern_id

    Usage:
        # Load default configuration
        config = UnifiedPatternConfig()

        # Get skeleton patterns for matching
        skeleton_patterns = config.get_skeleton_patterns()

        # Get HTP pattern metadata
        htp_patterns = config.get_htp_patterns()

        # Load IHV-specific configuration
        qnn_config = UnifiedPatternConfig(ihv_type="QNN")
        qnn_skeleton_patterns = qnn_config.get_skeleton_patterns()
        qnn_htp_patterns = qnn_config.get_htp_patterns()

        # Get alternatives for a pattern
        alternatives = qnn_config.get_alternatives(gelu_pattern)
    """

    def __init__(
        self,
        ihv_type: str | None = None,
        config_path: Path | str | None = None
    ) -> None:
        """Initialize the unified pattern configuration.

        Args:
            ihv_type: IHV type to load patterns for (e.g., "QNN", "Intel", "AMD").
                     If None, loads default configuration.
            config_path: Optional path to configuration file. If None, uses convention:
                        modelkit/pattern/rules/{ihv_type}.json
        """
        self._skeleton_patterns: list[Pattern] = []
        self._htp_patterns: list[SubgraphPattern] = []
        self._pattern_configs: dict[str, PatternConfig] = {}
        self._loaded = False
        self.ihv_type = ihv_type or "default"

        # Determine config file path
        if config_path is None:
            # Convention-based path: pattern_rules/{ihv_type}.json
            rules_dir = Path(__file__).parent / "rules"
            config_path = rules_dir / f"{self.ihv_type.lower()}.json"
        else:
            config_path = Path(config_path)

        self.config_path = config_path
        logger.debug(
            f"UnifiedPatternConfig initialized for IHV={self.ihv_type}, "
            f"config={self.config_path}"
        )

    def get_skeleton_patterns(self) -> list[Pattern]:
        """Get all enabled skeleton pattern instances for topology matching.

        Returns:
            List of Pattern class instances (e.g., Gelu1Pattern, MatMulAddPattern).

        Note:
            Patterns are lazily loaded on first call.
        """
        if not self._loaded:
            self._load_config()
        return self._skeleton_patterns.copy()

    def get_htp_patterns(self) -> list[SubgraphPattern]:
        """Get all HTP pattern metadata for semantic labeling.

        Returns:
            List of SubgraphPattern models for hierarchy tag matching.

        Note:
            Patterns are lazily loaded on first call.
        """
        if not self._loaded:
            self._load_config()
        return self._htp_patterns.copy()

    def get_alternatives(self, pattern: Pattern) -> list[PatternAlternative]:
        """Get alternative pattern metadata for the given pattern.

        Alternatives are used by runtime_checker to generate PatternTestResult,
        which then feeds into Information generation for user display.

        Args:
            pattern: The source pattern to find alternatives for

        Returns:
            List of PatternAlternative metadata, sorted by priority (highest first)
        """
        if not self._loaded:
            self._load_config()

        # Find config for this pattern using pattern class name
        pattern_class_name = pattern.__class__.__name__
        config = self._pattern_configs.get(pattern_class_name)

        if not config or not config.alternatives:
            return []

        # Return alternative metadata sorted by priority
        return sorted(config.alternatives, key=lambda x: x.priority)

    def _load_config(self) -> None:
        """Load and merge unified pattern configuration from JSON files.

        This method implements configuration inheritance:
        1. Loads default.json as the base configuration
        2. If IHV-specific config requested, loads and merges it on top
        3. Merge strategy: IHV config overrides default for same pattern_id

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
            ImportError: If a pattern module cannot be imported
            AttributeError: If a pattern class is not found
        """
        rules_dir = Path(__file__).parent / "rules"
        default_config_path = rules_dir / "default.json"

        # Load default configuration first
        if not default_config_path.exists():
            logger.error(f"Default configuration not found: {default_config_path}")
            raise FileNotFoundError(
                f"Default pattern configuration file not found: {default_config_path}"
            )

        logger.info(f"Loading default configuration from {default_config_path}")
        with default_config_path.open(encoding='utf-8') as f:
            merged_config = json.load(f)

        # If IHV-specific config requested, load and merge it
        if self.ihv_type.lower() != "default":
            if not self.config_path.exists():
                logger.warning(
                    f"IHV configuration file not found: {self.config_path}. "
                    f"Using default configuration only."
                )
            else:
                logger.info(f"Loading IHV configuration from {self.config_path}")
                with self.config_path.open(encoding='utf-8') as f:
                    ihv_config = json.load(f)

                # Merge HTPPatternRules: IHV overrides default by pattern_id
                merged_config['HTPPatternRules'] = self._merge_pattern_rules(
                    merged_config.get('HTPPatternRules', []),
                    ihv_config.get('HTPPatternRules', []),
                    key='pattern_id'
                )

                # Merge SkeletonPatternRules: IHV overrides default by pattern_class
                # Using pattern_class as key because multiple Pattern classes
                # can share the same pattern_id
                merged_config['SkeletonPatternRules'] = self._merge_pattern_rules(
                    merged_config.get('SkeletonPatternRules', []),
                    ihv_config.get('SkeletonPatternRules', []),
                    key='pattern_class'
                )

                logger.info(f"Merged default + {self.ihv_type} configurations")

        config_data = merged_config

        # Validate required sections
        if 'HTPPatternRules' not in config_data:
            logger.warning("Missing 'HTPPatternRules' section in configuration")
            config_data['HTPPatternRules'] = []

        if 'SkeletonPatternRules' not in config_data:
            logger.warning("Missing 'SkeletonPatternRules' section in configuration")
            config_data['SkeletonPatternRules'] = []

        # Load HTP pattern metadata
        for htp_data in config_data['HTPPatternRules']:
            try:
                # Import SubgraphPattern model
                from winml.modelkit.pattern.models import SubgraphPattern

                # Convert edge_topology if present
                if "edge_topology" in htp_data and isinstance(htp_data["edge_topology"], list):
                    htp_data["edge_topology"] = [tuple(edge) for edge in htp_data["edge_topology"]]

                # Provide default topology for semantic_label based patterns
                if "node_topology" not in htp_data:
                    htp_data["node_topology"] = {}
                if "edge_topology" not in htp_data:
                    htp_data["edge_topology"] = []

                htp_pattern = SubgraphPattern(**htp_data)
                self._htp_patterns.append(htp_pattern)
                logger.debug(f"Loaded HTP pattern: {htp_pattern.pattern_id}")
            except Exception as e:  # noqa: PERF203
                pattern_id = htp_data.get('pattern_id', 'unknown')
                logger.warning(
                    f"Failed to load HTP pattern {pattern_id}: {e}. Skipping."
                )

        # Load Skeleton pattern configurations
        skeleton_count = 0
        for pattern_data in config_data['SkeletonPatternRules']:
            # Parse alternatives (metadata only, not loaded as Pattern instances)
            alternatives = [
                PatternAlternative(
                    pattern_to_id=alt_data['pattern_to_id'],
                    priority=alt_data['priority'],
                    reason=alt_data.get('reason'),
                    pattern_class=alt_data.get('pattern_class'),
                    module=alt_data.get('module'),
                )
                for alt_data in pattern_data.get('alternatives', [])
            ]

            # Create pattern config
            config = PatternConfig(
                pattern_id=pattern_data['pattern_id'],
                pattern_class=pattern_data['pattern_class'],
                module=pattern_data['module'],
                enabled=pattern_data['enabled'],
                description=pattern_data.get('description'),
                alternatives=alternatives,
            )

            # Store config for get_alternatives() using pattern_class as key
            # This allows multiple Pattern classes with the same pattern_id
            self._pattern_configs[config.pattern_class] = config

            # Load pattern if enabled
            if config.enabled:
                try:
                    pattern = config.load_pattern()
                    self._skeleton_patterns.append(pattern)
                    skeleton_count += 1
                    logger.debug(f"Loaded skeleton pattern: {config.pattern_id}")
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        f"Failed to load skeleton pattern {config.pattern_id}: {e}. Skipping."
                    )

        self._loaded = True
        logger.info(
            f"Unified config loaded for IHV={self.ihv_type}: "
            f"{len(self._htp_patterns)} HTP patterns, {skeleton_count} skeleton patterns"
        )

    def _merge_pattern_rules(
        self,
        base_rules: list[dict],
        override_rules: list[dict],
        key: str = 'pattern_id'
    ) -> list[dict]:
        """Merge two pattern rule lists with override behavior.

        Args:
            base_rules: Base rules (typically from default.json)
            override_rules: Override rules (typically from IHV-specific config)
            key: The key to use for matching rules (e.g., 'pattern_id')

        Returns:
            Merged list where override_rules take precedence for matching keys

        Merge strategy:
            - If pattern_id exists in both: use override version (complete replacement)
            - If pattern_id only in override: add to result
            - If pattern_id only in base: add to result
        """
        # Build index of base rules
        base_index = {rule[key]: rule for rule in base_rules}

        # Build index of override rules
        override_index = {rule[key]: rule for rule in override_rules}

        # Merge: override takes precedence
        merged_index = {**base_index, **override_index}

        # Return as list (maintain insertion order from merged dict)
        return list(merged_index.values())

    def clear(self) -> None:
        """Clear all loaded patterns and reset state."""
        self._skeleton_patterns = []
        self._htp_patterns = []
        self._pattern_configs = {}
        self._loaded = False
        logger.debug("Unified pattern configuration cleared")
