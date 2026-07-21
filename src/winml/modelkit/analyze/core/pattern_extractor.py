# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""PatternExtractor - Extract operator and subgraph patterns from ONNX models.

Implements FR-003 (Extract patterns), FR-011 (Pattern detection), FR-004 (Subgraph patterns).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from ...pattern.base import InvalidPatternMatcherModelError, PatternMatcher
from ...pattern.config import PatternConfig, UnifiedPatternConfig
from ..models.onnx_model import ModelTag, ONNXModel
from ..models.output import extract_model_stats
from ..utils.timing_utils import make_timing_logger


if TYPE_CHECKING:
    import onnx

    from ...pattern.base import Pattern
    from ...pattern.match import PatternMatchResult
    from ...pattern.models import SubgraphPattern
    from ...utils.constants import EPNameOrAlias

    from ..models.ihv_type import IHVType
    from ..models.output import ModelStats


class PatternSourceStat(TypedDict):
    """Per-source skeleton extraction stats for debug reporting."""

    source: str
    cache_hit: bool
    pattern_class_count: int
    match_count: int
    elapsed_ms: int


class PatternSummary(TypedDict):
    """Type definition for pattern analysis summary."""

    summary: ModelStats
    subgraph_patterns: list[PatternMatchResult]
    subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]]
    source_stats: list[PatternSourceStat]
    total_extract_ms: int
    model_signature: str


# Type alias for HTP metadata structure
HTPMetadata = dict[str, dict[str, str] | dict[str, object]]

logger = logging.getLogger(__name__)
_log_timing = make_timing_logger(logger)


class PatternExtractor:
    """Extract operator and subgraph patterns from ONNX models.

    Responsibilities:
    - Detect subgraph patterns (GELU, LayerNorm, Attention)
    - Create PatternMatchResult instances for each detected pattern
    - Generate model metadata and statistics

    FR-003: Extract patterns from ONNX model
    FR-004: Detect subgraph-level patterns

    Attributes:
        model: ONNX model to analyze (ONNXModel)
    """

    # In-memory per-process caches.
    # - rules cache: source key -> loaded skeleton Pattern instances
    # - match cache: (model signature, source key) -> grouped PatternMatchResult
    _RULES_PATTERN_CACHE: dict[str, list[Pattern]] = {}
    _MATCH_CACHE: dict[tuple[str, str], dict[str, list[PatternMatchResult]]] = {}

    def __init__(self, model: ONNXModel, htp_metadata_path: str | None = None) -> None:
        """Initialize pattern extractor.

        Args:
            model: ONNX model to analyze (ONNXModel)
            htp_metadata_path: Optional path to HTP metadata JSON file

        Raises:
            TypeError: If model is invalid
        """
        if not isinstance(model, ONNXModel):
            raise TypeError(f"Expected ONNXModel, got {type(model)}")

        self._model = model
        self._htp_metadata_path = htp_metadata_path
        self._htp_metadata: HTPMetadata | None = None

        logger.info(
            "Initialized PatternExtractor for model: %s",
            model.model_path,
        )

        if htp_metadata_path:
            logger.info("HTP metadata path provided: %s", htp_metadata_path)

    @property
    def model(self) -> ONNXModel:
        """The ONNX model being analyzed."""
        return self._model

    def _compute_model_signature(self) -> str:
        """Build a stable in-process signature for cache keys."""
        model_path = self._model.model_path
        if model_path and model_path != "<memory>":
            path = Path(model_path)
            if path.exists():
                stat = path.stat()
                return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

        # Fallback for in-memory models or missing paths.
        model_bytes = self._model.get_model().SerializeToString()
        digest = hashlib.sha1(model_bytes).hexdigest()
        return f"in_memory:{digest}"

    @staticmethod
    def _ihv_to_rules_key(ihv_type: IHVType) -> str | None:
        """Map IHV enum to rules filename stem."""
        mapping = {
            "QC": "qnn",
            "INTEL": "openvino",
            "AMD": "quark",
            "NVIDIA": "nvidia",
            "MICROSOFT": "microsoft",
        }
        return mapping.get(ihv_type.name)

    def _resolve_sources_for_ep(self, ep: EPNameOrAlias | None) -> list[str]:
        """Return extraction sources for the target EP.

        The new flow keeps default and IHV-specific extraction independent.
        """
        sources = ["default"]
        if ep is None:
            return sources

        from ..utils import infer_ihv_from_ep_name
        from ..models.ihv_type import IHVType

        ihv_type = infer_ihv_from_ep_name(ep)
        if ihv_type is IHVType.UNKNOWN:
            return sources

        rules_key = self._ihv_to_rules_key(ihv_type)
        if rules_key and self._rules_file_for_source(rules_key).exists():
            sources.append(rules_key)
        return sources

    @staticmethod
    def _rules_dir() -> Path:
        """Return the pattern rules directory."""
        # .../modelkit/analyze/core/pattern_extractor.py -> .../modelkit/pattern/rules
        return Path(__file__).resolve().parents[2] / "pattern" / "rules"

    def _rules_file_for_source(self, source: str) -> Path:
        """Return rules JSON path for a source key."""
        return self._rules_dir() / f"{source}.json"

    def _load_skeleton_patterns_for_source(self, source: str) -> list[Pattern]:
        """Load skeleton pattern instances for one source, with in-memory cache."""
        cached = self._RULES_PATTERN_CACHE.get(source)
        if cached is not None:
            return cached

        patterns: list[Pattern] = []
        if source == "default":
            cfg = UnifiedPatternConfig(ihv_type="default")
            patterns = cfg.get_skeleton_patterns()
            self._RULES_PATTERN_CACHE[source] = patterns
            return patterns

        rules_file = self._rules_file_for_source(source)
        if not rules_file.exists():
            self._RULES_PATTERN_CACHE[source] = []
            return []

        try:
            with rules_file.open(encoding="utf-8") as f:
                source_cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load source rules config: %s", rules_file, exc_info=True)
            self._RULES_PATTERN_CACHE[source] = []
            return []

        for entry in source_cfg.get("SkeletonPatternRules", []):
            if not entry.get("enabled", False):
                continue
            try:
                pattern_cfg = PatternConfig(
                    pattern_id=entry["pattern_id"],
                    pattern_class=entry["pattern_class"],
                    module=entry["module"],
                    enabled=bool(entry["enabled"]),
                    description=entry.get("description"),
                    alternatives=[],
                )
                patterns.append(pattern_cfg.load_pattern())
            except Exception:
                logger.warning(
                    "Failed to load skeleton pattern from %s for source '%s': %s",
                    rules_file,
                    source,
                    entry.get("pattern_class", "<unknown>"),
                    exc_info=True,
                )

        self._RULES_PATTERN_CACHE[source] = patterns
        return patterns

    def _extract_skeleton_matches_for_source(
        self,
        *,
        source: str,
        model_signature: str,
    ) -> tuple[dict[str, list[PatternMatchResult]], PatternSourceStat]:
        """Extract skeleton matches for one source with model+source cache key."""
        cache_key = (model_signature, source)
        start = time.perf_counter()

        cached = self._MATCH_CACHE.get(cache_key)
        if cached is not None:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            hit_stat: PatternSourceStat = {
                "source": source,
                "cache_hit": True,
                "pattern_class_count": len(cached),
                "match_count": sum(len(v) for v in cached.values()),
                "elapsed_ms": elapsed_ms,
            }
            return {k: list(v) for k, v in cached.items()}, hit_stat

        grouped: dict[str, list[PatternMatchResult]] = {}
        pattern_instances = self._load_skeleton_patterns_for_source(source)
        if pattern_instances:
            model_proto = self._model.get_model()
            try:
                matcher = PatternMatcher(model_proto, model_path=self._model.model_path)
            except InvalidPatternMatcherModelError as e:
                logger.warning("Model validation failed for pattern matching: %s", str(e))
                self._model.model_tags[ModelTag(e.error_tag)] = str(e)
                matcher = None

            if matcher is not None:
                for pattern in pattern_instances:
                    matcher.register_pattern(pattern)

                matches = matcher.match()
                for match in matches:
                    # Keep explicit source for debug attribution.
                    match.attributes["source"] = source
                    pattern_class = match.pattern.__class__.__name__
                    grouped.setdefault(pattern_class, []).append(match)

        self._MATCH_CACHE[cache_key] = grouped
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        miss_stat: PatternSourceStat = {
            "source": source,
            "cache_hit": False,
            "pattern_class_count": len(grouped),
            "match_count": sum(len(v) for v in grouped.values()),
            "elapsed_ms": elapsed_ms,
        }
        return {k: list(v) for k, v in grouped.items()}, miss_stat

    def _load_htp_metadata(self) -> HTPMetadata:
        """Load HTP metadata from JSON file.

        Returns:
            Dictionary containing HTP metadata

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If JSON is invalid
        """
        if self._htp_metadata is not None:
            return self._htp_metadata

        if not self._htp_metadata_path:
            logger.debug("No HTP metadata path provided")
            return {}

        import json
        from pathlib import Path

        metadata_path = Path(self._htp_metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"HTP metadata file not found: {self._htp_metadata_path}")

        logger.info("Loading HTP metadata from: %s", self._htp_metadata_path)

        try:
            with metadata_path.open(encoding="utf-8") as f:
                self._htp_metadata = json.load(f)
            logger.info("Successfully loaded HTP metadata")
            return self._htp_metadata
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in HTP metadata file: {e}") from e

    def summary(self, ep: EPNameOrAlias | None = None) -> PatternSummary:
        """Generate comprehensive pattern analysis summary.

        Returns:
            PatternSummary with keys:
                - summary: ModelStats (from model_summary())
                - subgraph_patterns: List[PatternMatchResult] (from extract_subgraph_patterns())
        """
        logger.info("Generating pattern analysis summary")
        total_start = time.perf_counter()

        model_signature = self._compute_model_signature()
        sources = self._resolve_sources_for_ep(ep)

        subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]] = {}
        source_stats: list[PatternSourceStat] = []
        subgraph_patterns: list[PatternMatchResult] = []

        for source in sources:
            grouped_matches, stat = self._extract_skeleton_matches_for_source(
                source=source,
                model_signature=model_signature,
            )
            subgraph_patterns_by_source[source] = grouped_matches
            source_stats.append(stat)
            for matches in grouped_matches.values():
                subgraph_patterns.extend(matches)

        # Build pattern count dict: pattern_id -> count
        count_dict_start = time.perf_counter()
        pattern_count_dict: dict[str, int] = {}
        for pattern_match in subgraph_patterns:
            pattern_id = pattern_match.pattern.pattern_id
            pattern_count_dict[pattern_id] = pattern_count_dict.get(pattern_id, 0) + 1
        count_dict_ms = int((time.perf_counter() - count_dict_start) * 1000)

        # Generate model summary with pattern count dict
        model_summary_start = time.perf_counter()
        metadata = self.model_summary(detected_pattern_count=pattern_count_dict)
        model_summary_ms = int((time.perf_counter() - model_summary_start) * 1000)

        _log_timing(
            "pattern_extractor.summary",
            model=self._model.model_path,
            detected_subgraph_patterns=len(subgraph_patterns),
            unique_pattern_ids=len(pattern_count_dict),
            extract_subgraph_ms=sum(stat["elapsed_ms"] for stat in source_stats),
            build_count_dict_ms=count_dict_ms,
            model_summary_ms=model_summary_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        total_extract_ms = int((time.perf_counter() - total_start) * 1000)
        return {
            "summary": metadata,
            "subgraph_patterns": subgraph_patterns,
            "subgraph_patterns_by_source": subgraph_patterns_by_source,
            "source_stats": source_stats,
            "total_extract_ms": total_extract_ms,
            "model_signature": model_signature,
        }

    def extract_subgraph_patterns(self) -> list[PatternMatchResult]:
        """Extract subgraph patterns from model.

        Subgraph patterns represent multi-operator fusion opportunities
        (e.g., GELU, LayerNorm, Attention).

        Returns:
            List of PatternMatchResult objects

        Process:
            1. Load subgraph pattern definitions via get_subgraph_patterns()
            2. For each pattern, match against model graph
            3. For each match, create PatternMatchResult with node_topology mapping
            4. Return all detected subgraph patterns

        Note:
            - Pattern ID format: SUBGRAPH/<PatternName>
            - node_topology uses pattern-defined slot names as keys
            - Actual node names from the model graph as values
        """
        logger.info("Extracting subgraph patterns from model")
        total_start = time.perf_counter()

        # Get available subgraph pattern definitions
        get_pattern_defs_start = time.perf_counter()
        pattern_defs = self.get_subgraph_patterns()
        get_pattern_defs_ms = int((time.perf_counter() - get_pattern_defs_start) * 1000)

        # Match patterns against model graph
        detected_matches: list[PatternMatchResult] = []
        metadata_tag_match_start = time.perf_counter()

        for pattern_def in pattern_defs:
            # Try HTP metadata-based matching first if available
            if self._htp_metadata_path:
                htp_matches = self._match_subgraph_pattern_from_htp_metadata(pattern_def)
                if htp_matches:
                    detected_matches.extend(htp_matches)
                    continue

            # Fall back to hierarchy_tag attribute-based matching
            matches = self._match_subgraph_pattern_from_model_tags(pattern_def)
            detected_matches.extend(matches)
        metadata_tag_match_ms = int((time.perf_counter() - metadata_tag_match_start) * 1000)

        # Use PatternMatcher for skeleton-based pattern detection
        logger.info("Using PatternMatcher for skeleton-based pattern detection")
        pattern_matcher_start = time.perf_counter()
        pattern_matcher_matches = self.extract_subgraph_patterns_with_pattern_matcher()
        pattern_matcher_ms = int((time.perf_counter() - pattern_matcher_start) * 1000)

        # Deduplicate PatternMatcher results against existing matches
        # Priority: HTP metadata > hierarchy_tag > PatternMatcher
        # Collect node sets from existing matches (from HTP/tag)
        dedup_start = time.perf_counter()
        existing_node_sets: set[frozenset[str]] = {
            frozenset(match.matched_nodes) for match in detected_matches
        }

        # Filter PatternMatcher matches to exclude duplicates
        filtered_matcher_matches: list[PatternMatchResult] = []
        for match in pattern_matcher_matches:
            node_names = frozenset(match.matched_nodes)
            if node_names not in existing_node_sets:
                filtered_matcher_matches.append(match)
            else:
                # Log first few nodes (sorted for consistency)
                sample_nodes = sorted(node_names)[:3]
                logger.debug(
                    "Skipping PatternMatcher match with duplicate nodes: %s (pattern: %s)",
                    sample_nodes,
                    match.pattern_id,
                )

        dropped_count = len(pattern_matcher_matches) - len(filtered_matcher_matches)
        if dropped_count > 0:
            logger.info(
                "Dropped %d PatternMatcher matches that duplicate existing matches (from HTP/tag)",
                dropped_count,
            )

        # Add filtered PatternMatcher matches
        detected_matches.extend(filtered_matcher_matches)
        dedup_ms = int((time.perf_counter() - dedup_start) * 1000)

        logger.info(
            "Detected %d total subgraph pattern matches (including %d unique from PatternMatcher)",
            len(detected_matches),
            len(filtered_matcher_matches),
        )
        _log_timing(
            "pattern_extractor.extract_subgraph_patterns",
            model=self._model.model_path,
            pattern_defs=len(pattern_defs),
            matches_before_matcher=len(existing_node_sets),
            matcher_matches=len(pattern_matcher_matches),
            matcher_unique_added=len(filtered_matcher_matches),
            matcher_dropped_as_duplicate=dropped_count,
            get_pattern_defs_ms=get_pattern_defs_ms,
            metadata_tag_match_ms=metadata_tag_match_ms,
            pattern_matcher_ms=pattern_matcher_ms,
            dedup_ms=dedup_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )
        return detected_matches

    def extract_subgraph_patterns_with_pattern_matcher(self) -> list[PatternMatchResult]:
        """Extract subgraph patterns using PatternMatcher.

        This method uses the PatternMatcher class to perform skeleton-based
        pattern matching against registered patterns.

        Returns:
            List of PatternMatchResult objects

        Process:
            1. Create PatternMatcher instance with the model
            2. Load and register pattern instances from UnifiedPatternConfig
            3. Call matcher.match() to get PatternMatchResult objects
            4. Return all detected pattern matches
        """
        logger.info("Extracting subgraph patterns using PatternMatcher")
        total_start = time.perf_counter()

        # Get model proto for PatternMatcher
        get_model_start = time.perf_counter()
        model_proto = self._model.get_model()
        get_model_ms = int((time.perf_counter() - get_model_start) * 1000)

        # Create PatternMatcher instance - may raise InvalidPatternMatcherModelError
        try:
            matcher_init_start = time.perf_counter()
            matcher = PatternMatcher(model_proto, model_path=self._model.model_path)
            matcher_init_ms = int((time.perf_counter() - matcher_init_start) * 1000)
        except InvalidPatternMatcherModelError as e:
            # Model is invalid for pattern matching (e.g., nodes with empty names)
            logger.warning("Model validation failed for pattern matching: %s", str(e))
            # Mark model with the exception's associated tag and error message
            self._model.model_tags[ModelTag(e.error_tag)] = str(e)
            _log_timing(
                "pattern_extractor.pattern_matcher",
                model=self._model.model_path,
                failed=True,
                error_tag=e.error_tag,
                get_model_ms=get_model_ms,
                total_ms=int((time.perf_counter() - total_start) * 1000),
            )
            return []

        # Register patterns from the unified pattern config
        load_patterns_start = time.perf_counter()
        config = UnifiedPatternConfig()
        patterns_to_register = config.get_skeleton_patterns()
        load_patterns_ms = int((time.perf_counter() - load_patterns_start) * 1000)

        if not patterns_to_register:
            logger.warning("No patterns available in config")
            _log_timing(
                "pattern_extractor.pattern_matcher",
                model=self._model.model_path,
                failed=True,
                reason="no_patterns_in_config",
                get_model_ms=get_model_ms,
                matcher_init_ms=matcher_init_ms,
                load_patterns_ms=load_patterns_ms,
                total_ms=int((time.perf_counter() - total_start) * 1000),
            )
            return []

        register_start = time.perf_counter()
        for pattern in patterns_to_register:
            matcher.register_pattern(pattern)
        register_ms = int((time.perf_counter() - register_start) * 1000)

        logger.info("Registered %d patterns for matching", len(patterns_to_register))

        # Perform pattern matching
        logger.info("Calling PatternMatcher.match()...")
        match_start = time.perf_counter()
        pattern_matches = matcher.match()
        match_ms = int((time.perf_counter() - match_start) * 1000)
        logger.info("PatternMatcher found %d matches", len(pattern_matches))

        if not pattern_matches:
            logger.info("No pattern matches found by PatternMatcher")
            # Debug: try skeleton matching without validation
            skeleton_results = matcher.match_skeleton()
            logger.info(
                "Skeleton matching found %d potential matches (before validation)",
                len(skeleton_results),
            )
            if skeleton_results:
                matched_node_keys = skeleton_results[0].matched_node_keys
                sample_nodes = matched_node_keys[:3] if matched_node_keys else []
                logger.info(
                    "Sample skeleton match - Pattern: %s, Nodes: %s",
                    skeleton_results[0].pattern.__class__.__name__,
                    sample_nodes,
                )

        logger.info(
            "Extracted %d subgraph patterns using PatternMatcher",
            len(pattern_matches),
        )
        _log_timing(
            "pattern_extractor.pattern_matcher",
            model=self._model.model_path,
            patterns_registered=len(patterns_to_register),
            matches=len(pattern_matches),
            get_model_ms=get_model_ms,
            matcher_init_ms=matcher_init_ms,
            load_patterns_ms=load_patterns_ms,
            register_ms=register_ms,
            match_ms=match_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )
        return pattern_matches

    def _validate_pattern_for_matching(self, pattern: SubgraphPattern) -> bool:
        """Validate if pattern has required attributes for matching.

        Args:
            pattern: SubgraphPattern definition

        Returns:
            True if pattern is valid for matching, False otherwise
        """
        if not pattern.semantic_label:
            logger.debug(
                "Pattern %s has no semantic_label, skipping matching",
                pattern.pattern_id,
            )
            return False
        return True

    def _create_pattern_matches(
        self,
        pattern: SubgraphPattern,
        grouped_nodes: dict[str, list[tuple[str, str]]],
        source_type: str,
    ) -> list[PatternMatchResult]:
        """Create PatternMatchResult instances from grouped nodes.

        Args:
            pattern: SubgraphPattern definition
            grouped_nodes: Dict mapping tag to list of (node identifier, tag) tuples
            source_type: Source of the match ("hierarchy_tag" or "htp_metadata")

        Returns:
            List of PatternMatch instances
        """
        from ...pattern.match import PatternMatchResult, SkeletonMatchResult

        # Note: For hierarchy_tag and HTP metadata matches, we create a simplified
        # PatternMatchResult without full skeleton information since these matches
        # are based on tags rather than topology matching.

        detected_matches: list[PatternMatchResult] = []

        for tag, node_list in grouped_nodes.items():
            logger.debug(
                "Found %d nodes with tag '%s' containing pattern_label '%s'",
                len(node_list),
                tag,
                pattern.semantic_label,
            )

            # Resolve identifiers to NodeProto and normalize to stable keys.
            matched_node_identifiers = [node_identifier for node_identifier, _ in node_list]
            matched_nodes = []
            matched_node_keys = []
            for node_identifier in matched_node_identifiers:
                node_proto = self._model.get_node_by_key(node_identifier)
                if node_proto is None:
                    node_proto = self._model.get_node_by_name(node_identifier)
                if node_proto is None:
                    continue
                matched_nodes.append(node_proto)
                matched_node_keys.append(self._model.get_node_key(node_proto))

            # Create a minimal SkeletonMatchResult for API compatibility
            # This is a placeholder since hierarchy_tag matches don't have full skeleton info
            skeleton_result = SkeletonMatchResult(
                pattern=pattern,  # Use the SubgraphPattern directly
                matched_nodes=matched_nodes,
                matched_node_keys=matched_node_keys,
                matcher=None,  # type: ignore
                inputs=[],
                output="",
                removable=False,
            )

            # Create PatternMatchResult with source metadata
            attributes = {"source": source_type}
            if source_type == "htp_metadata":
                attributes["traced_tag"] = tag
            else:
                attributes["hierarchy_tag"] = tag

            pattern_match = PatternMatchResult(
                skeleton_match_result=skeleton_result,
                schema_input_to_value={},
                schema_output_to_value={},
                type_param_to_type={},
                attributes=attributes,
                input_infos={},
            )
            detected_matches.append(pattern_match)

        return detected_matches

    def _match_subgraph_pattern_from_model_tags(
        self, pattern: SubgraphPattern
    ) -> list[PatternMatchResult]:
        """Match a subgraph pattern against the model graph using hierarchy tags.

        Args:
            pattern: SubgraphPattern definition

        Returns:
            List of PatternMatchResult instances for detected matches

        Note:
            This implementation matches patterns based on hierarchy_tag attributes
            embedded in ONNX nodes. For nodes with hierarchy tags containing the
            pattern's semantic_label, it groups them by hierarchy_tag and creates
            PatternMatch instances.
        """
        # Validate pattern
        if not self._validate_pattern_for_matching(pattern):
            return []

        pattern_label = pattern.semantic_label
        assert pattern_label is not None  # ensured by _validate_pattern_for_matching

        # Get ONNX model
        model_proto = self._model.get_model()
        graph = model_proto.graph

        # Group nodes by hierarchy_tag that contains pattern_label
        grouped_nodes: dict[str, list[tuple[str, str]]] = {}

        for node in graph.node:
            # Extract hierarchy_tag attribute
            hierarchy_tag = self._extract_hierarchy_tag(node)
            if not hierarchy_tag:
                continue

            # Check if hierarchy_tag contains pattern_label
            if pattern_label in hierarchy_tag:
                if hierarchy_tag not in grouped_nodes:
                    grouped_nodes[hierarchy_tag] = []
                grouped_nodes[hierarchy_tag].append((self._model.get_node_key(node), hierarchy_tag))

        # Create PatternMatch instances
        detected_matches = self._create_pattern_matches(
            pattern=pattern,
            grouped_nodes=grouped_nodes,
            source_type="hierarchy_tag",
        )

        logger.info(
            "Pattern %s: found %d matches based on hierarchy_tag",
            pattern.pattern_id,
            len(detected_matches),
        )

        return detected_matches

    def _extract_hierarchy_tag(self, node: onnx.NodeProto) -> str | None:
        """Extract hierarchy_tag attribute from ONNX node.

        Args:
            node: ONNX NodeProto object

        Returns:
            Hierarchy tag string or None if not found
        """
        for attr in node.attribute:
            if attr.name == "hierarchy_tag":
                return attr.s.decode("utf-8") if attr.s else None
        return None

    def _match_subgraph_pattern_from_htp_metadata(
        self, pattern: SubgraphPattern
    ) -> list[PatternMatchResult]:
        """Match a subgraph pattern using HTP metadata.

        This method extracts patterns from HTP metadata JSON by analyzing
        the nodes mapping and module hierarchy.

        Args:
            pattern: SubgraphPattern definition

        Returns:
            List of PatternMatchResult instances for detected matches

        Note:
            Uses the 'nodes' section of HTP metadata which maps ONNX node names
            to their traced_tag (hierarchy path).
        """
        # Validate pattern
        if not self._validate_pattern_for_matching(pattern):
            return []

        # Load and validate HTP metadata
        htp_metadata = self._load_and_validate_htp_metadata()
        if not htp_metadata:
            return []

        pattern_label = pattern.semantic_label
        assert pattern_label is not None  # ensured by _validate_pattern_for_matching

        # The 'nodes' section of HTP metadata maps node names to traced tags (str -> str).
        nodes_mapping = cast("dict[str, str]", htp_metadata["nodes"])

        # Group nodes by traced_tag that contains pattern_label
        grouped_nodes = self._group_nodes_by_traced_tag(
            nodes_mapping=nodes_mapping,
            pattern_label=pattern_label,
        )

        # Create PatternMatch instances
        detected_matches = self._create_pattern_matches(
            pattern=pattern,
            grouped_nodes=grouped_nodes,
            source_type="htp_metadata",
        )

        logger.info(
            "Pattern %s: found %d matches from HTP metadata",
            pattern.pattern_id,
            len(detected_matches),
        )

        return detected_matches

    def _load_and_validate_htp_metadata(self) -> HTPMetadata | None:
        """Load and validate HTP metadata.

        Returns:
            HTP metadata dict or None if invalid/unavailable
        """
        try:
            htp_metadata = self._load_htp_metadata()
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Failed to load HTP metadata: %s", e)
            return None

        if not htp_metadata or "nodes" not in htp_metadata:
            logger.debug("No nodes section in HTP metadata")
            return None

        return htp_metadata

    def _group_nodes_by_traced_tag(
        self,
        nodes_mapping: dict[str, str],
        pattern_label: str,
    ) -> dict[str, list[tuple[str, str]]]:
        """Group nodes by their traced_tag that contains pattern_label.

        Args:
            nodes_mapping: Dict mapping node names to traced tags
            pattern_label: Pattern semantic label to match

        Returns:
            Dict mapping traced_tag to list of (node_name, traced_tag) tuples
        """
        grouped_nodes: dict[str, list[tuple[str, str]]] = {}

        for node_name, traced_tag in nodes_mapping.items():
            if pattern_label in traced_tag:
                if traced_tag not in grouped_nodes:
                    grouped_nodes[traced_tag] = []
                grouped_nodes[traced_tag].append((node_name, traced_tag))

        return grouped_nodes

    def get_subgraph_patterns(self) -> list[SubgraphPattern]:
        """Get available subgraph pattern definitions.

        Returns:
            List of SubgraphPattern objects with pattern definitions

        Note:
            Patterns are loaded from UnifiedPatternConfig (HTPPatternRules section).
        """
        logger.debug("Loading subgraph pattern definitions from UnifiedPatternConfig")

        # Load HTP patterns from UnifiedPatternConfig
        config = UnifiedPatternConfig()
        patterns = config.get_htp_patterns()

        if not patterns:
            logger.warning("No HTP patterns found in config, returning empty list")
            return []

        logger.debug("Loaded %d subgraph pattern definitions", len(patterns))
        return patterns

    def model_summary(
        self,
        detected_pattern_count: dict[str, int] | None = None,
    ) -> ModelStats:
        """Get model metadata and statistics.

        Args:
            detected_pattern_count: Pattern ID to count mapping (default: empty dict)

        Returns:
            ModelStats object containing model information
        """
        return extract_model_stats(
            self._model,
            detected_pattern_count=detected_pattern_count,
        )
