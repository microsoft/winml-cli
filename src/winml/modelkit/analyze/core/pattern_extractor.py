# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""PatternExtractor - Extract operator and subgraph patterns from ONNX models.

Implements FR-003 (Extract patterns), FR-011 (Pattern detection), FR-004 (Subgraph patterns).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

import numpy as np

from ...onnx import ONNXDomain
from ...pattern.base import InvalidPatternMatcherModelError, PatternMatcher, PatternMismatchedError
from ...pattern.config import PatternAlternative, PatternConfig, UnifiedPatternConfig
from ..models.onnx_model import ModelTag, ONNXModel
from ..models.output import extract_model_stats
from ..utils.model_utils import encode_rule_condition_value_for_parquet, make_hashable
from ..utils.rule_loader import get_runtime_rules_debug_search_dirs, get_runtime_rules_search_dirs
from ..utils.timing_utils import make_timing_logger


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ...pattern.base import Pattern
    from ...pattern.match import PatternMatchResult
    from ...utils.constants import EPNameOrAlias
    from ..models.ihv_type import IHVType
    from ..models.output import ModelStats
    from ..models.runtime_checks import RuntimeDebugDetails, RuntimeTestResult


class PatternSourceStat(TypedDict):
    """Per-source skeleton extraction stats for debug reporting."""

    source: str
    cache_hit: bool
    pattern_class_count: int
    match_count: int
    elapsed_ms: int


class PatternOptimizationHint(TypedDict):
    """Fallback optimization hint extracted from matched pattern alternatives."""

    source: str
    pattern_id: str
    pattern_to_id: str
    instances: int
    enabled: bool
    details: str | None
    reason: str | None
    action_items: list[dict[str, Any]]


class PatternSummary(TypedDict):
    """Type definition for pattern analysis summary."""

    summary: ModelStats
    subgraph_patterns: list[PatternMatchResult]
    subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]]
    source_stats: list[PatternSourceStat]
    merge_prep: list[PatternMergePrepEntry]
    model_signature: str
    parquet_lookup_supported: bool
    pattern_optimization_hints: list[PatternOptimizationHint]


class PatternRuleCompileRunResult(TypedDict):
    """Rule-table compile/run snapshot for one pattern candidate."""

    pattern_class: str
    pattern_id: str
    is_alternative: bool
    status: str
    mismatch_error: str | None
    compile: bool | None
    run: bool | None
    row_count: int
    table_file: str | None
    table_path: str | None
    domain: str | None
    opset_version: int | None
    compile_true_rows: int
    run_true_rows: int
    case_indices: list[Any] | None
    query_condition_count: int
    query_condition_keys: list[str]
    debug_details: RuntimeDebugDetails | None


class PatternMergePrepEntry(TypedDict):
    """Derived metadata used by upcoming pattern merge/dedup stage."""

    source: str
    pattern_class: str
    pattern_id: str
    match_count: int
    match_index: int
    match_id: str
    matched_node_keys: list[str]
    support_status: str
    alternatives: list[dict[str, Any]]
    candidates: list[PatternRuleCompileRunResult]

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
    # - merge prep cache: (model signature, ep, device, debug flag) -> merge prep entries
    _RULES_PATTERN_CACHE: ClassVar[dict[str, list[Pattern]]] = {}
    _MATCH_CACHE: ClassVar[dict[tuple[str, str], dict[str, list[PatternMatchResult]]]] = {}
    _DEDUPED_MATCH_CACHE: ClassVar[
        dict[
            tuple[str, str],
            tuple[
                dict[str, dict[str, list[PatternMatchResult]]],
                list[PatternMatchResult],
            ],
        ]
    ] = {}
    _MERGE_PREP_CACHE: ClassVar[
        dict[tuple[str, str, str, bool, bool], list[PatternMergePrepEntry]]
    ] = {}
    _VALID_EP_DEVICE_PAIRS_CACHE: set[tuple[str, str]] | None = None

    def __init__(self, model: ONNXModel) -> None:
        """Initialize pattern extractor.

        Args:
            model: ONNX model to analyze (ONNXModel)

        Raises:
            TypeError: If model is invalid
        """
        if not isinstance(model, ONNXModel):
            raise TypeError(f"Expected ONNXModel, got {type(model)}")

        self._model = model
        self._query_condition_build_cache: dict[
            tuple[str, str, tuple[tuple[str, int], ...]],
            tuple[dict[str, Any], Any],
        ] = {}

        logger.info(
            "Initialized PatternExtractor for model: %s",
            model.model_path,
        )

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
        digest = hashlib.sha1(model_bytes, usedforsecurity=False).hexdigest()
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

        from ..models.ihv_type import IHVType
        from ..utils import infer_ihv_from_ep_name

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

    @staticmethod
    def _available_providers_config_path() -> Path:
        """Return bundled EP/device validity mapping JSON path."""
        return (
            Path(__file__).resolve().parents[1]
            / "utils"
            / "avalizble_ep_device_ops"
            / "avaliable_providers.json"
        )

    @classmethod
    def _load_valid_ep_device_pairs(cls) -> set[tuple[str, str]]:
        """Load and cache valid EP/device pairs from provider config."""
        if cls._VALID_EP_DEVICE_PAIRS_CACHE is not None:
            return cls._VALID_EP_DEVICE_PAIRS_CACHE

        valid_pairs: set[tuple[str, str]] = set()
        config_path = cls._available_providers_config_path()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Failed to load available providers config: %s",
                config_path,
                exc_info=True,
            )
            cls._VALID_EP_DEVICE_PAIRS_CACHE = valid_pairs
            return valid_pairs

        if not isinstance(payload, dict):
            cls._VALID_EP_DEVICE_PAIRS_CACHE = valid_pairs
            return valid_pairs

        for ep_name, ep_payload in payload.items():
            if not isinstance(ep_name, str) or not isinstance(ep_payload, dict):
                continue

            devices_payload = ep_payload.get("devices")
            if not isinstance(devices_payload, dict):
                continue

            for device_name, device_payload in devices_payload.items():
                if not isinstance(device_name, str) or not isinstance(device_payload, dict):
                    continue
                if bool(device_payload.get("valid", False)):
                    valid_pairs.add((ep_name, device_name.upper()))

        cls._VALID_EP_DEVICE_PAIRS_CACHE = valid_pairs
        return valid_pairs

    def _is_valid_parquet_lookup_target(self, ep_name: str, device: str) -> bool:
        """Return True when parquet lookup should run for this EP/device pair."""
        valid_pairs = self._load_valid_ep_device_pairs()
        if not valid_pairs:
            return False
        return (ep_name, device.upper()) in valid_pairs

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

    @staticmethod
    def _copy_grouped_matches(
        grouped: dict[str, dict[str, list[PatternMatchResult]]],
    ) -> dict[str, dict[str, list[PatternMatchResult]]]:
        """Shallow-copy grouped-match containers while reusing match objects."""
        return {
            source: {
                pattern_class: list(matches)
                for pattern_class, matches in source_group.items()
            }
            for source, source_group in grouped.items()
        }

    @staticmethod
    def _cache_key_for_ep_dedup(ep: EPNameOrAlias | None) -> str:
        """Build cache key component for EP-scoped dedup results."""
        if ep is None:
            return "__default__"
        return str(ep)

    def _ordered_sources_for_ep_dedup(
        self,
        *,
        sources: list[str],
        ep: EPNameOrAlias | None,
    ) -> list[str]:
        """Return source traversal order with EP-specific source first when available."""
        if ep is None:
            return list(sources)

        from ..models.ihv_type import IHVType
        from ..utils import infer_ihv_from_ep_name

        ihv_type = infer_ihv_from_ep_name(ep)
        if ihv_type is IHVType.UNKNOWN:
            return list(sources)

        ep_source = self._ihv_to_rules_key(ihv_type)
        if not ep_source or ep_source not in sources:
            return list(sources)

        return [ep_source] + [source for source in sources if source != ep_source]

    def _dedup_grouped_matches_for_ep(
        self,
        *,
        subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]],
        sources: list[str],
        model_signature: str,
        ep: EPNameOrAlias | None,
    ) -> tuple[dict[str, dict[str, list[PatternMatchResult]]], list[PatternMatchResult]]:
        """Deduplicate matches by node key with EP-source traversal priority.

        Traversal order is EP cache first (when present), then default cache.
        Any pattern match touching a previously seen node key is filtered out.
        Results are cached by (model signature, EP) so same EP with different
        devices reuses dedup output.
        """
        cache_key = (model_signature, self._cache_key_for_ep_dedup(ep))
        cached = self._DEDUPED_MATCH_CACHE.get(cache_key)
        if cached is not None:
            cached_grouped, cached_flat = cached
            return self._copy_grouped_matches(cached_grouped), list(cached_flat)

        ordered_sources = self._ordered_sources_for_ep_dedup(sources=sources, ep=ep)

        seen_node_keys: set[str] = set()
        deduped_grouped: dict[str, dict[str, list[PatternMatchResult]]] = {}
        deduped_flat: list[PatternMatchResult] = []

        for source in ordered_sources:
            source_group = subgraph_patterns_by_source.get(source, {})
            kept_by_pattern_class: dict[str, list[PatternMatchResult]] = {}

            for pattern_class, matches in source_group.items():
                kept_matches: list[PatternMatchResult] = []

                for pattern_match in matches:
                    node_keys = list(pattern_match.matched_node_keys)
                    if any(node_key in seen_node_keys for node_key in node_keys):
                        continue

                    seen_node_keys.update(node_keys)
                    kept_matches.append(pattern_match)
                    deduped_flat.append(pattern_match)

                if kept_matches:
                    kept_by_pattern_class[pattern_class] = kept_matches

            if kept_by_pattern_class:
                deduped_grouped[source] = kept_by_pattern_class

        self._DEDUPED_MATCH_CACHE[cache_key] = (
            self._copy_grouped_matches(deduped_grouped),
            list(deduped_flat),
        )
        return deduped_grouped, deduped_flat

    def _domain_and_target_opset_for_pattern(
        self,
        pattern: Pattern,
        model_opsets: dict[ONNXDomain, int],
    ) -> tuple[str, int]:
        """Infer preferred domain/opset for locating pattern-level rule parquet files."""
        skeleton = pattern.get_skeleton()
        if not skeleton.node_domains:
            default_opset = model_opsets.get(ONNXDomain.AI_ONNX, 1)
            return ONNXDomain.AI_ONNX.value, default_opset

        preferred_domain = skeleton.node_domains[0]
        target_opset = model_opsets.get(
            preferred_domain,
            model_opsets.get(ONNXDomain.AI_ONNX, 1),
        )
        return preferred_domain.value, target_opset

    @staticmethod
    def _parse_pattern_rule_filename(
        filename: str,
        *,
        pattern_class: str,
        ep_name: str,
        device: str,
    ) -> tuple[str, int] | None:
        """Parse `<pattern>_<ep>_<device>_<domain>_opset<ver>.parquet` style names."""
        prefix = f"{pattern_class}_{ep_name}_{device.upper()}_"
        if not filename.startswith(prefix):
            return None

        suffix = filename[len(prefix) :]
        match = re.match(r"(?P<domain>.+)_opset(?P<opset>\d+)(?:_qdq)?\.parquet$", suffix)
        if match is None:
            return None

        return match.group("domain"), int(match.group("opset"))

    def _resolve_pattern_rule_table(
        self,
        *,
        pattern_class: str,
        ep_name: str,
        device: str,
        preferred_domain: str,
        target_opset: int,
        for_debug: bool,
    ) -> tuple[Path | None, str | None, int | None]:
        """Resolve the most suitable parquet table for one pattern candidate."""
        search_dirs: list[Path] = []
        if for_debug:
            search_dirs.extend(get_runtime_rules_debug_search_dirs())
        search_dirs.extend(get_runtime_rules_search_dirs())

        # Keep first-seen order and skip non-existing directories.
        dedup_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        for base_dir in search_dirs:
            try:
                resolved_dir = base_dir.resolve(strict=False)
            except OSError:
                continue
            if resolved_dir in seen_dirs or not resolved_dir.is_dir():
                continue
            seen_dirs.add(resolved_dir)
            dedup_dirs.append(resolved_dir)

        if not dedup_dirs:
            return None, None, None

        rule_subdir = f"{ep_name}_{device.upper()}"
        glob_pattern = f"{pattern_class}_{ep_name}_{device.upper()}_*_opset*.parquet"

        for base_dir in dedup_dirs:
            target_dir = base_dir / rule_subdir
            if not target_dir.is_dir():
                continue

            candidates: list[tuple[Path, str, int]] = []
            for path in target_dir.glob(glob_pattern):
                parsed = self._parse_pattern_rule_filename(
                    path.name,
                    pattern_class=pattern_class,
                    ep_name=ep_name,
                    device=device,
                )
                if parsed is None:
                    continue
                domain_name, opset_version = parsed
                candidates.append((path, domain_name, opset_version))

            if not candidates:
                continue

            # Prefer exact-domain rows; then closest opset not above target.
            same_domain_le = [
                c for c in candidates if c[1] == preferred_domain and c[2] <= target_opset
            ]
            if same_domain_le:
                return max(same_domain_le, key=lambda c: c[2])

            any_domain_le = [c for c in candidates if c[2] <= target_opset]
            if any_domain_le:
                return max(any_domain_le, key=lambda c: c[2])

            same_domain_gt = [
                c for c in candidates if c[1] == preferred_domain and c[2] > target_opset
            ]
            if same_domain_gt:
                return min(same_domain_gt, key=lambda c: c[2])

            return min(candidates, key=lambda c: c[2])

        return None, None, None

    @staticmethod
    def _normalize_compile_run_cell(value: Any) -> tuple[bool, bool] | None:
        """Normalize one `compile_run_success` cell to `(compile, run)` booleans."""
        raw_value = value
        if not isinstance(raw_value, (list, tuple)) and hasattr(raw_value, "tolist"):
            try:
                raw_value = raw_value.tolist()
            except Exception:
                return None

        if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
            return None

        return bool(raw_value[0]), bool(raw_value[1])

    @staticmethod
    def _extract_rule_condition_columns(column_names: list[str]) -> list[str]:
        """Return parquet condition columns (excluding output metadata columns)."""
        output_cols = {
            "row_index",
            "compile_run_success",
            "compile_reason",
            "run_reason",
            "rule_row_count",
            "case_indices",
        }
        return [col for col in column_names if col not in output_cols]

    @staticmethod
    def _normalize_case_indices(case_indices: Any) -> list[Any] | None:
        """Normalize case_indices to list form for debug payloads."""
        if case_indices is None:
            return None

        normalized = case_indices
        if hasattr(normalized, "tolist"):
            try:
                normalized = normalized.tolist()
            except Exception:
                normalized = case_indices

        if isinstance(normalized, list):
            return normalized
        if isinstance(normalized, tuple):
            return list(normalized)
        return [normalized]

    def _load_pattern_rule_table(
        self,
        parquet_path: Path,
        table_cache: dict[str, Any],
    ) -> tuple[str, Any | None]:
        """Load + sanitize parquet table with a per-summary cache."""
        cache_key = str(parquet_path.resolve(strict=False)).casefold()
        if cache_key in table_cache:
            return "ok", table_cache[cache_key]

        try:
            import pandas as pd
        except Exception:
            return "pandas_unavailable", None

        try:
            table_df = pd.read_parquet(parquet_path)
        except Exception:
            logger.debug("Failed to read pattern parquet: %s", parquet_path, exc_info=True)
            return "read_error", None

        table_df = table_df.where(table_df.notna(), None)
        for col in table_df.columns:
            raw = table_df[col].to_numpy()
            table_df[col] = [make_hashable(v) for v in raw]

        table_cache[cache_key] = table_df
        return "ok", table_df

    def _probe_candidate_pattern_mismatch(
        self,
        *,
        candidate_pattern_obj: Any | None,
        pattern_match: PatternMatchResult,
        model_opsets: dict[ONNXDomain, int],
    ) -> tuple[bool, str | None]:
        """Probe candidate pattern preconditions via get_internal_constants_and_attributes.

        If a pattern explicitly raises PatternMismatchedError for this match,
        we stop before parquet lookup and surface the mismatch reason directly.
        """
        if candidate_pattern_obj is None:
            return False, None

        try:
            schema = candidate_pattern_obj.get_schema()
        except Exception:
            return False, None

        inputs: dict[str, np.ndarray] = {}
        is_constant_map: dict[str, bool] = {}

        for input_param in schema.inputs:
            input_name = input_param.name
            info = pattern_match.input_infos.get(input_name)

            # Missing/unknown input facts means probe is inconclusive.
            if info is None:
                return False, None

            is_constant_map[input_name] = info.is_constant

            if info.value is not None:
                inputs[input_name] = info.value
                continue

            if info.shape is None:
                return False, None

            safe_shape = tuple(
                int(dim) if isinstance(dim, (int, np.integer)) and int(dim) > 0 else 1
                for dim in info.shape
            )
            inputs[input_name] = np.zeros(safe_shape, dtype=np.float32)

        try:
            candidate_pattern_obj.get_internal_constants_and_attributes(
                inputs=inputs,
                attributes=pattern_match.attributes,
                is_constant_map=is_constant_map,
                domain_versions=model_opsets,
            )
        except PatternMismatchedError as mismatch_error:
            return True, str(mismatch_error)
        except Exception:
            logger.debug(
                "Candidate mismatch probe failed for %s; continue parquet lookup",
                candidate_pattern_obj.__class__.__name__,
                exc_info=True,
            )

        return False, None

    def _query_pattern_rule_compile_run_for_match(
        self,
        *,
        parquet_path: Path,
        pattern_match: PatternMatchResult,
        candidate_pattern_name: str,
        model_opsets: dict[ONNXDomain, int],
        table_cache: dict[str, Any],
        opset_signature: tuple[tuple[str, int], ...],
        query_lookup_cache: dict[
            tuple[str, str, tuple[tuple[str, Any], ...]],
            tuple[
                str,
                bool | None,
                bool | None,
                int,
                int,
                int,
                list[Any] | None,
                int,
                list[str],
                RuntimeDebugDetails | None,
            ],
        ],
    ) -> tuple[
        str,
        bool | None,
        bool | None,
        int,
        int,
        int,
        list[Any] | None,
        int,
        list[str],
        RuntimeDebugDetails | None,
    ]:
        """Query one candidate parquet table using one match's constraints."""
        result: tuple[
            str,
            bool | None,
            bool | None,
            int,
            int,
            int,
            list[Any] | None,
            int,
            list[str],
            RuntimeDebugDetails | None,
        ]
        from .runtime_checker_query import get_query_conditions_for_pattern, query_table_exact_match

        load_status, table_df = self._load_pattern_rule_table(parquet_path, table_cache)
        if load_status != "ok":
            return load_status, None, None, 0, 0, 0, None, 0, [], None
        if table_df is None:
            return "read_error", None, None, 0, 0, 0, None, 0, [], None

        row_count = len(table_df)
        if row_count == 0:
            return "empty_table", None, None, 0, 0, 0, None, 0, [], None

        if "compile_run_success" not in table_df.columns:
            return "missing_compile_run_success", None, None, row_count, 0, 0, None, 0, [], None

        match_identity = "|".join(str(key) for key in pattern_match.matched_node_keys)
        if not match_identity:
            match_identity = str(getattr(pattern_match, "match_id", ""))

        condition_build_cache_key = (
            match_identity,
            candidate_pattern_name,
            opset_signature,
        )
        cached_conditions = self._query_condition_build_cache.get(condition_build_cache_key)
        try:
            if cached_conditions is None:
                conditions, infinite_properties = get_query_conditions_for_pattern(
                    pattern_match=pattern_match,
                    pattern_name=candidate_pattern_name,
                    opset_versions=model_opsets,
                )
                self._query_condition_build_cache[condition_build_cache_key] = (
                    conditions,
                    infinite_properties,
                )
            else:
                conditions, infinite_properties = cached_conditions
        except Exception:
            logger.debug(
                "Failed to build query conditions for pattern '%s'",
                candidate_pattern_name,
                exc_info=True,
            )
            return "query_build_error", None, None, row_count, 0, 0, None, 0, [], None

        condition_columns = self._extract_rule_condition_columns(list(table_df.columns))
        query_conditions: dict[str, Any] = {}
        for col in condition_columns:
            if col in infinite_properties:
                continue
            if col not in conditions:
                return (
                    "query_key_missing",
                    None,
                    None,
                    row_count,
                    0,
                    0,
                    None,
                    len(query_conditions),
                    sorted(query_conditions.keys()),
                    None,
                )

            encoded_value = encode_rule_condition_value_for_parquet(conditions[col])
            query_conditions[col] = make_hashable(encoded_value)

        query_lookup_cache_key = (
            str(parquet_path.resolve(strict=False)).casefold(),
            candidate_pattern_name,
            tuple(sorted(query_conditions.items())),
        )
        cached_query_result = query_lookup_cache.get(query_lookup_cache_key)
        if cached_query_result is not None:
            return cached_query_result

        if query_conditions:
            matched_df = query_table_exact_match(table_df, query_conditions)
            if matched_df.empty:
                debug_steps: list[dict[str, Any]] = []
                current_df = table_df
                first_zero_column: str | None = None
                for col, value in query_conditions.items():
                    rows_before = len(current_df)
                    if col in current_df.columns:
                        current_df = current_df[current_df[col] == value]
                    rows_after = len(current_df)

                    debug_steps.append(
                        {
                            "column": col,
                            "value": repr(value),
                            "rows_before": rows_before,
                            "rows_after": rows_after,
                        }
                    )
                    if first_zero_column is None and rows_after == 0:
                        first_zero_column = col

                debug_details: RuntimeDebugDetails = {
                    "type": "properties_not_found",
                    "pattern_name": candidate_pattern_name,
                    "table_path": str(parquet_path.resolve(strict=False)),
                    "table_file": parquet_path.name,
                    "total_rows": row_count,
                    "query_condition_count": len(query_conditions),
                    "query_conditions": {
                        key: repr(value) for key, value in query_conditions.items()
                    },
                    "first_zero_column": first_zero_column,
                    "steps": debug_steps,
                }
                result = (
                    "properties_not_found",
                    None,
                    None,
                    row_count,
                    0,
                    0,
                    None,
                    len(query_conditions),
                    sorted(query_conditions.keys()),
                    debug_details,
                )
                query_lookup_cache[query_lookup_cache_key] = result
                return result
            matched_row = matched_df.iloc[0]
        else:
            matched_row = table_df.iloc[0]

        compile_run = self._normalize_compile_run_cell(matched_row.get("compile_run_success"))
        if compile_run is None:
            return (
                "invalid_compile_run_success",
                None,
                None,
                row_count,
                0,
                0,
                None,
                len(query_conditions),
                sorted(query_conditions.keys()),
                None,
            )

        compile_ok, run_ok = compile_run
        result = (
            "ok",
            compile_ok,
            run_ok,
            row_count,
            int(compile_ok),
            int(run_ok),
            self._normalize_case_indices(matched_row.get("case_indices")),
            len(query_conditions),
            sorted(query_conditions.keys()),
            None,
        )
        query_lookup_cache[query_lookup_cache_key] = result
        return result

    @staticmethod
    def _canonical_supported_status(value: str | None) -> str:
        """Normalize support labels to canonical lowercase values."""
        if value is None:
            return "unknown"

        normalized = str(value).strip().lower()
        if normalized == "unknow":
            return "unknown"
        if normalized in {"supported", "partial", "unsupported", "unknown"}:
            return normalized
        return "unknown"

    @classmethod
    def _supported_status_rank(cls, value: str | None) -> int:
        """Rank support labels for descending preference ordering."""
        status = cls._canonical_supported_status(value)
        rank_map = {
            "supported": 3,
            "partial": 2,
            "unsupported": 1,
            "unknown": 0,
        }
        return rank_map.get(status, 0)

    @classmethod
    def _candidate_supported_status(
        cls,
        candidate: PatternRuleCompileRunResult | None,
    ) -> str:
        """Derive support status from one candidate compile/run snapshot."""
        if candidate is None:
            return "unknown"

        if candidate.get("status") not in {"ok", "local_ep_check"}:
            return "unknown"

        compile_ok = bool(candidate.get("compile"))
        run_ok = bool(candidate.get("run"))

        if compile_ok and run_ok:
            return "supported"
        if (not compile_ok) and run_ok:
            return "partial"
        return "unsupported"

    @classmethod
    def _match_supported_status_from_candidates(
        cls,
        *,
        pattern_id: str,
        candidate_results: list[PatternRuleCompileRunResult],
    ) -> str:
        """Derive one support status for a pattern match from candidate snapshots."""
        base_candidate = next(
            (
                candidate
                for candidate in candidate_results
                if not bool(candidate.get("is_alternative", False))
                and str(candidate.get("pattern_id", "")) == pattern_id
            ),
            None,
        )
        if base_candidate is not None:
            return cls._candidate_supported_status(base_candidate)

        first_non_alternative = next(
            (
                candidate
                for candidate in candidate_results
                if not bool(candidate.get("is_alternative", False))
            ),
            None,
        )
        if first_non_alternative is not None:
            return cls._candidate_supported_status(first_non_alternative)

        if candidate_results:
            return cls._candidate_supported_status(candidate_results[0])

        return "unknown"

    @staticmethod
    def _priority_sort_key(priority: Any) -> int:
        """Convert alternative priority to sortable integer (smaller is better)."""
        if isinstance(priority, bool):
            return int(priority)
        if isinstance(priority, int):
            return priority
        if isinstance(priority, str):
            try:
                return int(priority)
            except ValueError:
                pass
        return 1_000_000

    @staticmethod
    def _derive_pattern_class_from_id(pattern_id: str) -> str:
        """Fallback pattern class from pattern id suffix."""
        if "/" not in pattern_id:
            return pattern_id
        return pattern_id.split("/")[-1]

    @classmethod
    def _find_alternative_candidate(
        cls,
        *,
        candidate_results: list[PatternRuleCompileRunResult],
        alt_pattern_id: str,
        alt_pattern_class: str,
    ) -> PatternRuleCompileRunResult | None:
        """Find candidate snapshot for one configured alternative."""
        strict_match = next(
            (
                candidate
                for candidate in candidate_results
                if bool(candidate.get("is_alternative", False))
                and str(candidate.get("pattern_id", "")) == alt_pattern_id
                and str(candidate.get("pattern_class", "")) == alt_pattern_class
            ),
            None,
        )
        if strict_match is not None:
            return strict_match

        return next(
            (
                candidate
                for candidate in candidate_results
                if bool(candidate.get("is_alternative", False))
                and str(candidate.get("pattern_id", "")) == alt_pattern_id
            ),
            None,
        )

    @classmethod
    def _select_and_filter_alternatives(
        cls,
        *,
        alternatives_meta: list[dict[str, Any]],
        candidate_results: list[PatternRuleCompileRunResult],
    ) -> tuple[list[dict[str, Any]], list[PatternRuleCompileRunResult]]:
        """Keep only one best alternative and drop unsupported-selected branches.

        Selection keys:
            1) supported_status rank: supported > partial > unsupported > unknown
            2) priority: smaller integer first

        After selecting the top alternative, if its status is ``unsupported``,
        remove alternatives entirely for this pattern match.
        """
        if not alternatives_meta:
            base_candidates = [
                candidate
                for candidate in candidate_results
                if not bool(candidate.get("is_alternative", False))
            ]
            return [], base_candidates

        ranked_alternatives: list[
            tuple[
                int,
                int,
                str,
                str,
                dict[str, Any],
                PatternRuleCompileRunResult | None,
                str,
            ]
        ] = []

        for alternative in alternatives_meta:
            alt_pattern_id = str(alternative.get("pattern_to_id", ""))
            alt_pattern_class = str(
                alternative.get("pattern_class")
                or cls._derive_pattern_class_from_id(alt_pattern_id)
            )
            matched_candidate = cls._find_alternative_candidate(
                candidate_results=candidate_results,
                alt_pattern_id=alt_pattern_id,
                alt_pattern_class=alt_pattern_class,
            )
            alt_status = cls._candidate_supported_status(matched_candidate)

            ranked_alternatives.append(
                (
                    cls._supported_status_rank(alt_status),
                    cls._priority_sort_key(alternative.get("priority")),
                    alt_pattern_id,
                    alt_pattern_class,
                    alternative,
                    matched_candidate,
                    alt_status,
                )
            )

        ranked_alternatives.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
                item[3],
            )
        )

        best_status = ranked_alternatives[0][6]
        selected_alternatives: list[dict[str, Any]] = []
        selected_candidate: PatternRuleCompileRunResult | None = None

        if best_status != "unsupported":
            selected_alternatives = [ranked_alternatives[0][4]]
            selected_candidate = ranked_alternatives[0][5]

        filtered_candidates: list[PatternRuleCompileRunResult] = []
        for candidate in candidate_results:
            if not bool(candidate.get("is_alternative", False)):
                filtered_candidates.append(candidate)
                continue

            if selected_candidate is not None and candidate is selected_candidate:
                filtered_candidates.append(candidate)

        return selected_alternatives, filtered_candidates

    def _build_merge_prep_metadata(
        self,
        *,
        subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]],
        model_signature: str,
        ep: EPNameOrAlias | None,
        device: str | None,
        for_debug: bool,
        on_pattern_query_result: Callable[[str, str], None] | None = None,
        local_pattern_checker: (
            Callable[[PatternMatchResult, str, bool], RuntimeTestResult | None] | None
        ) = None,
    ) -> list[PatternMergePrepEntry]:
        """Build alternatives + parquet compile/run snapshots for merge/dedup preparation."""
        if ep is None or device is None:
            return []

        ep_name = str(ep)
        device_name = device.upper()
        parquet_lookup_supported = self._is_valid_parquet_lookup_target(
            ep_name,
            device_name,
        )
        if not parquet_lookup_supported and local_pattern_checker is None:
            logger.info(
                "Skip pattern parquet lookup for invalid EP/device pair: %s_%s",
                ep_name,
                device_name,
            )
            return []

        cache_key = (
            model_signature,
            ep_name,
            device_name,
            bool(for_debug),
            local_pattern_checker is not None,
        )
        cached_merge_prep = self._MERGE_PREP_CACHE.get(cache_key)
        if cached_merge_prep is not None:
            cloned = copy.deepcopy(cached_merge_prep)

            def _emit_cached_pattern_query_result(pattern_id: str, support_status: str) -> None:
                if on_pattern_query_result is None:
                    return
                try:
                    on_pattern_query_result(pattern_id, support_status)
                except Exception:
                    logger.debug("on_pattern_query_result callback failed", exc_info=True)

            if on_pattern_query_result is not None:
                for entry in cloned:
                    _emit_cached_pattern_query_result(
                        str(entry.get("pattern_id", "")),
                        str(entry.get("support_status", "unknown")),
                    )
            return cloned

        model_opsets = ONNXDomain.get_model_domain_opset_versions(self._model.get_model())
        source_configs: dict[str, UnifiedPatternConfig] = {}
        entries: list[PatternMergePrepEntry] = []
        table_cache: dict[str, Any] = {}
        query_lookup_cache: dict[
            tuple[str, str, tuple[tuple[str, Any], ...]],
            tuple[
                str,
                bool | None,
                bool | None,
                int,
                int,
                int,
                list[Any] | None,
                int,
                list[str],
                RuntimeDebugDetails | None,
            ],
        ] = {}
        parquet_resolution_cache: dict[
            tuple[str, str, str, str, int, bool],
            tuple[Path | None, str | None, int | None],
        ] = {}
        opset_signature = tuple(
            sorted((domain.value, int(version)) for domain, version in model_opsets.items())
        )

        for source, source_group in sorted(subgraph_patterns_by_source.items()):
            if not source_group:
                continue

            config = source_configs.get(source)
            if config is None:
                config = UnifiedPatternConfig(ihv_type=source)
                source_configs[source] = config

            for pattern_class, matches in sorted(source_group.items()):
                if not matches:
                    continue

                representative = matches[0]
                pattern_obj = representative.pattern
                pattern_id = pattern_obj.pattern_id

                config_alternatives = config.get_alternatives(pattern_obj)
                alternatives_meta = [
                    {
                        "pattern_to_id": alt.pattern_to_id,
                        "pattern_class": alt.pattern_class,
                        "priority": alt.priority,
                        "enabled": alt.enabled,
                        "details": alt.details,
                        "reason": alt.reason,
                        "action_items": alt.action_items,
                    }
                    for alt in config_alternatives
                ]

                alternative_priority_by_key: dict[tuple[str, str], int] = {}
                alternative_priority_by_id: dict[str, int] = {}
                for alternative in alternatives_meta:
                    alt_pattern_id = str(alternative.get("pattern_to_id", ""))
                    alt_pattern_class = str(
                        alternative.get("pattern_class")
                        or self._derive_pattern_class_from_id(alt_pattern_id)
                    )
                    priority = self._priority_sort_key(alternative.get("priority"))

                    key = (alt_pattern_id, alt_pattern_class)
                    previous_priority = alternative_priority_by_key.get(key)
                    if previous_priority is None or priority < previous_priority:
                        alternative_priority_by_key[key] = priority

                    previous_id_priority = alternative_priority_by_id.get(alt_pattern_id)
                    if previous_id_priority is None or priority < previous_id_priority:
                        alternative_priority_by_id[alt_pattern_id] = priority

                candidate_specs: list[tuple[str, str, bool, Any | None]] = [
                    (pattern_class, pattern_id, False, pattern_obj)
                ]
                seen_candidates: set[tuple[str, str]] = {(pattern_class, pattern_id)}

                for alt in config_alternatives:
                    alt_pattern_class = alt.pattern_class or alt.pattern_to_id.split("/")[-1]
                    alt_pattern_id = alt.pattern_to_id
                    dedup_key = (alt_pattern_class, alt_pattern_id)
                    if dedup_key in seen_candidates:
                        continue
                    seen_candidates.add(dedup_key)

                    alt_pattern_obj: Any | None = None
                    if alt.pattern_class and alt.module:
                        try:
                            alt_pattern_obj = PatternConfig(
                                pattern_id=alt_pattern_id,
                                pattern_class=alt.pattern_class,
                                module=alt.module,
                                enabled=True,
                            ).load_pattern()
                        except Exception:
                            logger.debug(
                                "Failed to load alternative pattern %s from %s",
                                alt.pattern_class,
                                alt.module,
                                exc_info=True,
                            )

                    candidate_specs.append(
                        (alt_pattern_class, alt_pattern_id, True, alt_pattern_obj)
                    )

                candidate_runtime_specs: list[tuple[str, str, bool, Any | None, str, int]] = []
                for candidate_class, candidate_id, is_alt, candidate_pattern_obj in candidate_specs:
                    if candidate_pattern_obj is not None:
                        preferred_domain, target_opset = self._domain_and_target_opset_for_pattern(
                            candidate_pattern_obj,
                            model_opsets,
                        )
                    else:
                        preferred_domain = ONNXDomain.AI_ONNX.value
                        target_opset = model_opsets.get(ONNXDomain.AI_ONNX, 1)

                    candidate_runtime_specs.append(
                        (
                            candidate_class,
                            candidate_id,
                            is_alt,
                            candidate_pattern_obj,
                            preferred_domain,
                            int(target_opset),
                        )
                    )

                base_candidate_runtime_specs = [
                    spec for spec in candidate_runtime_specs if not spec[2]
                ]
                alternative_runtime_specs = [spec for spec in candidate_runtime_specs if spec[2]]

                def _alternative_runtime_sort_key(
                    runtime_spec: tuple[str, str, bool, Any | None, str, int],
                    _priority_by_key: dict[tuple[str, str], int] = alternative_priority_by_key,
                    _priority_by_id: dict[str, int] = alternative_priority_by_id,
                ) -> tuple[int, str, str]:
                    candidate_class, candidate_id, *_ = runtime_spec
                    priority = _priority_by_key.get(
                        (candidate_id, candidate_class),
                        _priority_by_id.get(candidate_id, 1_000_000),
                    )
                    return (priority, candidate_id, candidate_class)

                ordered_candidate_runtime_specs = base_candidate_runtime_specs + sorted(
                    alternative_runtime_specs,
                    key=_alternative_runtime_sort_key,
                )

                for match_index, pattern_match in enumerate(matches, start=1):
                    candidate_results: list[PatternRuleCompileRunResult] = []
                    for (
                        candidate_class,
                        candidate_id,
                        is_alt,
                        candidate_pattern_obj,
                        preferred_domain,
                        target_opset,
                    ) in ordered_candidate_runtime_specs:
                        is_mismatch, mismatch_error = self._probe_candidate_pattern_mismatch(
                            candidate_pattern_obj=candidate_pattern_obj,
                            pattern_match=pattern_match,
                            model_opsets=model_opsets,
                        )

                        if is_mismatch:
                            candidate_result: PatternRuleCompileRunResult = {
                                "pattern_class": candidate_class,
                                "pattern_id": candidate_id,
                                "is_alternative": is_alt,
                                "status": "mismatch_error",
                                "mismatch_error": mismatch_error,
                                "compile": None,
                                "run": None,
                                "row_count": 0,
                                "table_file": None,
                                "table_path": None,
                                "domain": None,
                                "opset_version": None,
                                "compile_true_rows": 0,
                                "run_true_rows": 0,
                                "case_indices": None,
                                "query_condition_count": 0,
                                "query_condition_keys": [],
                                "debug_details": None,
                            }
                        else:
                            resolution_cache_key = (
                                candidate_class,
                                ep_name,
                                device_name,
                                preferred_domain,
                                target_opset,
                                bool(for_debug),
                            )

                            resolved = parquet_resolution_cache.get(resolution_cache_key)
                            if resolved is None:
                                if parquet_lookup_supported:
                                    resolved = self._resolve_pattern_rule_table(
                                        pattern_class=candidate_class,
                                        ep_name=ep_name,
                                        device=device_name,
                                        preferred_domain=preferred_domain,
                                        target_opset=target_opset,
                                        for_debug=for_debug,
                                    )
                                else:
                                    resolved = (None, preferred_domain, target_opset)
                                parquet_resolution_cache[resolution_cache_key] = resolved

                            table_path, resolved_domain, resolved_opset = resolved

                            if table_path is None:
                                candidate_result = {
                                    "pattern_class": candidate_class,
                                    "pattern_id": candidate_id,
                                    "is_alternative": is_alt,
                                    "status": "table_not_found",
                                    "mismatch_error": None,
                                    "compile": None,
                                    "run": None,
                                    "row_count": 0,
                                    "table_file": None,
                                    "table_path": None,
                                    "domain": resolved_domain,
                                    "opset_version": resolved_opset,
                                    "compile_true_rows": 0,
                                    "run_true_rows": 0,
                                    "case_indices": None,
                                    "query_condition_count": 0,
                                    "query_condition_keys": [],
                                    "debug_details": None,
                                }
                            else:
                                candidate_pattern_name = (
                                    candidate_pattern_obj.__class__.__name__
                                    if candidate_pattern_obj is not None
                                    else candidate_class
                                )

                                (
                                    status,
                                    compile_ok,
                                    run_ok,
                                    row_count,
                                    compile_true_rows,
                                    run_true_rows,
                                    case_indices,
                                    query_condition_count,
                                    query_condition_keys,
                                    debug_details,
                                ) = self._query_pattern_rule_compile_run_for_match(
                                    parquet_path=table_path,
                                    pattern_match=pattern_match,
                                    candidate_pattern_name=candidate_pattern_name,
                                    model_opsets=model_opsets,
                                    table_cache=table_cache,
                                    opset_signature=opset_signature,
                                    query_lookup_cache=query_lookup_cache,
                                )

                                candidate_result = {
                                    "pattern_class": candidate_class,
                                    "pattern_id": candidate_id,
                                    "is_alternative": is_alt,
                                    "status": status,
                                    "mismatch_error": None,
                                    "compile": compile_ok,
                                    "run": run_ok,
                                    "row_count": row_count,
                                    "table_file": table_path.name,
                                    "table_path": str(table_path.resolve(strict=False)),
                                    "domain": resolved_domain,
                                    "opset_version": resolved_opset,
                                    "compile_true_rows": compile_true_rows,
                                    "run_true_rows": run_true_rows,
                                    "case_indices": case_indices,
                                    "query_condition_count": query_condition_count,
                                    "query_condition_keys": query_condition_keys,
                                    "debug_details": debug_details,
                                }

                        if (
                            not is_alt
                            and self._candidate_supported_status(candidate_result) == "unknown"
                            and local_pattern_checker is not None
                        ):
                            fallback_reason = str(candidate_result["status"])
                            local_result = local_pattern_checker(
                                pattern_match,
                                fallback_reason,
                                for_debug,
                            )
                            if local_result is not None:
                                candidate_result.update(
                                    {
                                        "status": "local_ep_check",
                                        "compile": local_result.compile,
                                        "run": local_result.run,
                                        "compile_true_rows": int(local_result.compile),
                                        "run_true_rows": int(local_result.run),
                                        "debug_details": local_result.debug_details,
                                    }
                                )

                        candidate_results.append(candidate_result)

                        # Alternatives are processed by priority, so the first
                        # supported alternative is already the optimal pick.
                        if (
                            is_alt
                            and self._candidate_supported_status(candidate_result) == "supported"
                        ):
                            break

                    (
                        filtered_alternatives,
                        filtered_candidates,
                    ) = self._select_and_filter_alternatives(
                        alternatives_meta=alternatives_meta,
                        candidate_results=candidate_results,
                    )

                    support_status = self._match_supported_status_from_candidates(
                        pattern_id=pattern_id,
                        candidate_results=filtered_candidates,
                    )

                    entries.append(
                        {
                            "source": source,
                            "pattern_class": pattern_class,
                            "pattern_id": pattern_id,
                            "match_count": len(matches),
                            "match_index": match_index,
                            "match_id": pattern_match.match_id,
                            "matched_node_keys": list(pattern_match.matched_node_keys),
                            "support_status": support_status,
                            "alternatives": filtered_alternatives,
                            "candidates": filtered_candidates,
                        }
                    )

                    if on_pattern_query_result is not None:
                        try:
                            on_pattern_query_result(pattern_id, support_status)
                        except Exception:
                            logger.debug("on_pattern_query_result callback failed", exc_info=True)

        self._MERGE_PREP_CACHE[cache_key] = copy.deepcopy(entries)
        return entries

    @staticmethod
    def _normalize_optimization_action_items(
        action_items: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Normalize optimization action items to snake_case option keys."""
        normalized_items: list[dict[str, Any]] = []
        for raw_item in action_items or []:
            raw_options = raw_item.get("optimization_options")
            if not isinstance(raw_options, dict) or not raw_options:
                continue

            normalized_options: dict[str, bool] = {}
            for option_key, option_value in raw_options.items():
                if isinstance(option_value, bool):
                    normalized_options[str(option_key).replace("-", "_")] = option_value

            if not normalized_options:
                continue

            normalized_items.append(
                {
                    "type": str(raw_item.get("type", "GraphOptimization")),
                    "optimization_options": normalized_options,
                }
            )

        return normalized_items

    def _build_pattern_optimization_hints(
        self,
        *,
        subgraph_patterns: list[PatternMatchResult],
        pattern_count_dict: dict[str, int],
    ) -> list[PatternOptimizationHint]:
        """Collect fallback optimization hints from matched pattern alternatives.

        For each matched pattern ID, pick the first enabled alternative that
        carries optimization_options and export those action_items.
        """
        hints: list[PatternOptimizationHint] = []
        processed_pattern_ids: set[str] = set()
        source_config_cache: dict[str, UnifiedPatternConfig] = {}

        for pattern_match in subgraph_patterns:
            pattern_id = str(pattern_match.pattern.pattern_id)
            if not pattern_id or pattern_id in processed_pattern_ids:
                continue

            source = str(pattern_match.attributes.get("source", "default")).strip().lower()
            if not source:
                source = "default"

            source_config = source_config_cache.get(source)
            if source_config is None:
                source_config = UnifiedPatternConfig(ihv_type=source)
                source_config_cache[source] = source_config

            alternatives = source_config.get_alternatives(pattern_match.pattern)
            selected_alternative: PatternAlternative | None = None
            selected_action_items: list[dict[str, Any]] = []

            for alternative in alternatives:
                if not alternative.enabled:
                    continue

                normalized_action_items = self._normalize_optimization_action_items(
                    alternative.action_items,
                )
                if not normalized_action_items:
                    continue

                selected_alternative = alternative
                selected_action_items = normalized_action_items
                break

            if selected_alternative is None or not selected_action_items:
                continue

            hints.append(
                {
                    "source": source,
                    "pattern_id": pattern_id,
                    "pattern_to_id": selected_alternative.pattern_to_id,
                    "instances": int(pattern_count_dict.get(pattern_id, 0)),
                    "enabled": bool(selected_alternative.enabled),
                    "details": selected_alternative.details,
                    "reason": selected_alternative.reason,
                    "action_items": selected_action_items,
                }
            )
            processed_pattern_ids.add(pattern_id)

        return hints

    def summary(
        self,
        ep: EPNameOrAlias | None = None,
        device: str | None = None,
        for_debug: bool = False,
        on_pattern_query_start: Callable[[Mapping[str, int], bool], None] | None = None,
        on_pattern_query_result: Callable[[str, str], None] | None = None,
        local_pattern_checker: (
            Callable[[PatternMatchResult, str, bool], RuntimeTestResult | None] | None
        ) = None,
    ) -> PatternSummary:
        """Generate comprehensive pattern analysis summary.

        Returns:
            PatternSummary with keys:
                - summary: ModelStats (from model_summary())
                - subgraph_patterns: List[PatternMatchResult]
                  (skeleton extraction + EP-priority dedup)
        """
        logger.info("Generating pattern analysis summary")
        total_start = time.perf_counter()

        model_signature = self._compute_model_signature()
        sources = self._resolve_sources_for_ep(ep)

        subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]] = {}
        source_stats: list[PatternSourceStat] = []

        for source in sources:
            grouped_matches, stat = self._extract_skeleton_matches_for_source(
                source=source,
                model_signature=model_signature,
            )
            subgraph_patterns_by_source[source] = grouped_matches
            source_stats.append(stat)

        (
            subgraph_patterns_by_source,
            subgraph_patterns,
        ) = self._dedup_grouped_matches_for_ep(
            subgraph_patterns_by_source=subgraph_patterns_by_source,
            sources=sources,
            model_signature=model_signature,
            ep=ep,
        )

        # Build pattern count dict: pattern_id -> count
        count_dict_start = time.perf_counter()
        pattern_count_dict: dict[str, int] = {}
        for pattern_match in subgraph_patterns:
            pattern_id = pattern_match.pattern.pattern_id
            pattern_count_dict[pattern_id] = pattern_count_dict.get(pattern_id, 0) + 1
        count_dict_ms = int((time.perf_counter() - count_dict_start) * 1000)

        # Pattern matching is EP-specific, so preserve the owning EP in metadata.
        detected_pattern_count: dict[str, dict[str, int]] = {}
        if ep is not None:
            detected_pattern_count[str(ep)] = pattern_count_dict
        metadata = self.model_summary(detected_pattern_count=detected_pattern_count)

        parquet_lookup_supported = True
        if ep is not None and device is not None:
            parquet_lookup_supported = self._is_valid_parquet_lookup_target(
                str(ep),
                str(device).upper(),
            )

        pattern_optimization_hints = self._build_pattern_optimization_hints(
            subgraph_patterns=subgraph_patterns,
            pattern_count_dict=pattern_count_dict,
        )

        if on_pattern_query_start is not None:
            try:
                pattern_check_supported = (
                    parquet_lookup_supported or local_pattern_checker is not None
                )
                on_pattern_query_start(pattern_count_dict, pattern_check_supported)
            except Exception:
                logger.debug("on_pattern_query_start callback failed", exc_info=True)

        _log_timing(
            "pattern_extractor.summary",
            model=self._model.model_path,
            detected_subgraph_patterns=len(subgraph_patterns),
            unique_pattern_ids=len(pattern_count_dict),
            extract_subgraph_ms=sum(stat["elapsed_ms"] for stat in source_stats),
            build_count_dict_ms=count_dict_ms,
            total_ms=int((time.perf_counter() - total_start) * 1000),
        )

        merge_prep = self._build_merge_prep_metadata(
            subgraph_patterns_by_source=subgraph_patterns_by_source,
            model_signature=model_signature,
            ep=ep,
            device=device,
            for_debug=for_debug,
            on_pattern_query_result=on_pattern_query_result,
            local_pattern_checker=local_pattern_checker,
        )
        return {
            "summary": metadata,
            "subgraph_patterns": subgraph_patterns,
            "subgraph_patterns_by_source": subgraph_patterns_by_source,
            "source_stats": source_stats,
            "merge_prep": merge_prep,
            "model_signature": model_signature,
            "parquet_lookup_supported": parquet_lookup_supported,
            "pattern_optimization_hints": pattern_optimization_hints,
        }

    def model_summary(
        self,
        detected_pattern_count: dict[str, dict[str, int]] | None = None,
    ) -> ModelStats:
        """Get model metadata and statistics.

        Args:
            detected_pattern_count: EP to pattern ID count mapping (default: empty dict)

        Returns:
            ModelStats object containing model information
        """
        return extract_model_stats(
            self._model,
            detected_pattern_count=detected_pattern_count,
        )
