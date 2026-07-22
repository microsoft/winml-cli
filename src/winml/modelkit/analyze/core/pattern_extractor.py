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
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from ...onnx import ONNXDomain
from ...pattern.base import InvalidPatternMatcherModelError, PatternMatcher, PatternMismatchedError
from ...pattern.config import PatternConfig, UnifiedPatternConfig
from ..models.onnx_model import ModelTag, ONNXModel
from ..models.output import extract_model_stats
from ..utils.model_utils import encode_rule_condition_value_for_parquet, make_hashable
from ..utils.rule_loader import get_runtime_rules_debug_search_dirs, get_runtime_rules_search_dirs
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
    merge_prep: list["PatternMergePrepEntry"]
    total_extract_ms: int
    model_signature: str


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


class PatternMergePrepEntry(TypedDict):
    """Derived metadata used by upcoming pattern merge/dedup stage."""

    source: str
    pattern_class: str
    pattern_id: str
    match_count: int
    match_index: int
    match_id: str
    matched_node_keys: list[str]
    alternatives: list[dict[str, Any]]
    candidates: list[PatternRuleCompileRunResult]


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
    _VALID_EP_DEVICE_PAIRS_CACHE: set[tuple[str, str]] | None = None

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
            same_domain_le = [c for c in candidates if c[1] == preferred_domain and c[2] <= target_opset]
            if same_domain_le:
                picked = max(same_domain_le, key=lambda c: c[2])
                return picked

            any_domain_le = [c for c in candidates if c[2] <= target_opset]
            if any_domain_le:
                picked = max(any_domain_le, key=lambda c: c[2])
                return picked

            same_domain_gt = [c for c in candidates if c[1] == preferred_domain and c[2] > target_opset]
            if same_domain_gt:
                picked = min(same_domain_gt, key=lambda c: c[2])
                return picked

            picked = min(candidates, key=lambda c: c[2])
            return picked

        return None, None, None

    @staticmethod
    def _normalize_compile_run_cell(value: Any) -> tuple[bool, bool] | None:
        """Normalize one `compile_run_success` cell to `(compile, run)` booleans."""
        raw_value = value
        if not isinstance(raw_value, (list, tuple)) and hasattr(raw_value, "tolist"):
            try:
                raw_value = raw_value.tolist()
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return "pandas_unavailable", None

        try:
            table_df = pd.read_parquet(parquet_path)
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    ) -> tuple[str, bool | None, bool | None, int, int, int, list[Any] | None, int, list[str]]:
        """Query one candidate parquet table using one match's constraints."""
        from .runtime_checker_query import get_query_conditions_for_pattern, query_table_exact_match

        load_status, table_df = self._load_pattern_rule_table(parquet_path, table_cache)
        if load_status != "ok":
            return load_status, None, None, 0, 0, 0, None, 0, []
        if table_df is None:
            return "read_error", None, None, 0, 0, 0, None, 0, []

        row_count = int(len(table_df))
        if row_count == 0:
            return "empty_table", None, None, 0, 0, 0, None, 0, []

        if "compile_run_success" not in table_df.columns:
            return "missing_compile_run_success", None, None, row_count, 0, 0, None, 0, []

        try:
            conditions, infinite_properties = get_query_conditions_for_pattern(
                pattern_match=pattern_match,
                pattern_name=candidate_pattern_name,
                opset_versions=model_opsets,
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Failed to build query conditions for pattern '%s'",
                candidate_pattern_name,
                exc_info=True,
            )
            return "query_build_error", None, None, row_count, 0, 0, None, 0, []

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
                )

            encoded_value = encode_rule_condition_value_for_parquet(conditions[col])
            query_conditions[col] = make_hashable(encoded_value)

        if query_conditions:
            matched_df = query_table_exact_match(table_df, query_conditions)
            if matched_df.empty:
                return (
                    "properties_not_found",
                    None,
                    None,
                    row_count,
                    0,
                    0,
                    None,
                    len(query_conditions),
                    sorted(query_conditions.keys()),
                )
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
            )

        compile_ok, run_ok = compile_run
        return (
            "ok",
            compile_ok,
            run_ok,
            row_count,
            int(compile_ok),
            int(run_ok),
            self._normalize_case_indices(matched_row.get("case_indices")),
            len(query_conditions),
            sorted(query_conditions.keys()),
        )

    def _build_merge_prep_metadata(
        self,
        *,
        subgraph_patterns_by_source: dict[str, dict[str, list[PatternMatchResult]]],
        ep: EPNameOrAlias | None,
        device: str | None,
        for_debug: bool,
    ) -> list[PatternMergePrepEntry]:
        """Build alternatives + parquet compile/run snapshots for merge/dedup preparation."""
        if ep is None or device is None:
            return []

        ep_name = str(ep)
        device_name = device.upper()
        if not self._is_valid_parquet_lookup_target(ep_name, device_name):
            logger.info(
                "Skip pattern parquet lookup for invalid EP/device pair: %s_%s",
                ep_name,
                device_name,
            )
            return []

        model_opsets = ONNXDomain.get_model_domain_opset_versions(self._model.get_model())
        source_configs: dict[str, UnifiedPatternConfig] = {}
        entries: list[PatternMergePrepEntry] = []
        table_cache: dict[str, Any] = {}

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
                    }
                    for alt in config_alternatives
                ]

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
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "Failed to load alternative pattern %s from %s",
                                alt.pattern_class,
                                alt.module,
                                exc_info=True,
                            )

                    candidate_specs.append(
                        (alt_pattern_class, alt_pattern_id, True, alt_pattern_obj)
                    )

                for match_index, pattern_match in enumerate(matches, start=1):
                    candidate_results: list[PatternRuleCompileRunResult] = []
                    for candidate_class, candidate_id, is_alt, candidate_pattern_obj in candidate_specs:
                        is_mismatch, mismatch_error = self._probe_candidate_pattern_mismatch(
                            candidate_pattern_obj=candidate_pattern_obj,
                            pattern_match=pattern_match,
                            model_opsets=model_opsets,
                        )
                        if is_mismatch:
                            candidate_results.append(
                                {
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
                                }
                            )
                            continue

                        if candidate_pattern_obj is not None:
                            preferred_domain, target_opset = self._domain_and_target_opset_for_pattern(
                                candidate_pattern_obj,
                                model_opsets,
                            )
                        else:
                            preferred_domain = ONNXDomain.AI_ONNX.value
                            target_opset = model_opsets.get(ONNXDomain.AI_ONNX, 1)

                        table_path, resolved_domain, resolved_opset = self._resolve_pattern_rule_table(
                            pattern_class=candidate_class,
                            ep_name=ep_name,
                            device=device_name,
                            preferred_domain=preferred_domain,
                            target_opset=target_opset,
                            for_debug=for_debug,
                        )

                        if table_path is None:
                            candidate_results.append(
                                {
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
                                }
                            )
                            continue

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
                        ) = self._query_pattern_rule_compile_run_for_match(
                            parquet_path=table_path,
                            pattern_match=pattern_match,
                            candidate_pattern_name=candidate_pattern_name,
                            model_opsets=model_opsets,
                            table_cache=table_cache,
                        )

                        candidate_results.append(
                            {
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
                            }
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
                            "alternatives": alternatives_meta,
                            "candidates": candidate_results,
                        }
                    )

        return entries

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
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("HTP metadata root must be a JSON object")
            self._htp_metadata = cast("HTPMetadata", loaded)
            logger.info("Successfully loaded HTP metadata")
            return cast("HTPMetadata", self._htp_metadata)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in HTP metadata file: {e}") from e

    def summary(
        self,
        ep: EPNameOrAlias | None = None,
        device: str | None = None,
        for_debug: bool = False,
    ) -> PatternSummary:
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

        merge_prep = (
            self._build_merge_prep_metadata(
                subgraph_patterns_by_source=subgraph_patterns_by_source,
                ep=ep,
                device=device,
                for_debug=for_debug,
            )
            if for_debug
            else []
        )

        total_extract_ms = int((time.perf_counter() - total_start) * 1000)
        return {
            "summary": metadata,
            "subgraph_patterns": subgraph_patterns,
            "subgraph_patterns_by_source": subgraph_patterns_by_source,
            "source_stats": source_stats,
            "merge_prep": merge_prep,
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
