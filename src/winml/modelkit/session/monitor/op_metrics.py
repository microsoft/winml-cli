# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpTraceResult + OperatorMetrics — structured profiling output.

Relocated from ``optracing/result.py`` as part of the op-tracing refactor.
Extended with ``status`` / ``error`` fields for failure reporting.
"""

from __future__ import annotations

import json
import statistics as _stats
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


#: Closed set of values for :attr:`OpTraceResult.status`.
#:
#: * ``"ok"`` — trace parsed cleanly.
#: * ``"no_data"`` — expected artifacts (e.g. profiling CSV) never appeared.
#: * ``"parse_failed"`` — artifacts were present but unparseable; ``error``
#:   carries the message.
#: * ``"basic_fallback"`` — caller asked for ``detail`` mode but the backend
#:   could only produce basic data (e.g. QHAS unavailable).
#: * ``"not_run"`` — :py:meth:`__exit__` has not been called yet.
#:
#: ``Literal`` is enforced statically (mypy / ruff); at runtime ``status`` is
#: still a plain ``str`` so :py:meth:`OpTraceResult.to_dict` and JSON
#: serialization are unaffected.
TraceStatus = Literal["ok", "no_data", "parse_failed", "basic_fallback", "not_run"]


@dataclass
class OperatorMetrics:
    """Per-operator profiling metrics."""

    # Identity
    name: str  # Resolved op type. Vocabulary depends on v2.4 fallback chain
    # in QNNMonitor._resolve_op_type:
    # L1: ONNX node.op_type (e.g. "Conv", "Add", "MaxPool")
    # L2: EP-authoritative (e.g. QHAS qnn_op_type "Conv2d")
    # L3: heuristic leaf-split
    # L4: raw op_path
    op_path: str  # Framework path ("/layer1/conv/Conv")
    op_id: int | None = None

    # P0: Temporal Localization
    start_time_us: float | None = None
    duration_us: float = 0.0
    percent_of_total: float = 0.0

    # P1: Roofline Analysis (detail only)
    hardware_time_us: float | None = None
    memory_time_us: float | None = None
    dominant_path_us: float | None = None

    # P2: DMA Traffic (detail only, per-op)
    dram_read_bytes: int | None = None
    dram_write_bytes: int | None = None
    vtcm_read_bytes: int | None = None
    vtcm_write_bytes: int | None = None

    # P3: Cache Efficiency (detail only, derived)
    vtcm_hit_ratio: float | None = None

    # Context
    num_htp_ops: int | None = None
    data_type: str | None = None
    dims: list[int] | None = None

    # Per-sample timings retained for downstream stats (p90, total, count).
    # Empty when source parser only produced an aggregated avg.
    samples_us: list[float] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Number of retained per-sample timings."""
        return len(self.samples_us)

    @property
    def avg_us(self) -> float:
        """Mean of ``samples_us`` (0.0 when empty)."""
        return sum(self.samples_us) / len(self.samples_us) if self.samples_us else 0.0

    @property
    def total_us(self) -> float:
        """Sum of ``samples_us`` (0.0 when empty)."""
        return sum(self.samples_us)

    @property
    def p90_us(self) -> float:
        """Inclusive 90th-percentile of ``samples_us`` (0.0 when empty)."""
        n = len(self.samples_us)
        if n == 0:
            return 0.0
        if n == 1:
            return self.samples_us[0]
        # statistics.quantiles with n=10 method="inclusive" gives 9 cut points;
        # index 8 is the 90th percentile.
        return _stats.quantiles(self.samples_us, n=10, method="inclusive")[8]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, preserving None for unavailable fields."""
        return asdict(self)


@dataclass
class OpTraceResult:
    """Complete op-tracing result."""

    # Required
    model: str | None
    device: str
    tracing_level: str  # "basic" or "detail"
    operators: list[OperatorMetrics] = field(default_factory=list)

    # Optional metadata
    ep: str = ""
    tracing_backend: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    num_samples: int = 0

    # Summary (model-level aggregates)
    summary: dict[str, Any] = field(default_factory=dict)

    # Per-operator multi-sample statistics
    statistics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Raw artifact paths
    artifacts: dict[str, str] = field(default_factory=dict)

    # Status of the trace. See :data:`TraceStatus` for the closed set of
    # legal values; static type checkers enforce the alias.
    status: TraceStatus = "ok"
    # Populated when status == "parse_failed".
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to structured dict.

        Preserves existing nested schema; adds top-level ``status`` and
        ``error`` keys additively.
        """
        return {
            "metadata": {
                "model": self.model,
                "device": self.device,
                "ep": self.ep,
                "tracing_level": self.tracing_level,
                "tracing_backend": self.tracing_backend,
                "timestamp": self.timestamp,
                "num_samples": self.num_samples,
            },
            "summary": self.summary,
            "operators": [op.to_dict() for op in self.operators],
            "statistics": self.statistics,
            "artifacts": self.artifacts,
            # ---- Additive ----
            "status": self.status,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
