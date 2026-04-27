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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    name: str  # QNN op type ("Conv2d", "LayerNorm")
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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
