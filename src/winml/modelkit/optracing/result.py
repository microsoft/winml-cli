# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Op-tracing result dataclasses for structured profiling output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ..utils.constants import EPName


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
    model: str
    device: str
    tracing_level: str  # "basic" or "detail"
    operators: list[OperatorMetrics] = field(default_factory=list)

    # Optional metadata
    ep: EPName | Literal[""] = ""
    tracing_backend: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_samples: int = 0

    # Summary (model-level aggregates)
    summary: dict[str, Any] = field(default_factory=dict)

    # Per-operator multi-sample statistics
    statistics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Raw artifact paths
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to structured dict for JSON output."""
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
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
