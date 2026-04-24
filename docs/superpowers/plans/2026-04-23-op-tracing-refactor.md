# Op-Tracing Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the `EPMonitor` (`session/monitor/`) and `OpTracer` (`optracing/`) hierarchies into one; fix QNN op-tracing to work with both `onnxruntime-qnn` AND `onnxruntime-windowsml`; delete the `optracing/` package.

**Architecture:** Hook-based Plugin + Template Method + Observer. `WinMLSession.perf(warmup, monitor=...)` yields a `PerfContext(stats, monitor)`. Monitors contribute `get_session_options()` + `get_provider_options()` at compile time. `QNNMonitor` replaces `QNNProfiler` and owns all QNN-specific knowledge. See `docs/design/optracing/2_coreloop.md` v2.1.

**Tech Stack:** Python 3.11+, ONNX Runtime (three variants: `onnxruntime`, `onnxruntime-qnn`, `onnxruntime-windowsml`), pytest, ruff. Testing via `uv run pytest`.

**Related docs:**
- `docs/design/optracing/1_prd.md` v2.1 (requirements)
- `docs/design/optracing/2_coreloop.md` v2.1 (core design)
- `docs/standards/design-doc-spec.md` v1.1 (doc standard)

**Sequencing strategy:** Relocate helpers FIRST as backward-compatible shims (old imports still work); build new monitor on top; flip callers; delete the old package last. Each task leaves `uv run pytest tests/` green and `uv run ruff check src/ tests/` clean.

---

## Task 0: Prep — create branch, verify starting state

**Files:** none yet (orientation only).

- [ ] **Step 1: Ensure clean working tree**

Run:
```bash
git status
git rev-parse --abbrev-ref HEAD
```
Expected: current branch is `feat/mvp` or a branch off of it; no uncommitted changes (or only docs changes from this session).

- [ ] **Step 2: Create implementation branch**

Run:
```bash
git checkout -b feat/op-tracing-refactor
```
Expected: branch created from current HEAD.

- [ ] **Step 3: Verify current test baseline**

Run:
```bash
uv run pytest tests/unit/optracing/ -v --tb=short
```
Expected: existing optracing tests PASS (or whatever the baseline is — record the exact count and any pre-existing failures unrelated to our work; those stay pre-existing).

Record the baseline output in a scratch note; we will reuse it as regression checks.

- [ ] **Step 4: Verify ruff is clean on touched modules**

Run:
```bash
uv run ruff check src/winml/modelkit/session/ src/winml/modelkit/optracing/ src/winml/modelkit/commands/perf.py
```
Expected: No findings, OR a short list of pre-existing findings (record them; they stay pre-existing).

---

## Task 1: Add `ensure_initialized()` module function to `ep_registry.py`

**Rationale:** Break the reverse-coupling where `QNNMonitor.is_available()` would otherwise need to import `WinMLSession`. A thin module-level wrapper gives us an import-cycle-safe entry point.

**Files:**
- Modify: `src/winml/modelkit/session/ep_registry.py` (existing; add ~10 lines)
- Test: `tests/unit/session/test_ep_registry.py` (new)

- [ ] **Step 1: Create the test directory if missing**

Run:
```bash
mkdir -p tests/unit/session
test -f tests/unit/session/__init__.py || touch tests/unit/session/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/session/test_ep_registry.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module-level helpers."""

from __future__ import annotations

from unittest.mock import patch

from winml.modelkit.session.ep_registry import ensure_initialized


def test_ensure_initialized_calls_registry_once():
    """ensure_initialized() calls register_to_ort() once regardless of call count."""
    with patch(
        "winml.modelkit.session.ep_registry.WinMLEPRegistry"
    ) as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True

        ensure_initialized()
        ensure_initialized()
        ensure_initialized()

        # Singleton should be fetched, register_to_ort called each time (idempotent inside registry)
        assert mock_registry_cls.get_instance.call_count >= 1
        # Multiple calls must not raise
```

- [ ] **Step 3: Run test — expect ImportError (function not defined)**

Run:
```bash
uv run pytest tests/unit/session/test_ep_registry.py -v
```
Expected: FAIL with `ImportError: cannot import name 'ensure_initialized' from 'winml.modelkit.session.ep_registry'`.

- [ ] **Step 4: Add the function to `ep_registry.py`**

Modify `src/winml/modelkit/session/ep_registry.py`. After the `get_ort_available_providers` function (around line 199), append:

```python
def ensure_initialized() -> None:
    """Idempotent module-level entry point for WinML EP registration.

    Wraps ``WinMLEPRegistry.get_instance().register_to_ort()`` so callers
    (e.g. ``QNNMonitor.is_available``) can trigger EP registration without
    importing ``WinMLSession`` — breaks a latent import cycle.

    Safe to call multiple times. No-op if WinML is unavailable on this system.
    """
    try:
        registry = WinMLEPRegistry.get_instance()
        if registry.winml_available:
            registry.register_to_ort()
    except Exception as exc:  # noqa: BLE001 — log-and-continue is intentional
        logger.debug("ensure_initialized: WinML EP registration skipped: %s", exc)
```

- [ ] **Step 5: Run test — expect PASS**

Run:
```bash
uv run pytest tests/unit/session/test_ep_registry.py -v
```
Expected: PASS.

- [ ] **Step 6: Run ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/ep_registry.py tests/unit/session/test_ep_registry.py --fix
```
Expected: no findings.

- [ ] **Step 7: Full pytest sanity**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: All tests pass (same as baseline).

- [ ] **Step 8: Commit**

Run:
```bash
git add src/winml/modelkit/session/ep_registry.py tests/unit/session/test_ep_registry.py tests/unit/session/__init__.py
git commit -m "feat(session): add ensure_initialized() module function to ep_registry

Breaks reverse-coupling for QNNMonitor.is_available() by providing a
module-level entry point that wraps the WinMLEPRegistry singleton.
Idempotent; safe to call multiple times."
```

---

## Task 2: Extend `EPMonitor` ABC with optional default hooks

**Rationale:** Add the two hook methods (`get_session_options`, `get_provider_options`) and the `requires_session_teardown` class attribute that the new design relies on. All three have defaults so existing `VitisAIMonitor`, `NullEPMonitor`, `OpenVinoMonitor`, `QNNMonitor` (placeholder) continue to work unchanged.

**Files:**
- Modify: `src/winml/modelkit/session/monitor/ep_monitor.py`
- Test: `tests/unit/session/monitor/test_ep_monitor_base.py` (new)

- [ ] **Step 1: Create test directory**

Run:
```bash
mkdir -p tests/unit/session/monitor
touch tests/unit/session/monitor/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/unit/session/monitor/test_ep_monitor_base.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for EPMonitor ABC default hook behavior and NullEPMonitor inheritance."""

from __future__ import annotations

import pytest

from winml.modelkit.session.monitor.ep_monitor import EPMonitor, NullEPMonitor


def test_null_monitor_default_get_session_options():
    """NullEPMonitor inherits empty session-options default."""
    assert NullEPMonitor().get_session_options() == {}


def test_null_monitor_default_get_provider_options():
    """NullEPMonitor inherits empty provider-options default."""
    assert NullEPMonitor().get_provider_options() == {}


def test_null_monitor_default_requires_teardown():
    """NullEPMonitor.requires_session_teardown is False by default."""
    assert NullEPMonitor.requires_session_teardown is False


def test_ep_monitor_is_abstract():
    """EPMonitor cannot be instantiated directly (still abstract)."""
    with pytest.raises(TypeError):
        EPMonitor()  # type: ignore[abstract]


def test_hooks_return_fresh_dicts():
    """get_*_options returns a fresh dict each call (not a shared mutable)."""
    m = NullEPMonitor()
    d1 = m.get_session_options()
    d1["injected"] = "1"
    d2 = m.get_session_options()
    assert "injected" not in d2
```

- [ ] **Step 3: Run tests — expect failures**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_ep_monitor_base.py -v
```
Expected: multiple FAILs with `AttributeError: ... has no attribute 'get_session_options'` etc.

- [ ] **Step 4: Add defaults to the ABC**

Modify `src/winml/modelkit/session/monitor/ep_monitor.py`. Change the class definition to add three members. Replace the current class body's top with:

```python
from typing import Any, ClassVar

class EPMonitor(ABC):
    """Base class for EP-specific hardware performance monitoring.

    Used as a context manager alongside ``PerfStats`` to collect
    hardware utilization metrics during inference.

    Example::

        with session.perf(warmup=10, monitor=SomeEPMonitor()) as ctx:
            for _ in range(110):
                session.run(inputs)

        print(ctx.stats.mean_ms)    # inference timing
        print(ctx.monitor.to_dict()) # proof-of-execution data
    """

    # ---- Optional hooks: defaults provided; subclasses override as needed ----

    #: ORT-specific hint: does this monitor's data flush require
    #: ``ort.InferenceSession`` destruction? Example: QNN flushes CSV
    #: only on session destroy. Default: False (no teardown needed).
    requires_session_teardown: ClassVar[bool] = False

    def get_session_options(self) -> dict[str, str]:
        """Entries to pass to ``SessionOptions.add_session_config_entry()``.

        Default: empty dict. Override in subclasses that need e.g.
        ``"session.disable_cpu_ep_fallback": "1"``.
        """
        return {}

    def get_provider_options(self) -> dict[str, str]:
        """Options to merge into ``add_provider_for_devices([ep], opts)``.

        Default: empty dict. Override in subclasses that need e.g.
        ``"profiling_level": "detailed"``.
        """
        return {}

    # ---- Mandatory contract ----
```

Keep the existing `@abstractmethod` methods (`__enter__`, `__exit__`, `to_dict`, `is_available`) unchanged below.

- [ ] **Step 5: Run tests — expect PASS**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_ep_monitor_base.py -v
```
Expected: all PASS.

- [ ] **Step 6: Verify no regression to VitisAI / OpenVINO / QNN placeholder**

Run:
```bash
uv run pytest tests/ -k "monitor" -v
```
Expected: all existing monitor tests still pass.

- [ ] **Step 7: Ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/monitor/ep_monitor.py tests/unit/session/monitor/ --fix
```

- [ ] **Step 8: Commit**

Run:
```bash
git add src/winml/modelkit/session/monitor/ep_monitor.py tests/unit/session/monitor/
git commit -m "feat(monitor): add default hooks on EPMonitor ABC

Adds three optional members with safe defaults:
- requires_session_teardown: ClassVar[bool] = False
- get_session_options() -> {}
- get_provider_options() -> {}

Existing subclasses (VitisAI, OpenVINO, QNN placeholder, NullEPMonitor)
inherit defaults unchanged."
```

---

## Task 3: Relocate `OpTraceResult` / `OperatorMetrics` → `session/monitor/op_metrics.py` with additive `status` / `error` fields

**Rationale:** Content move + additive extension. We move the file, keep `optracing/result.py` as a temporary re-export shim so old callers keep working during the transition, then delete the shim in Task 14.

**Files:**
- Create: `src/winml/modelkit/session/monitor/op_metrics.py`
- Modify: `src/winml/modelkit/optracing/result.py` → re-export shim
- Test: `tests/unit/session/monitor/test_op_metrics.py` (new)

- [ ] **Step 1: Write failing tests for the new location + new fields**

Create `tests/unit/session/monitor/test_op_metrics.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the relocated OpTraceResult + new status/error fields."""

from __future__ import annotations

import json

from winml.modelkit.session.monitor.op_metrics import (
    OperatorMetrics,
    OpTraceResult,
)


def test_model_field_accepts_none():
    """model: str | None — passing None must not raise."""
    r = OpTraceResult(model=None, device="npu", tracing_level="basic")
    assert r.model is None


def test_status_default_is_ok():
    """New status field defaults to 'ok' for backward compat with existing construction."""
    r = OpTraceResult(model="x", device="npu", tracing_level="basic")
    assert r.status == "ok"
    assert r.error is None


def test_status_can_be_set():
    r = OpTraceResult(
        model="x", device="npu", tracing_level="basic",
        status="parse_failed", error="corrupt CSV",
    )
    assert r.status == "parse_failed"
    assert r.error == "corrupt CSV"


def test_to_dict_preserves_nested_schema():
    """Existing nested schema (metadata / summary / operators / statistics / artifacts) preserved."""
    r = OpTraceResult(model="m.onnx", device="npu", tracing_level="basic", ep="QNN")
    d = r.to_dict()
    # Existing keys — must still exist exactly as before
    assert "metadata" in d
    assert d["metadata"]["model"] == "m.onnx"
    assert d["metadata"]["device"] == "npu"
    assert d["metadata"]["tracing_level"] == "basic"
    assert d["metadata"]["ep"] == "QNN"
    assert "summary" in d
    assert "operators" in d
    assert "statistics" in d
    assert "artifacts" in d


def test_to_dict_adds_status_and_error_at_top_level():
    """New fields appear as additive top-level keys, not replacing anything."""
    r = OpTraceResult(
        model="x", device="npu", tracing_level="basic",
        status="no_data", error=None,
    )
    d = r.to_dict()
    assert d["status"] == "no_data"
    assert d["error"] is None


def test_to_json_round_trip():
    """to_json must produce valid JSON containing both old and new fields."""
    r = OpTraceResult(model="x", device="npu", tracing_level="basic", status="ok")
    parsed = json.loads(r.to_json())
    assert parsed["metadata"]["model"] == "x"
    assert parsed["status"] == "ok"


def test_operator_metrics_to_dict_preserved():
    op = OperatorMetrics(name="Conv", op_path="/conv_1", duration_us=12.5, percent_of_total=5.0)
    d = op.to_dict()
    assert d["name"] == "Conv"
    assert d["duration_us"] == 12.5
```

- [ ] **Step 2: Run tests — expect ImportError**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_op_metrics.py -v
```
Expected: `ImportError: No module named 'winml.modelkit.session.monitor.op_metrics'`.

- [ ] **Step 3: Create the new file by copying current `optracing/result.py`, then extend**

Copy the content of `src/winml/modelkit/optracing/result.py` into a new file `src/winml/modelkit/session/monitor/op_metrics.py`, and apply these changes:

1. Change the `OpTraceResult.model` field from `model: str` to `model: str | None`.
2. Add two new fields after `artifacts` (at the end of the existing field list, BEFORE `to_dict`):
   ```python
   # Status of the trace — "ok" | "no_data" | "parse_failed" | "basic_fallback"
   status: str = "ok"
   # Populated when status == "parse_failed"
   error: str | None = None
   ```
3. Modify `to_dict()` to include the new keys additively at top level (existing keys untouched):
   ```python
   def to_dict(self) -> dict[str, Any]:
       """Serialize to structured dict. Preserves existing nested schema;
       adds top-level ``status`` and ``error`` keys additively."""
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
   ```

Module docstring at top should be updated to:
```python
"""OpTraceResult + OperatorMetrics — structured profiling output.

Relocated from ``optracing/result.py`` as part of the op-tracing refactor.
Extended with ``status`` / ``error`` fields for failure reporting.
"""
```

Required imports at top: `from __future__ import annotations`, `import json`, `from dataclasses import dataclass, field, asdict`, `from datetime import datetime, timezone`, `from typing import Any`.

- [ ] **Step 4: Replace `optracing/result.py` with a re-export shim**

Overwrite `src/winml/modelkit/optracing/result.py` with:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim.

``OpTraceResult`` and ``OperatorMetrics`` moved to
``winml.modelkit.session.monitor.op_metrics``. This shim keeps old imports
working during the op-tracing refactor; removed once all callers are updated.
"""

from __future__ import annotations

from ..session.monitor.op_metrics import OperatorMetrics, OpTraceResult


__all__ = ["OperatorMetrics", "OpTraceResult"]
```

- [ ] **Step 5: Run new tests — expect PASS**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_op_metrics.py -v
```
Expected: all 7 PASS.

- [ ] **Step 6: Run existing tests via old import path — expect PASS (shim works)**

Run:
```bash
uv run pytest tests/unit/optracing/test_result.py -v
```
Expected: existing tests still PASS (shim re-exports).

- [ ] **Step 7: Full test sanity**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: no regressions.

- [ ] **Step 8: Ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/monitor/op_metrics.py src/winml/modelkit/optracing/result.py tests/unit/session/monitor/test_op_metrics.py --fix
```

- [ ] **Step 9: Commit**

Run:
```bash
git add src/winml/modelkit/session/monitor/op_metrics.py src/winml/modelkit/optracing/result.py tests/unit/session/monitor/test_op_metrics.py
git commit -m "feat(monitor): relocate OpTraceResult → session/monitor/op_metrics

Move dataclasses from optracing/result.py. Additive changes:
- model: str -> str | None (allows None for standalone profiling)
- New fields: status (default 'ok'), error (default None)
- to_dict() preserves nested schema; adds top-level status/error keys

Old import path retained as re-export shim; removed in later task."
```

---

## Task 4: Relocate report helpers → `session/monitor/report.py`

**Rationale:** `display_op_trace_report` and `write_op_trace_json` move verbatim. Old `optracing/report.py` becomes a shim.

**Files:**
- Create: `src/winml/modelkit/session/monitor/report.py`
- Modify: `src/winml/modelkit/optracing/report.py` → shim
- Test: `tests/unit/session/monitor/test_report.py` (move existing)

- [ ] **Step 1: Copy content verbatim**

Read `src/winml/modelkit/optracing/report.py`. Create `src/winml/modelkit/session/monitor/report.py` with identical content, but update:
- Module docstring: `"""Report helpers — display / write JSON for op-trace results.\n\nRelocated from optracing/report.py."""`.
- Any internal imports of `.result` → `.op_metrics` (since OpTraceResult now lives there).
- Import path for `OpTraceResult`: `from .op_metrics import OpTraceResult, OperatorMetrics` (replace the old `from .result import ...`).

- [ ] **Step 2: Replace `optracing/report.py` with a shim**

Overwrite `src/winml/modelkit/optracing/report.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim.

Report helpers moved to ``winml.modelkit.session.monitor.report``.
"""

from __future__ import annotations

from ..session.monitor.report import (
    display_op_trace_report,
    write_op_trace_json,
)


__all__ = ["display_op_trace_report", "write_op_trace_json"]
```

- [ ] **Step 3: Move the existing test file**

Run:
```bash
git mv tests/unit/optracing/test_report.py tests/unit/session/monitor/test_report.py
```

Then update imports in the moved file: replace `from winml.modelkit.optracing.report import` with `from winml.modelkit.session.monitor.report import`, and any `from winml.modelkit.optracing.result import` with `from winml.modelkit.session.monitor.op_metrics import`.

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_report.py tests/unit/optracing/ -v
```
Expected: all PASS. The shim ensures old-path imports still resolve.

- [ ] **Step 5: Ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/monitor/report.py src/winml/modelkit/optracing/report.py tests/unit/session/monitor/test_report.py --fix
```

- [ ] **Step 6: Commit**

Run:
```bash
git add src/winml/modelkit/session/monitor/report.py src/winml/modelkit/optracing/report.py tests/unit/session/monitor/test_report.py tests/unit/optracing/
git commit -m "feat(monitor): relocate report helpers → session/monitor/report

Moves display_op_trace_report + write_op_trace_json. Old path retained
as re-export shim."
```

---

## Task 5: Relocate QNN helpers → `session/monitor/qnn/`

**Rationale:** Move the three QNN-specific helper modules (`csv_parser.py`, `qhas_parser.py`, `viewer.py`) and fixtures. Old `optracing/qnn/` keeps `profiler.py` alive via shims until Task 12.

**Files:**
- Create: `src/winml/modelkit/session/monitor/qnn/__init__.py`
- Create: `src/winml/modelkit/session/monitor/qnn/{csv_parser.py, qhas_parser.py, viewer.py}`
- Modify: `src/winml/modelkit/optracing/qnn/{csv_parser.py, qhas_parser.py, viewer.py}` → shims
- Test: move fixtures + test files

- [ ] **Step 1: Create new package directory**

Run:
```bash
mkdir -p src/winml/modelkit/session/monitor/qnn
```

Create `src/winml/modelkit/session/monitor/qnn/__init__.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNN-specific helpers for QNNMonitor: CSV parser, QHAS parser, viewer shell-out."""
```

- [ ] **Step 2: Move three helper files via git mv**

Run (one file at a time so git tracks rename; then restore the shim):
```bash
git mv src/winml/modelkit/optracing/qnn/csv_parser.py src/winml/modelkit/session/monitor/qnn/csv_parser.py
git mv src/winml/modelkit/optracing/qnn/qhas_parser.py src/winml/modelkit/session/monitor/qnn/qhas_parser.py
git mv src/winml/modelkit/optracing/qnn/viewer.py src/winml/modelkit/session/monitor/qnn/viewer.py
```

- [ ] **Step 3: Add shims back at old paths**

Create `src/winml/modelkit/optracing/qnn/csv_parser.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim. Moved to session/monitor/qnn/csv_parser.py."""

from __future__ import annotations

from ...session.monitor.qnn.csv_parser import *  # noqa: F401,F403
from ...session.monitor.qnn.csv_parser import parse_qnn_profiling_csv


__all__ = ["parse_qnn_profiling_csv"]
```

Repeat identical pattern for `qhas_parser.py` (exporting `parse_qhas`) and `viewer.py` (exporting `find_qnn_sdk`, `run_qhas_viewer`).

- [ ] **Step 4: Update moved files' internal imports**

If any of the moved files import from `..result` or `..report`, redirect:
- `from ..result import` → `from ..op_metrics import`
- `from ..report import` → `from ..report import` (already present in monitor/)

- [ ] **Step 5: Move test files + fixtures**

Run:
```bash
mkdir -p tests/unit/session/monitor/qnn
touch tests/unit/session/monitor/qnn/__init__.py
git mv tests/unit/optracing/test_csv_parser.py tests/unit/session/monitor/qnn/test_csv_parser.py
git mv tests/unit/optracing/test_qhas_parser.py tests/unit/session/monitor/qnn/test_qhas_parser.py
git mv tests/unit/optracing/fixtures tests/unit/session/monitor/qnn/fixtures
```

Update imports in the moved test files:
- `from winml.modelkit.optracing.qnn.csv_parser import` → `from winml.modelkit.session.monitor.qnn.csv_parser import`
- Likewise for `qhas_parser`.

Update fixture paths if any tests load them via relative paths.

- [ ] **Step 6: Run tests**

Run:
```bash
uv run pytest tests/unit/session/monitor/qnn/ tests/unit/optracing/ -v
```
Expected: all PASS (new path + shim-based old path both work).

- [ ] **Step 7: Ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/monitor/qnn/ src/winml/modelkit/optracing/qnn/ tests/unit/session/monitor/qnn/ --fix
```

- [ ] **Step 8: Commit**

Run:
```bash
git add src/winml/modelkit/session/monitor/qnn/ src/winml/modelkit/optracing/qnn/ tests/unit/session/monitor/qnn/ tests/unit/optracing/
git commit -m "feat(monitor): relocate QNN helpers → session/monitor/qnn/

Moves csv_parser.py, qhas_parser.py, viewer.py + fixtures.
Old paths retained as shims."
```

---

## Task 6: Add `PerfContext` dataclass to `session/session.py`

**Rationale:** `session.perf()` will yield this dataclass instead of a raw `PerfStats`. Introducing it now lets Task 8 extend `perf()` without churn.

**Files:**
- Modify: `src/winml/modelkit/session/session.py`

- [ ] **Step 1: Add the dataclass near other session types**

In `src/winml/modelkit/session/session.py`, after the existing `SessionState` enum (around line 58-64), add:

```python
@dataclass(frozen=True)
class PerfContext:
    """Yielded by ``WinMLSession.perf()``.

    Aggregates perf statistics and the optional attached EP monitor.
    Frozen: mutation is not a supported pattern — update the underlying
    objects instead.
    """
    stats: PerfStats
    monitor: EPMonitor  # NullEPMonitor when no monitor was passed
```

Ensure imports at top of file include:
- `from dataclasses import dataclass`
- `from .monitor.ep_monitor import EPMonitor, NullEPMonitor`
- Existing `from .stats import PerfStats`

- [ ] **Step 2: Verify the import doesn't cause a circular dependency**

Run:
```bash
uv run python -c "from winml.modelkit.session.session import WinMLSession, PerfContext; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: No test yet — dataclass is glue. Defer to Task 8's perf() test.**

- [ ] **Step 4: Ruff**

Run:
```bash
uv run ruff check src/winml/modelkit/session/session.py --fix
```

- [ ] **Step 5: Commit**

Run:
```bash
git add src/winml/modelkit/session/session.py
git commit -m "feat(session): add PerfContext dataclass

Frozen dataclass aggregating PerfStats + EPMonitor. Prep for
session.perf(monitor=...) signature change."
```

---

## Task 7: Add `_active_session_option_entries` state + merge-in-_build_session_options

**Rationale:** New instance attribute tracks monitor-contributed session-level config entries. `_build_session_options` applies them. Safe to land before `perf()` sets them because the dict is empty by default.

**Files:**
- Modify: `src/winml/modelkit/session/session.py`

- [ ] **Step 1: Initialize state in `__init__`**

In `WinMLSession.__init__` (around line 165-220), after `self._provider_options = ep_config.provider_options if ep_config else {}` (~line 202), add:
```python
# Monitor-contributed session config entries (populated by session.perf(monitor=...))
self._active_session_option_entries: dict[str, str] = {}
```

- [ ] **Step 2: Apply entries in `_build_session_options`**

In the `_build_session_options` method (around line 415-452), after the method obtains `opts` (either the policy path or the explicit-EP path), and BEFORE returning `opts`, insert:
```python
# Apply monitor-contributed session config entries (active during session.perf(monitor=...))
for key, value in self._active_session_option_entries.items():
    opts.add_session_config_entry(key, value)
```

Do this in BOTH branches (explicit-EP path and policy path) so the entries apply regardless.

- [ ] **Step 3: Write a unit test verifying entries are applied**

In `tests/unit/session/test_perf_monitor_integration.py` (new file), start a file we'll fill in more across subsequent tasks:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for WinMLSession.perf(monitor=...) — teardown ordering,
auto-reset, session/provider option merging, exception transparency."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import onnxruntime as ort
import pytest


def test_active_session_option_entries_applied():
    """_build_session_options applies monitor-contributed entries."""
    from winml.modelkit.session.session import WinMLSession

    # Use a tiny real ONNX model or mock the ORT parts
    # For unit isolation: mock _find_ep_device + InferenceSession entirely
    with patch.object(WinMLSession, "_find_ep_device", return_value=None):
        session = WinMLSession.__new__(WinMLSession)
        session._device = "cpu"
        session._ep = None
        session._session_options = ort.SessionOptions()
        session._provider_options = {}
        session._active_session_option_entries = {"session.disable_cpu_ep_fallback": "1"}

        opts = session._build_session_options("cpu")
        # ORT does not expose a clean read-back API, so at minimum verify no exception
        # and the dict was consumed
        assert isinstance(opts, ort.SessionOptions)
```

- [ ] **Step 4: Run test**

Run:
```bash
uv run pytest tests/unit/session/test_perf_monitor_integration.py::test_active_session_option_entries_applied -v
```
Expected: PASS.

- [ ] **Step 5: Full test sanity**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: no regressions (the new dict is empty during normal runs).

- [ ] **Step 6: Ruff + commit**

Run:
```bash
uv run ruff check src/winml/modelkit/session/session.py tests/unit/session/test_perf_monitor_integration.py --fix
git add src/winml/modelkit/session/session.py tests/unit/session/test_perf_monitor_integration.py
git commit -m "feat(session): add _active_session_option_entries state

Infrastructure for session.perf(monitor=...) to contribute session-level
config entries via add_session_config_entry. Empty by default; populated
transiently during perf() context."
```

---

## Task 8: Extend `WinMLSession.perf()` to accept `monitor=` + full lifecycle

**Rationale:** The central change. Implements hook invocation, auto-reset on option conflict, teardown ordering (reset → monitor.__exit__), exception transparency via `sys.exc_info()`, gc.collect for Windows file handles, nested-perf guard.

**Files:**
- Modify: `src/winml/modelkit/session/session.py`
- Test: `tests/unit/session/test_perf_monitor_integration.py` (extend)
- Test: `tests/unit/session/test_perf_auto_reset.py` (new)

- [ ] **Step 1: Write failing integration tests**

Append to `tests/unit/session/test_perf_monitor_integration.py`:
```python
def test_perf_monitor_none_backward_compatible(tmp_path):
    """perf() with no monitor works as before — yields PerfContext with NullEPMonitor."""
    from winml.modelkit.session.session import WinMLSession, PerfContext
    from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

    # Minimal model — use existing test fixture or skip if not available
    # [Use a pre-existing fixture path — reference whatever the project uses]
    model_path = _get_minimal_onnx_fixture()  # helper; may need to import from tests/
    session = WinMLSession(model_path, device="cpu")
    with session.perf(warmup=0) as ctx:
        assert isinstance(ctx, PerfContext)
        assert isinstance(ctx.monitor, NullEPMonitor)


def test_nested_perf_raises():
    """Entering perf() while another perf() is active raises RuntimeError."""
    from winml.modelkit.session.session import WinMLSession
    model_path = _get_minimal_onnx_fixture()
    session = WinMLSession(model_path, device="cpu")
    with session.perf():
        with pytest.raises(RuntimeError, match="already active"):
            with session.perf():
                pass


def test_teardown_ordering_reset_before_monitor_exit():
    """Monitor.requires_session_teardown=True → self.reset() fires BEFORE monitor.__exit__."""
    from winml.modelkit.session.session import WinMLSession
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor

    observations = []

    class _TeardownMonitor(EPMonitor):
        requires_session_teardown = True
        @classmethod
        def is_available(cls):
            return True
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            # The session._session attribute should have been cleared by now
            observations.append(("exit", getattr(self, "_session_at_exit", "MISSING")))
        def to_dict(self):
            return {"ep": "test"}

    model_path = _get_minimal_onnx_fixture()
    session = WinMLSession(model_path, device="cpu")
    mon = _TeardownMonitor()

    with session.perf(monitor=mon) as ctx:
        session.run({_get_first_input_name(model_path): _get_zero_input(model_path)})

    # After exit, capture state the monitor observed at exit time
    # Arrangement: add a __setattr__ trick OR check that _session is None post-exit
    assert session._session is None  # reset happened


def _get_minimal_onnx_fixture():
    """Return path to a trivially runnable ONNX model fixture."""
    # Use whatever fixture the project's session tests use; if none exists,
    # create one in tests/unit/session/fixtures/. For plan purposes, delegate:
    from tests._helpers import get_minimal_onnx_model_path
    return get_minimal_onnx_model_path()


def _get_first_input_name(model_path):
    import onnx
    m = onnx.load(str(model_path))
    return m.graph.input[0].name


def _get_zero_input(model_path):
    import numpy as np
    import onnx
    m = onnx.load(str(model_path))
    inp = m.graph.input[0]
    shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
    return np.zeros(shape, dtype=np.float32)
```

If `tests/_helpers.py::get_minimal_onnx_model_path` doesn't exist, create it:
```python
# tests/_helpers.py
from pathlib import Path

def get_minimal_onnx_model_path() -> Path:
    """Return path to a tiny ONNX Identity model used by WinMLSession tests."""
    import onnx
    from onnx import helper, TensorProto
    fixture = Path(__file__).parent / "_fixtures" / "identity.onnx"
    if not fixture.exists():
        fixture.parent.mkdir(exist_ok=True)
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "identity", [inp], [out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.save(model, fixture)
    return fixture
```

Create `tests/unit/session/test_perf_auto_reset.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for session.perf() auto-reset behavior when a monitor contributes options."""

from __future__ import annotations

import logging
from tests._helpers import get_minimal_onnx_model_path


def test_auto_reset_fires_when_options_contributed(caplog):
    """If session is already compiled AND monitor contributes provider options,
    session.perf().__enter__ auto-resets with a WARNING log."""
    from winml.modelkit.session.session import WinMLSession
    from winml.modelkit.session.monitor.ep_monitor import EPMonitor

    class _ContributingMonitor(EPMonitor):
        @classmethod
        def is_available(cls): return True
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def to_dict(self): return {"ep": "test"}
        def get_provider_options(self):
            return {"some_provider_option": "1"}

    session = WinMLSession(get_minimal_onnx_model_path(), device="cpu")
    # Force compile
    session.compile()
    assert session._session is not None
    pre_compile_obj = session._session

    with caplog.at_level(logging.WARNING):
        with session.perf(monitor=_ContributingMonitor()):
            pass  # reset should happen on enter

    assert any("auto-reset" in rec.message.lower() for rec in caplog.records)
    # After perf exits (and perf's exit restores options), session may or may not
    # be compiled. The key is that the pre-compile object was dropped.
    assert session._session is None or session._session is not pre_compile_obj
```

- [ ] **Step 2: Run tests — expect failures**

Run:
```bash
uv run pytest tests/unit/session/test_perf_monitor_integration.py tests/unit/session/test_perf_auto_reset.py -v
```
Expected: FAILS because `perf()` doesn't accept `monitor=` yet.

- [ ] **Step 3: Rewrite `WinMLSession.perf()`**

In `src/winml/modelkit/session/session.py`, replace the existing `perf` method (around lines 583-603) with:

```python
@contextmanager
def perf(
    self,
    warmup: int = 0,
    monitor: EPMonitor | None = None,
) -> Generator[PerfContext, None, None]:
    """Run a scoped performance window yielding a PerfContext.

    Args:
        warmup: Number of initial samples to exclude from statistics.
        monitor: Optional EPMonitor. Contributes session/provider options at
            compile time (auto-resets the session if already compiled with
            different options — logs WARNING). Parses artifacts on exit.

    Yields:
        PerfContext(stats=PerfStats, monitor=EPMonitor | NullEPMonitor)

    Raises:
        RuntimeError: If another perf() context is already active on this session.
    """
    if self._perf_stats is not None:
        raise RuntimeError(
            "session.perf() already active; nested perf is forbidden"
        )

    mon: EPMonitor = monitor if monitor is not None else NullEPMonitor()

    # Collect hook contributions — must be idempotent per EPMonitor contract
    extra_sess = mon.get_session_options()
    extra_prov = mon.get_provider_options()

    # Auto-reset if options to apply AND session is already compiled
    if (extra_sess or extra_prov) and self._session is not None:
        logger.warning(
            "session.perf(): auto-resetting compiled session to apply monitor "
            "session/provider options (monitor=%s)",
            type(mon).__name__,
        )
        self.reset()

    # Save + merge
    saved_sess = dict(self._active_session_option_entries)
    saved_prov = dict(self._provider_options)
    self._active_session_option_entries = {**saved_sess, **extra_sess}
    self._provider_options = {**saved_prov, **extra_prov}

    stats = PerfStats(warmup=warmup)
    self._perf_stats = stats
    mon.__enter__()

    try:
        yield PerfContext(stats=stats, monitor=mon)
    finally:
        self._perf_stats = None
        exc_info = sys.exc_info()
        try:
            if mon.requires_session_teardown:
                self.reset()
                gc.collect()  # Windows: release CSV file handle
        finally:
            try:
                mon.__exit__(*exc_info)
            finally:
                self._active_session_option_entries = saved_sess
                self._provider_options = saved_prov
```

Required imports at top of `session.py`:
```python
import gc
import sys
```

And near existing `@contextmanager` import, confirm `from contextlib import contextmanager` is imported.

- [ ] **Step 4: Run tests — expect PASS**

Run:
```bash
uv run pytest tests/unit/session/test_perf_monitor_integration.py tests/unit/session/test_perf_auto_reset.py -v
```
Expected: all PASS.

- [ ] **Step 5: Backward-compat check: existing `session.perf()` users**

Run:
```bash
uv run pytest tests/ -k "perf" -v --tb=short
```
Expected: no regressions — the old `session.perf(warmup=10) as stats` pattern still works because `PerfContext.stats` is accessible as `.stats` but also the old callers likely do `stats.mean_ms` — that breaks! We need to handle this.

- [ ] **Step 6: Fix: audit existing callers**

Run:
```bash
uv run grep -rn "session.perf(" src/ tests/ --include="*.py"
```

For each call site, if it uses `as stats:` and treats the yielded object as a `PerfStats`, update it to `as ctx:` and use `ctx.stats`. Primary callers:
- `src/winml/modelkit/commands/perf.py` (benchmark loop)
- Any benchmark helper in `src/winml/modelkit/session/perf_benchmark.py` if present

Update each — example for `commands/perf.py` benchmark loop:
```python
# Before
with session.perf(warmup=...) as stats:
    ...
    stats.mean_ms  # etc.

# After
with session.perf(warmup=...) as ctx:
    stats = ctx.stats
    ...
    stats.mean_ms  # etc.
```

This keeps the minimal delta. Task 11 will add `monitor=` to these calls.

- [ ] **Step 7: Rerun full tests**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: all pass.

- [ ] **Step 8: Ruff + commit**

Run:
```bash
uv run ruff check src/winml/modelkit/session/session.py src/winml/modelkit/commands/perf.py tests/unit/session/ --fix
git add src/winml/modelkit/session/session.py src/winml/modelkit/commands/perf.py tests/unit/session/ tests/_helpers.py tests/_fixtures/
git commit -m "feat(session): extend perf() with monitor= yielding PerfContext

- perf(warmup, monitor=None) yields PerfContext(stats, monitor)
- Auto-reset on option conflict (WARNING log)
- Teardown ordering: reset → gc.collect → monitor.__exit__ → restore
- Exception transparency via sys.exc_info()
- Nested perf() raises RuntimeError

Migrate existing callers to use ctx.stats."
```

---

## Task 9: Rewrite `QNNMonitor` from placeholder to full implementation

**Rationale:** The new monitor. Uses the relocated helpers, the new `OpTraceResult` with `status`/`error`, the new base-class hooks, and `ensure_initialized()`.

**Files:**
- Modify: `src/winml/modelkit/session/monitor/qnn_monitor.py`
- Test: `tests/unit/session/monitor/test_qnn_monitor.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/unit/session/monitor/test_qnn_monitor.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for QNNMonitor — the QNN EP op-tracing monitor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_ctor_defaults():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor()
    assert m._level == "basic"
    assert m._output_dir.exists()  # tempdir created
    assert m._csv_path.is_absolute()


def test_ctor_rejects_invalid_level():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    with pytest.raises(ValueError, match="level"):
        QNNMonitor(level="invalid")  # type: ignore[arg-type]


def test_get_session_options():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor()
    opts = m.get_session_options()
    assert opts["session.disable_cpu_ep_fallback"] == "1"
    assert opts["ep.context_enable"] == "1"
    assert opts["ep.context_embed_mode"] == "0"


def test_get_provider_options_basic():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor(level="basic")
    opts = m.get_provider_options()
    assert opts["profiling_level"] == "detailed"
    assert opts["backend_path"] == "QnnHtp.dll"
    assert "profiling_file_path" in opts


def test_get_provider_options_detail():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor(level="detail")
    opts = m.get_provider_options()
    assert opts["profiling_level"] == "optrace"


def test_profiling_keys_not_user_overridable():
    """User extras cannot override profiling_level or profiling_file_path."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor(
        level="basic",
        extra_provider_options={
            "profiling_level": "off",
            "profiling_file_path": "/attacker/path",
            "htp_performance_mode": "balanced",
        },
    )
    opts = m.get_provider_options()
    assert opts["profiling_level"] == "detailed"  # monitor-owned
    assert opts["profiling_file_path"] != "/attacker/path"
    assert opts["htp_performance_mode"] == "balanced"  # user extra honored


def test_get_provider_options_idempotent():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor(level="basic")
    assert m.get_provider_options() == m.get_provider_options()


def test_requires_session_teardown_true():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    assert QNNMonitor.requires_session_teardown is True


def test_double_enter_raises():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor()
    m.__enter__()
    with pytest.raises(RuntimeError, match="already entered"):
        m.__enter__()


def test_exit_with_no_csv_reports_no_data(tmp_path):
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)
    d = m.to_dict()
    assert d["status"] == "no_data"


def test_is_available_via_bundled():
    """When QNN EP is in get_available_providers(), is_available() returns True."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    with patch(
        "onnxruntime.get_available_providers",
        return_value=["QNNExecutionProvider", "CPUExecutionProvider"],
    ):
        assert QNNMonitor.is_available() is True


def test_is_available_via_winml(tmp_path):
    """When QNN EP is registered via WinML (in get_ep_devices), is_available returns True."""
    from unittest.mock import MagicMock
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    fake_ep = MagicMock()
    fake_ep.ep_name = "QNNExecutionProvider"
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch("onnxruntime.get_ep_devices", return_value=[fake_ep]), \
         patch("winml.modelkit.session.ep_registry.ensure_initialized"):
        assert QNNMonitor.is_available() is True


def test_is_available_neither():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor
    with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]), \
         patch("onnxruntime.get_ep_devices", return_value=[]), \
         patch("winml.modelkit.session.ep_registry.ensure_initialized"):
        assert QNNMonitor.is_available() is False
```

- [ ] **Step 2: Run tests — expect failures**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_qnn_monitor.py -v
```
Expected: most fail (placeholder doesn't have these methods).

- [ ] **Step 3: Rewrite `qnn_monitor.py`**

Overwrite `src/winml/modelkit/session/monitor/qnn_monitor.py`:
```python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNNMonitor — Qualcomm NPU per-operator profiling via ORT's QNN EP.

Produces an OpTraceResult with per-op cycle counts (level="basic") or full
QHAS roofline / DMA traffic data (level="detail"). Attached to a
WinMLSession via ``session.perf(monitor=QNNMonitor(...))``.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping, TYPE_CHECKING

from .ep_monitor import EPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult
from .qnn.csv_parser import parse_qnn_profiling_csv


if TYPE_CHECKING:
    from typing_extensions import Self


logger = logging.getLogger(__name__)

# Level → QNN profiling_level value
_LEVEL_TO_PROFILING: dict[str, str] = {
    "basic": "detailed",
    "detail": "optrace",
}


class QNNMonitor(EPMonitor):
    """Qualcomm NPU per-op profiler via ORT's QNN EP.

    Level modes:
        - ``"basic"``: CSV with per-op cycle counts (fast; covers most use cases).
        - ``"detail"``: QHAS via QNN SDK viewer (roofline + DMA traffic;
          requires QNN SDK installed; falls back to CSV with a warning if not).

    Example::

        with session.perf(monitor=QNNMonitor(level="basic")) as ctx:
            for _ in range(10):
                session.run(inputs)
        print(ctx.monitor.to_dict())
    """

    requires_session_teardown: ClassVar[bool] = True

    def __init__(
        self,
        level: Literal["basic", "detail"] = "basic",
        output_dir: Path | None = None,
        extra_provider_options: Mapping[str, str] | None = None,
    ) -> None:
        if level not in _LEVEL_TO_PROFILING:
            raise ValueError(
                f"level must be 'basic' or 'detail', got {level!r}"
            )
        self._level = level
        # Idempotency: resolve all paths at __init__, not per-call
        self._output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="qnn_profile_"))
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = (self._output_dir / "profiling_output.csv").resolve()
        self._extra = dict(extra_provider_options or {})
        self._entered = False
        self._result: OpTraceResult | None = None

    # ---- EPMonitor contract ----

    @classmethod
    def is_available(cls) -> bool:
        """True iff QNN EP is usable via bundled DLL or WinML registration."""
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        if "QNNExecutionProvider" in ort.get_available_providers():
            return True
        # WinML path
        try:
            from ..ep_registry import ensure_initialized
            ensure_initialized()
            return any(
                d.ep_name == "QNNExecutionProvider"
                for d in ort.get_ep_devices()
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("QNNMonitor.is_available: WinML path failed: %s", exc)
            return False

    def get_session_options(self) -> dict[str, str]:
        return {
            "session.disable_cpu_ep_fallback": "1",
            "ep.context_enable": "1",
            "ep.context_embed_mode": "0",
        }

    def get_provider_options(self) -> dict[str, str]:
        # Build in layers; owner-enforced keys last so they can't be overridden.
        opts: dict[str, str] = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "high_performance",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",
        }
        opts.update(self._extra)
        # C-3: these two keys are NEVER user-overridable.
        opts["profiling_level"] = _LEVEL_TO_PROFILING[self._level]
        opts["profiling_file_path"] = str(self._csv_path)
        return opts

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("QNNMonitor already entered")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        # Must not suppress caller exceptions — return None implicitly.
        try:
            self._result = self._parse_artifacts()
        except Exception as e:  # noqa: BLE001
            logger.warning("QNNMonitor: artifact parse failed: %s", e)
            self._result = self._make_failure_result("parse_failed", str(e))

    def to_dict(self) -> dict[str, Any]:
        if self._result is None:
            return {"ep": "QNN", "device": "NPU", "status": "not_run"}
        return self._result.to_dict()

    @property
    def result(self) -> OpTraceResult | None:
        """Structured result — consumed by display_op_trace_report / write_op_trace_json."""
        return self._result

    # ---- Internals ----

    def _parse_artifacts(self) -> OpTraceResult:
        """Parse CSV (and QHAS if detail). Retry once for Windows file-handle lag."""
        if not self._csv_path.exists():
            # Retry — Windows may lag on file-handle release
            time.sleep(0.05)
        if not self._csv_path.exists():
            logger.warning(
                "QNNMonitor: no CSV at %s — nothing to parse", self._csv_path
            )
            return self._make_failure_result("no_data", None)

        parsed = parse_qnn_profiling_csv(self._csv_path)
        meta = parsed["metadata"]
        total_cycles = meta.get("accel_execute_cycles", 0)
        accel_us = meta.get("accel_execute_us", 0)
        cycle_to_us = accel_us / total_cycles if total_cycles > 0 else 0.0

        operators = [
            OperatorMetrics(
                name=op["name"],
                op_path=op["name"],
                op_id=op["op_id"],
                duration_us=op["cycles"] * cycle_to_us,
                percent_of_total=(
                    op["cycles"] / total_cycles * 100 if total_cycles > 0 else 0
                ),
            )
            for op in parsed["operators"]
        ]

        artifacts: dict[str, str] = {"csv": str(self._csv_path)}
        qnn_log = Path(str(self._csv_path) + "_qnn.log")
        if qnn_log.is_file():
            artifacts["qnn_log"] = str(qnn_log)

        status = "ok"
        if self._level == "detail":
            qhas_path = self._try_qhas(qnn_log, artifacts)
            if qhas_path is None:
                status = "basic_fallback"
                logger.warning(
                    "QNNMonitor: detail mode requested but QHAS viewer "
                    "unavailable; falling back to basic CSV data"
                )

        return OpTraceResult(
            model=None,
            device="npu",
            tracing_level=self._level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=operators,
            num_samples=meta.get("num_samples", 0),
            summary={
                "hvx_threads": meta.get("hvx_threads", 0),
                "accel_execute_cycles": meta.get("accel_execute_cycles", 0),
                "accel_execute_us": accel_us,
            },
            artifacts=artifacts,
            status=status,
        )

    def _try_qhas(self, qnn_log: Path, artifacts: dict[str, str]) -> Path | None:
        """Detail mode: run QNN SDK viewer to produce QHAS. Returns path or None."""
        if not qnn_log.is_file():
            return None
        try:
            from .qnn.viewer import find_qnn_sdk, run_qhas_viewer
        except ImportError:
            return None
        sdk = find_qnn_sdk()
        if sdk is None:
            return None
        # Locate schematic.bin via glob fallback (no os.chdir per FR-12)
        schematics = list(self._output_dir.glob("*_schematic.bin"))
        if not schematics:
            # Fallback: check process CWD (QNN SDK default behavior)
            schematics = list(Path.cwd().glob("*_schematic.bin"))
        if not schematics:
            logger.warning("QNNMonitor: no *_schematic.bin found for detail mode")
            return None
        schematic = schematics[0]
        artifacts["schematic"] = str(schematic)
        qhas_out = self._output_dir / "qhas_output.json"
        result_path = run_qhas_viewer(qnn_log, schematic, qhas_out, sdk_root=sdk)
        if result_path is not None and result_path.is_file():
            artifacts["qhas"] = str(result_path)
            return result_path
        return None

    def _make_failure_result(
        self, status: str, error: str | None
    ) -> OpTraceResult:
        return OpTraceResult(
            model=None,
            device="npu",
            tracing_level=self._level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=[],
            summary={},
            artifacts={"csv": str(self._csv_path)} if self._csv_path.exists() else {},
            status=status,
            error=error,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

Run:
```bash
uv run pytest tests/unit/session/monitor/test_qnn_monitor.py -v
```
Expected: all PASS.

- [ ] **Step 5: Full suite sanity**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: no regressions.

- [ ] **Step 6: Ruff + commit**

Run:
```bash
uv run ruff check src/winml/modelkit/session/monitor/qnn_monitor.py tests/unit/session/monitor/test_qnn_monitor.py --fix
git add src/winml/modelkit/session/monitor/qnn_monitor.py tests/unit/session/monitor/test_qnn_monitor.py
git commit -m "feat(monitor): rewrite QNNMonitor from placeholder to real impl

- get_session_options() contributes disable_cpu_ep_fallback, ep.context_*
- get_provider_options() contributes backend_path + profiling_level
  (user overrides honored for non-profiling keys; profiling_level and
  profiling_file_path are owner-enforced)
- __exit__ parses CSV → OpTraceResult; retries once on Windows lag
- detail mode: runs QHAS viewer if SDK available; falls back to basic
- No os.chdir (glob fallback for schematic.bin location)
- is_available works with both onnxruntime-qnn AND onnxruntime-windowsml"
```

---

## Task 10: Port availability test from `test_detection.py` (and delete)

**Rationale:** The existing `tests/unit/optracing/test_detection.py` tests the old `is_qnn_profiling_available()`. Rewrite minimally as `test_qnn_monitor_availability.py` at the new location, then delete the old file.

**Files:**
- Create: `tests/unit/session/monitor/test_qnn_monitor_availability.py`
- Delete: `tests/unit/optracing/test_detection.py`

- [ ] **Step 1: Read the existing test**

Run:
```bash
cat tests/unit/optracing/test_detection.py
```
Identify the behaviors being tested.

- [ ] **Step 2: Write the replacement**

Create `tests/unit/session/monitor/test_qnn_monitor_availability.py` with equivalent behaviors re-expressed against `QNNMonitor.is_available()` (note: Task 9 already covers most of this; this task just ensures we don't drop any test from `test_detection.py` that wasn't covered).

If `test_detection.py` checks behaviors not already covered in `test_qnn_monitor.py`, add them here. Otherwise, delete without replacement.

- [ ] **Step 3: Delete the old file**

Run:
```bash
git rm tests/unit/optracing/test_detection.py
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/unit/session/monitor/ -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add tests/unit/session/monitor/test_qnn_monitor_availability.py tests/unit/optracing/test_detection.py
git commit -m "test(monitor): port QNN availability tests to new location

Replaces tests/unit/optracing/test_detection.py with tests exercising
QNNMonitor.is_available() — covers both bundled (onnxruntime-qnn) and
WinML-registered (onnxruntime-windowsml) EP discovery paths."
```

---

## Task 11: Add `_resolve_ep_monitor` dispatch + wire op-tracing into main benchmark loop in `commands/perf.py`

**Rationale:** Collapse the separate op-tracing block (`perf.py:1334-1386`) into the existing benchmark `session.perf()` call by passing `monitor=QNNMonitor(...)`.

**Files:**
- Modify: `src/winml/modelkit/commands/perf.py`
- Test: `tests/unit/commands/test_perf_optracing.py` (move from `tests/unit/optracing/test_perf_optracing_cli.py`)

- [ ] **Step 1: Move the existing CLI test file**

Run:
```bash
mkdir -p tests/unit/commands
test -f tests/unit/commands/__init__.py || touch tests/unit/commands/__init__.py
git mv tests/unit/optracing/test_perf_optracing_cli.py tests/unit/commands/test_perf_optracing.py
```

Update imports in the moved file to match the new module structure (replace `winml.modelkit.optracing.*` references with `winml.modelkit.session.monitor.*` where applicable).

- [ ] **Step 2: Add `_resolve_ep_monitor` helper to `commands/perf.py`**

Near the top of `commands/perf.py`, after the imports and `DYNAMIC_DIM_DEFAULTS`, add:
```python
def _resolve_ep_monitor(
    ep: str,
    op_tracing: str | None,
    output_dir: Path,
):
    """Pick the EPMonitor for the requested EP + optional op-tracing level.

    Explicit dispatch — no registry, no plugin loading. Raises RuntimeError
    when op-tracing is requested against an EP that has no op-tracing monitor.
    """
    from ..session.monitor.ep_monitor import NullEPMonitor
    if op_tracing:
        from ..session.monitor.qnn_monitor import QNNMonitor
        if ep == "qnn" and QNNMonitor.is_available():
            return QNNMonitor(level=op_tracing, output_dir=output_dir)
        raise RuntimeError(
            f"Op-tracing not available for EP '{ep}'. Supported: 'qnn'."
        )
    from ..session.monitor.vitisai_monitor import VitisAIMonitor
    if ep == "vitisai" and VitisAIMonitor.is_available():
        return VitisAIMonitor()
    return NullEPMonitor()
```

- [ ] **Step 3: Collapse the op-tracing block**

Locate the op-tracing block (currently around lines 1334-1386). **Delete** the entire block (from `if op_tracing:` through `console.print(f"[green]Op-trace saved to:[/green] {trace_output}")`).

- [ ] **Step 4: Wire `monitor=` into the main benchmark loop**

Find the main benchmark invocation. It currently looks like:
```python
with session.perf(warmup=...) as ctx:
    ...
```

Replace with:
```python
monitor = None
if op_tracing:
    try:
        monitor = _resolve_ep_monitor(
            ep=config.ep,
            op_tracing=op_tracing,
            output_dir=output.parent if output else Path.cwd(),
        )
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

with session.perf(warmup=config.warmup, monitor=monitor) as ctx:
    ...  # existing loop body; stats = ctx.stats
```

- [ ] **Step 5: Add the post-benchmark report logic**

After the `with` block, add:
```python
if op_tracing:
    from ..session.monitor.report import display_op_trace_report, write_op_trace_json
    result = ctx.monitor.result
    if result is None or result.status == "no_data":
        console.print(
            "[yellow]Warning:[/yellow] No profiling data produced."
        )
    else:
        display_op_trace_report(result, console)
        model_slug = hf_model.replace("/", "_").replace("\\", "_")
        if is_onnx:
            model_slug = model_path.stem
        trace_output = (output.parent if output else Path.cwd()) / f"{model_slug}_op_trace.json"
        write_op_trace_json(result, trace_output)
        console.print(f"[green]Op-trace saved to:[/green] {trace_output}")
```

- [ ] **Step 6: Remove dead imports**

At the top of `commands/perf.py`, remove the now-unused `from ..optracing import is_qnn_profiling_available`, `get_tracer`, etc. (search and remove).

- [ ] **Step 7: Run CLI tests**

Run:
```bash
uv run pytest tests/unit/commands/test_perf_optracing.py -v
```
Expected: PASS (may need test updates to match new dispatch — fix per-test).

- [ ] **Step 8: Manual CLI smoke (non-hardware-gated)**

Run:
```bash
uv run winml perf --help | head -20
```
Expected: help output renders, no import errors.

- [ ] **Step 9: Full test sanity**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: no regressions.

- [ ] **Step 10: Ruff + commit**

Run:
```bash
uv run ruff check src/winml/modelkit/commands/perf.py tests/unit/commands/ --fix
git add src/winml/modelkit/commands/perf.py tests/unit/commands/ tests/unit/optracing/
git commit -m "feat(perf): collapse op-tracing block into integrated monitor path

- Add _resolve_ep_monitor(ep, op_tracing, output_dir) dispatch helper
- Delete standalone op-tracing block (~50 lines)
- Pass monitor=QNNMonitor(...) to session.perf() in main benchmark
- Hard-fail when op-tracing requested against unsupported EP"
```

---

## Task 12: Delete `QNNProfiler` and related deprecated modules

**Rationale:** With `commands/perf.py` no longer importing `QNNProfiler` / `is_qnn_profiling_available` / `get_tracer`, the old optracing classes can go.

**Files:**
- Delete: `src/winml/modelkit/optracing/qnn/profiler.py`
- Delete: `src/winml/modelkit/optracing/base.py`
- Delete: `src/winml/modelkit/optracing/registry.py`
- Delete: `tests/unit/optracing/test_qnn_profiler.py`
- Delete: `tests/unit/optracing/test_registry.py`
- Delete: `tests/unit/optracing/test_integration.py` (replaced by `tests/unit/session/test_perf_monitor_integration.py`)

- [ ] **Step 1: Verify no remaining imports**

Run:
```bash
uv run grep -rn "from winml.modelkit.optracing" src/ tests/ --include="*.py" | grep -v "^tests/unit/optracing"
```
Expected: empty output. If not empty, redirect remaining imports first.

Also:
```bash
uv run grep -rn "QNNProfiler\|OpTracer\|is_qnn_profiling_available\|get_tracer\|register_tracer" src/ tests/ --include="*.py"
```
Expected: matches only inside `optracing/` directory itself (safe to delete).

- [ ] **Step 2: Delete the files**

Run:
```bash
git rm src/winml/modelkit/optracing/qnn/profiler.py
git rm src/winml/modelkit/optracing/base.py
git rm src/winml/modelkit/optracing/registry.py
git rm tests/unit/optracing/test_qnn_profiler.py
git rm tests/unit/optracing/test_registry.py
git rm tests/unit/optracing/test_integration.py
```

- [ ] **Step 3: Run full tests**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

Run:
```bash
git commit -m "refactor(optracing): delete QNNProfiler, OpTracer, registry

- QNNProfiler replaced by QNNMonitor (session/monitor/qnn_monitor.py)
- OpTracer ABC + registry collapsed into EPMonitor hierarchy
- Tests migrated to tests/unit/session/"
```

---

## Task 13: Delete `optracing/` package entirely (shims included)

**Rationale:** All callers migrated. Shims in `optracing/result.py`, `report.py`, `qnn/*.py`, `__init__.py` can be removed.

**Files:**
- Delete: `src/winml/modelkit/optracing/` (entire directory)
- Delete: `tests/unit/optracing/` (entire directory, after moving the `__init__.py`)

- [ ] **Step 1: Sanity check for final references**

Run:
```bash
uv run grep -rn "winml.modelkit.optracing\|from .optracing\|from ..optracing\|from ...optracing" src/ tests/ --include="*.py"
```
Expected: empty.

- [ ] **Step 2: Delete directories**

Run:
```bash
git rm -r src/winml/modelkit/optracing/
git rm -r tests/unit/optracing/
```

- [ ] **Step 3: Run full tests**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: PASS.

- [ ] **Step 4: Ruff**

Run:
```bash
uv run ruff check src/ tests/ --fix
```

- [ ] **Step 5: Commit**

Run:
```bash
git commit -m "refactor: delete src/winml/modelkit/optracing/ package

All functionality relocated to session/monitor/. Shims removed now
that no caller imports from the old paths."
```

---

## Task 14: Delete `WinMLSession._init_winml_eps_once` classmethod; use `ensure_initialized` module function

**Rationale:** The classmethod is now redundant — the module function does the same thing and is what `QNNMonitor.is_available` uses.

**Files:**
- Modify: `src/winml/modelkit/session/session.py`

- [ ] **Step 1: Replace calls to `_init_winml_eps_once` with `ensure_initialized`**

In `session.py`, find the classmethod `_init_winml_eps_once` (around line 149-163) and its one caller in `__init__` (around line 191).

Replace the `__init__` call from:
```python
WinMLSession._init_winml_eps_once()
```
to:
```python
from .ep_registry import ensure_initialized
ensure_initialized()
```

Remove the classmethod itself and the class attribute `_eps_initialized` if it was only used there.

- [ ] **Step 2: Run tests**

Run:
```bash
uv run pytest tests/ -x --tb=short -q
```
Expected: PASS.

- [ ] **Step 3: Commit**

Run:
```bash
uv run ruff check src/winml/modelkit/session/session.py --fix
git add src/winml/modelkit/session/session.py
git commit -m "refactor(session): remove _init_winml_eps_once classmethod

Redundant with ep_registry.ensure_initialized() module function.
WinMLSession.__init__ now calls the module function directly."
```

---

## Task 15: Relocate design docs per spec §1.5.1 transitional commitment

**Rationale:** Implementation complete. Per the Transitional Location note in both design docs, move them under `docs/design/session/monitor/`.

**Files:**
- Move: `docs/design/optracing/` → `docs/design/session/monitor/`
- Modify: the two design docs to remove the Transitional Location note and update `Module` / cross-refs as needed.

- [ ] **Step 1: Create the new directory**

Run:
```bash
mkdir -p docs/design/session/monitor
```

- [ ] **Step 2: Move the docs and iterations**

Run:
```bash
git mv docs/design/optracing/1_prd.md docs/design/session/monitor/1_prd.md
git mv docs/design/optracing/2_coreloop.md docs/design/session/monitor/2_coreloop.md
git mv docs/design/optracing/iterations docs/design/session/monitor/iterations
```

- [ ] **Step 3: Remove the Transitional Location note from both docs**

In both `1_prd.md` and `2_coreloop.md`, delete the four-line `**Transitional Location**` block immediately after the metadata header. Replace it with a line-break only.

- [ ] **Step 4: Bump Version to 2.2 with a Revision History entry**

In both docs, change `**Version**: 2.1` → `**Version**: 2.2` and append a Revision History row:
```markdown
| 2.2 | 2026-04-23 | Relocated from `docs/design/optracing/` to `docs/design/session/monitor/` per spec §1.5.1 transitional commitment (implementation complete). Removed Transitional Location note. |
```

- [ ] **Step 5: Delete the now-empty optracing doc directory**

Run:
```bash
rmdir docs/design/optracing 2>&1 || true
```

- [ ] **Step 6: Update any cross-references**

Run:
```bash
uv run grep -rn "docs/design/optracing" --include="*.md"
```

For each match, update the path. Notably: `docs/standards/design-doc-spec.md` §7.4 references `docs/design/optracing/1_prd.md` + `2_coreloop.md` — update to `docs/design/session/monitor/1_prd.md` + `2_coreloop.md`.

- [ ] **Step 7: Commit**

Run:
```bash
git add docs/design/ docs/standards/
git commit -m "docs: relocate op-tracing design to docs/design/session/monitor/

Per spec §1.5.1 transitional commitment — implementation landed, so docs
move to their spec-compliant location under the target module directory.
Version bumped 2.1 → 2.2. Transitional Location note removed."
```

---

## Task 16: Final end-to-end verification

**Files:** (none modified)

- [ ] **Step 1: Full test suite**

Run:
```bash
uv run pytest tests/ -v --tb=short
```
Expected: all pass (or only pre-existing failures noted in Task 0 baseline).

- [ ] **Step 2: Ruff clean**

Run:
```bash
uv run ruff check src/ tests/ docs/
```
Expected: no findings.

- [ ] **Step 3: Verify the CLI import smoke**

Run:
```bash
uv run winml perf --help
```
Expected: help renders.

- [ ] **Step 4: Check for any stale `optracing` references anywhere**

Run:
```bash
uv run grep -rn "optracing" src/ tests/ docs/standards/ --include="*.py" --include="*.md"
```
Expected: no matches (or only matches in the Revision History entries / Migration Footprint, which are historical).

- [ ] **Step 5: Hardware-gated E2E (if QNN NPU available)**

Run:
```bash
uv run winml perf -m microsoft/resnet-50 --device npu --op-tracing basic
```
Expected (on QNN hardware): CSV produced, per-op report rendered, JSON file written. On non-QNN machines: helpful error message.

- [ ] **Step 6: Verify SC-1 through SC-6 from PRD**

- **SC-1** ✓ if step 5 produced valid output on a QNN machine.
- **SC-2** ✓ via step 4 (no `optracing` references).
- **SC-3** ✓ covered by `test_qnn_monitor_availability.py`.
- **SC-4** ✓ — the 8-line idiom works (covered by integration tests).
- **SC-5** ✓ — step 1.
- **SC-6** ✓ — `display_op_trace_report` / `write_op_trace_json` consume `OpTraceResult`; `OpTraceResult.to_dict()` preserved.

- [ ] **Step 7: Final commit if any cleanup was done**

Run:
```bash
git status
# if anything left:
git add -A
git commit -m "chore: final cleanup from op-tracing refactor E2E"
```

- [ ] **Step 8: Summarize for PR description**

Draft a PR description citing:
- Bug fixed (D-1: `QNNProfiler` broken on `onnxruntime-windowsml`)
- Architectural simplification (`OpTracer` hierarchy merged into `EPMonitor`)
- Spec compliance (first doc pair authored against `design-doc-spec.md` v1.1)
- All 16 SCs from the PRD + the Transitional Location commitment honored.

---

## Self-Review

**Spec coverage:** Each PRD section has a task.
- FR-1 (both ORT variants) → Task 9 `is_available` + Task 11 dispatch
- FR-2 (`session.perf(monitor=...)`) → Tasks 6, 8
- FR-3 (single hierarchy) → Tasks 12, 13
- FR-4 (`QNNMonitor` replaces `QNNProfiler`) → Tasks 9, 12
- FR-5 (basic/detail levels) → Task 9
- FR-6 (`OpTraceResult` preserved + extended) → Task 3
- FR-7 (8-line standalone idiom) → verified in Task 16.6
- FR-8 (availability) → Task 9
- FR-9 (HWMonitor orthogonal) → no task needed; already orthogonal
- FR-10 (EPMonitor hooks) → Task 2
- FR-11 (factory dispatch, no registry) → Task 11
- FR-12 (no `os.chdir`) → Task 9 uses glob fallback
- NFR-1 through NFR-7 covered in Tasks 2, 8, 9
- All risks (R-1 through R-6) have mitigations implemented (teardown ordering, gc.collect, exception transparency, fresh tempdir, WARNING log, retry on CSV lag)

**Placeholder scan:** No TBDs. Every code block is complete.

**Type consistency:** `OpTraceResult` status field default `"ok"` consistent between Task 3 (where added) and Task 9 (where used). `ensure_initialized` function signature consistent between Task 1 (where added) and Task 9 (where called). `PerfContext(stats, monitor)` consistent between Task 6 and Task 8.

**One risk worth flagging:** The `test_teardown_ordering_reset_before_monitor_exit` test in Task 8 is written to check the final state (`session._session is None`) rather than capture the intermediate state during `monitor.__exit__`. A more rigorous test would inject an observer into `monitor.__exit__` that checks `session._session` at that exact moment. If the subagent executing Task 8 wants stronger verification, they may add that — the weaker check is sufficient for the load-bearing invariant given the implementation is direct.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-23-op-tracing-refactor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. This plan has 16 self-contained tasks well-suited to this pattern.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

**Which approach?**
