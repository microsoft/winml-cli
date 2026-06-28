# src/winml/modelkit/session/monitor/__init__.py

## TL;DR
New package marker for the `session/monitor/` hierarchy introduced by this commit. Contains only the standard MS copyright header and a one-line module docstring; deliberately empty of re-exports — callers import concrete monitor classes from their submodules.

## Diff metrics
- Lines added: 5
- Lines removed: 0
- New file

## Role before vs after
- **Before:** did not exist. The legacy `src/winml/modelkit/optracing/` tree (deleted wholesale by this commit) provided the rough functional equivalent.
- **After:** marks `session/monitor/` as a Python package; functions as the documentation anchor (`"""Per-EP monitors and op-tracing post-processing."""`) for the new sub-tree containing `ep_monitor.py`, `op_metrics.py`, `report.py`, plus the per-EP monitors (`qnn_monitor.py`, `openvino_monitor.py`, `vitisai_monitor.py`, `hw_monitor.py`).

## Symbol-level changes
None — file declares no symbols. There is no `__all__` and no public re-exports.

## Behavior / contract changes
None directly. Indirectly, by existing it allows the `from .op_metrics import OpTraceResult`-style relative imports used in `ep_monitor.py` and `report.py` to resolve, and lets external code import via `from winml.modelkit.session.monitor.ep_monitor import EPMonitor` etc.

## Cross-file impact
- **Used by which modules:** every submodule under `session/monitor/` (`ep_monitor.py`, `op_metrics.py`, `report.py`, plus the per-EP monitors); also indirectly any consumer like `WinMLSession.perf()` and `commands/perf.py` that import from `winml.modelkit.session.monitor.*`.
- **Depends on which modules:** none.

## Risks / subtleties
- Per CLAUDE.md "Import Rules", every symbol external code needs must be exported via the package's `__init__.py`. This file currently exports nothing, so all external callers must reach into submodules directly. This is a deliberate (or accidental) deviation from the project convention — worth flagging.
- No `__all__` declared.

## Open questions / TODOs surfaced
- Should this `__init__.py` re-export the public monitor API (`EPMonitor`, `NullEPMonitor`, `OpTraceResult`, `OperatorMetrics`, `TraceStatus`, `display_op_trace_report`, `write_op_trace_json`, and the concrete `QNNMonitor` / `VitisAIMonitor` / `OpenVinoMonitor`) to comply with the project's package-API convention?
