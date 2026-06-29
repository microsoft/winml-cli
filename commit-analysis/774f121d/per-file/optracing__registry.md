# src/winml/modelkit/optracing/registry.py (DELETED)

## TL;DR
This file is removed. It implemented a global substring-pattern registry mapping `(ep_pattern, level) -> OpTracer subclass`, with self-registration triggered on import. The registry is **dropped entirely** — replaced by explicit `if/elif` dispatch in `commands/perf.py::_resolve_ep_monitor`.

## Diff metrics
- Lines deleted: 64
- Status: DELETED

## What this file did (pre-state)
Maintained a module-level `_TRACERS: dict[str, dict[str, type[OpTracer]]]` keyed by EP-name substring (e.g. `"QNN"`) and profiling level (`"basic"` / `"detail"`). Provided:
- `register_tracer(pattern, level, cls)` — install a tracer class.
- `get_tracer(ep_name, level)` — look up by substring match (`"QNN"` matches `"QNNExecutionProvider"`).
- `_register_defaults()` — eagerly registered `QNNProfiler` for both `"basic"` and `"detail"` levels on import.

The substring match was the extension point: third parties could `register_tracer("CustomEP", "basic", MyTracer)` without modifying source.

## Public symbols (pre-deletion)
- `register_tracer(ep_pattern: str, level: str, tracer_class: type[OpTracer]) -> None`
- `get_tracer(ep_name: str, level: str) -> type[OpTracer] | None`
- `_register_defaults()` (private, called at import time)
- `_TRACERS` (private module-global dict)

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `_TRACERS` registry dict | **Removed entirely.** No replacement data structure. |
| `register_tracer(...)` | **Dropped.** No extension hook — adding a new monitor requires editing `_resolve_ep_monitor`. |
| `get_tracer(ep_name, level)` | **Replaced by `_resolve_ep_monitor(ep, device, op_tracing, output_dir)` at `src/winml/modelkit/commands/perf.py`** (~line 118). It hard-codes the QNN branch (`if ep_norm == "qnn"` or auto-detect on `device_norm in ("npu", "auto", "")`) and instantiates `QNNMonitor` directly. The fallback returns `NullEPMonitor()`. |
| `_register_defaults()` | **Dropped.** No equivalent eager-registration step on import. |

## Net behavior change
- **Substring matching is gone.** The new dispatch keys off exact normalized EP names (`"qnn"`), not a substring of the ORT provider string. This is more predictable but loses the loose match (`"OpenVINO"` → `"OpenVINOExecutionProvider"` would have worked under the old registry).
- **Auto-resolution by device added.** `_resolve_ep_monitor` selects `QNNMonitor` when no EP is given and `device_norm in ("npu", "auto", "")` AND `QNNMonitor.is_available()` returns True. This is new behavior — the old `get_tracer` required the caller to know the EP name.
- **Levels are no longer registry keys.** The level (`"basic"` / `"detail"`) is now a constructor argument passed to the monitor (`QNNMonitor(level=op_tracing, ...)`), not a dispatch key.

## Risks
- Out-of-tree code that called `register_tracer(...)` or `get_tracer(...)` will get `ImportError`. There is no extension hook — adding a custom EP monitor now requires patching `commands/perf.py`.
- The substring-match flexibility is lost; any caller that relied on partial matches (e.g. registering against `"Custom"` and expecting it to match `"CustomExecutionProvider_v2"`) must now use the exact normalized EP name.
