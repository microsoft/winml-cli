# src/winml/modelkit/optracing/registry.py (DELETED)

## TL;DR
This file is removed. The EP-pattern → tracer-class registry is **dropped entirely** with no replacement. The new architecture has no runtime registry layer; callers explicitly instantiate the concrete `EPMonitor` subclass (e.g. `QNNMonitor()`) and pass it to `WinMLSession.perf(monitor=...)`.

## Diff metrics
- Lines deleted: 64
- Status: DELETED

## What this file did (pre-state)
Implemented a substring-matched registry that mapped `(ep_pattern, level) -> OpTracer` subclass. It:
- Maintained a module-level dict `_TRACERS: dict[str, dict[str, type[OpTracer]]]`.
- Provided `register_tracer(ep_pattern, level, tracer_class)` for adding entries.
- Provided `get_tracer(ep_name, level)` which iterated `_TRACERS.items()` and returned the first class whose pattern was a substring of `ep_name` (so `"QNN"` matched `"QNNExecutionProvider"` without hardcoding the full name).
- Eagerly registered `QNNProfiler` on import via the `_register_defaults()` helper — bound to both `("QNN", "basic")` and `("QNN", "detail")`.
- Returned `None` for unmatched lookups.

## Public symbols (pre-deletion)
- `_TRACERS: dict[str, dict[str, type[OpTracer]]]` — module-private registry storage.
- `register_tracer(ep_pattern: str, level: str, tracer_class: type[OpTracer]) -> None` — public registration API.
- `get_tracer(ep_name: str, level: str) -> type[OpTracer] | None` — public lookup API with substring matching.
- `_register_defaults()` — private auto-registration of `QNNProfiler` on import.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `_TRACERS` registry dict | **Dropped entirely.** No runtime registry exists in the new architecture. |
| `register_tracer` | **Dropped entirely.** No replacement; the registration pattern is removed from the codebase. |
| `get_tracer` | **Dropped entirely.** The substring-matched (EP-name, level) → tracer-class lookup is gone. Callers now select a monitor explicitly: `QNNMonitor(level="basic")`. |
| `_register_defaults` | **Dropped entirely.** No import-time side effects in the new monitor tree. |
| Auto-registration of `QNNProfiler` for `("QNN", *)` | **Dropped.** The single replacement subclass `QNNMonitor` is exposed via `from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor` and instantiated by name. |

The CLI / `commands/perf.py` code that previously called `get_tracer(ep_name, level)` to find the right tracer class has been rewritten to construct a `QNNMonitor` (or the appropriate monitor) directly. See `src/winml/modelkit/commands/perf.py` for the new wiring.

## Net behavior change
- The indirection layer between "an EP name" and "the tracer that handles it" is gone. Callers must know what monitor they want and import it directly.
- No more substring matching: `"QNNExecutionProvider"` → `"QNN"` pattern resolution is replaced by the caller hardcoding `QNNMonitor` at the call site (which is correct because `QNNMonitor.ep_name = "qnn"` is now a class-level invariant that pins the session to the QNN EP).
- No more import-time registry mutation. The optracing tree had a module-load side effect that registered `QNNProfiler`; the new tree has no such side effects.

## Risks
- Any out-of-tree extension that registered a custom tracer via `register_tracer("MYEP", "basic", MyTracer)` has no replacement API. The migration is: write an `EPMonitor` subclass and have callers import/instantiate it directly. This is a *deliberate* simplification but is breaking for plugin-style consumers.
- Callers that wrote dispatch code like `tracer_cls = get_tracer(session.get_providers()[0], "basic"); tracer = tracer_cls(model, output_dir=...)` need to be rewritten to switch on EP name themselves and instantiate the right monitor class. The dispatch logic moves from the registry to the caller.
- The auto-registration import-time side effect is gone — any code that depended on the side effect (e.g. "merely importing `winml.modelkit.optracing` populates the registry") will not find an equivalent. The new code has no equivalent global state.
