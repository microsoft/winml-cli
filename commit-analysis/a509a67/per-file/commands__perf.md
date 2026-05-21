# src/winml/modelkit/commands/perf.py

## TL;DR
By far the heaviest change in this group (570 lines / +338 / -176 — roughly a 60% rewrite of the perf CLI). Three orthogonal migrations land together:

1. **EPDevice migration at the CLI boundary.** `WinMLSession(...)` is no longer constructed with `device=<str>` and `ep=<str|None>` kwargs. The perf command now calls `session.resolve_device(ep, device)` — a single typed resolver that accepts `device="auto"` and dispatches to auto-detection internally — to produce an `EPDevice` descriptor, then hands the descriptor to `WinMLSession`, `WinMLAutoModel.from_pretrained`, `WinMLAutoModel.from_onnx`, and `_run_onnx_benchmark`. This replaces the previous `sysinfo.resolve_device` call sites.
2. **PerfContext consumption.** `session.perf(warmup=…)` now yields a `PerfContext` (with `.stats` and `.monitor` attributes), not a bare `PerfStats`. Every `with session.perf(...) as stats` was rewritten to `as ctx`, with downstream code reading `ctx.stats` and `ctx.monitor`. The monitored path additionally passes `monitor=ep_monitor` into `session.perf(...)` so the EP monitor wraps the real benchmark iterations.
3. **Op-tracing console + CLI redesign.** The legacy `optracing/` tree (`get_tracer`, `is_qnn_profiling_available`, `display_op_trace_report`, `write_op_trace_json` from `..optracing`) is wholesale replaced by `..session.monitor.report` and explicit `_resolve_ep_monitor()` dispatch. New `--top-k` CLI option (default 5, validated to require `--op-tracing` and be `>= 1`). Smart default: `--op-tracing` without explicit `--iterations` collapses to 1 iteration. New pre-bench identity block (`_pre_bench.print_pre_bench_block`) replaces the previous bespoke `_print_model_info`. The benchmark JSON `write_json_report(result, output)` is **moved from before** the op-trace report **to after** the op-trace status check (the A3 fix the commit body advertises).

## Diff metrics
- 570 lines changed (337 insertions / 175 deletions per `--stat`).
- Largest single-file diff in the four-file group.
- Three top-level functions added (`_monitor_to_json_dict`, `_resolve_ep_monitor`, `_io_specs_from_config`, `_print_save_to_footer`); one removed (`_print_model_info`).

## Role before vs after
Role unchanged: `winml perf` benchmarks HF or `.onnx` models, renders a Rich console report, writes JSON, optionally lights up live HW monitoring and op-tracing. What changed is the wiring:

- **Pre-state**: built a `WinMLSession(onnx_path, device=str)` (or via `WinMLAutoModel`), then ran `with session.perf(warmup=…) as stats:`. Op-tracing was a *separate* synthetic pass after the benchmark, driven by `..optracing.get_tracer(...).run(iterations=min(iterations,10), warmup=min(warmup,3))` — i.e. profiling and benchmarking observed different runs.
- **Post-state**: `WinMLSession(onnx_path=…, ep_device=EPDevice)`, `with session.perf(warmup=…, monitor=ep_monitor) as ctx:`. Op-tracing is *integrated into the benchmark loop* — the EP monitor observes the same iterations that produce the latency numbers. The "post-benchmark" op-tracing section becomes a *report consumer* of `ctx.monitor.result` rather than a separate profiling driver.

## Symbol-level changes

### Imports (new)
- `from ..session import VALID_DEVICES` (top of file)
- `from ._pre_bench import print_pre_bench_block` (top of file)
- `TYPE_CHECKING` adds `EPDevice` and `EPMonitor` (`from ..session import EPDevice`, `from ..session.monitor.ep_monitor import EPMonitor`)

### Imports (removed)
- The previous `from ..optracing import ...` set (`get_tracer`, `is_qnn_profiling_available`, `display_op_trace_report`, `write_op_trace_json`) is gone — replaced by `from ..session.monitor.report import display_op_trace_report, write_op_trace_json` lazily inside the perf callback.

### New top-level helpers

- **`_monitor_to_json_dict(monitor: EPMonitor) -> dict[str, Any]`** (lines ~57–96)
  Typed accessor → transitional `to_dict()` → empty-dict dispatch (v2.4). Order: (1) `monitor.result` (for op-tracing monitors that produce `OpTraceResult`), (2) `monitor.to_dict()` (transitional for VitisAI/OpenVINO proof-of-execution), (3) `{}` (NullEPMonitor). Wraps the whole thing in `try/except` and emits `{"error": "monitor_serialization_failed: …"}` on regression rather than crashing `wmk perf` mid-output. Bundle B (error containment).

- **`_resolve_ep_monitor(ep, op_tracing, output_dir, device=None) -> EPMonitor`** (lines ~117–187 in diff)
  Explicit dispatch — no registry, no plugin loading. When `op_tracing` is set:
    - Auto-infers `ep='qnn'` when `ep` is empty and `device in ('npu','auto','')` and `QNNMonitor.is_available()`.
    - Returns `QNNMonitor(level=op_tracing, output_dir=output_dir)` if `ep_norm == 'qnn'` and available.
    - Raises `RuntimeError` with remediation hints ("install onnxruntime-qnn or onnxruntime-windowsml with QNN runtime") otherwise.
  When `op_tracing` is unset:
    - Returns `VitisAIMonitor()` for `ep_norm == 'vitisai'` if available.
    - Returns `NullEPMonitor()` otherwise.

- **`_io_specs_from_config(io_config, *, prefix) -> list[tuple[name, dtype, shape]] | None`**
  Projects the session's io_config dict into the `(name, dtype, shape)` triple shape expected by `print_pre_bench_block`. Dynamic dims (`None`) render as the string sentinel `"?"` rather than collapsing to integer `0` — fix for the previous `_print_model_info` which let dynamic dims look like `batch=0`.

- **`_print_save_to_footer(console, *, trace_json, profiling_csv)`**
  Renders the post-op-trace footer with `[dim]…[/dim]` labels. Each line is rendered only if its path is supplied.

### Removed top-level helper
- **`_print_model_info(io_config, *, task, device)`** — replaced by `print_pre_bench_block` + `_io_specs_from_config`. The new pre-bench block has model identity + device sub-blocks (model_id, task, opset, inputs, outputs, cached_onnx_path, onnx_file, device, ep) rather than just device + task + inputs/outputs.

### `BenchmarkConfig`
- Adds `op_tracing: str | None = None` field. Carried through to `PerfBenchmark._run_benchmark` for the dispatch decision.

### `PerfBenchmark`

- **`run()`**: instead of `_print_model_info(io_config, task=…, device=…)`, calls `print_pre_bench_block(...)` with full identity (`model_id`, `task`, `opset=None`, inputs/outputs via `_io_specs_from_config`, `cached_onnx_path`, `onnx_file=None`, `device=str(self._model.device)`, `ep=str(self.config.ep) if self.config.ep else "auto"`).

- **`_load_model()`** — the **EPDevice construction site**:
  ```python
  from ..session import resolve_device
  ep_device = resolve_device(ep=self.config.ep or None, device=self.config.device)
  ```
  Single call — `resolve_device` accepts `device="auto"` and dispatches to auto-detection internally. See `src/winml/modelkit/commands/perf.py:461,469`. Then `common_kwargs` now passes `"ep_device": ep_device` instead of the old `"device": <str>, "ep": <str>`. So **`WinMLAutoModel.from_pretrained` / `from_onnx` receive an `EPDevice` descriptor**, not strings.

- **`_run_benchmark()`** — dispatch flag widened: `if self.config.monitor or self.config.op_tracing: return self._run_benchmark_monitored()`. Routing both flags through the monitored path is what guarantees op-tracing has an EP monitor wrapped around `session.perf()`.

- **`_run_benchmark_simple()`** — `with session.perf(warmup=…) as ctx` (was `as stats`); stores `self._perf_ctx = ctx` and returns `ctx.stats`. The `_perf_ctx` is what the post-benchmark op-trace consumer reads from later, though the simple path won't produce a `monitor.result`.

- **`_run_benchmark_monitored()`** — fully rewritten:
  1. Resolve `ep_monitor` via `_resolve_ep_monitor(ep, op_tracing, output_dir, device)`. RuntimeError → stderr red + `SystemExit(1)`.
  2. Compute `hw_available = HWMonitor.is_available()`. If `--monitor` was set but HW is unavailable, print a yellow warning **but do not bail out** (previous code fell back to `_run_benchmark_simple` here, which would have dropped op-tracing silently — the new logic preserves op-tracing).
  3. If `hw_available`: nested `with` over `session.perf(warmup=…, monitor=ep_monitor) as ctx, hw_monitor as hw`, run `_run_monitored_loop`, capture `self._hw_metrics = hw.to_dict()`. After the context exits, call `_monitor_to_json_dict(ctx.monitor)` and stash into `self._hw_metrics["ep_proof"]` when non-empty.
  4. If not `hw_available` but op-tracing is requested: `with session.perf(warmup=…, monitor=ep_monitor) as ctx:` plus `_run_simple_loop` — the EP-monitor-only path. `_monitor_to_json_dict(ctx.monitor)` stashed into `self._hw_metrics = {"ep_proof": ep_dict}`.
  5. Stash `self._perf_ctx = ctx`, return `ctx.stats`.

- **`_perf_modules()`** (per-module mode): now constructs `WinMLSession(str(build_result.final_onnx_path), ep_device=resolve_device("cpu", "cpu"))` — explicit CPU sniff with an explicit EPDevice. `with session.perf(warmup=…) as ctx, hw_ctx as hw:` / `as ctx:`, `mod_stats = ctx.stats`. The previous indirection through a bare `stats` variable is gone.

- **`_run_onnx_benchmark()`** signature changes: `device: str` → `ep_device: EPDevice`. Internally `WinMLSession(onnx_path=…, ep_device=ep_device)`. Pre-bench identity block uses the raw ONNX path: `print_pre_bench_block(Console(stderr=True), model_id=None, task=None, opset=None, inputs=None, outputs=None, cached_onnx_path=None, onnx_file=str(onnx_path), device=str(session.device), ep=str(config.ep) if config.ep else "auto")`. `_run_monitored_loop`'s `device=` arg now sources `ep_device.device`.

### `perf()` Click callback

- **New flag `--top-k / top_k`**: `click.option("--top-k", "top_k", type=int, default=None, help="Number of top operator instances to show in the op-tracing table (default: 5, per mockup spec OP_TRACING_TOP_K_DEFAULT). Requires --op-tracing.")`.
- **`--device` Choice**: same swap as `eval.py` / `config.py` — `["auto", *sorted(VALID_DEVICES)]`.
- **`--iterations` help** updated to mention op-tracing's smart default of 1.
- **`--op-tracing` help** updated: "Currently supported only for HuggingFace model IDs and built model directories — not for direct .onnx file inputs."
- **Validation new**:
  - `--top-k` without `--op-tracing` → `click.UsageError("--top-k requires --op-tracing to be set.")`.
  - `--top-k < 1` → `click.UsageError("--top-k must be >= 1.")`.
  - `--op-tracing` on a direct `.onnx` input → `click.UsageError("--op-tracing is not yet supported for direct ONNX file inputs. Use a HuggingFace model ID or a built model directory.")` (NFR-2).
- **Smart default**:
  ```python
  if op_tracing and ctx.get_parameter_source("iterations") == click.core.ParameterSource.DEFAULT:
      iterations = 1
  ```
  Drives the "single inference produces a usable per-op trace" UX.

### ONNX-direct branch (inside `perf()`)
- Replaces `from ..sysinfo import resolve_device` with `from ..session import resolve_device`. Single-step resolution: `ep_device = resolve_device(ep=config.ep or None, device=config.device)` (`resolve_device` accepts `device="auto"` directly). Hands `ep_device` (not a string) into `_run_onnx_benchmark`. See `src/winml/modelkit/commands/perf.py:1540,1544`.

### Post-benchmark op-tracing block (the A3 fix)
- **Crucial ordering change**: in pre-state, `write_json_report(result, output)` ran **before** the `if op_tracing:` block. In post-state, the JSON write is **moved inside both branches**:
  - **Op-tracing branch**: after the status-check fail-fast guards (statuses `no_data`, `parse_failed`, the trace_result `None` case all `sys.exit(4)` *before* the JSON write), `write_json_report` only runs when the trace status is valid (`ok` or `basic_fallback`).
  - **No-op-tracing branch**: `else: write_json_report(result, output)` immediately after the console report.
  Inline doc-comment makes the intent explicit: *"Writing after the guard means a failed op-trace (exit 4 above) leaves no JSON artifact on disk."*

## Behavior / contract changes

### (a) How it constructs the EPDevice now
Two places — both at the *CLI boundary*, both single-step:

```python
# HF / PerfBenchmark._load_model  (perf.py:461,469)
from ..session import resolve_device
ep_device = resolve_device(ep=self.config.ep or None, device=self.config.device)

# ONNX direct path in perf()  (perf.py:1540,1544)
from ..session import resolve_device
ep_device = resolve_device(ep=config.ep or None, device=config.device)
```

Compared with the original `sysinfo.resolve_device` (which returned a `(category, info)` tuple), the new `session.resolve_device(ep, device)` returns a typed `EPDevice` descriptor, accepts `device="auto"` directly (dispatching to internal auto-detection), and takes either kwarg as `None` to deduce the missing one from `EP_DEVICE_SPECS` (in catalog order — first matching spec wins, which is the determinism property the commit body advertises over the old `_find_ep_device`). The resulting `EPDevice` is a frozen dataclass with at least `.ep` and `.device` fields (used as `ep_device.device` in `_run_onnx_benchmark`). Callers that only need the str category (no `EPDevice`) use the top-level `auto_detect_device()` helper instead.

There is a small extra construction in `_perf_modules`: a literal `resolve_device("cpu", "cpu")` is built for the CPU-sniff session used to inspect submodule IO. Comment marks it as "future opt: cache".

### (b) PerfContext consumption pattern
Every `with session.perf(warmup=…) as <name>:` was rewritten:
- `<name>` is now `ctx` (was `stats`). `ctx` is a `PerfContext` with at least `.stats: PerfStats` and `.monitor: EPMonitor` attributes.
- The benchmark loop reads `ctx.stats` to compute throughput / mean / etc.
- The monitored path additionally passes `monitor=ep_monitor` as a kwarg to `session.perf(...)`, so the same `PerfContext` exposes a populated `.monitor` after exit. `ctx.monitor.result` is the `OpTraceResult` for QNNMonitor (None for VitisAI / Null).
- `PerfBenchmark` stashes `self._perf_ctx = ctx` in both paths (simple and monitored) so the post-benchmark op-tracing report can read `benchmark._perf_ctx.monitor.result` (`perf()`).

### (c) Ordering of "benchmark JSON write" vs "op-trace status check"
Pre-state (broken): `write_json_report(result, output)` ran unconditionally right after `display_console_report(...)`, *before* the `if op_tracing:` block. A failed op-trace (exit 1 in the old code) therefore left a JSON artifact on disk.

Post-state (fixed):

```text
display_console_report(result, console)

if op_tracing:
    # Read benchmark._perf_ctx.monitor.result
    if trace_result is None: console.print("[red]Error:[/red] …"); sys.exit(4)
    if trace_result.status == "no_data": …; sys.exit(4)
    if trace_result.status == "parse_failed": …; sys.exit(4)
    if trace_result.status == "basic_fallback": …  # yellow notice, falls through

    write_json_report(result, output)            # ← MOVED HERE
    console.print(f"[green]Results saved to:[/green] {output}")

    if top_k is not None:
        display_op_trace_report(trace_result, console, top_n=top_k)
    else:
        display_op_trace_report(trace_result, console)
    write_op_trace_json(trace_result, trace_output)
    _print_save_to_footer(...)
else:
    write_json_report(result, output)            # No-op-tracing: write immediately
    console.print(f"[green]Results saved to:[/green] {output}")
```

Exit codes: failed op-trace ⇒ `sys.exit(4)` (NFR-2). The only "degraded-success" status is `basic_fallback` (yellow notice, exit 0 implied via fall-through).

### (d) New / changed CLI flags
- **NEW**: `--top-k INT` (default `None` → render's own default of 5 applied by `display_op_trace_report`). Validated to require `--op-tracing` and to be `>= 1`.
- **NEW (de-facto)**: `--op-tracing basic|detail` is no longer a CLI option for `optracing/` (which is now deleted) — it now drives the session.monitor pipeline via `_resolve_ep_monitor`.
- **CHANGED**: `--device` Choice list is now `["auto", *sorted(VALID_DEVICES)]` instead of a hard-coded `["auto","cpu","gpu","npu"]`.
- **CHANGED help text**: `--iterations`, `--op-tracing` (both more detailed).
- **REJECTED (new validation)**: `--op-tracing` + direct `.onnx` model path ⇒ `click.UsageError`.

### Other behavior changes
- **Pre-bench block**: every entry path now prints a structured identity block (model + device sub-blocks) via `print_pre_bench_block` before the benchmark starts. Replaces the old free-form `_print_model_info`. Dynamic dims render as `"?"` rather than `0`.
- **Module mode session construction**: the inline session for per-module sniff is now constructed with an explicit `ep_device=resolve_device("cpu", "cpu")` rather than the implicit-default behavior of the old `WinMLSession(str(path))` call (the old call worked because `WinMLSession.__init__` allowed kwarg defaults; the new `__init__` requires `ep_device` positional).
- **`_run_benchmark_monitored` no longer falls back to `_run_benchmark_simple`** when `HWMonitor.is_available()` is `False` — instead it runs with the EP monitor only. Op-tracing must not be silently degraded.

## Cross-file impact

- **Hard dependency on the new session-layer public surface**: `VALID_DEVICES`, `EPDevice`, `resolve_device` from `..session` (and `auto_detect_device` for any str-only callers); `EPMonitor`, `NullEPMonitor` from `..session.monitor.ep_monitor`; `HWMonitor` from `..session.monitor.hw_monitor`; `QNNMonitor` from `..session.monitor.qnn_monitor`; `VitisAIMonitor` from `..session.monitor.vitisai_monitor`; `display_op_trace_report`, `write_op_trace_json` from `..session.monitor.report`.
- **Deleted dependency**: `..optracing` (entire tree). The commit body confirms the old `src/winml/modelkit/optracing/` was deleted wholesale. Any caller still importing from there is now broken.
- **Sibling helper file added in same commit**: `commands/_pre_bench.py` (`print_pre_bench_block`) — `perf.py` is the only consumer.
- **`WinMLAutoModel.from_pretrained / from_onnx`** must accept `ep_device=…` kwarg and reject the old `device=…, ep=…` pair. The commit body confirms this is the case for `models/auto.py` and `models/winml/base.py`.
- **`WinMLSession.__init__`** must accept `ep_device=` positional/kwarg. `perf.py` passes it as a kwarg in two places (`_perf_modules`, `_run_onnx_benchmark`); both rely on the commit's "WinMLSession.__init__ requires ep_device positional" hard break.
- **`session.perf(warmup=…, monitor=…)` is the new signature** — the optional `monitor=` kwarg drives the EP-monitor wrap. The commit body documents this on the `WinMLSession.perf()` PerfContext yield.

## Risks / subtleties

- **`benchmark._perf_ctx` is set as an attribute on `PerfBenchmark`, not declared in `__init__`**. The post-benchmark op-tracing reader uses `getattr(benchmark, "_perf_ctx", None)` defensively, but the simple no-monitor path also stashes it now (line 533 in post-state) "for parity". If a future refactor forgets to stash it in either branch, op-tracing will silently report "no profiling data was produced" with exit 4 — a confusing failure mode.
- **`ctx` is bound by `with`** and used after the `with` block (`self._perf_ctx = ctx`). This works only if `WinMLSession.perf()`'s context manager keeps the `PerfContext` object alive after `__exit__` (it must, since `ctx.monitor.result` is the only artifact). Worth verifying in the session-layer review that `PerfContext` doesn't aggressively null-out its `.monitor` on exit.
- **Op-tracing on direct ONNX is rejected upstream** with `click.UsageError`, so `getattr(benchmark, "_perf_ctx", None)` is only ever evaluated on the HF/`PerfBenchmark` branch — but the rejection is enforced *before* the `try:` block, so a future relaxation of that rule must also thread `_perf_ctx` through `_run_onnx_benchmark`.
- **`_resolve_ep_monitor` only knows about QNN and VitisAI**. OpenVINO is mentioned in the commit body as a "placeholder for parity", but `_resolve_ep_monitor` has no `openvino` branch — so `--ep openvino --op-tracing basic` raises `RuntimeError("Op-tracing not available for EP 'openvino' …")`. Behavior is consistent with the broader commit narrative ("op-tracing currently requires QNN") but the error message says "device {device!r}", which renders as `device None` if the user didn't pass `--device` — minor UX glitch.
- **`int(d) if d is not None else "?"` in `_io_specs_from_config`** — if `shape` contains string sentinels (e.g. `"batch_size"` from ONNX dynamic-axis names) that are not `None`, the `int()` will raise `ValueError`. The pre-bench helper is then crashed inside `_io_specs_from_config`. Worth confirming that the caller normalizes string-sentinels to `None` before reaching this function (likely yes, since the old `_print_model_info` rendered shapes as `{shape!s}` and accepted any value).
- **`ctx.get_parameter_source("iterations")` is Click ≥ 8.0 API.** If the repo pins an older Click, this raises `AttributeError`. Worth a pyproject pin verification.
- **`raise SystemExit(1) from None` and `sys.exit(4)`** — two different exit-code conventions inside the same callback: `SystemExit(1)` for monitor-resolution failure (Bundle B), `sys.exit(4)` for op-trace status fail. The asymmetry is intentional per the commit body (op-tracing failure is exit 4), but a casual reader may not infer that exit 1 / 4 mean different things to CI.
- **`_monitor_to_json_dict` catches `Exception`** — broad swallow with a logged warning and a `{"error": ...}` sentinel. Good for run continuity, bad if a monitor's serializer raises `KeyboardInterrupt` (caught and turned into an "error" sentinel). Minor.
- **`output_dir = self.config.output_path.parent if self.config.output_path else Path.cwd()`** in `_run_benchmark_monitored` doesn't match `output_dir = output.parent if output else Path.cwd()` used in the post-benchmark section: the former dereferences `self.config.output_path`, the latter uses the `output` Click param (which is also written into `config.output_path` upstream, so they should agree — but they're separately computed and could drift in a future edit).

## Open questions / TODOs surfaced

- **In-code comment, `_perf_modules`**: `# CPU sniff — uses live resolve_device; future opt: cache`. The `resolve_device("cpu", "cpu")` call is per-module-call; a single cached `EPDevice` for the CPU sniff loop would save catalog lookups.
- **`opset=None`** is passed to `print_pre_bench_block` in both call sites with the comment "opset is not currently extracted on this path; pass None." Pre-bench has the rendering scaffold but no data source.
- **NFR-2 carve-out**: `--op-tracing` on direct `.onnx` is rejected. The TODO is implicit — `_run_onnx_benchmark` needs to thread the EP monitor through `session.perf` to enable this combination.
- **Proof-of-execution monitors (VitisAI / OpenVINO) still use `to_dict()`** rather than the typed `result` accessor. Doc-comment on `_monitor_to_json_dict` explicitly defers this: *"to be replaced by a typed `proof` accessor in a follow-up PR (see PRD OQ-6)."*
- **`--compare-devices`** is still listed as "Not yet implemented" and short-circuits with a yellow warning. Unchanged from pre-state.
- **`--hf-model` deprecation** is unchanged — still printed as a `DeprecationWarning` and aliased into `model_id`.
