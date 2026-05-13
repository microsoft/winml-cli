# Review: `src/winml/modelkit/commands/perf.py`

**Status:** modified (large — CLI orchestrator)
**Lines added/removed:** ~230+ / ~130-

## 1. Purpose

`perf.py` is the CLI orchestrator for `wmk perf`. It manages two benchmark paths:
the HF/PerfBenchmark path (model load → build → compile → benchmark) and the direct
ONNX path (`_run_onnx_benchmark`). This PR integrates op-tracing as a first-class
feature alongside the existing `--monitor` flag, threads `EPDevice` through all
session-creation call sites replacing the legacy `device=` string, introduces the new
`--top-k` option, and rewires the op-tracing report section to consume `OpTraceResult`
via the `perf_ctx.monitor.result` accessor rather than invoking a standalone profiler.

## 2. Changes summary

- Import `print_pre_bench_block` from new `_pre_bench` module; remove the old
  `_print_model_info` function and replace all its call sites.
- New TYPE_CHECKING imports: `EPDevice`, `EPMonitor`.
- New module-level function `_monitor_to_json_dict`: typed dispatch from EPMonitor to
  JSON-serializable dict with error containment.
- New module-level function `_resolve_ep_monitor`: explicit EP→monitor dispatch for
  both op-tracing and proof-of-execution monitors; replaces the inline dict lookup.
- `BenchmarkConfig.op_tracing` field added.
- `PerfBenchmark.run`: replaces `_print_model_info` with `print_pre_bench_block`.
- `PerfBenchmark._load_model`: imports `resolve_device_category` (sysinfo) and
  `resolve_device` (ep_device); resolves to `EPDevice`; passes `ep_device=` instead
  of `device=` + `ep=` to `WinMLAutoModel`.
- `PerfBenchmark._run_benchmark`: routes to monitored path when `op_tracing` is set
  (not just `monitor`).
- `PerfBenchmark._run_benchmark_simple`: stores `ctx` as `self._perf_ctx`; returns
  `ctx.stats`.
- `PerfBenchmark._run_benchmark_monitored`: major rewrite — uses `_resolve_ep_monitor`,
  passes `monitor=ep_monitor` to `session.perf()`, handles HW-unavailable gracefully
  without falling back to simple path.
- New `_io_specs_from_config`: extracts typed `(name, dtype, shape)` triples for the
  pre-bench panel; replaces the ad-hoc inline loop.
- New `_print_save_to_footer`: renders op-trace / CSV save paths after the report.
- `_run_onnx_benchmark`: signature change from `device: str` to `ep_device: EPDevice`;
  uses `print_pre_bench_block`; returns `ctx.stats`.
- `_perf_modules` sniff path: now calls `resolve_device("cpu", "cpu")` for the
  per-module WinMLSession; `ctx.stats` extraction aligned.
- `perf` CLI function: adds `--top-k`, smart iteration default when `--op-tracing` is
  set without explicit `--iterations`, early `UsageError` for `--op-tracing` on direct
  ONNX, rewired op-tracing post-report block.

## 3. Per-symbol review

### `_monitor_to_json_dict`

- **Role:** Bridge between the typed `EPMonitor` hierarchy and the JSON report,
  with error containment.
- **Signature:** `def _monitor_to_json_dict(monitor: EPMonitor) -> dict[str, Any]`
- **Behavior:** Checks `monitor.result` first (op-tracing monitors); falls through to
  `hasattr(monitor, "to_dict")` for transitional proof-of-execution monitors; returns
  `{}` for `NullEPMonitor`. All paths are wrapped in a broad `except Exception` that
  logs at WARNING and returns a sentinel error dict.
- **Invariants:** Never raises; always returns a dict.
- **Risks / concerns:**
  - The broad `except Exception` swallows real bugs in serializers silently at WARNING
    level. The sentinel `{"error": "monitor_serialization_failed: ..."}` is injected
    into the JSON report, which is the right signal for automated consumers but could
    surprise a user reading the JSON manually.
  - `monitor.result` is accessed without a `hasattr` guard. If a future `EPMonitor`
    subclass defines `result` as a property that raises (not `None`-returning on
    unavailability), the catch block absorbs the exception. This is by design per the
    docstring ("Bundle B containment"), but means misbehaving monitors are silently
    degraded.
- **Tests:** `tests/unit/session/test_perf_monitor_integration.py`

### `_resolve_ep_monitor`

- **Role:** Explicit, registry-free dispatch from `(ep, op_tracing, output_dir, device)`
  to a concrete `EPMonitor` instance.
- **Signature:**
  ```python
  def _resolve_ep_monitor(
      ep: str | None,
      op_tracing: str | None,
      output_dir: Path,
      device: str | None = None,
  ) -> Any
  ```
- **Behavior:** When `op_tracing` is set, auto-infers `ep_norm = "qnn"` when `ep` is
  empty and `device` is `"npu"`, `"auto"`, or `""`, and `QNNMonitor.is_available()`.
  Raises `RuntimeError` when op-tracing is requested but QNN is unavailable or the EP
  has no op-tracing monitor. When `op_tracing` is `None`, checks VitisAI availability
  and returns `NullEPMonitor` otherwise.
- **Invariants:** Only raises `RuntimeError` on the op-tracing unsupported path;
  returns a valid monitor or `NullEPMonitor` for the proof-of-execution path.
- **Risks / concerns:**
  - Auto-inference of `ep_norm = "qnn"` when `device_norm == "auto"` (line ~155 in
    context) means that on a machine without QNN but with `--op-tracing`, the
    `QNNMonitor.is_available()` guard raises correctly. However, if QNN is installed
    but the device is not an NPU, `resolve_device("qnn", "auto")` called by the caller
    will fail at the EPDevice level — the error surfaces later and with a different
    message than the `_resolve_ep_monitor` RuntimeError. The error chain is still
    correct but the UX could be improved.
  - `device_norm == ""` as an auto-infer trigger (line ~155) means programmatic callers
    that pass `device=""` get QNN when available. This is documented in the docstring
    but is a footgun for any future caller that inadvertently passes an empty string.
  - The function returns `Any` (not `EPMonitor`) to avoid a circular-import cycle at
    module load time. The type safety gap is acceptable given the context but should be
    noted.
- **Tests:** `tests/unit/commands/test_perf_optracing.py`

### `BenchmarkConfig.op_tracing`

- **Role:** Carries the `--op-tracing` level through `PerfBenchmark`.
- **Signature:** `op_tracing: str | None = None`
- **Behavior:** Used in `_run_benchmark` routing decision and passed to
  `_resolve_ep_monitor`. `None` means no op-tracing.
- **Invariants:** If set, must be `"basic"` or `"detail"` (validated by CLI before
  reaching `BenchmarkConfig`).
- **Risks / concerns:** No validation at the dataclass level; programmatic callers
  passing an arbitrary string get `_resolve_ep_monitor` called with that string,
  which passes it to `QNNMonitor(level=...)`. Whether `QNNMonitor` validates it is
  out of scope here.
- **Tests:** `tests/unit/commands/test_perf_cli.py`

### `PerfBenchmark._load_model`

- **Role:** Resolve device + EP → `EPDevice`, then load model via `WinMLAutoModel`.
- **Signature:** `def _load_model(self) -> None`
- **Behavior:** Calls `resolve_device_category` to get `resolved_device` string, then
  builds `ep_str` from the `_default_ep_for_device` fallback map or the user's
  `config.ep`. Calls `resolve_device(ep=ep_str, device=resolved_device)` to get an
  `EPDevice`. Passes `ep_device=ep_device` to `WinMLAutoModel.from_pretrained` /
  `from_onnx`, dropping the old `device=` and `ep=` kwargs.
- **Invariants:** EPDevice resolution happens at the CLI boundary here, not inside
  WinMLAutoModel.
- **Risks / concerns:**
  - **HF auto-build crash path (audit):** `WinMLAutoModel.from_pretrained` calls into
    `compile.py` which uses `_build_session_options` and a premature
    `ort.InferenceSession` construction inside `WinMLSession.__init__`. This path was
    flagged in the audit as broken for QNN+NPU. This PR does not fix it; it changes
    only the argument name from `device=` to `ep_device=`. The crash surface is
    unchanged.
  - `_default_ep_for_device` is hardcoded inline here, duplicated in `perf.py:perf`
    (ONNX path) and `evaluate.py:_load_model`. Three copies of the same dict is a
    DRY violation.
  - `config.ep` is already `.lower()`-ized at the CLI boundary (`ep=ep.lower() if ep
    else None` in `perf()`). `resolve_device` calls `expand_ep_name` which does its
    own case-fold, so the double lower is harmless but redundant.
- **Tests:** `tests/unit/commands/test_perf_cli.py` (mock-based)

### `PerfBenchmark._run_benchmark`

- **Role:** Dispatch to simple or monitored benchmark path.
- **Signature:** `def _run_benchmark(self) -> PerfStats`
- **Behavior:** Now routes to `_run_benchmark_monitored` when either `self.config.monitor`
  or `self.config.op_tracing` is set.
- **Invariants:** The simple path is taken only when both `monitor` and `op_tracing`
  are `None`/`False`.
- **Risks / concerns:** None. The routing is clean and the docstring explains the
  parity rationale.
- **Tests:** `tests/unit/commands/test_perf_cli.py`

### `PerfBenchmark._run_benchmark_simple`

- **Role:** Execute bare benchmark with no hardware monitoring.
- **Behavior:** Now stores `ctx` as `self._perf_ctx = ctx` before returning `ctx.stats`.
  The `_perf_ctx` attribute is used by the op-tracing report section in `perf()`.
- **Risks / concerns:** `self._perf_ctx` is set as an instance attribute after the
  `with` block exits. If `_run_benchmark_simple` raises before the `with` block
  completes (e.g., inside `_run_simple_loop`), `_perf_ctx` is never set. The op-tracing
  report section uses `getattr(benchmark, "_perf_ctx", None)` which safely handles the
  missing-attribute case.
- **Tests:** `tests/unit/commands/test_perf_cli.py`

### `PerfBenchmark._run_benchmark_monitored`

- **Role:** Execute benchmark with EP monitor (op-tracing and/or VitisAI proof) and
  optionally HWMonitor for live chart.
- **Behavior:** Resolves EP monitor via `_resolve_ep_monitor`. When HW is available,
  runs both `session.perf(warmup=..., monitor=ep_monitor)` and `hw_monitor` in a
  combined `with` block, stores `hw.to_dict()` results, then dispatches
  `_monitor_to_json_dict(ctx.monitor)` for ep data. When HW is unavailable, runs with
  EP monitor only via `_run_simple_loop`.
- **Risks / concerns:**
  - When `hw_available` is `False` but `self.config.monitor` is `True`, the warning is
    printed but execution continues with the EP-monitor-only path. The user sees a
    warning but no live chart — this is the correct degraded behavior.
  - The `_run_simple_loop` call in the HW-unavailable branch (line ~605) does not call
    `_run_monitored_loop`, so the live chart is never shown even if `--monitor` was
    set. This is correct (no HW data → no chart), but the warning message does not
    explicitly say "the live chart will not be shown."
  - `self._hw_metrics` is assigned inside the HW-available branch at line 595. In the
    HW-unavailable branch it is assigned only when `ep_dict` is non-empty (line 608).
    If `ep_dict` is empty (NullEPMonitor, no op-tracing) and HW is unavailable,
    `self._hw_metrics` is never set. `_collect_results` uses
    `getattr(self, "_hw_metrics", None)` which handles the missing-attr case correctly.
- **Tests:** `tests/unit/session/test_perf_monitor_integration.py`

### `_io_specs_from_config`

- **Role:** Project `io_config` dict into typed `(name, dtype, shape)` triples for
  the pre-bench panel.
- **Signature:**
  ```python
  def _io_specs_from_config(
      io_config: dict, *, prefix: str
  ) -> list[tuple[str, str, tuple[int | str, ...]]] | None
  ```
- **Behavior:** Returns `None` when `{prefix}_names` is absent (pre-bench helper omits
  the row). Dynamic dims (`None` in shape) render as `"?"` string sentinels.
- **Invariants:** Pure function; no mutation of `io_config`.
- **Risks / concerns:** If `io_config` has unequal length name/shape/type lists
  (malformed config), the `shapes[i]` index guards (`if i < len(shapes)`) fall back to
  `()` and `""` for shape and dtype respectively. The output will be silently
  incomplete rather than raising.
- **Tests:** `tests/unit/commands/test_perf_cli.py` (implicitly via
  `print_pre_bench_block` call path)

### `_print_save_to_footer`

- **Role:** Print save-path footer lines after the op-trace report.
- **Signature:**
  ```python
  def _print_save_to_footer(
      console: Console, *, trace_json: str | None, profiling_csv: str | None
  ) -> None
  ```
- **Behavior:** Emits one line per non-None arg. Silent when both are `None`.
- **Risks / concerns:** None.
- **Tests:** `tests/unit/commands/test_perf_save_footer.py`

### `_run_onnx_benchmark`

- **Role:** Benchmark a direct ONNX file without HF model build.
- **Signature change:** `device: str` → `ep_device: EPDevice`
- **Behavior:** Creates `WinMLSession(onnx_path=..., ep_device=ep_device)`. Passes
  `ep_device.device` to `_run_monitored_loop`. No op-tracing monitor integration (the
  `--op-tracing` + ONNX path is blocked upstream with `click.UsageError`).
- **Risks / concerns:**
  - The `ep` shown in `print_pre_bench_block` is `str(config.ep) if config.ep else "auto"`.
    This is the *requested* short ep name from CLI, not `ep_device.ep` (canonical). For
    the display case this is fine — but there is a subtle inconsistency: `ep_device.ep`
    is always the canonical name (e.g. `"QNNExecutionProvider"`), while the panel shows
    `"qnn"`. Users reading the panel see the short name; users reading the JSON report
    see the canonical. Not a bug, but worth noting for documentation consistency.
  - The monitored loop passes `device=ep_device.device` (string) to `_run_monitored_loop`
    which passes it to `LiveMonitorDisplay`. This is the correct device *kind* string,
    not the canonical EP name.
- **Tests:** `tests/unit/commands/test_perf_cli.py`

### `_perf_modules` — CPU sniff path (line 773)

- **Role:** Per-module benchmark uses a CPU-only `WinMLSession` for shape sniffing.
- **Behavior:** `resolve_device("cpu", "cpu")` is called inline at line 773 to produce
  an `EPDevice` for the per-module CPU session. The comment notes "future opt: cache."
- **Risks / concerns:**
  - `resolve_device("cpu", "cpu")` is called fresh for each module instance in the loop
    (there may be tens of instances). Each call invokes `WinMLEPRegistry.get_instance().register_ep(...)`.
    This is the noted "future opt: cache" — if registry operations are expensive, the
    per-loop overhead accumulates. Not a correctness issue.
  - The per-module path never passes op-tracing through (no `op_tracing` kwarg to
    `_perf_modules`). If a user passes `--op-tracing --module BertAttention`, the
    op-tracing config field is set but `_perf_modules` is called before the op-tracing
    block, which exits early via `return`. The op-tracing report section is never
    reached. This is correct by design — op-tracing is not supported in `--module`
    mode — but is not documented in the help text for `--module`.
- **Tests:** `tests/unit/commands/test_perf_module.py`

### `perf` CLI function

- **Role:** Top-level Click command; orchestrates all paths.
- **New behavior:**
  - `--top-k` option added; validated against `--op-tracing` presence and `>= 1`.
  - Smart iteration default: when `--op-tracing` is set without explicit `--iterations`,
    collapses to `1` via `ctx.get_parameter_source`.
  - Early `UsageError` for `--op-tracing` on ONNX direct input.
  - Op-tracing report section: accesses `benchmark._perf_ctx.monitor.result`;
    dispatches `display_op_trace_report` with `top_n=top_k` or default.
  - `op_tracing` is now included in `BenchmarkConfig`.
- **Risks / concerns:**
  - `ctx.get_parameter_source("iterations") == click.core.ParameterSource.DEFAULT`
    is the smart-default check (line 1437). This is the correct Click idiom for
    distinguishing a user-supplied value from the CLI default. Risk: if Click changes
    the `ParameterSource` enum in a future version, this check silently stops working
    and all op-tracing runs get 100 iterations. Medium risk for a minor UX regression.
  - `perf_ctx = getattr(benchmark, "_perf_ctx", None)` (post-ONNX guard). On the ONNX
    path, `benchmark` is not defined in scope — the `is_onnx` branch does not create a
    `PerfBenchmark` object. The op-tracing code is inside `if op_tracing:`, and the
    ONNX + op-tracing combination is blocked earlier by `UsageError`, so this dead code
    path is never reached. But it is fragile: if the guard is ever removed, the code
    accesses an undefined `benchmark` name.
  - `sys.exit(4)` is used for missing/failed op-tracing data (lines ~1590–1605). This
    is a non-standard exit code. It is documented in the inline comment but not in
    `--help` output. Scripts consuming `wmk perf` need to handle exit 4.
  - The op-tracing report is emitted *after* `write_json_report` (which succeeds).
    If `trace_result.status == "no_data"` fires, the benchmark JSON is already saved
    but the process exits 4. The caller therefore gets a JSON file AND a non-zero exit
    code. This dual outcome may confuse CI pipelines.
  - The `--ep` help text says "overrides device-to-provider mapping" but does not
    enumerate valid short names in `--help`. Users guessing `--ep DmlExecutionProvider`
    would pass a canonical name; `expand_ep_name` would fall through to
    `canonicalize_ep_name` and pass it unmodified, which is correct but not obvious.
- **Tests:** `tests/unit/commands/test_perf_cli.py`, `tests/unit/commands/test_perf_optracing.py`

## 4. Cross-cutting concerns

- **Audit gap — HF auto-build crash:** The audit identified that
  `compile.py:_build_session_options` and a premature `ort.InferenceSession` in
  `WinMLSession.__init__` crash for QNN+NPU models. This PR does not change that path;
  it only renames the kwargs. The crash is still live on the HF `from_pretrained`
  path. The fix belongs in `session.py` / `compile.py`, not here.
- **`_default_ep_for_device` duplication:** Defined inline at three sites:
  `_load_model` (line ~472), the ONNX path inside `perf()` (line ~1552), and
  `evaluate.py:_load_model` (line ~138). Should be a module-level constant or a
  helper in `ep_device.py`.
- **Legacy `device=` callers:** All removed from this file. `WinMLSession` and
  `WinMLAutoModel` now receive `ep_device=` everywhere within this file.
- **CLI help text / mental model:**
  - `--device` help: "Device to run benchmark on" — unchanged, fine.
  - `--ep` help: "Overrides device-to-provider mapping" — correct but terse; does not
    mention that the value is the short name (not canonical).
  - `--op-tracing` help: correctly notes "not for direct .onnx file inputs."
  - `--top-k` help: mentions the default (5) but the actual default is `None` and the
    5 is enforced inside `display_op_trace_report`. If `display_op_trace_report`
    changes its default, the help text becomes stale.

## 5. Confidence level

**Medium.** The core refactor (EPDevice threading, `_resolve_ep_monitor`, monitored
path rewrite) is logically correct and well-tested. The two highest-risk items —
the HF auto-build crash path and the `sys.exit(4)` post-JSON race — are pre-existing
or design choices, not new regressions. The `_default_ep_for_device` duplication is
a maintainability debt.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| **High** | `perf.py:~501` + `compile.py` | HF auto-build path (`WinMLAutoModel.from_pretrained`) still crashes for QNN+NPU due to `_build_session_options` / premature `ort.InferenceSession` in `WinMLSession.__init__`. This PR does not fix it. |
| **Medium** | `perf.py:1577–1605` | Benchmark JSON is written before op-trace status is checked. If `trace_result.status == "no_data"`, the process exits 4 but a partial JSON artifact already exists on disk. CI pipelines may misinterpret the exit code. |
| **Medium** | `perf.py:1437` | Smart-default logic uses `click.core.ParameterSource.DEFAULT` — a Click internals check. Breaking change in Click would silently apply 100 iterations to all op-tracing runs. |
| **Medium** | `perf.py:~472` + `perf.py:~1552` + `evaluate.py:138` | `_default_ep_for_device` duplicated at 3 inline sites. A new device key (e.g. `"fpga"`) requires 3 edits. |
| **Low** | `perf.py:~1580` | `benchmark` name is referenced inside `if op_tracing:` after the `try` block. On the ONNX path `benchmark` is never defined; the op-tracing + ONNX guard at line 1527 prevents this from being reached, but the code is fragile to guard removal. |
| **Low** | `perf.py:~155` (`_resolve_ep_monitor`) | `device_norm == ""` triggers QNN auto-infer. Programmatic callers passing empty string get QNN silently. |
| **Low** | `perf.py:1128` (`_run_onnx_benchmark`) | `ep` label in pre-bench panel shows short name / `"auto"` (requested), not the resolved `EPDevice.ep` canonical name. Cosmetic inconsistency with JSON report. |
