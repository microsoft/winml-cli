# src/winml/modelkit/commands/perf.py

## TL;DR
By far the heaviest change in this batch (620 lines / +474 / -146 per
`--stat`; a near-total rewrite of the perf CLI's resolver + monitor wiring).
Two major migrations land alongside the prior v2.4 op-tracing redesign:

1. **`EPDeviceTarget` resolver at the CLI boundary** replaces the typed-string
   `device=<str>, ep=<str>` pair. Three call sites (`PerfBenchmark._load_model`,
   `_perf_modules`, and the ONNX-direct branch in `perf()`) now go through
   `resolve_device(EPDeviceTarget(ep=..., device=..., source=...))` →
   `WinMLEPRegistry.instance().auto_device(target)` to bind a typed
   `WinMLEPDevice` from the unified Tier-1 / Tier-2 dispatch
   (see `2_coreloop.md` §5.3 + §5.6).
2. **`--ep <name>@<source-tag>` syntax** via the new shared
   `EpAtSourceParamType` click ParamType (in `commands/_ep_arg.py`).
   `--ep openvino@pypi` parses to `(ep, source)` at click time, threads the
   source-tag into `EPDeviceTarget`, enabling Scenarios A.5 / A.6 from
   `2_coreloop.md` §6.2. The `--device` Choice continues to import
   `VALID_DEVICES` from `..session`.

Everything else (op-tracing report dispatch, monitor JSON, smart-default
iterations, pre-bench block, A3 ordering fix) was already in place from the
prior commit and is unchanged or lightly touched.

## Diff metrics
- 620 lines changed (474 insertions / 146 deletions per `--stat`).
- Largest single-file diff in the commands batch, alongside `sys.py`'s 618.
- New imports at top: `VALID_DEVICES`, `EpAtSourceParamType`,
  `print_pre_bench_block`. `TYPE_CHECKING` adds `WinMLEPDevice`,
  `WinMLEPMonitor` (renamed from `EPMonitor` in the previous commit).
- `BenchmarkConfig.ep_source` field added (carries the `@<tag>` half).

## Role before vs after
Role unchanged: `winml perf` benchmarks HF or `.onnx` models. What changed
is the resolver layer between CLI parsing and `WinMLSession(...)` /
`WinMLAutoModel.from_*(...)`.

- **Pre-state** (already past v2.4 op-tracing redesign): construct a
  `BenchmarkConfig` from raw `--ep <str> --device <str>` strings, hand
  the strings into `WinMLAutoModel.from_pretrained(..., ep=..., device=...)`,
  let downstream do the dispatch.
- **Post-state**: parse `--ep` via `EpAtSourceParamType` →
  `(ep_part, ep_source_part)`. Build a typed `EPDeviceTarget(ep=..., device=...,
  source=...)`. Run it through `resolve_device(...)` (pure deduction, no
  DLL load) then `WinMLEPRegistry.instance().auto_device(target)` (the
  Tier-2 dispatcher with precedence-ordered retry). The resulting
  `WinMLEPDevice` is passed via `ep_device=ep_device` kwarg to
  `WinMLAutoModel.from_*()` and `WinMLSession(...)`.

## Symbol-level changes

### Imports (added)
- Top-level: `from ..session import VALID_DEVICES`
- Top-level: `from ._ep_arg import EpAtSourceParamType`
- Top-level: `from ._pre_bench import print_pre_bench_block` (kept)
- `TYPE_CHECKING`: `from ..session import WinMLEPDevice`; the EP-monitor
  type hint is renamed from `EPMonitor` → `WinMLEPMonitor` per the
  session-layer rename.

### Imports (removed)
- `from ..utils import cli as cli_utils` (the `@cli_utils.build_config_option`
  decorator on `perf()` is gone — `perf` does not consume the global build
  config file directly).
- Old free-form `device=str` resolution helpers — replaced by
  `resolve_device + auto_device` calls.

### `BenchmarkConfig`
Single field added: `ep_source: str | None = None`. Carries the parsed
`@<tag>` half from `EpAtSourceParamType`. Threaded into both `_load_model`
and the ONNX-direct branch via `EPDeviceTarget(source=...)`.

`BenchmarkResult.to_dict()` records `config.ep_source` under
`benchmark_info.ep_source` so the persisted JSON captures the pin (helpful
for re-running with the same source).

### `PerfBenchmark._load_model()` — the EPDevice construction site
```python
from ..session import EPDeviceTarget, WinMLEPRegistry, resolve_device

target = resolve_device(
    EPDeviceTarget(
        ep=self.config.ep or "auto",
        device=self.config.device or "auto",
        source=self.config.ep_source,
    )
)
ep_device = WinMLEPRegistry.instance().auto_device(target)
```

Two-step: `resolve_device` does pure deduction (§5.3 in 2_coreloop.md —
*"Pure deduction; no DLL load, no filesystem scan, no registry I/O.
`source` passes through unchanged"*); `auto_device` does the Tier 2
DLL-load + precedence-retry (§5.6). The resulting `ep_device` is a
`WinMLEPDevice` with `.ep` (WinMLEP) and `.device` (WinMLDevice) — the
typed mirror of `ort.OrtEpDevice` (§4 class taxonomy table).

`common_kwargs["ep_device"] = ep_device` — passes the typed pair to
`WinMLAutoModel.from_pretrained` / `from_onnx`. The legacy `"device":
<str>, "ep": <str>` kwargs are gone from this call.

### `_perf_modules()` — the per-module path
```python
from ..session import EPDeviceTarget, WinMLEPRegistry, WinMLSession, resolve_device

# CPU sniff — uses live resolve_device; future opt: cache
cpu_target = resolve_device(EPDeviceTarget(ep="cpu", device="cpu"))
cpu_ep_device = WinMLEPRegistry.instance().auto_device(cpu_target)
session = WinMLSession(
    str(build_result.final_onnx_path),
    ep_device=cpu_ep_device,
)
```

The per-module sniff still hardcodes CPU (the inline comment marks the
opt: cache the resolved EPDevice across modules). The `WinMLSession`
constructor now requires `ep_device=` positionally/kw — the pre-state's
implicit-default `WinMLSession(str(path))` is gone (commit body confirms
this as a hard break in `session/session.py`).

### `_run_onnx_benchmark()`
Signature change: `device: str` → `ep_device: WinMLEPDevice`. Internally
`WinMLSession(onnx_path=onnx_path, ep_device=ep_device)`. The pre-bench
block uses `ep_device.device.device_type.lower()` for the chart's device
label — note the chain dereferences the `WinMLDevice.device_type` enum.

### `perf()` Click callback — boundary changes
- `--device` Choice list: `["auto", *sorted(VALID_DEVICES)]` — same as
  `eval.py` / `config.py` / `compile.py`. Single source of truth.
- `--ep` type: `EpAtSourceParamType()` — produces `(ep, source)` tuple
  or `None`. Help text updated:
  *"Optional ``@<source-tag>`` pins the source (e.g. ``openvino@pypi``).
  Overrides device-to-provider mapping."* — matches the §6.2 / Scenario
  A.5/A.6 spec from `2_coreloop.md`.
- `@cli_utils.build_config_option` decorator removed; `config_file`
  param dropped from the callback signature (the pre-state's config
  file consumption path is gone — `perf` is now self-contained).
- After parsing: `ep_part, ep_source_part = ep if ep else (None, None)`
  → flows into `BenchmarkConfig`.
- The ONNX-direct branch builds a fresh `EPDeviceTarget` + calls
  `resolve_device` + `auto_device` (mirrors `_load_model`).

### Op-tracing dispatch (unchanged from prior commit)
The `_resolve_ep_monitor(ep, op_tracing, output_dir, device=None)` helper
is unchanged: explicit dispatch, no registry, auto-infers `ep='qnn'` when
`device in ('npu','auto','')` and QNN is available, raises `RuntimeError`
with remediation hints otherwise. Note that this lives independently of
the new `EPDevice` resolver path — `_resolve_ep_monitor` still consumes
the raw `config.ep` string (not the resolved `WinMLEPDevice.ep`). That's a
deliberate split: the monitor needs the short EP name, not the full Tier-2
resolution, because it makes its own QNNMonitor.is_available() check
against the system-installed QNN runtime.

### `_monitor_to_json_dict(monitor: WinMLEPMonitor) -> dict[str, Any]`
Type-hint update: `EPMonitor` → `WinMLEPMonitor`. Otherwise unchanged from
the prior commit. The dispatch order is preserved: `monitor.result.to_dict()`
(typed accessor; QNNMonitor) → `monitor.to_dict()` (transitional;
VitisAI/OpenVINO) → `{}` (NullEPMonitor) → `{"error": "..."}` sentinel
on exception (Bundle B error containment).

### `BenchmarkResult.to_dict()`
Adds `"ep_source": self.config.ep_source` under `benchmark_info`. Persists
the source-tag the user typed (or None) into the perf JSON — useful for
reproducing the same Scenario B pin in a later run.

## Behavior / contract changes

### (a) `--ep openvino@pypi` is now legal
The user-facing CLI gains the `@<source-tag>` form (Scenarios A.5 / A.6
in `2_coreloop.md` §6.2). The parse happens at click-validation time via
`EpAtSourceParamType.convert()` (in `commands/_ep_arg.py`). Invalid
source-tags or malformed `@` syntax surface as `click.UsageError`
**before** the callback runs — so the user sees the error in stderr with
the normal click formatting, not a stack trace.

`BenchmarkConfig.ep_source` carries the parsed source tag into
`_load_model` / `_run_onnx_benchmark` and the persisted JSON.

### (b) Two-step resolve → auto_device
Pre-state used a single `sysinfo.resolve_device(device)` call that
returned a `(category, info)` tuple. Post-state separates intent
resolution (pure, `resolve_device`) from registration (DLL load,
`auto_device`).

This matters because:
- `resolve_device` is now safe to call from contexts that mustn't load
  DLLs (e.g., listing, dry-run, JSON config validation).
- `auto_device` does the precedence-ordered retry (which the commit
  body advertises: *"auto_device's precedence retry"*) and can be
  bypassed when the caller already has a target it trusts (e.g.,
  config-loaded JSON has the source-tag baked in).
- When the resolver can't pick an EP, the failure surfaces as
  `WinMLEPNotDiscovered` / `DeviceNotFound` (named exceptions) rather
  than a generic ValueError. `perf.py` doesn't catch them explicitly
  — they bubble up through the outer `except Exception as e` in
  `perf()` and surface as `sys.exit(4)`. Note: `compile.py` DOES
  catch them with remediation hints (cross-file inconsistency).

### (c) `WinMLSession` constructor break
The pre-state's `WinMLSession(str(path))` (no kwargs) used to work via
implicit defaults. Post-state requires `ep_device=` — three call sites
in this file rely on the hard break.

### (d) The `EPDeviceTarget` "auto" default
`ep=self.config.ep or "auto"` and `device=self.config.device or "auto"`
— both axes accept `"auto"`, and the resolver does the right thing
(§5.3 in `2_coreloop.md`: *"`"auto"` allowed on either axis"*). When
the user types neither, `--device` defaults to `"auto"` (Click default),
`--ep` parses to `None`, and the resolver's catalog ordering picks the
first compatible `(ep, device)` pair.

### (e) Smart default + validations (unchanged from prior commit)
- `--top-k` requires `--op-tracing`; `top-k < 1` rejected.
- `--op-tracing` + direct `.onnx` rejected (NFR-2).
- `--op-tracing` without explicit `--iterations` collapses to 1
  iteration (smart default).
- `ctx.get_parameter_source("iterations")` is Click ≥ 8.0; pre-existing
  constraint.

### (f) JSON write ordering (A3 fix, unchanged from prior commit)
`write_json_report(result, output)` is still moved inside both branches
of the post-benchmark `if op_tracing:` block — written only after a
valid trace status (`ok` / `basic_fallback`). Failed op-trace
(`sys.exit(4)`) leaves no JSON artifact.

## Cross-file impact
- **`session.__init__` must re-export** `VALID_DEVICES`, `EPDeviceTarget`,
  `WinMLEPRegistry`, `WinMLSession`, `resolve_device`, and (TYPE_CHECKING)
  `WinMLEPDevice`. The git status confirms `session/__init__.py` is
  modified.
- **`session.monitor.ep_monitor` is renamed** internally from
  `EPMonitor` → `WinMLEPMonitor` (the `TYPE_CHECKING` import shows this).
- **`commands/_ep_arg.py` (NEW)** is the click ParamType. `perf.py`,
  `compile.py`, `build.py`, `config.py` all consume it.
- **`commands/_pre_bench.py`** unchanged consumer.
- **`WinMLAutoModel.from_pretrained` / `from_onnx`** must accept
  `ep_device=` kwarg (was `device=, ep=` pair). Confirmed by
  `models/auto.py` modification in `git status`.
- **`WinMLSession.__init__`** requires `ep_device=`. Confirmed by
  `session/session.py` modification.
- **Op-tracing report module**: `..session.monitor.report` is lazily
  imported inside `perf()`'s op_tracing branch. Same as prior commit.

## Risks / subtleties
- **`benchmark = None` sentinel**: `_perf_ctx` is read via
  `getattr(benchmark, "_perf_ctx", None)`. If the ONNX-direct branch
  ever stops rejecting `--op-tracing` upstream, the `benchmark = None`
  sentinel hides the bug (silent `trace_result=None` → `sys.exit(4)`
  with the generic "no profiling data" message). Commit body notes
  this in the docstring on the line.
- **`resolve_device` + `auto_device` is a two-call sequence** in three
  places. A single helper `resolve_and_bind_device(target)` would
  collapse the duplication. Cf. `compile.py` which currently uses
  `resolve_device(...)` only (it doesn't call `auto_device` because
  `compile_onnx` doesn't take a `WinMLEPDevice` — it takes a config
  built from the resolved target). Cross-file inconsistency worth
  noting.
- **`config.ep` is the raw user string, not the resolved EP name**:
  the post-benchmark `display_console_report` reads `result.config.ep`
  to show the requested EP. This is correct (we want to show the
  user's typed value, not the resolved one) but the field name `ep`
  with a different shape from `actual_device` (which IS resolved) is
  inconsistent. `config.ep` and `actual_ep` would be a clearer
  symmetry with `config.device` / `actual_device`. The diff doesn't
  add `actual_ep`.
- **`_resolve_ep_monitor(ep=..., device=...)` takes the raw config
  strings** rather than the resolved `WinMLEPDevice`. This is
  intentional (the monitor doesn't need the full Tier-2 resolution),
  but it means there are two parallel resolution paths in `perf.py`:
  one for `WinMLSession` (typed) and one for `_resolve_ep_monitor`
  (string). If a user does `--ep openvino@pypi --op-tracing basic`,
  `_resolve_ep_monitor` ignores the source-tag entirely. That's OK
  for now (op-tracing is QNN-only), but the asymmetry deserves
  documentation.
- **The `_perf_modules` CPU sniff** rebuilds an EPDevice per module
  call (`resolve_device("cpu", "cpu") → auto_device(...)`). The inline
  comment marks this: *"future opt: cache"*. For a model with 24
  attention layers and `--module BertAttention`, that's 24 DLL-load
  calls; each `register_ep` should be idempotent (commit body confirms
  `register_ep` returns the cached `WinMLEP` on dll_path hit), so the
  cost is a dict lookup × 24, not 24 LoadLibrary calls. Acceptable
  for now.
- **`config_file` parameter dropped silently**: any user who relied on
  `winml perf --config foo.json` to seed defaults loses that path.
  Not flagged in the diff. The migration story is "perf's defaults are
  CLI-only; persisted configs live in `winml build`".
- **`PerfBenchmark.__init__` doesn't declare `_perf_ctx` or
  `_hw_metrics`** — both set as instance attributes during run().
  `getattr(benchmark, "_perf_ctx", None)` is the defensive read. A
  type-annotated `_perf_ctx: PerfContext | None = None` in `__init__`
  would catch typos and tighten the IDE story.
- **`from ..session import EPDeviceTarget, WinMLEPRegistry,
  resolve_device`** is duplicated as a local import in three places
  (`_load_model`, `_perf_modules`, the ONNX branch). The TYPE_CHECKING
  block already imports `WinMLEPDevice`; promoting the runtime imports
  to top-level would remove the duplication (cost: CLI startup time
  bumps by one session-package import, which is already loaded for
  `VALID_DEVICES` at the top, so the cost is effectively zero).

## Simplification opportunities
- **Three near-identical `resolve_device + auto_device` blocks**:
  `_load_model` (lines ~473-480), `_perf_modules` (lines ~778-779),
  the ONNX-direct branch (lines ~1568-1575). A
  `_bind_ep_device(ep, device, source=None) -> WinMLEPDevice` helper
  at module scope would collapse 21 lines to 3.
- **`_resolve_ep_monitor` doesn't share the Tier-2 resolution**:
  if the EP monitor selection is conceptually downstream of the
  resolved `WinMLEPDevice`, it should take the resolved EPDevice
  rather than re-parsing strings. A future refactor could collapse
  monitor selection into a method on `WinMLEPDevice` itself
  (`ep_device.monitor(op_tracing=...)`).
- **`config.ep` vs `config.ep_source` are split fields**: keeping
  them on `BenchmarkConfig` (and threading both through `EPDeviceTarget`)
  works, but the boundary between "user input" and "resolved EPDevice"
  is now blurry — `BenchmarkConfig.target: EPDeviceTarget` would
  collapse the two fields into the typed primitive.
- **`config_file` dropped**: if `perf` ever needs persisted defaults
  (e.g., a `~/.config/winml/perf.json` for CI), the right shape is
  a thin `WinMLPerfConfig` dataclass that mirrors the click options.
  Current state: nothing.
- **`benchmark = None` sentinel**: the comment says "the ONNX branch
  is rejected upstream when op_tracing is on". The sentinel could go
  away if `_run_onnx_benchmark` returns a `(result, ctx_or_none)`
  tuple — the op-tracing reader would consume `ctx_or_none` directly
  without the `getattr(benchmark, ...)` indirection.
- **Two console.print "Results saved to" lines** (one inside each
  branch of `if op_tracing:`). Same string, different code path —
  could be hoisted after the `if/else` block. Cost: a one-line if to
  check whether the JSON was actually written; current shape is
  simpler to read.

## Open questions / TODOs surfaced
- **`# CPU sniff — uses live resolve_device; future opt: cache`** in
  `_perf_modules`. Per-module loop; cache the resolved EPDevice once
  before the module-iteration starts.
- **`# opset is not currently extracted on this path; pass None.`**
  Pre-bench block has the field; no data source upstream. The
  build pipeline knows the opset (it's in the ONNX file's metadata);
  threading it through to perf's pre-bench is a small follow-up.
- **NFR-2 carve-out**: `--op-tracing` on direct `.onnx` is rejected.
  The TODO is implicit — `_run_onnx_benchmark` needs to thread the
  EP monitor through `session.perf` to enable this combination.
- **`--compare-devices` "Not yet implemented"** unchanged.
- **`--hf-model` deprecation** unchanged.
- **`_resolve_ep_monitor` only handles QNN + VitisAI** (matching prior
  commit). OpenVINO is mentioned in the commit body as a "placeholder
  for parity" but has no branch. `--ep openvino --op-tracing basic`
  raises `RuntimeError("Op-tracing not available for EP 'openvino' on
  device {device!r}...")`. Behavior consistent with the broader narrative,
  but the error renders as `device None` if the user didn't pass
  `--device` — minor UX glitch.
- **`source` axis not threaded into op-trace dispatch**: if a future
  user does `--ep qnn@msix --op-tracing basic`, the source-tag is
  consumed by `WinMLSession`'s EP binding but completely ignored by
  `QNNMonitor` (which uses its own `is_available()` probe against
  installed packages). This is the same observation as the
  `_resolve_ep_monitor` two-paths note above — not a bug, but a
  documentation gap.
