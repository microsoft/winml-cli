# Batch 01 Verification Report — commit 774f121d

**Reviewed:** 18 per-file docs (assigned batch)
**Repo:** `C:/Users/zhengte/BYOM/ModelKits/winml`
**Commit under review:** `774f121d` vs. mergebase `7a66c024`
**Verification date:** 2026-06-28

---

## Critical claims (D-01 & D-02)

### D-01 — `compiler/configs.py` missing `import warnings` with 8 live `warnings.warn(...)` call sites

**Verdict: REAL REGRESSION — CONFIRMED.**

Direct read of `src/winml/modelkit/compiler/configs.py`:

- Lines 17-22 — entire import block:
  ```python
  from __future__ import annotations

  from dataclasses import dataclass, field
  from pathlib import Path
  from typing import TYPE_CHECKING, Final
  ```
  **No `import warnings`. Also no `from typing import Any`** despite `dict[str, Any]` annotations at lines 133, 267, 273, 289 (survives only via `from __future__ import annotations` string-stringification).

- `warnings.warn(` is called at 8 sites in the eight `for_*` factories:
  - Line 161: `for_qnn`
  - Line 173: `for_cpu`
  - Line 187: `for_cuda`
  - Line 201: `for_dml`
  - Line 215: `for_nv_tensorrt_rtx`
  - Line 229: `for_openvino`
  - Line 243: `for_vitisai`
  - Line 257: `for_migraphx`

- `git diff 7a66c024..774f121d` confirms `-import warnings` was deleted in this squash and not re-added.

**Empirical confirmation:**
```
$ uv run python -c "from winml.modelkit.compiler import configs; configs.WinMLCompileConfig.for_qnn(quantize=True)"
NameError: name 'warnings' is not defined. Did you mean: 'Warning'?
```

The bare module-level import (`from winml.modelkit.compiler import configs`) succeeds — `warnings.warn` is only inside factory methods, so importing the module does not fire the NameError. The bug is dormant until any caller passes the deprecated `quantize=` kwarg to one of the `for_*` factories. **No test in `tests/unit/compiler/` exercises the `quantize=` path** (verified via grep — only `for_qnn()` / `for_provider()` calls without `quantize=` kwarg appear), so CI passes despite the latent NameError. Lethal for anyone exercising the legacy API.

---

### D-02 — `compiler/cli.py` imports `CalibrationConfig` / `QDQConfig` that no longer exist

**Verdict: REAL REGRESSION — CONFIRMED.**

Direct read of `src/winml/modelkit/compiler/cli.py`:

- Lines 14-19:
  ```python
  from .configs import (
      CalibrationConfig,
      EPConfig,
      QDQConfig,
      WinMLCompileConfig,
  )
  ```

- Direct read of `src/winml/modelkit/compiler/configs.py`: only `EPConfig` (line 35) and `WinMLCompileConfig` (line 58) are defined. **No `CalibrationConfig` class. No `QDQConfig` class.**

- Call sites in `cli.py` that reference the removed symbols:
  - Line 179: `qdq_config = QDQConfig(...)`
  - Line 184: `calibration_config = CalibrationConfig(...)`
  - Lines 193-194: `WinMLCompileConfig(qdq_config=..., calibration_config=...)` — but `WinMLCompileConfig.__init__` no longer accepts those kwargs either.

- The commit body's docstring on `configs.py` confirms intent: *"Quantization concerns (QDQ, calibration) have been moved to WinMLQuantizationConfig in modelkit.quant.config (#241)."* The deletion of the classes is intentional; the `cli.py` import update was missed.

**Empirical confirmation:**
```
$ uv run python -c "from winml.modelkit.compiler import cli"
ImportError: cannot import name 'CalibrationConfig' from 'winml.modelkit.compiler.configs'
```

The sub-CLI `python -m winml.modelkit.compiler compile ...` is **completely broken at import time** as shipped. The top-level `winml compile` entry point (in `commands/compile.py`) is unaffected — it doesn't import from `compiler/cli.py`.

---

## Per-doc verification

### 1. `analyze__runtime_checker__ep_checker.md`

Source file: `src/winml/modelkit/analyze/runtime_checker/ep_checker.py`.

**Verified:**
- Module-level `from ... import winml` removed; lazy 4-symbol import (`EPDeviceTarget`, `WinMLEPRegistry`, `resolve_device`, `short_ep_name`) inside `_get_sess_options` (lines 42-47).
- New call chain `target → resolve_device(target) → registry.auto_device(resolved) → sess_options.add_provider_for_devices([ep_device.device.ort_handle], options)` (lines 54-69) matches doc exactly.
- `self.device_type.name.lower()` and `short_ep_name(self.ep_name)` round-trips described correctly.
- `options = dict(self._provider_options[0])` — confirms "takes only the first element" claim (line 64).
- `__init__` signature unchanged at line 31-39 (still `Sequence[dict[Any, Any]] | None`).

**No false/overstated claims found.**

---

### 2. `build__hf.md`

Source file: `src/winml/modelkit/build/hf.py`.

**Verified:**
- Line 368: `"timestamp": datetime.datetime.now(datetime.UTC).isoformat()` — matches doc exactly.
- `git diff` confirms the only change in this file is the `datetime.timezone.utc` → `datetime.UTC` one-line swap (single hunk).
- Doc accurately describes this as "single-line modernization … zero behavior change."

**No false/overstated claims found.**

---

### 3. `build__onnx.md`

Source file: `src/winml/modelkit/build/onnx.py`.

**Verified:**
- Line 267: `"timestamp": datetime.datetime.now(datetime.UTC).isoformat()` — matches doc.
- `git diff` confirms the *only* change in this file is the same one-line modernization. Doc's "1 line changed" claim verified.

**No false/overstated claims found.**

---

### 4. `commands___ep_arg.md`

Source file: `src/winml/modelkit/commands/_ep_arg.py`.

**Verified:**
- `git diff 7a66c024..774f121d` confirms NEW file (`new file mode`), 98 lines per `wc`.
- `split_ep_at_source` (lines 21-63): five validation steps as documented (whitespace, multi-`@`, no-`@` shortcut, split-and-validate, source-lowercase + `VALID_SOURCE_TAGS` match).
- `EpAtSourceParamType.convert` (lines 85-98): `None`/empty pass-through, tuple-idempotency short-circuit, try/except + `self.fail`.
- `name = "ep_at_source"` (line 83).
- `from ..session.ep_device import VALID_SOURCE_TAGS` (line 18) — reaches into submodule, doc flags this as a future cleanup.

**No false/overstated claims found.**

---

### 5. `commands___live_chart.md`

Source file: `src/winml/modelkit/commands/_live_chart.py`.

**Verified:**
- Line 20: `_CHART_WINDOW_SECONDS = 15.0` (was 10.0 per `git diff`).
- Line 38: `chart_width: int = 120` (was 80 per `git diff`).
- Doc's "4 lines changed (2 insertions / 2 deletions)" matches `git diff` output.
- The 50%-more-samples math (`int(_CHART_WINDOW_SECONDS / self._poll_interval_s)` at line 134) is verified.

**No false/overstated claims found.**

---

### 6. `commands___pre_bench.md`

Source file: `src/winml/modelkit/commands/_pre_bench.py`.

**Verified:**
- `git diff` confirms NEW file (`new file mode`), 85 lines (lines 0-85 added).
- `print_pre_bench_block` signature (lines 26-38): keyword-only with 8 content fields, matches doc.
- Two render branches (model_id → HF identity card; onnx_file → ONNX-file card) at lines 46-67.
- Surface placeholder comment `# 2. Surface (placeholder; ...)` at line 69 — confirmed.
- Device sub-block (lines 72-77) renders `Device:` + `EP:` rows in a Panel titled "Device".
- `_fmt_io` helper at lines 80-85 joins `(name, dtype, shape)` triples.

**No false/overstated claims found.**

---

### 7. `commands__analyze.md`

Source file: `src/winml/modelkit/commands/analyze.py`.

**Verified:**
- `git diff` confirms exactly two hunks: device_option default `"NPU"`→`"npu"` (line 407-408 area) and docstring example `--ep ov --device GPU`→`--ep openvino --device gpu` (line 479).
- 4 lines changed per diff stat. Matches doc.

**No false/overstated claims found.**

---

### 8. `commands__build.md`

Source file: `src/winml/modelkit/commands/build.py`.

**Verified:**
- Line 30: `from ._ep_arg import EpAtSourceParamType`.
- Lines 270-276: `--ep` decorator uses `EpAtSourceParamType()` with rejection-documented help text.
- Lines 379-387: source-tag rejection block with `click.UsageError`.
- Lines 392-409: auto-select block calls `resolve_device(EPDeviceTarget(ep="auto", device="auto"))`, broad `except Exception`, logger.warning.
- Line 281: `--device` help text *still* says "Default: NPU." — doc correctly flags this as "stale doc drift."

**Note (minor overstated detail):** Doc claims `--device` Choice is changed to `["auto", *sorted(VALID_DEVICES)]`. Actual code at lines 278-282 shows `--device` *does not have a Choice type at all* (just `default=None, help=...`). The Choice swap claim is **OVERSTATED** for build.py — the swap happened for the OTHER commands (compile, config, eval, perf) but not for build.

**Overstated:** `--device` Choice swap claim — `build.py` has no Choice on `--device`; passes through as free-form `str | None`. The doc's "Choice list source-of-truth swap" framing doesn't apply here.

---

### 9. `commands__compile.md`

Source file: `src/winml/modelkit/commands/compile.py`.

**Verified:**
- Line 29: `from ..session import VALID_DEVICES, EPDeviceTarget, resolve_device`.
- Line 30: `from ..session.ep_device import DeviceNotFound, WinMLEPNotDiscovered, WinMLEPRegistrationFailed`.
- Line 33: `from ._ep_arg import EpAtSourceParamType`.
- Lines 57-60: `--device` Choice is `["auto", *sorted(VALID_DEVICES)]` with `default=None`.
- Lines 63-67: `--ep` uses `EpAtSourceParamType()`.
- Lines 174-197: resolver block with all 4 named exception handlers (DeviceNotFound, WinMLEPNotDiscovered, WinMLEPRegistrationFailed, ValueError → UsageError).
- Line 203: `click.echo(list_compilers(ep_device_resolved.device))`.
- Line 220: `config = WinMLCompileConfig.for_ep_device(ep_device_resolved)`.
- `_resolve_compile_provider` helper deletion confirmed: `git show 7a66c024:.../compile.py` had it; current file has 0 hits via grep.
- `git diff` confirms `from ..config import VALID_EPS` and `from ..config.precision import _DEVICE_TO_PROVIDER, _EP_TO_DEVICE` deleted at top.

**No false/overstated claims found.**

---

### 10. `commands__config.md`

Source file: `src/winml/modelkit/commands/config.py`.

**Verified:**
- Line 32: `from ..session import VALID_DEVICES`.
- Line 33: `from ._ep_arg import EpAtSourceParamType`.
- Line 119: `--device` Choice is `["auto", *sorted(VALID_DEVICES)]` with `default="auto"`.
- Lines 124-126: `--ep` uses `EpAtSourceParamType()`.
- Line 132: help-text rejects source-pinning with `(Source-pinning ``@<source-tag>`` is rejected: ...)`.
- Lines 258-269: source-tag rejection block — verbatim duplicate of `build.py`'s shape with command name "config".
- Lines 477-479: `from ..session import auto_detect_device; _resolved_dev = auto_detect_device() if device.lower() == "auto" else device.lower()`.

**No false/overstated claims found.**

---

### 11. `commands__eval.md`

Source file: `src/winml/modelkit/commands/eval.py`.

**Verified:**
- Line 16: `from ..session import VALID_DEVICES` added.
- Line 59: `--device` Choice is `["auto", *sorted(VALID_DEVICES)]` (per `git diff`).
- Lines 229-247: `from ..sysinfo import resolve_device; resolved_device, _ = resolve_device(device)` is **removed** (confirmed by `git diff` deleting 3 lines).
- Line 247: `device=device` (not `device=resolved_device`).
- `git diff --stat` confirms: `8 +++++---  1 file changed, 3 insertions(+), 5 deletions(-)` matches doc's "8 lines changed (3/5)".

**No false/overstated claims found.**

---

### 12. `commands__perf.md`

Source file: `src/winml/modelkit/commands/perf.py`.

**Verified:**
- Line 30: `from ..session import VALID_DEVICES`.
- Line 31: `from ._ep_arg import EpAtSourceParamType`.
- Line 33: `from ._pre_bench import print_pre_bench_block`.
- Lines 37-40 (TYPE_CHECKING): `WinMLEPDevice`, `WinMLEPMonitor` per session-layer rename.
- Lines 465-494: `_load_model()` uses `resolve_device(EPDeviceTarget(ep=..., device=..., source=self.config.ep_source))` → `auto_device(target)` → `common_kwargs["ep_device"] = ep_device`. Matches doc.
- Lines 775-782: `_perf_modules` CPU sniff with `resolve_device(EPDeviceTarget(ep="cpu", device="cpu"))` → `auto_device` → `WinMLSession(..., ep_device=cpu_ep_device)`. Matches doc.
- Lines 1102, 1115, 1165: `_run_onnx_benchmark` takes `ep_device: WinMLEPDevice`; uses `WinMLSession(onnx_path=onnx_path, ep_device=ep_device)`; reads `ep_device.device.device_type.lower()`.
- Lines 1565-1579: ONNX-direct branch in `perf()` mirrors the same resolve_device + auto_device pattern.

**Note (UNVERIFIED):** Doc's "620 lines changed" stat matches commit diff. Doc claim that this is "near-total rewrite" is meta-aesthetic.

**No false/overstated claims found.**

---

### 13. `commands__sys.md`

Source file: `src/winml/modelkit/commands/sys.py`.

**Verified:**
- Lines 43-50: imports `EPEntry`, `PyPISource`, `MSIXPackageSource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource` from `..ep_path`. Matches doc.
- Line 51: `from ..session import WinMLEP, WinMLEPRegistry`.
- Lines 66-68: `_INDENT_L2 = "    "`, `_INDENT_L3 = " " * 14`, `_INDENT_L4 = " " * 16` constants present.
- Line 541: `_describe_source(entry: EPEntry) -> dict[str, Any]`.
- Line 571: `_gather_ep_info() -> dict[str, dict[str, Any]]`.
- Line 602: lazy `from ..ep_path import EP_CATALOG`.
- Line 621-625: walks `registry.all_discovered()` → `register_ep(entry)` → `EP_CATALOG.is_compatible(entry.ep_name)`.
- Line 719: `_SOURCE_KIND_LABEL` dict.
- Line 729: `_format_devices_from_handles`.
- Line 391: `_gather_device_info()`.
- Line 459-470: TODO + `for winml_ep in registry._registered.values()` enrichment loop — confirms the "reaches into registry-internal" smell.
- Line 825: `_gather` function.
- Lines 869, 881, 886: `_render_text`, `_render_json`, `_render_compact`.
- Line 898: `_RENDERERS` dispatch table.
- Lines 994-1006, 1013: `sysinfo()` callback compressed shape.

**No false/overstated claims found.**

---

### 14. `compiler__cli.md`

Source file: `src/winml/modelkit/compiler/cli.py`.

**See "Critical claims — D-02" above. CONFIRMED.**

Additional verified claims from the doc:
- `git diff` confirms only 2 functional line changes: `+from ..session import VALID_EPS` (top of imports) and `type=click.Choice(sorted(VALID_EPS))` replacing the four-element hardcoded list.
- All other code (CalibrationConfig/QDQConfig usage at lines 175-196) is byte-identical to pre-state per `git diff` — the squash didn't touch it.

**No false/overstated claims found.**

---

### 15. `compiler__configs.md`

Source file: `src/winml/modelkit/compiler/configs.py`.

**See "Critical claims — D-01" above. CONFIRMED.**

Additional verified claims:
- Lines 17-21: imports are `from __future__ import annotations`, `from dataclasses import dataclass, field`, `from pathlib import Path`, `from typing import TYPE_CHECKING, Final` — both `warnings` and `Any` deleted as doc claims.
- Lines 24-25: `if TYPE_CHECKING: from ..session import EPDeviceTarget  # noqa: TC004`.
- Lines 28-32: `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]] = frozenset({"qnn", "openvino"})`. Grep across the codebase confirms **the constant is NEVER referenced** anywhere else — dead code as doc claims.
- Lines 96-99: `device` property preserved.
- Lines 101-119: new `for_ep_device` classmethod with lazy `from ..session import short_ep_name`.
- Lines 149-265: all 8 `for_*` factories with the deprecated `warnings.warn` call sites preserved.
- Lines 267-286: `to_dict` with `d: dict[str, Any] = {...}` + conditional `ep_device` key.
- Lines 288-322: `from_dict` with `from ..session import EPDeviceTarget as _EPDeviceTarget` lazy import.

**No false/overstated claims found.**

---

### 16. `compiler__stages__compile.md`

Source file: `src/winml/modelkit/compiler/stages/compile.py`.

**Verified:**
- Lines 18-24: `from ...session import (EPDeviceTarget, WinMLEPRegistry, WinMLQairtSession, WinMLSession, resolve_device,)`. Matches doc's "import switch" claim.
- Lines 70-92: `process()` body has the 3-way resolver, `auto_device(target)`, `WinMLSession(... ep_device=ep_device, ep_config=ep_config)`.
- Line 86: `f"Creating {session_cls.__name__} for {target.ep}/{target.device}"`.
- Lines 234-249: `_finalize_output` has lazy `from ...session import ep_to_device`, try/except for `ep_to_device(context.execution_provider)`, `ctx_patterns.insert(0, ...)` prepending the device-category pattern.
- Lines 261-263: output filename uses provider short name (`device = context.execution_provider.lower()`, `final_ctx_name = f"{original_stem}_{device}_ctx.onnx"`).
- Line 165: `_build_provider_options` is **defined but never called** within `process()` — doc claim "Dead method" confirmed (grep across the file shows only the definition, no call sites).

**No false/overstated claims found.**

---

### 17. `config__build.md`

Source file: `src/winml/modelkit/config/build.py`.

**Verified:**
- Lines 276-278: lazy imports `from ..session import auto_detect_device`, `from ..sysinfo.hardware import get_available_devices`, `from .precision import resolve_precision`.
- Lines 280-281: `available_devices = get_available_devices()`, `resolved_device = auto_detect_device() if device.lower() == "auto" else device.lower()`. Matches doc.
- Lines 570-577 in `generate_hf_build_config`: same split (lazy imports + two-helper split).
- Lines 610-614: the `else` branch with `_canonical = default_ep_for_device(resolved_device)`, `_short = short_ep_name(_canonical) if _canonical is not None else None`, `hw_provider = _short if _short != "cpu" else None`. Matches doc exactly.
- No top-level imports added/removed. Sigs unchanged.

**No false/overstated claims found.**

---

### 18. `config__precision.md`

Source file: `src/winml/modelkit/config/precision.py`.

**Verified:**
- Lines 17-27: top-level `from ..session import (VALID_DEVICES, VALID_EPS, default_ep_for_device, ep_to_device, short_ep_name)` with the documenting comment block. Matches doc.
- `_DEVICE_TO_PROVIDER`, `_EP_TO_DEVICE`, `_VALID_DEVICES`, `get_provider_for_device` are all absent (grep confirms 0 hits) — all four deletions confirmed.
- Line 226-230: `if ep not in VALID_EPS: raise ValueError(...)`, `device = ep_to_device(ep)`.
- Line 245-246: `if device not in VALID_DEVICES: raise ValueError(...)`.
- Lines 268-277: compile_provider derivation block — matches doc verbatim, including the `_short if _short != "cpu" else None` CPU guard.
- Line 167: `PrecisionPolicy.compile_provider` docstring "Short EP name (e.g. \"qnn\", \"dml\") or None for CPU." — matches doc.
- Line 202: `available_devices` docstring "from sysinfo.get_available_devices()" — matches doc.

**No false/overstated claims found.**

---

## Batch overall

| Verdict | Count |
|---------|-------|
| ✓ VERIFIED (claims fully grounded) | 17 docs |
| ⚠ OVERSTATED (1 minor) | 1 doc (commands__build.md — Choice swap claim on `--device` doesn't apply to build.py; the file has no Choice on --device) |
| ✗ FALSE | 0 docs |
| ? UNVERIFIED | 0 docs |

**Critical regressions D-01 and D-02 are both REAL — empirically confirmed via direct file reads, `git diff` against mergebase, and live Python imports.** Both fail at runtime: D-01 fires `NameError` when any caller passes the deprecated `quantize=` kwarg to a `for_*` factory; D-02 fails at module import (`ImportError: cannot import name 'CalibrationConfig'`), bricking the `python -m winml.modelkit.compiler` sub-CLI entry point. CI passes both today only because no test exercises the deprecated `quantize=` path and no test imports `compiler.cli`. The top-level `winml compile` (via `commands/compile.py`) is unaffected.

The only overstatement found in the batch is a minor mis-attribution in `commands__build.md` — it claims the `--device` Choice list was migrated to `["auto", *sorted(VALID_DEVICES)]`, but `build.py`'s `--device` option has no Choice type at all (just `default=None` and a help string). The migration happened for compile/config/eval/perf but not build. Doc is correct that other commands made the swap; only the assertion that build did too is inaccurate.

All other 16 docs check out cleanly against the source — file paths, line numbers, symbol names, signatures, deletion claims, and behavioral characterizations all match the actual code.
