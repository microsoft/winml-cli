# src/winml/modelkit/session/qairt/qairt_session.py

## TL;DR

QAIRT SDK session subclass — overrides `compile()` to run the QAIRT SDK pipeline (subprocess in isolated venv → .bin → cache_info.json → EPContext ONNX → InferenceSession). Constructor change: `ep_device: WinMLEPDevice | None = None`, defaulting to `auto_device(resolve_device(EPDeviceTarget(ep="qnn", device="npu")))` when omitted. The `_create_inference_session` private helper imports `_build_session_options` from the parent package `..session` — a fragile import that exposes a session-internal helper across module boundaries.

## Diff metrics

- 251 lines (parent: 233 → +18 net per commit stat).
- One class `WinMLQairtSession(WinMLSession)`.
- 6 methods on top of inheritance: `__init__`, `compile`, `_resolve_sdk_path`, `_compile_to_qnn_bin`, `_create_context_bin_info`, `_wrap_bin_to_onnx`, `_create_inference_session`.

## Role before vs after

**Before.** `WinMLQairtSession.__init__` took loose `(device: str, ep: str)` strings; internally called `WinMLEPRegistry.register_ep` to derive the handle.

**After.** Constructor takes `ep_device: WinMLEPDevice | None`. When `None`, defaults to `auto_device(resolve_device(EPDeviceTarget(ep="qnn", device="npu")))`. The constructor then delegates to `super().__init__(onnx_path, ep_device, ep_config=ep_config)`.

## Symbol-level changes

### Top-level

- `QAIRT_DEPENDENCIES: list[str]` — pinned pip requirements for the isolated venv-winml virtualenv.
- `COMPILE_QAIRT_BIN_SCRIPT: Path` — sibling-file path to `compile_qairt_bin.py`.

### `WinMLQairtSession.__init__(self, onnx_path, ep_device=None, ep_config=None)` (lines 52-74)

1. If `ep_device is None`: `target = resolve_device(EPDeviceTarget(ep="qnn", device="npu"))`; `ep_device = WinMLEPRegistry.instance().auto_device(target)`.
2. Call `super().__init__(onnx_path, ep_device, ep_config=ep_config)`.
3. Cache three artifact paths:
   - `self._bin_path = parent / f"{stem}_qnn_ctx_qnn.bin"`.
   - `self._bin_info_path = parent / f"{stem}_cache_info.json"`.
   - `self._ctx_path = parent / f"{stem}_qnn_ctx.onnx"`.
4. Resolve QNN SDK root: prefer `ep_config.qnn_sdk_root`, else `_resolve_sdk_path()`.
5. Log INFO.

### `compile()` (lines 76-117)

Override. Five steps (per docstring):
1. `ensure_venv(root_path=self._qnn_sdk_root, venv_name="venv-winml", python_version="3.10", requirements=QAIRT_DEPENDENCIES)` → returns `venv_python: Path`.
2. `self._compile_to_qnn_bin(venv_python)` → produces `.bin`.
3. `self._create_context_bin_info()` → generates `cache_info.json`.
4. `self._wrap_bin_to_onnx()` → wraps into EPContext ONNX.
5. `self._create_inference_session()` → creates ORT session from EPContext.

Idempotent on `self._session is not None`.

### `_resolve_sdk_path()` (lines 119-129)

Checks `QNN_SDK_ROOT` then `QAIRT_SDK_ROOT` env vars. Returns the first existing path. Raises `FileNotFoundError` otherwise.

### `_compile_to_qnn_bin(venv_python)` (lines 131-158)

Spawns `subprocess.run` with the venv Python + `compile_qairt_bin.py` + `--qairt-root` + `--model` + `--output-dir`. Stdout suppressed unless DEBUG; stderr captured. 600s timeout. Renames generated bin from `{stem}.bin` to `_bin_path` if necessary.

### `_create_context_bin_info()` (lines 160-189)

If `_bin_info_path` exists, skip. Else: locate `qnn-context-binary-utility.exe` under the SDK's `bin/aarch64-windows-msvc/`. Spawn subprocess with `--context_binary` and `--json_file` args. 120s timeout.

### `_wrap_bin_to_onnx()` (lines 191-232)

Imports `gen_qnn_ctx_onnx_model` from `onnxruntime.tools.qnn`. Loads `_bin_info_path` JSON, iterates `info.graphs`, calls `parse_qnn_graph`, post-processes to set `id` fields (because `parse_qnn_graph` "doesn't set id field, extract from raw JSON" — workaround for ORT bug). Calls `generate_wrapper_onnx_file` with the parsed tensors. Breaks after first graph ("Only process first graph").

### `_create_inference_session()` (lines 234-250)

Imports `onnxruntime` and `_build_session_options` from `..session`. Builds `sess_options` (passing `None` for monitor at compile time, matching the parent's compile-path). Creates `InferenceSession` from `_ctx_path`. Sets `_state = SessionState.COMPILED`. Logs INFO with `get_providers()`.

## Behavior / contract changes

1. **Default `ep_device` is QNN/NPU, fully resolved.** No string-based defaults. Cleaner than the prior pattern.
2. **The QAIRT compile flow is multi-process.** The venv-winml subprocess runs `compile_qairt_bin.py` as a child, then the parent process subprocesses `qnn-context-binary-utility.exe` for cache info, then in-process wraps the bin to ONNX, then creates the InferenceSession. Each step is bounded by a fresh subprocess timeout.
3. **Compile artifacts have specific filenames** (`_qnn_ctx_qnn.bin`, `_cache_info.json`, `_qnn_ctx.onnx`) that the qairt pipeline knows about. If a user passes a model whose parent dir already has stale artifacts with these names, the pipeline reuses them silently. Acceptable cache behavior.
4. **`compile_to_qnn_bin`'s rename step is unconditional** (line 156-158). If the generated bin already matches `_bin_path`, the rename is a no-op. Subtle but correct.
5. **`_wrap_bin_to_onnx` breaks after the first graph** (line 232). Multi-graph QNN models are not supported. Comment is explicit. Acceptable but worth documenting elsewhere.
6. **`_create_inference_session` imports `_build_session_options` from `..session`** — this is the private name from the parent package. See Risks.

## Cross-file impact

- Depends on parent `WinMLSession.__init__` accepting the `ep_device` kwarg.
- Imports `EPDeviceTarget`, `WinMLEPDevice`, `WinMLEPRegistry`, `resolve_device` from the session facade (`from .. import ...`).
- Imports `SessionState`, `WinMLSession` from the parent `session.py` module (`from ..session import ...`).
- Imports `_build_session_options` from the parent **package** `..session` (line 238) — but it's not in `session/__init__.py.__all__`. See Risks #1.
- Calls `ensure_venv` from `utils.python_env`.
- Subprocess spawns `compile_qairt_bin.py` (sibling file) and `qnn-context-binary-utility.exe` (from SDK).

## Risks / subtleties

1. **The `_build_session_options` import is fragile.** Line 238 does `from ..session import _build_session_options`. Python's `from package import name` looks up `name` as an attribute on `package` — which is `session/__init__.py`. That module does NOT export `_build_session_options`. The import succeeds only because Python falls through to `session.session._build_session_options` (the submodule attribute that's been loaded because `session/__init__.py` does `from .session import InferenceError, SessionState, WinMLSession` — which loads the `.session` submodule, making it discoverable via attribute lookup on the package). This is technically valid Python but fragile: any reorg of `session/__init__.py` that doesn't load `.session` early would break this import silently at runtime (when `_create_inference_session` runs). Safer alternatives: explicit `from ..session.session import _build_session_options` OR add `_build_session_options` to `session/__init__.py` `__all__` OR move it to a public name. See `session__session.md` Simplification #1.
2. **`compile_to_qnn_bin` does not capture stdout when DEBUG**, but pipes stderr. If the QAIRT compiler logs progress to stderr (some do), it accumulates in `result.stderr` until the subprocess exits. For a 600s compile, this could buffer megabytes. Acceptable for a compile step but worth knowing.
3. **`_create_context_bin_info` does NOT use `_suppress_native_output`** (the parent's stdout-redirect ctxmanager). The native utility might write to fd 1 directly. The current code captures via `text=True` mode but doesn't redirect. Probably fine since the utility runs in a child process, but inconsistent with the parent's compile path.
4. **`_resolve_sdk_path` checks `path.exists()` not `path.is_dir()`.** A file at `QNN_SDK_ROOT` would pass the check. Worth tightening.
5. **The `aarch64-windows-msvc` hardcoded SDK subdir** in `_create_context_bin_info` (line 171) assumes ARM64. On an x64 host using a cross-compiled SDK, the path is wrong. Acceptable today (QAIRT is ARM-focused) but worth a constant.
6. **`_wrap_bin_to_onnx` does `qnn_input_tensor_dic[tensor_name].id = ...` mutation** on the output of `parse_qnn_graph`. If `parse_qnn_graph` is fixed in a later ORT release to set `id` properly, this overwrite is benign but wasteful. The comment ("Fix: parse_qnn_graph doesn't set id field, extract from raw JSON") documents the rationale.
7. **The `disable_embed_mode` toggle inside `generate_wrapper_onnx_file`** uses `not self._embed_context`. The `_embed_context` attribute is set in the parent's `__init__` from `ep_config.embed_context if ep_config else False`. When `ep_config is None`, it's `False`, so `disable_embed_mode=True` — the bin is referenced via relative path `./{name}.bin`. When True (embed mode), the bin is embedded in the ONNX. Consistent.
8. **`compile()` does not wrap in CompilationError** like the parent does. Failures from `ensure_venv`, the subprocess timeouts, the WinRT-style JSON parsing, the ONNX wrapper, etc., all surface as their native exceptions. The parent's `compile()` catches and converts to `CompilationError` with structured context; this subclass does not. Inconsistency.
9. **`_create_inference_session` swallows the exception path** — there's no try/except. If `InferenceSession` construction fails, the user gets a raw `ort.RuntimeException` (or whatever). The parent's `compile()` wraps in `CompilationError` (line 352-363); this subclass's override does not.
10. **`subprocess.run(..., timeout=600)` raises `subprocess.TimeoutExpired`** — not caught here. The user sees a Python traceback. Acceptable for a compile timeout but worth documenting as a failure mode.

## Simplification opportunities

1. **Move `_build_session_options` import to the top of the module** (with the other imports). The lazy import inside `_create_inference_session` (line 238) is in a hot-ish path. Importing at module load is cheaper. Same for `import onnxruntime as ort` (line 236).
2. **Fix the fragile `_build_session_options` import** by either making it public in `session/__init__.py` OR importing from `..session.session` directly. The current path is implicit Python behavior.
3. **Wrap `compile()` in a try/except** that converts subprocess / JSON / ONNX failures to `CompilationError` for parity with the parent. As-is, a QAIRT compile failure gives a different exception class than an ORT ModelCompiler failure, complicating downstream error handling.
4. **`_resolve_sdk_path` could iterate `("QNN_SDK_ROOT", "QAIRT_SDK_ROOT")` once** with a single loop and check `is_dir()` instead of `exists()`. The two-line loop is fine; just tighten the predicate.
5. **`QAIRT_DEPENDENCIES` list** could be in a sibling `requirements.txt` (the venv builder might already support file paths). Currently the list is inline in a module that's primarily concerned with compile orchestration.
6. **`_wrap_bin_to_onnx`'s `break  # Only process first graph`** could be an early return after the first iteration, making the intent clearer. As-is the for-loop with break suggests the rest of the loop body would run if not for the break — but there's only one iteration of useful work.
7. **The `compile_to_qnn_bin` rename step** is conditional (`if generated_bin != self._bin_path`). The condition is almost always true (the subprocess emits a generic name; we rename to a unique one). Could be unconditional + `Path.rename` for clarity. But the explicit `if` defends against the subprocess being changed to emit the canonical name directly — defensible.
8. **The three artifact paths (`_bin_path`, `_bin_info_path`, `_ctx_path`)** could be properties (computed on demand) instead of `__init__`-set instance attrs. Marginal — they're cheap dict lookups.
9. **`compile()`'s logging is verbose** (one INFO per step). Could be one INFO at start + one at end with timing. Marginal.

## Open questions / TODOs surfaced

- Should the QAIRT compile failure modes be unified with the parent's `CompilationError` taxonomy? Today there's an asymmetry: ORT ModelCompiler failures → CompilationError; QAIRT failures → raw native exceptions.
- Multi-graph QNN models are silently truncated to the first graph. Is there a CLI-level surfacing of "this model has N graphs, only N=1 was processed"?
- The `aarch64-windows-msvc` hardcoded subdir should be parameterized OR validated against the SDK's actual layout. A SDK organized differently silently fails.
- Why does `__init__` build `_bin_path`, `_bin_info_path`, `_ctx_path` from the ONNX stem? If the user passes a model with a slash or special chars in the stem, the paths might break — best practice would be `Path(self._onnx_path.stem).with_suffix(...)` but the current code is OK for typical inputs.
- The `_build_session_options` private-name import is the most actionable cleanup; the fragility risks future breakage.
