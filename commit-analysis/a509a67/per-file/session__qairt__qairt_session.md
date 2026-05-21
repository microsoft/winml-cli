# src/winml/modelkit/session/qairt/qairt_session.py

## TL;DR
`WinMLQairtSession` is a thin `WinMLSession` subclass that swaps the ORT `ModelCompiler` pipeline for the Qualcomm QAIRT SDK (subprocess + venv + bin->ONNX wrapper). The commit migrates its surface to the new `EPDevice` API: `__init__` now takes `ep_device: EPDevice | None` (defaulting to `resolve_device("qnn", "npu")`) instead of `device: str = "qnn"`, and `_create_inference_session` switches from the deleted instance method `self._build_session_options(self._device)` to the new free function `_build_session_options(self._ep_device, self._ep_config, None, self._base_session_options)`.

## Diff metrics
- Lines added: 10
- Lines removed: 3

## Role before vs after
- **Before:** `WinMLQairtSession(onnx_path, device="qnn", ep_config=None)` — passed `device="qnn"` straight through to `WinMLSession`, which would map it through `DEVICE_POLICY_MAP`/`_EP_NAME_MAP`. `_create_inference_session` called the now-deleted `self._build_session_options(self._device)` instance method.
- **After:** `WinMLQairtSession(onnx_path, ep_device: EPDevice | None = None, ep_config=None)` — when `ep_device is None`, defaults to `resolve_device("qnn", "npu")` and forwards an `EPDevice` to `WinMLSession`. `_create_inference_session` calls the new module-level `_build_session_options` free function with the full `(ep_device, ep_config, monitor=None, base_session_options)` tuple.

## Symbol-level changes

- **`WinMLQairtSession.__init__`** — signature-changed
  - Old: `(onnx_path, device: str = "qnn", ep_config=None)`.
  - New: `(onnx_path, ep_device: EPDevice | None = None, ep_config=None)`.
  - Added a default-resolution shim: `if ep_device is None: ep_device = resolve_device("qnn", "npu")`.
  - Super-call switched from `super().__init__(onnx_path, device=device, ep_config=ep_config)` to `super().__init__(onnx_path, ep_device, ep_config=ep_config)`.
  - Note: the default `ep_device=None` + auto-resolve preserves backward "no-arg construction" ergonomics, which is **inconsistent with `WinMLSession`** (where `ep_device` is required positional with no default). Intentional convenience asymmetry for the QAIRT-specific subclass since the EP is fixed to QNN.

- **`WinMLQairtSession._create_inference_session`** — refactored
  - Old call: `sess_options = self._build_session_options(self._device)` (instance method, now deleted).
  - New call: local lazy import `from ..session import _build_session_options`, then `sess_options = _build_session_options(self._ep_device, self._ep_config, None, self._base_session_options)`.
  - Behavior otherwise unchanged: `ort.InferenceSession(str(self._ctx_path), sess_options=sess_options)`, set state `SessionState.COMPILED`, log providers.

- **All other QAIRT methods** (`compile`, `_resolve_sdk_path`, `_compile_to_qnn_bin`, `_create_context_bin_info`, `_wrap_bin_to_onnx`) — unchanged.

## Behavior / contract changes
- Callers can no longer pass `device="qnn"`; the kwarg is removed (hard break). The replacement is `ep_device=resolve_device("qnn", "npu")` or simply omitting the argument to accept the QNN+NPU default.
- Auto-default `resolve_device("qnn", "npu")` runs at construction time — will raise from `.ep_device` if QNN isn't discoverable on the current host (previously a `device="qnn"` ctor was lazy and only surfaced device issues at compile/run time).
- `_create_inference_session` inherits the new `_build_session_options` contract: may raise `DeviceNotFound` / `AmbiguousMatch` if the (qnn, npu, vendor_id, device_id) handle isn't present.

## Cross-file impact
- **Imports added:**
  - `from .. import EPDevice, resolve_device` (at module level)
  - `from ..session import _build_session_options` (lazy, inside `_create_inference_session`)
- **Imports removed:** none.
- **Depends on:** `..session` (`WinMLSession`, `SessionState`, `_build_session_options`), parent package `..` (`EPDevice`, `resolve_device`), `...utils.python_env.ensure_venv`, `onnxruntime`, `onnxruntime.tools.qnn.gen_qnn_ctx_onnx_model`.
- **Depended on by:** callers that explicitly select the QAIRT compile pipeline (likely `compiler/stages/compile.py` or model variants — search `WinMLQairtSession` across the tree to confirm). Not surveyed in this scope.
- **Confirms:** This file is a thin parallel subclass, not a divergent reimplementation. It reuses everything from `WinMLSession` except `compile()` (overridden for QAIRT SDK) and `_create_inference_session` (because EPContext model is wrapped from the QAIRT-produced .bin, not produced by `ort.ModelCompiler`).

## Risks / subtleties
- **Constructor asymmetry vs parent:** `WinMLQairtSession(onnx_path)` works (auto-defaults to qnn/npu); `WinMLSession(onnx_path)` is a TypeError. This is intentional convenience but a source of future confusion.
- **`resolve_device("qnn", "npu")` at import-time / construct-time** depends on `WinMLEPRegistry` having been bootstrapped. If QNN isn't installed locally, construction now fails earlier than before.
- **`_ep_config` is read for `qnn_sdk_root`** (`ep_config.qnn_sdk_root if ep_config else None`) but the rest of QAIRT pipeline (`_compile_to_qnn_bin`, etc.) ignores it. The new `_build_session_options` call passes `ep_config` through, so `ep_config.provider_options` (if any) will now flow into the QNN runtime session — previously the deleted instance method took only `device`, so any user `provider_options` were silently dropped here. **This is a behavior change**: callers who set `EPConfig.provider_options` will now see them applied to the QAIRT-EPContext runtime session.
- **Lazy import inside `_create_inference_session`** (`from ..session import _build_session_options`) avoids a circular import risk; the module-level import path goes `.. import EPDevice, resolve_device` already, which presumably re-exports without pulling in `_build_session_options`.
- **The `_build_session_options` import targets a private (`_`-prefixed) symbol** in a sibling module. The CLAUDE.md cardinal rules don't forbid this within the `session/` package, but the call from `qairt/qairt_session.py` is one step outside `session/session.py`. Since `qairt/` is a subpackage of `session/`, the internal reach is acceptable; flagged for awareness.

## Open questions / TODOs surfaced
- Should `_build_session_options` be re-exported via `session/__init__.py` (or `session._internal`) so subpackages don't import a `_`-prefixed symbol directly?
- Now that `ep_config.provider_options` flows through `_build_session_options` at the QAIRT runtime session, are there QNN-SDK-specific options that would conflict? Worth a regression test on the QAIRT path.
- Default `ep_device=None` + auto-resolve hides the EP/device choice — should `WinMLQairtSession` instead expose only a `device` parameter (since EP is fixed to "qnn") to make the asymmetry-with-parent intentional and discoverable?
