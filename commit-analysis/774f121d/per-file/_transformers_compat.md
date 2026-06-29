# src/winml/modelkit/_transformers_compat.py

## TL;DR
New file (+304). A side-effecting compatibility module that re-injects transformers 4.x symbols into transformers 5.x's `_LazyModule._objects` registry so that **optimum-onnx 0.1.0** — which hardcodes imports against transformers 4 internals — can still be imported. Loaded once at package-init from `winml/modelkit/__init__.py` (immediately after `_warnings`). Shims fall into three categories: (1) top-level transformers names (`CLIPFeatureExtractor`, `MT5Tokenizer`, `AutoModelForVision2Seq`) injected via `_objects` setdefault, (2) submodule attributes (`is_offline_mode`, `get_parameter_dtype`, `_CAN_RECORD_REGISTRY`, `OutputRecorder`) injected via plain setattr because submodules are regular `ModuleType`, (3) a runtime-behavior monkey-patch to `optimum.exporters.onnx.model_patcher.sdpa_mask_without_vmap` re-targeting it at the transformers 5.x mask_interface signature. Author labels the entire module a temporary band-aid: "Drop this file (and the corresponding override in pyproject.toml) once optimum-onnx 0.2+ ships with transformers 5.x compatibility."

## Diff metrics
- Lines: +304 / -0 (entire file new)
- Hunks: N/A (new file)
- Symbols defined: 2 module-level classes (`CLIPFeatureExtractor`, `MT5Tokenizer`), 3 inner classes/funcs conditionally defined (`is_offline_mode`, `get_parameter_dtype`, `OutputRecorder`, `_sdpa_mask_without_vmap_tf5`), 1 module-level attribute (`_top_objects`).

## Role before vs after
- Before: file did not exist. Optimum-onnx 0.1.0 imports would fail with `ImportError: cannot import name 'CLIPFeatureExtractor' from 'transformers'` (and similar) whenever winml.modelkit's optimum-backed conversion path was exercised on transformers 5.x. The project's pyproject.toml pinning of transformers 5.x without an offsetting compat layer made any optimum-backed export broken.
- After: importing `winml.modelkit` (which happens unconditionally for any `winml.modelkit.*` consumer) installs all required shims as a side effect. Optimum-onnx 0.1.0 imports cleanly thereafter, and the export codepath that goes through `optimum.exporters.onnx.model_patcher` uses the new transformers-5-compatible `sdpa_mask_without_vmap` instead of the broken 4-era version.

## Symbol-level changes
- **`CLIPFeatureExtractor`** (class, lines 62–74): Subclass of `CLIPImageProcessor` that emits a `UserWarning` deprecation nudge on construction, then delegates to the parent constructor. Documented trade-off: `isinstance(x, CLIPFeatureExtractor)` is False for objects built directly via `CLIPImageProcessor(...)` — accepted because no in-tree consumer performs that check.
- **`MT5Tokenizer`** (class, lines 91–114): Pure stub. `__new__`, `__init__`, and `from_pretrained` all raise `RuntimeError` with a long explanation. Justified by the vocab divergence between T5 (~32K English) and MT5 (~250K multilingual); aliasing to T5 would silently corrupt HunYuanDiT prompts. Import succeeds; instantiation fails loud.
- **`AutoModelForVision2Seq`**: aliased to `AutoModelForImageTextToText` via `_top_objects.setdefault("AutoModelForVision2Seq", AutoModelForImageTextToText)` (line 132). Note caveat: the successor's model registry isn't a 1:1 superset — `ORTModelForVision2Seq.from_pretrained` may fail for some Vision2Seq checkpoints. Accepted because winml.modelkit doesn't exercise Vision2Seq directly; the alias only unblocks the *import cascade*.
- **`is_offline_mode()`** (lines 146–148): re-implements `transformers.utils.is_offline_mode` exactly as the 4.57 version did — `TRANSFORMERS_OFFLINE` env-var check returning a bool. Patched into `transformers.utils`.
- **`get_parameter_dtype(parameter)`** (lines 158–169): tries `next(parameter.parameters()).dtype`; on `StopIteration|AttributeError` falls back to `parameter.dtype`; final fallback returns `torch.float32`. Patched into `transformers.modeling_utils`. Called by optimum's `onnx_export_from_model` at line 933.
- **`_CAN_RECORD_REGISTRY = {}`** (line 180): empty dict satisfies the import; the recorder branch never fires for any model, so output_attentions/output_hidden_states capture during export is silently skipped (acceptable per author).
- **`OutputRecorder`** (lines 190–207): a fully unused placeholder class (since `_CAN_RECORD_REGISTRY` is empty). Methods exist purely to satisfy import-time `from transformers.utils.generic import OutputRecorder`.
- **`_sdpa_mask_without_vmap_tf5(...)`** (lines 249–302): the critical runtime patch. Replaces `optimum.exporters.onnx.model_patcher.sdpa_mask_without_vmap` with a function matching transformers 5.x's mask_interface calling convention (`q_length: int`, `q_offset: int`, `device`, `**kwargs` instead of `cache_position: tensor`). Body identical to optimum's 0.1.0 implementation modulo the q_indices derivation and `prepare_padding_mask` call (drops the dropped-in-5.x `_slice=False` kwarg). Crucially: this is wrapped in `try/except ImportError` so the patch is a no-op when optimum is not installed.

## Behavior / contract changes
- Loading the module has process-global side effects on the live `transformers` package's `_objects` registry and submodule `__dict__`s. The author calls this out:
  ```
  Implementation note: transformers 5.x's top-level package is a `_LazyModule`,
  and `from transformers import <SomeClass>` triggers `_LazyModule.__getattr__`,
  which can replace `sys.modules["transformers"]` with a fresh `_LazyModule`
  instance as a side effect.
  ```
- The required order is: (1) do every `from transformers import …` upfront so replacements settle, (2) only then capture `sys.modules["transformers"]._objects` and inject. This is the durable injection point — plain `setattr(transformers, name, value)` would write to a stale `__dict__` after a subsequent lazy-module replacement.
- `_top_objects.setdefault(...)` is used (not assignment) so re-imports are idempotent and a real future transformers reintroduction of the name wouldn't be clobbered.
- For submodules (`transformers.utils`, `transformers.modeling_utils`, `transformers.utils.generic`) plain setattr is correct because they are regular `ModuleType`, not `_LazyModule`.
- The optimum monkey-patch is *runtime* behavior change, not import-only — it changes how every causal-mask-using HF model is exported. Any verification of generated ONNX produced by optimum-onnx 0.1.0 + winml.modelkit on transformers 5.x is exercising this function, not optimum's bundled one.

## Cross-file impact
- Loaded exactly once from `src/winml/modelkit/__init__.py` (added in the same commit; see `__init__.md`).
- Affects all callers that use any optimum-backed conversion (`models/auto.py`, `models/hf/*.py`, `utils/optimum_loader.py`, `compiler/stages/compile.py` indirectly). Without this file, any `from optimum.onnxruntime import ORTModelFor*` would fail at import time.
- Reaches into transformers' private internals (`_objects`, `_LazyModule`). Future transformers refactors of the lazy-loader could silently break the injection without raising an error — the shims would be set but unreachable via `from transformers import …`.

## Risks / subtleties
- **Process-global mutation of third-party private state.** `transformers._objects` and `transformers.utils.generic._CAN_RECORD_REGISTRY` are not part of transformers' public API. Any transformers minor-version bump could rename `_objects` and silently no-op the top-level shims.
- **MT5Tokenizer "loud failure" is partial.** The class blocks `__new__`, `__init__`, and `from_pretrained`, but a determined caller using `pickle.loads(...)` or `copy.deepcopy(...)` to reconstruct an instance bypasses all three. The deception window is narrow but exists.
- **AutoModelForVision2Seq alias is silently lossy.** Author flags that the successor's model registry is not a 1:1 superset. Any Vision2Seq checkpoint that the new class can't load will fail at `from_pretrained` time with an opaque error — and the user will see "AutoModelForImageTextToText" in the traceback, not the original alias they typed. Worth a sentinel that re-raises with a "this is the Vision2Seq alias" hint.
- **OutputRecorder is dead code by design.** With `_CAN_RECORD_REGISTRY = {}` it's never instantiated. But a future transformers version that hardcodes `OutputRecorder` construction *outside* the registry check would silently get this stub and produce empty recordings. The shim has no fidelity.
- **The optimum sdpa patch was written against optimum 0.1.0's body.** If `_optimum_mp.sdpa_mask_without_vmap` were updated by a 0.1.x patch release (not yet shipped) with a different body, this patch would silently replace the upgrade with the older logic. No version pin enforced.
- **Import-time cost.** Pre-loading `AutoModelForImageTextToText` and `CLIPImageProcessor` from transformers triggers transformers' full lazy-loader chain. Every `winml.modelkit` consumer now pays this even if they never touch optimum. The comment "each `from transformers import …` may swap sys.modules" suggests the author is aware.
- **`transformers.masking_utils` imports are unconditional inside the `try…except ImportError` block**, but only the optimum branch — so a transformers version that lacks `_ignore_causal_mask_sdpa` would raise a different ImportError that escapes (the `try` only catches `import optimum.exporters.onnx.model_patcher`). Defensive narrowing or a nested try would be safer.

## Simplification opportunities
- **Hoist the optimum patch into its own module** (`_optimum_compat.py`) so the top-level shims and the runtime patch can be reasoned about (and toggled) independently. The file is doing two distinct jobs today.
- **Replace the `_top_objects.setdefault(name, …)` triple with a small helper** (`def _inject_top(name, value): _top_objects.setdefault(name, value)`) — currently it's three near-identical lines with the same risk surface (typo in the name string is silent).
- **Add a version-gate** at the top: `if transformers.__version__.startswith("4."): return` — the entire file is a no-op on transformers 4.x. Today it runs unconditionally and pays the import cost regardless.
- **Make `_sdpa_mask_without_vmap_tf5` an isolated helper file** so it can be unit-tested without importing `winml.modelkit`. The current packaging makes that hard.
- **The `MT5Tokenizer` error message is duplicated as both `cls._ERROR` and `self._ERROR` references** — easy to drift. Extract to a module-level constant.

## Open questions / TODOs surfaced
- The whole file's existence depends on optimum-onnx not catching up. Tracking issue: is there an upstream optimum-onnx PR for transformers 5 support that this branch could subscribe to? Comment-only TODO at line 22–23.
- No test exercises `_sdpa_mask_without_vmap_tf5` directly. Worth a minimal unit test that exports a causal-mask-using HF model under transformers 5 and asserts the ONNX graph contains the expected mask op pattern.
- Should this file ever be unloaded? Currently the patches are permanent for the process — if a test wants to verify "without the shim, optimum-onnx fails", there's no `_uninstall_compat()` helper.
- The `AutoModelForVision2Seq` shim should probably emit a warning *somewhere* — silent aliasing means a user who deliberately reaches for Vision2Seq gets ImageTextToText with no nudge. A `__getattr__`-based proxy could log at first access.
