# Review: `src/winml/modelkit/models/auto.py`

**Status:** modified
**Lines added/removed:** 22+ / 16-

## 1. Purpose

`WinMLAutoModel` is the top-level factory for constructing inference-ready
`WinMLPreTrainedModel` subclasses, mirroring HuggingFace's `AutoModel`
pattern. It provides two public entry points: `from_pretrained()` for the
full HF pipeline (config → load → export → optimize → [quantize] →
[compile]) and `from_onnx()` for the pre-exported ONNX fast path. This
diff removes the loose `device: str` and `ep: str | None` arguments from
both entry points and replaces them with a single, pre-resolved `EPDevice`
value object.

## 2. Changes summary

- `from_onnx()` and `from_pretrained()` — remove `device: str = "auto"` and
  `ep: str | None` parameters; add `ep_device: EPDevice` (positional, no
  default, keyword-forbidden via `*`-separator on `from_onnx`, positional
  on `from_pretrained`).
- Import `short_ep_name` from `session.ep_device` (runtime) and `EPDevice`
  from the same module (type-checking only).
- All internal pass-through sites (`generate_onnx_build_config`,
  `generate_hf_build_config`, `build_onnx_model`, `build_hf_model`,
  `winml_class(...)`) updated to extract `ep_device.device` and
  `short_ep_name(ep_device.ep)` instead of forwarding the raw strings.
- `from_pretrained()`'s ONNX fast-path no longer pops `kwargs["ep"]`; the
  `ep` key was the only `kwargs` consumer for that branch, so the `kwargs`
  forwarding is now cleaner.

## 3. Per-symbol review

### `WinMLAutoModel.from_onnx`

- **Role:** Build from a pre-exported ONNX file (optimize → [quantize] → [compile]).
- **Signature:** `def from_onnx(cls, onnx_path, *, ep_device: EPDevice, task, config, precision, cache_dir, use_cache, force_rebuild, skip_build, **kwargs) -> WinMLPreTrainedModel`
- **Behavior:** Generates a `WinMLBuildConfig` using `ep_device.device` and
  `short_ep_name(ep_device.ep)`, checks for compiled-model skip, runs
  `build_onnx_model`, and wraps the result in the appropriate
  `WinMLPreTrainedModel` subclass. The `skip_build` / `is_compiled_onnx`
  branch passes `ep_device` directly to `winml_class(...)`.
- **Invariants:**
  - `ep_device` must be pre-resolved before the call; the factory no longer
    resolves hardware internally.
  - `short_ep_name(ep_device.ep)` converts the canonical EP name (e.g.,
    `"QNNExecutionProvider"`) to the short form (e.g., `"qnn"`) consumed
    by `generate_onnx_build_config(ep=...)` and `build_onnx_model(ep=...)`.
    This bridge call is present at lines 149 and 191.
- **Risks / concerns:**
  - `ep_device` is keyword-only (after the `*`) so callers cannot pass it
    positionally. Existing call sites that previously supplied `device=` or
    `ep=` as kwargs will get `TypeError` at runtime — no static check will
    catch this until callers are updated.
  - The `kwargs.pop("ep", None)` removal in `from_pretrained()` at line 278
    is correct but silent: if any call site still passes `ep=` as a kwarg
    to `from_pretrained()`, it will now reach `from_onnx(**kwargs)` and
    raise `TypeError` (unexpected keyword). The old pop silently discarded
    it. Prefer an explicit check or rely on tests to catch this.
  - `skip_build` branch (line 165-169) constructs `winml_class(onnx_path,
    config=None, ep_device=ep_device)`. If `winml_class.__init__` has a
    different parameter order, this will silently bind incorrectly. Since
    `base.py` now takes `(onnx_path, ep_device, config)` (see base review),
    the keyword argument here prevents the positional mismatch.
- **Tests:** `tests/unit/models/auto/test_auto_onnx.py` — `TestFromOnnx`
  suite covers config generation, `short_ep_name` forwarding, `ep_device`
  pass-through, and the ONNX fast-path delegation.

---

### `WinMLAutoModel.from_pretrained`

- **Role:** Full pipeline factory: HF model → inference wrapper, or ONNX fast-path delegate.
- **Signature:** `def from_pretrained(cls, model_id_or_path, ep_device: EPDevice, *, task, config, precision, cache_dir, use_cache, force_rebuild, trust_remote_code, shape_config, **kwargs) -> WinMLPreTrainedModel`
- **Behavior:** If `model_id_or_path` is an `.onnx` file, delegates directly
  to `from_onnx()` with `ep_device`. Otherwise runs
  `generate_hf_build_config`, `load_hf_model`, `build_hf_model`, then
  wraps in `winml_class(onnx_path, config=hf_config, ep_device=ep_device)`.
  The `ep` forwarded to `build_hf_model` (line 347) comes from
  `config.compile.ep_config.provider` (the resolved build config), not from
  `ep_device.ep` — this is intentional and correct.
- **Invariants:**
  - `ep_device` is positional (position 2 after `cls` and
    `model_id_or_path`), not keyword-only. Callers must supply it. The
    docstring states "Required."
  - `short_ep_name(ep_device.ep)` is called at line 295 when constructing
    the build config, converting the canonical EP to the short form.
- **Risks / concerns:**
  - **Breaking positional API**: `ep_device` is in position 2. Any existing
    call site that positionally passed `task` or `config` (unlikely but
    possible via positional style) would silently bind to `ep_device`
    instead. All callers must be audited.
  - **Audit Gap #1 — EPDevice end-to-end**: `build_hf_model(ep=resolved_ep,
    device=ep_device.device)` at line 339-348 takes `resolved_ep` from the
    compile config, not from `ep_device.ep`. If `config.compile` is `None`
    (fp32/CPU path), `resolved_ep` is `None`. This is intentional but
    represents a subtle split: `ep_device.ep` is used for the config-gen
    phase; the compile phase uses what the config system resolved. Any
    mismatch between the two would go undetected.
  - There is no validation that `ep_device.ep` and the compile config's
    `ep_config.provider` are consistent. A caller that passes a QNN
    `ep_device` but a CPU `config` override could produce a silent
    mismatch.
- **Tests:** `tests/unit/models/auto/test_auto_onnx.py` (`TestFromPretrained`),
  `tests/unit/commands/test_perf_cli.py` (mock-level coverage).

---

### `short_ep_name` (imported, not defined here)

- **Role:** Bridge: canonical EP name → short alias, consumed by legacy `ep=` parameters.
- **Usage locations:** lines 149, 191, 295 — all three call sites are present and correct.
- **Risks:** If `ep_device.ep` holds an unknown canonical name not in
  `_CANONICAL_TO_SHORT`, `short_ep_name` falls back to
  `canonical.removesuffix("ExecutionProvider").lower()`. This will produce
  an unrecognized short name that downstream validators (e.g.,
  `resolve_precision`'s `VALID_EPS` check) will reject with `ValueError`.
  The failure is explicit, not silent.

## 4. Cross-cutting

- The `from_pretrained()` docstring still references `device="auto"` and
  `"npu > GPU > CPU"` auto-detection in the class-level docstring
  (`from_pretrained("model.onnx", device="npu")` example at line 78). This
  is stale and will mislead readers. Should be updated to show the
  `ep_device=resolve_device(...)` pattern.
- Deferred work: `# TODO: run analyze_onnx` at line 163 predates this diff.
- The `from_onnx` parameter for `skip_build` is not present in `from_pretrained`'s
  delegation call (line 269-279). If `skip_build=True` is needed from the
  HF path it must be passed via `**kwargs`. This is a pre-existing gap not
  introduced by this diff.

## 5. Confidence level

High — the mechanical substitution is complete and consistent. The
remaining risk is caller-side breakage from the positional `ep_device`
addition to `from_pretrained`, which tests partially cover.

## 6. Verbatim risk inventory

| # | Location | Risk |
|---|----------|------|
| R1 | `auto.py:210` | `ep_device` is positional on `from_pretrained`; any caller that previously relied on positional order for subsequent args will silently misroute. |
| R2 | `auto.py:347-348` | `resolved_ep` from compile config vs `ep_device.ep` — no consistency assertion; mismatch possible when user passes partial `config` override. |
| R3 | `auto.py:78` (docstring) | Class-level docstring still shows old `device="npu"` API; will mislead callers. |
| R4 | `auto.py:149,191,295` | `short_ep_name(ep_device.ep)` — unknown canonicals silently fall back to suffix-strip; downstream `VALID_EPS` check is the only guard. |
