# Inspect Command Consolidation Design

> **Status:** Draft — pending user approval before implementation
> **Prerequisite:** Phase 1 changes (I/O extraction fix, device detection, dynamic IO config) — done in this session
> **GitHub Issues:** #247 (MUST-rule violations), #412 (config device bug), #354 (ONNX inspect groundwork)

## Problem Statement

`inspect/resolver.py` contains thin wrapper functions that re-call existing module APIs
with different names. Five functions (`detect_task`, `validate_task`, `resolve_loader`,
`resolve_exporter`, `get_build_config`) are <10 lines of real logic — they call
`loader/task.py` or `export/io.py` functions and wrap results with source labels.

Meanwhile, `loader/config.py:resolve_loader_config()` already does Steps 1-3 (load HF config,
detect task, resolve model class) in a single call that inspect doesn't use.

## What Inspect Currently Does (10 steps)

```
inspect_model(model_id)
  1. AutoConfig.from_pretrained()           ← duplicates resolve_loader_config Step 1
  2. detect_task()                          ← wraps loader/task._detect_task_from_config()
  3. resolve_loader()                       ← wraps loader/task._get_custom_model_class()
  4. resolve_exporter()                     ← wraps export/io.resolve_io_specs()
  5. resolve_winml()                        ← reads models/winml dicts
  6. compile_support_status()               ← aggregates support levels
  7. get_build_config()                     ← MODEL_BUILD_CONFIGS.get()
  8. resolve_cache()                        ← cache module
  9. resolve_processor()                    ← HF hub + Auto classes
 10. resolve_io_config()                    ← dynamic OnnxConfig attr discovery
```

## Target Architecture

```
inspect_model(model_id, task, model_type, model_class)
  1. resolve_loader_config()                ← SHARED with config command
  2. resolve_io_specs()                     ← SHARED with config command
  3. get_winml_class()                      ← from models/winml (existing)
  4. resolve_processor()                    ← inspect-only (stays)
  5. resolve_io_config()                    ← inspect-only (stays)
  6. resolve_cache()                        ← inspect-only (stays)
  7. Derive display metadata                ← source labels, support levels
  8. Format and display
```

## Profiling (why not call generate_build_config directly)

| Step | Time | Inspect needs? |
|------|------|----------------|
| resolve_loader_config() | 8.55s | Yes |
| MODEL_BUILD_CONFIGS.get() | 0.00s | Yes |
| _resolve_export_config_from_specs() | 0.25s | No — inspect calls resolve_io_specs() directly |
| _assemble_config() | 0.00s | No |
| resolve_device() | **4.83s** | **No — pure overhead** |
| resolve_precision() | 0.00s | No |

Inspect needs **loader config + I/O specs**, not the full build config.
Calling `generate_build_config()` adds 4.83s of device detection overhead for zero benefit.

## Functions to DELETE from resolver.py

| Function | Lines | Why delete |
|----------|-------|------------|
| `detect_task()` | 101-130 | `resolve_loader_config()` detects task |
| `validate_task()` | 84-98 | Inline validation into `inspect_model()` or CLI |
| `resolve_loader()` | 133-172 | `resolve_loader_config()` returns model_class |
| `resolve_exporter()` | 271-393 | Call `resolve_io_specs()` directly + `_build_tensor_infos_from_io_specs()` |
| `get_build_config()` | 481-496 | Inline `MODEL_BUILD_CONFIGS.get()` |

**~250 lines deleted.**

## Functions to KEEP in resolver.py

| Function | Lines | Why keep |
|----------|-------|----------|
| `get_known_tasks()` | 55-81 | Aggregates 3 sources for --list-tasks UI |
| `_shape_to_desc()` | 175-208 | Display helper |
| `_build_tensor_infos_from_io_specs()` | 211-268 | Converts export types → inspect display types |
| `resolve_winml()` | 396-435 | Reads models/winml dicts, returns display metadata |
| `compile_support_status()` | 438-478 | Aggregates support levels for display |
| `resolve_cache()` | 499-611 | Inspect-only, manifest + filename scanning |
| `_find_nested_configs()` | 614-639 | Dynamic nested config discovery |
| `_discover_io_attrs_from_onnx_config()` | 642-697 | Dynamic IO attr discovery from NormalizedConfig |
| `resolve_io_config()` | 700-807 | Model config attrs for display |
| `resolve_processor()` + helpers | 810-1063 | 3-strategy processor resolution |

### Note on resolve_winml()

`get_winml_class()` in `models/winml/` does the same 3-level lookup but returns the
actual class type for instantiation. `resolve_winml()` returns `WinMLInfo` (class name
string + source label + support level) for display. Different return types, same dicts.

**Future consideration:** Could call `get_winml_class()` and derive metadata post-hoc,
but current implementation is straightforward and not a duplication of logic — it's a
different consumption of the same data.

## Changes to inspect_model()

### Before (current)
```python
def inspect_model(model_id, include_hierarchy=False, task_override=None):
    hf_config = AutoConfig.from_pretrained(model_id)
    model_type = hf_config.model_type
    task, task_source = detect_task(hf_config)
    loader_info = resolve_loader(model_type, task)
    exporter_info = resolve_exporter(model_type, task, hf_config, model_id=model_id)
    ...
```

### After (proposed)
```python
def inspect_model(
    model_id=None, include_hierarchy=False, task_override=None,
    model_type=None, model_class=None,
):
    # Step 1: Shared loader resolution (same as config command)
    loader_config, hf_config, resolved_class = resolve_loader_config(
        model_id, task=task_override, model_type=model_type, model_class=model_class,
    )
    model_type = loader_config.model_type
    task = loader_config.task

    # Step 2: I/O specs via shared path
    io_specs = resolve_io_specs(model_type, task, hf_config, model_id=model_id)
    input_tensors, output_tensors = _build_tensor_infos_from_io_specs(io_specs)

    # Step 3: Derive display metadata (source labels, support levels)
    loader_info = _derive_loader_info(model_type, task, loader_config)
    exporter_info = _derive_exporter_info(model_type, task, input_tensors, output_tensors)

    # Step 4-8: Inspect-only enrichment (unchanged)
    winml_info = resolve_winml(model_type, task)
    processor_info = resolve_processor(model_id, model_type=model_type)
    io_config_info = resolve_io_config(hf_config, model_id=model_id, model_type=model_type, task=task)
    cache_info = resolve_cache(model_id)
    ...
```

### Display metadata derivation

```python
def _derive_loader_info(model_type, task, loader_config):
    """Derive LoaderInfo display metadata from resolve_loader_config results."""
    mt = model_type.lower().replace("_", "-")
    if (mt, task) in HF_MODEL_CLASS_MAPPING:
        source, level = "MODEL_CLASS_MAPPING", SupportLevel.SUPPORTED
    elif task in HF_TASK_DEFAULTS:
        source, level = "HF_TASK_DEFAULTS", SupportLevel.DEFAULT
    else:
        source, level = "TasksManager", SupportLevel.DEFAULT
    return LoaderInfo(
        hf_model_class=loader_config.model_class or "Auto (TasksManager)",
        hf_model_class_source=source,
        support_level=level,
    )

def _derive_task_source(model_type, task):
    """Derive task detection source label for display."""
    mt = model_type.lower().replace("_", "-")
    if (mt, task) in HF_MODEL_CLASS_MAPPING:
        return "HF_MODEL_CLASS_MAPPING"
    return "TasksManager"
```

## Known Output Changes

### Loader class display string
**Before:** `"Auto (TasksManager)"` for models without explicit registry entry
**After:** `"AutoModelForImageClassification"` (actual class name from resolve_loader_config)

This is intentionally more informative — users see the actual class that will be used.
If the old string must be preserved, add: `if source == "TasksManager": display = "Auto (TasksManager)"`.

## CLI Changes

Extend `wmk inspect` to match `wmk config` flags:

```
wmk inspect -m microsoft/resnet-50                    # existing
wmk inspect --model-type bert                         # NEW: model_type without model_id
wmk inspect --model-type bert --task fill-mask        # NEW: model_type + task
wmk inspect -m custom-model --model-class BertForCTC  # NEW: model_class override
```

**Note:** When `model_id` is None (e.g., `--model-type` only), `resolve_cache()` and
`resolve_processor()` must be skipped since they require a model_id. These sections
will show as empty in the output — same as how `wmk config --model-type bert` works
without a model_id.

## Error Handling

Current `inspect_model()` wraps `AutoConfig.from_pretrained()` with `ModelNotFoundError`
and `NetworkError`. After switching to `resolve_loader_config()`, which raises generic
`ValueError`, inspect must catch and re-wrap:

```python
try:
    loader_config, hf_config, resolved_class = resolve_loader_config(...)
except ValueError as e:
    if "not found" in str(e).lower() or "404" in str(e):
        raise ModelNotFoundError(str(e)) from e
    raise InspectError(str(e)) from e
except OSError as e:
    raise NetworkError(str(e)) from e
```

## Multimodal Sub-Config Note

After `resolve_loader_config()`, `hf_config` may be a sub-config (e.g., `CLIPTextConfig`
instead of `CLIPConfig`) and `model_type` may be the sub-model type (e.g., `clip_text_model`).
This is correct for I/O spec resolution — `resolve_io_specs()` needs the narrowed config.
But `resolve_io_config()` (which shows model-level attrs like vocab_size) should receive the
**parent config** for multimodal models. Implementation must preserve the parent config before
calling `resolve_loader_config()` if the parent is needed downstream.

## Dependency on Phase 1 (this session's changes)

This design builds on the Phase 1 changes already implemented:

- [x] `resolve_io_specs()` consolidation (deleted `_extract_tensor_specs_from_onnx_config`)
- [x] Dynamic IO config discovery from NormalizedConfig
- [x] Dynamic nested config discovery (`_find_nested_configs`)
- [x] Processor source attribution + HF registry lookup
- [x] `--list-tasks` flag
- [x] ONNX file detection
- [x] Config device detection fix (#412)

## Implementation Order

1. **Add `--model-type` and `--model-class` to inspect CLI**
2. **Replace Steps 1-3 in `inspect_model()` with `resolve_loader_config()` call**
   - Wrap with `ModelNotFoundError`/`NetworkError`
   - Derive `task_source` via `_derive_task_source()`
3. **Handle MODEL_BUILD_CONFIGS registry path:**
   - Check `MODEL_BUILD_CONFIGS.get(model_type)` before calling `resolve_io_specs()`
   - If registered config has `input_tensors`, build `TensorInfo` from those directly
   - Otherwise fall through to `resolve_io_specs()`
4. **Replace `resolve_exporter()` with direct `resolve_io_specs()` + `_build_tensor_infos_from_io_specs()`**
5. **Add `_derive_loader_info()`, `_derive_task_source()`, `_derive_exporter_info()` helpers**
6. **Make `resolve_cache()` and `resolve_processor()` conditional on `model_id is not None`**
7. **Delete `detect_task()`, `validate_task()`, `resolve_loader()`, `resolve_exporter()`, `get_build_config()`**
8. **Update tests**
9. **Verify `wmk inspect -m microsoft/resnet-50` output — expect loader class string change**

## Resolved Questions

### Q: MODEL_BUILD_CONFIGS registry path
**Answer:** Concrete step added (Step 3). Check registry first. If registered config has
`input_tensors`, use them. Otherwise call `resolve_io_specs()`. Same priority as
`generate_build_config()` Step 2-3.

### Q: Should resolve_loader_config return source metadata?
**Answer:** No. Source labels are a display concern. Inspect derives them post-hoc by
checking `HF_MODEL_CLASS_MAPPING` and `HF_TASK_DEFAULTS`. Config doesn't need this.

### Q: task_source required field
**Answer:** `_derive_task_source()` added. Checks same dicts as old `detect_task()`
to produce the source label.
