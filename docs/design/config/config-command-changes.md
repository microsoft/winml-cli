# Config Command Console Output — Change Explanation

Every diff block in `modelkit/commands/config.py` explained.

---

## Block 1: Import replacement

```diff
-from rich.console import Console
+from ..utils.console import (
+    get_console,
+    print_command_header,
+    print_error,
+    print_io_specs_detail,
+    print_io_specs_na,
+    print_kv,
+    print_success,
+)

-console = Console(stderr=True)
+console = get_console()
```

**Why**: Instead of creating `Console(stderr=True)` directly, we import shared formatting functions from `modelkit/utils/console.py`. This ensures consistent output style across all commands (config, build, analyze). `get_console()` returns the same `Console(stderr=True)` — just centralized.

**Functions imported**:
- `print_command_header` — `═══` separator + title (used by analyze too)
- `print_kv` — key-value line like `📦 Model: bert-base-uncased`
- `print_io_specs_detail` — aligned I/O tensor table (Input/Output with name, shape, dtype)
- `print_io_specs_na` — "N/A" line for ONNX mode (no I/O specs available)
- `print_error` — `❌` error line with `💡` hint
- `print_success` — `✅` success line

---

## Block 2: Error display before validation error

```diff
     if hf_model is None and model_type is None and model_class is None:
+        print_command_header(console, "\U0001f4cb CONFIG GENERATION")
+        print_error(
+            console,
+            "Missing required input",
+            hint="Provide one of: -m/--model, --model-type, or --model-class",
+        )
+        console.print()
         raise click.UsageError(...)
```

**Why**: Before, the error was just a plain click error message. Now we show the styled header + a hint line before raising, so the user sees:
```
════════════════════════════
📋 CONFIG GENERATION
════════════════════════════
   ❌ Missing required input
   💡 Provide one of: -m/--model, --model-type, or --model-class
```
The `click.UsageError` still raises for proper exit code handling.

---

## Block 3: Store override filenames for later display

```diff
         override = None
+        _override_file: str | None = None
+        _shape_config_file: str | None = None
```

```diff
-            console.print(f"[dim]Loaded overrides from {config_path.name}[/dim]")
+            _override_file = config_path.name
```

```diff
-            console.print(f"[dim]Loaded I/O config from {shape_config_path.name}[/dim]")
+            _shape_config_file = shape_config_path.name
```

**Why**: The old code printed "Loaded overrides from X" immediately when the file was parsed. This broke the visual flow — the message appeared before the command header.

Now we store the filename and display it later in the proper location (after the header, under "Override files" section). The variables are initialized to `None` at the top so they're always defined even when no override file is provided.

---

## Block 4: Remove inline "Generating..." messages

```diff
-            console.print(f"[dim]Generating ONNX build config for {hf_model}...[/dim]")
+            # Header printed after all config generation completes (below)
```

```diff
-            label = hf_model or model_type
-            console.print(f"[dim]Generating config for {label}...[/dim]")
+            _is_onnx_mode = False
```

**Why**: The old code printed "Generating config for X..." before calling `generate_build_config()`. This is a transient message that provides no lasting value — the operation is fast (<3s typically). Instead, we now print a complete summary AFTER generation completes, which is more informative.

---

## Block 5: Collect metadata instead of printing inline

```diff
-            console.print("[green]Generated ONNX build config (export=None)[/green]")
             output_data = config_obj.to_dict()
+            _is_onnx_mode = True
+            _resolved_task = None
+            _resolved_model_class = None
+            _export_cfg = None
+            _n_modules = 0
```

```diff
             if module:
                 configs = result
-                # Apply --no-quant / --no-compile overrides to each config
                 for cfg in configs:
                     _apply_stage_overrides(cfg, ...)
-                console.print(f"[green]Found {len(configs)} submodules matching '{module}'[/green]")
                 output_data = [cfg.to_dict() for cfg in configs]
+                _n_modules = len(configs)
+                config_obj = configs[0] if configs else None
```

```diff
             else:
                 config_obj = result
-                # Apply --no-quant / --no-compile overrides
                 _apply_stage_overrides(config_obj, ...)
-                if not task and not module:
-                    auto_task = config_obj.loader.task
-                    source = model_type or hf_model
-                    console.print(f"[dim]Auto-selected task: {auto_task} (from '{source}')[/dim]")
-                console.print(f"[green]Generated config for task '{config_obj.loader.task}'[/green]")
-                output_data = config_obj.to_dict()
+                output_data = config_obj.to_dict()
+                _n_modules = 0
+
+            _resolved_task = config_obj.loader.task if config_obj else None
+            _resolved_model_class = config_obj.loader.model_class if config_obj else None
+            _export_cfg = config_obj.export if config_obj else None
```

**Why**: The old code printed messages scattered throughout the generation logic — "Generated ONNX build config", "Found N submodules", "Auto-selected task: X", "Generated config for task X". Each was a separate `console.print()` call with inconsistent formatting.

Now we collect all metadata into variables (`_is_onnx_mode`, `_resolved_task`, `_resolved_model_class`, `_export_cfg`, `_n_modules`) and print everything together in a structured block below. This separation of "compute" from "display" makes the code easier to follow.

---

## Block 6: Structured Rich console output

This is the main new section — replaces all the scattered `console.print()` calls.

### 6a: Command header

```python
subtitle = "ONNX mode" if _is_onnx_mode else ("module mode" if module else None)
print_command_header(console, "📋 CONFIG GENERATION", subtitle)
```

Produces: `════════ 📋 CONFIG GENERATION (ONNX mode) ════════`

### 6b: Model identity

```python
model_label = hf_model or model_type or model_class or "?"
print_kv(console, "Model:", model_label, icon="📦")
```

Shows the primary model identifier — whatever the user provided.

### 6c: Model class + Task (or ONNX mode)

```python
if _is_onnx_mode:
    print_kv(console, "Mode:", "Direct ONNX", note="export=None", icon="🔧")
else:
    # Model class before Task (matches build mock convention)
    if module:
        print_kv(console, "Module:", module, icon="🧩")
    elif _resolved_model_class:
        mc_note = None if model_class else "auto-detected"
        print_kv(console, "Model class:", _resolved_model_class, note=mc_note, icon="🧩")
    if _resolved_task:
        task_note = None if task else "auto-detected"
        print_kv(console, "Task:", _resolved_task, note=task_note, icon="🏷️")
```

**Key design**: The `(auto-detected)` suffix only appears when the user did NOT provide `--task` or `--model-class`. When the user explicitly provided it, no suffix — they already know. This is determined by checking the original CLI args (`task`, `model_class`) against the resolved values.

### 6d: Override files

```python
if config_file:
    console.print(f"   📁 Overrides:    {_override_file}  ✓")
if shape_config_file:
    console.print(f"   📁 Shape config: {_shape_config_file}  ✓")
```

Only shown when override files were actually provided. Uses the filenames stored in Block 3.

### 6e: I/O specs (always full detail)

```python
if _is_onnx_mode:
    print_io_specs_na(console)
elif _export_cfg is not None:
    print_io_specs_detail(console, _export_cfg)
```

- ONNX mode: shows "N/A — inferred from ONNX graph at build time"
- HF mode: shows aligned columns with each input/output tensor name, shape, dtype

`print_io_specs_detail` reads `export_config.input_tensors` and `export_config.output_tensors` directly — these are populated by `generate_build_config()` during Optimum OnnxConfig resolution.

### 6f: Resolution (from config object, no hardcoding)

```python
if _ref_config is not None:
    _quant = _ref_config.quant
    _compile = _ref_config.compile

    if _quant is not None or _compile is not None:
        console.print("   ⚙️  Resolution:")

        if _compile and hasattr(_compile, "ep_config") and _compile.ep_config:
            _provider = _compile.ep_config.provider
            from ..utils.constants import normalize_ep_name
            _ep_full = normalize_ep_name(_provider) or _provider
            console.print(f"      EP:         {_ep_full}")

        if _quant:
            console.print(f"      Quant:      {_quant.weight_type}/{_quant.activation_type}")
```

**Critical design decision**: This section reads directly from the config object — `config.compile.ep_config.provider` and `config.quant.weight_type/activation_type`. No reverse mapping, no hardcoded strings, no inference.

- EP display name uses `normalize_ep_name()` from `modelkit/utils/constants.py` (existing API, also used by `wmk analyze`)
- Quant types are displayed exactly as stored in the config
- The section only shows when quant or compile is configured (not shown for default CPU/fp32 builds)

### 6g: Submodule list

```python
if module and not _is_onnx_mode and _n_modules > 0:
    console.print(f"   🧩 Submodules: {_n_modules} matching '{module}'")
```

Only shown in module mode when submodules were found.

---

## Block 7: Output line (single line)

```diff
-            console.print(f"[green]Config saved to:[/green] {output}")
+            suffix = f"  [dim]({_n_modules} submodules)[/dim]" if _n_modules else ""
+            print_success(console, f"Config saved to: [bold]{output}[/bold]{suffix}")
         else:
+            print_success(console, "Config written to stdout")
             # Print to stdout (not stderr where console prints)
             print(config_json)
+
+        console.print()
```

**Why**: Merged the success indicator + save location into a single line with `✅`. Two variants:
- File output: `✅ Config saved to: output.json`
- Stdout: `✅ Config written to stdout` (then JSON follows on stdout)

Module mode appends the count: `✅ Config saved to: output.json  (3 submodules)`

---

## Data Flow Summary

```
CLI args (hf_model, task, model_class, device, precision, ep)
    │
    ▼
generate_build_config()  ← blocking, resolves everything
    │
    ▼
WinMLBuildConfig  ← contains all resolved values
    │
    ├── .loader.task              → "Task: fill-mask (auto-detected)"
    ├── .loader.model_class       → "Model class: BertForMaskedLM (auto-detected)"
    ├── .export.input_tensors     → "Input: input_ids [1, 128] int64"
    ├── .export.output_tensors    → "Output: logits ? ?"  (see Known Limitations #1)
    ├── .compile.ep_config.provider → normalize_ep_name() → "EP: QNNExecutionProvider"
    └── .quant.weight_type/activation_type → "Quant: uint8/uint8"
```

Every displayed value comes from the config object or existing APIs. No hardcoded model-specific logic or mapping tables. Display-only strings like `"auto-detected"`, `"Direct ONNX"`, `"module mode"` are UI labels, not model logic.

---

## Known Limitations

### 1. OutputTensorSpec lacks shape and dtype

`OutputTensorSpec` (from `modelkit/onnx/io.py`) only carries `name` — no `shape` or `dtype`. This is because output shapes are model-dependent and not always known until export time. The display code uses `getattr(t, "shape", None)` with a `"?"` fallback, so output tensors will show:

```
   Output:       logits             ?              ?
```

This is accurate — the config genuinely doesn't know output shapes at generation time.

### 2. Resolution section hidden when quant=None and compile=None

When the user runs `wmk config -m some-model` and the model has no registered build config with quant/compile defaults, the Resolution section is not shown at all. This is intentional — there's nothing to resolve. The generated JSON config will have `quant: null` and `compile: null`.

### 3. ONNX + --module is rejected

`--module` requires a HuggingFace model for submodule discovery via torchinfo. ONNX files don't have a PyTorch module tree. An explicit `UsageError` is raised if both are provided.

### 4. `print_resolution()` removed from console.py

An earlier version of `console.py` had a `print_resolution()` function with hardcoded device/precision display logic (inferred from EP mappings). This was removed because:
- It duplicated `_EP_TO_DEVICE` and `_WEIGHT_TYPE` from `precision.py`
- The config command now reads resolution directly from the config object
- No other command imports it

If the build command needs a resolution display in the future, it should also read from the config object directly.
